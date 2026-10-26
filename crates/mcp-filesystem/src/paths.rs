//! Path validation for the filesystem MCP server.
//!
//! Mirrors the security model of `@modelcontextprotocol/server-filesystem`:
//! every path is checked against an allow-list, symlinks are resolved before
//! the check so a link cannot escape the sandbox, and new files must live in
//! an existing parent directory.

use std::fs;
use std::path::{Component, Path, PathBuf};

use anyhow::{anyhow, Result};

/// Expand a leading `~` to the user's home directory.
pub fn expand_home(input: &str) -> String {
    if input == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().to_string();
        }
        return input.to_string();
    }
    if let Some(rest) = input.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().to_string();
        }
    }
    input.to_string()
}

/// Lexically resolve `.` and `..` without touching the filesystem.
pub fn normalize(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Prefix(p) => out.push(p.as_os_str()),
            Component::RootDir => out.push(Component::RootDir.as_os_str()),
            Component::CurDir => {}
            Component::ParentDir => {
                if !out.pop() {
                    out.push("..");
                }
            }
            Component::Normal(part) => out.push(part),
        }
    }
    if out.as_os_str().is_empty() {
        out.push(".");
    }
    out
}

/// True when `path` is `allowed` itself or lives below it.
pub fn is_path_within_allowed_directories(path: &Path, allowed: &[PathBuf]) -> bool {
    if allowed.is_empty() {
        return false;
    }
    let normalized_path = normalize(path);
    allowed.iter().any(|dir| {
        let normalized_dir = normalize(dir);
        normalized_path == normalized_dir || normalized_path.starts_with(&normalized_dir)
    })
}

/// Resolve a requested path against the allow-list.
///
/// * `~` is expanded,
/// * relative paths are resolved against the allowed directories,
/// * the result must stay inside an allowed directory,
/// * symlinks are resolved and re-checked,
/// * a path that does not exist yet must have an existing parent.
pub fn validate_path(requested: &str, allowed: &[PathBuf]) -> Result<PathBuf> {
    let expanded = expand_home(requested);

    let absolute = if Path::new(&expanded).is_absolute() {
        PathBuf::from(&expanded)
    } else {
        resolve_relative(&expanded, allowed)
    };

    let normalized = normalize(&absolute);

    if !is_path_within_allowed_directories(&normalized, allowed) {
        let dirs = allowed
            .iter()
            .map(|d| d.to_string_lossy().to_string())
            .collect::<Vec<_>>()
            .join(", ");
        return Err(anyhow!(
            "Access denied - path outside allowed directories: {} not in {}",
            normalized.display(),
            dirs
        ));
    }

    match std::fs::canonicalize(&normalized) {
        Ok(real) => {
            if !is_path_within_allowed_directories(&real, allowed) {
                let dirs = allowed
                    .iter()
                    .map(|d| d.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(anyhow!(
                    "Access denied - symlink target outside allowed directories: {} not in {}",
                    real.display(),
                    dirs
                ));
            }
            Ok(real)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            resolve_missing_tail(&normalized, allowed)
        }
        Err(e) => Err(anyhow!("{}", e)),
    }
}

/// Resolve a path whose trailing components do not exist yet.
///
/// The nearest existing ancestor is canonicalised and re-checked against the
/// allow-list, then the missing tail is appended. This is what lets
/// `create_directory` build nested directories.
fn resolve_missing_tail(normalized: &Path, allowed: &[PathBuf]) -> Result<PathBuf> {
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut current = normalized.to_path_buf();

    loop {
        if fs::symlink_metadata(&current).is_ok() {
            let real = fs::canonicalize(&current)
                .map_err(|_| anyhow!("Parent directory does not exist: {}", current.display()))?;
            if !is_path_within_allowed_directories(&real, allowed) {
                let dirs = allowed
                    .iter()
                    .map(|d| d.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                return Err(anyhow!(
                    "Access denied - symlink target outside allowed directories: {} not in {}",
                    real.display(),
                    dirs
                ));
            }
            let mut result = real;
            for part in tail.iter().rev() {
                result.push(part);
            }
            return Ok(result);
        }

        let Some(name) = current.file_name().map(|n| n.to_os_string()) else {
            return Err(anyhow!(
                "Parent directory does not exist: {}",
                current.display()
            ));
        };
        tail.push(name);
        let Some(parent) = current.parent().map(|p| p.to_path_buf()) else {
            return Err(anyhow!(
                "Parent directory does not exist: {}",
                current.display()
            ));
        };
        current = parent;
    }
}

/// Resolve a relative path against the allowed directories.
fn resolve_relative(relative: &str, allowed: &[PathBuf]) -> PathBuf {
    if allowed.is_empty() {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
        return normalize(&cwd.join(relative));
    }
    for dir in allowed {
        let candidate = normalize(&dir.join(relative));
        if is_path_within_allowed_directories(&candidate, allowed) {
            return candidate;
        }
    }
    normalize(&allowed[0].join(relative))
}

/// Collect the allowed directories from the command line and the environment.
///
/// Positional arguments win; `OPENCLAWC_FS_DIRS` (colon separated) adds more.
/// With neither, the current working directory is allowed.
pub fn allowed_directories_from_cli() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Some(env_dirs) = std::env::var_os("OPENCLAWC_FS_DIRS") {
        for entry in std::env::split_paths(&env_dirs) {
            if !entry.as_os_str().is_empty() {
                push_unique(&mut dirs, entry);
            }
        }
    }

    for arg in std::env::args().skip(1) {
        if arg.starts_with('-') {
            continue;
        }
        push_unique(&mut dirs, PathBuf::from(expand_home(&arg)));
    }

    if dirs.is_empty() {
        if let Ok(cwd) = std::env::current_dir() {
            push_unique(&mut dirs, cwd);
        }
    }

    dirs
}

/// Add a directory to the allow-list unless it is already there.
fn push_unique(dirs: &mut Vec<PathBuf>, candidate: PathBuf) {
    let normalized = normalize(&candidate);
    if !dirs.contains(&normalized) {
        dirs.push(normalized);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn within_allowed() {
        let allowed = vec![PathBuf::from("/tmp/allowed")];
        assert!(is_path_within_allowed_directories(
            &PathBuf::from("/tmp/allowed/file.txt"),
            &allowed
        ));
        assert!(is_path_within_allowed_directories(
            &PathBuf::from("/tmp/allowed"),
            &allowed
        ));
        assert!(!is_path_within_allowed_directories(
            &PathBuf::from("/tmp/allowed-evil/file.txt"),
            &allowed
        ));
        assert!(!is_path_within_allowed_directories(
            &PathBuf::from("/etc/passwd"),
            &allowed
        ));
    }

    #[test]
    fn normalize_resolves_dot_dot() {
        assert_eq!(
            normalize(Path::new("/tmp/allowed/../other")),
            PathBuf::from("/tmp/other")
        );
    }
}
