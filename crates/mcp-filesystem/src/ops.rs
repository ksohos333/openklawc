//! File operations for the filesystem MCP server.

use std::fs;
use std::io::{Read, Seek, SeekFrom};
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use chrono::{DateTime, Local};
use glob::{MatchOptions, Pattern};
use serde_json::{json, Value};
use similar::TextDiff;

/// One replacement applied by `edit_file`.
#[derive(Debug, Clone)]
pub struct Edit {
    pub old_text: String,
    pub new_text: String,
}

/// Human readable byte size, matching the upstream server's formatting.
pub fn format_size(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    if bytes == 0 {
        return "0 B".to_string();
    }
    let mut value = bytes as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit < UNITS.len() - 1 {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", value, UNITS[unit])
    }
}

/// Normalise CRLF to LF.
pub fn normalize_line_endings(text: &str) -> String {
    text.replace("\r\n", "\n")
}

/// The line ending a file predominantly uses.
fn detect_line_ending(text: &str) -> &'static str {
    let crlf = text.matches("\r\n").count();
    let lf = text.matches('\n').count() - crlf;
    if crlf > lf {
        "\r\n"
    } else {
        "\n"
    }
}

/// Read a text file, optionally limited to the first or last `n` lines.
pub fn read_text_file(path: &Path, head: Option<usize>, tail: Option<usize>) -> Result<String> {
    if head.is_some() && tail.is_some() {
        return Err(anyhow!(
            "Cannot specify both head and tail parameters simultaneously"
        ));
    }
    if let Some(n) = tail {
        return tail_file(path, n);
    }
    if let Some(n) = head {
        return head_file(path, n);
    }
    fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))
}

/// Last `num_lines` lines of a file, read backwards in 1 KiB chunks.
fn tail_file(path: &Path, num_lines: usize) -> Result<String> {
    if num_lines == 0 {
        return Ok(String::new());
    }
    let mut file = fs::File::open(path).with_context(|| format!("reading {}", path.display()))?;
    let size = file.metadata()?.len();
    if size == 0 {
        return Ok(String::new());
    }

    const CHUNK: usize = 1024;
    let mut position = size;
    let mut chunks: Vec<Vec<u8>> = Vec::new();
    let mut newlines = 0usize;

    while position > 0 && newlines < num_lines {
        let take = std::cmp::min(CHUNK as u64, position) as usize;
        position -= take as u64;
        let mut buffer = vec![0u8; take];
        file.seek(SeekFrom::Start(position))?;
        file.read_exact(&mut buffer)?;
        newlines += buffer.iter().filter(|b| **b == b'\n').count();
        chunks.push(buffer);
    }

    chunks.reverse();
    let text = normalize_line_endings(&String::from_utf8_lossy(&chunks.concat()));
    let lines: Vec<&str> = text.split('\n').collect();
    let start = lines.len().saturating_sub(num_lines);
    Ok(lines[start..].join("\n"))
}

/// First `num_lines` lines of a file.
fn head_file(path: &Path, num_lines: usize) -> Result<String> {
    if num_lines == 0 {
        return Ok(String::new());
    }
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    Ok(text
        .split('\n')
        .take(num_lines)
        .collect::<Vec<_>>()
        .join("\n"))
}

/// Guess a MIME type from the file extension.
pub fn mime_type_for(path: &Path) -> &'static str {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "webp" => "image/webp",
        "bmp" => "image/bmp",
        "svg" => "image/svg+xml",
        "mp3" => "audio/mpeg",
        "wav" => "audio/wav",
        "ogg" => "audio/ogg",
        "flac" => "audio/flac",
        _ => "application/octet-stream",
    }
}

/// Read a binary file and turn it into an MCP content block.
pub fn read_media_file(path: &Path) -> Result<Value> {
    let bytes = fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    let mime = mime_type_for(path);
    let data = base64::engine::general_purpose::STANDARD.encode(&bytes);

    if mime.starts_with("image/") {
        Ok(json!({ "type": "image", "data": data, "mimeType": mime }))
    } else if mime.starts_with("audio/") {
        Ok(json!({ "type": "audio", "data": data, "mimeType": mime }))
    } else {
        Ok(json!({
            "type": "resource",
            "resource": {
                "uri": format!("file://{}", path.display()),
                "mimeType": mime,
                "blob": data,
            }
        }))
    }
}

/// Write a file, creating or truncating it.
pub fn write_file(path: &Path, content: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() && !parent.exists() {
            return Err(anyhow!(
                "Parent directory does not exist: {}",
                parent.display()
            ));
        }
    }
    fs::write(path, content).with_context(|| format!("writing {}", path.display()))
}

/// The indentation unit used by a set of lines.
fn indent_unit(lines: &[&str]) -> String {
    let mut narrowest = usize::MAX;
    for line in lines {
        let indent = leading_whitespace(line);
        if indent.is_empty() || line.trim().is_empty() {
            continue;
        }
        if indent.contains('\t') {
            return "\t".to_string();
        }
        narrowest = narrowest.min(indent.len());
    }
    if narrowest == usize::MAX {
        String::new()
    } else {
        " ".repeat(narrowest)
    }
}

fn leading_whitespace(line: &str) -> &str {
    let trimmed = line.trim_start();
    &line[..line.len() - trimmed.len()]
}

/// Re-indent a replacement line so it matches the indentation of the file.
fn reindent_line(
    line: &str,
    file_line: &str,
    old_line: &str,
    edit_unit: &str,
    file_unit: &str,
) -> String {
    let body = line.trim_start();
    if body.is_empty() {
        return line.to_string();
    }
    let file_indent = leading_whitespace(file_line);
    let new_indent = leading_whitespace(line);
    let delta = new_indent.len() as isize - leading_whitespace(old_line).len() as isize;

    if !edit_unit.is_empty() && !file_unit.is_empty() && (edit_unit == "\t") != (file_unit == "\t")
    {
        let levels = delta / edit_unit.len() as isize;
        let indent = if levels >= 0 {
            format!("{}{}", file_indent, file_unit.repeat(levels as usize))
        } else {
            let cut = (levels.unsigned_abs() as usize) * file_unit.len();
            file_indent[..file_indent.len().saturating_sub(cut)].to_string()
        };
        return format!("{}{}", indent, body);
    }

    let indent = if delta >= 0 {
        let keep = delta as usize;
        let start = new_indent.len().saturating_sub(keep);
        format!("{}{}", file_indent, &new_indent[start..])
    } else {
        let cut = delta.unsigned_abs() as usize;
        file_indent[..file_indent.len().saturating_sub(cut)].to_string()
    };
    format!("{}{}", indent, body)
}

/// Apply a list of edits to the content of a file.
///
/// Returns the new content. `edits` are applied in order; each one either
/// matches exactly or, failing that, matches line-by-line ignoring leading
/// and trailing whitespace.
pub fn apply_edits(content: &str, edits: &[Edit]) -> Result<String> {
    let mut modified = content.to_string();

    for edit in edits {
        let normalized_old = normalize_line_endings(&edit.old_text);
        let normalized_new = normalize_line_endings(&edit.new_text);

        if modified.contains(&normalized_old) {
            modified = modified.replacen(&normalized_old, &normalized_new, 1);
            continue;
        }

        let old_lines: Vec<&str> = normalized_old.split('\n').collect();
        let mut content_lines: Vec<String> = modified.split('\n').map(|s| s.to_string()).collect();
        let file_unit = indent_unit(&content_lines.iter().map(|s| s.as_str()).collect::<Vec<_>>());

        let mut found: Option<usize> = None;
        if old_lines.len() <= content_lines.len() {
            'outer: for i in 0..=(content_lines.len() - old_lines.len()) {
                for (j, old_line) in old_lines.iter().enumerate() {
                    if old_line.trim() != content_lines[i + j].trim() {
                        continue 'outer;
                    }
                }
                found = Some(i);
                break;
            }
        }

        let Some(index) = found else {
            return Err(anyhow!(
                "Could not find exact match for edit:\n{}",
                edit.old_text
            ));
        };

        let new_lines: Vec<&str> = normalized_new.split('\n').collect();
        let edit_unit = indent_unit(
            &old_lines
                .iter()
                .copied()
                .chain(new_lines.iter().copied())
                .collect::<Vec<_>>(),
        );

        let replacement: Vec<String> = new_lines
            .iter()
            .enumerate()
            .map(|(j, line)| {
                let k = std::cmp::min(j, old_lines.len() - 1);
                reindent_line(
                    line,
                    &content_lines[index + k],
                    old_lines[k],
                    &edit_unit,
                    &file_unit,
                )
            })
            .collect();

        content_lines.splice(index..index + old_lines.len(), replacement);
        modified = content_lines.join("\n");
    }

    Ok(modified)
}

/// Build a git-style unified diff between two versions of a file.
pub fn unified_diff(original: &str, modified: &str, file_path: &str) -> String {
    let normalized_original = normalize_line_endings(original);
    let normalized_modified = normalize_line_endings(modified);
    let diff = TextDiff::from_lines(normalized_original.as_str(), normalized_modified.as_str());

    let mut unified = diff.unified_diff();
    unified.context_radius(3).header(
        &format!("{}\toriginal", file_path),
        &format!("{}\tmodified", file_path),
    );

    let mut out = String::new();
    out.push_str(&format!("Index: {}\n", file_path));
    out.push_str("===================================================================\n");
    out.push_str(&unified.to_string());
    out
}

/// Edit a file in place and return the diff. With `dry_run` the file is left
/// untouched.
pub fn edit_file(path: &Path, edits: &[Edit], dry_run: bool) -> Result<String> {
    let raw = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let eol = detect_line_ending(&raw);
    let content = normalize_line_endings(&raw);
    let modified = apply_edits(&content, edits)?;
    let diff = unified_diff(&content, &modified, &path.to_string_lossy());

    if !dry_run {
        let to_write = if eol == "\r\n" {
            modified.replace('\n', "\r\n")
        } else {
            modified
        };
        fs::write(path, to_write).with_context(|| format!("writing {}", path.display()))?;
    }

    Ok(diff)
}

/// Create a directory and any missing parents.
pub fn create_directory(path: &Path) -> Result<()> {
    fs::create_dir_all(path).with_context(|| format!("creating directory {}", path.display()))
}

/// List a directory as `[DIR] name` / `[FILE] name` lines.
pub fn list_directory(path: &Path) -> Result<String> {
    let entries = fs::read_dir(path).with_context(|| format!("listing {}", path.display()))?;
    let mut names: Vec<(bool, String)> = Vec::new();
    for entry in entries {
        let entry = entry?;
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        names.push((is_dir, entry.file_name().to_string_lossy().to_string()));
    }
    names.sort_by(|a, b| a.1.cmp(&b.1));
    Ok(names
        .iter()
        .map(|(is_dir, name)| {
            if *is_dir {
                format!("[DIR] {}", name)
            } else {
                format!("[FILE] {}", name)
            }
        })
        .collect::<Vec<_>>()
        .join("\n"))
}

/// List a directory with sizes and a summary.
pub fn list_directory_with_sizes(path: &Path, sort_by_size: bool) -> Result<String> {
    let entries = fs::read_dir(path).with_context(|| format!("listing {}", path.display()))?;
    let mut detailed: Vec<(String, bool, u64)> = Vec::new();

    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let size = fs::metadata(entry.path()).map(|m| m.len()).unwrap_or(0);
        detailed.push((name, is_dir, size));
    }

    if sort_by_size {
        detailed.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    } else {
        detailed.sort_by(|a, b| a.0.cmp(&b.0));
    }

    let formatted: Vec<String> = detailed
        .iter()
        .map(|(name, is_dir, size)| {
            let padded_name = format!("{:<30}", name);
            if *is_dir {
                format!("[DIR] {} ", padded_name)
            } else {
                format!("[FILE] {} {:>10}", padded_name, format_size(*size))
            }
        })
        .collect();

    let total_files = detailed.iter().filter(|e| !e.1).count();
    let total_dirs = detailed.iter().filter(|e| e.1).count();
    let total_size: u64 = detailed.iter().filter(|e| !e.1).map(|e| e.2).sum();

    let mut lines = formatted;
    lines.push(String::new());
    lines.push(format!(
        "Total: {} files, {} directories",
        total_files, total_dirs
    ));
    lines.push(format!("Combined size: {}", format_size(total_size)));
    Ok(lines.join("\n"))
}

/// Build a recursive JSON tree of a directory.
pub fn directory_tree(root: &Path, exclude_patterns: &[String]) -> Result<Value> {
    let root_display = root.to_path_buf();
    let tree = build_tree(root, &root_display, exclude_patterns)?;
    Ok(tree)
}

fn build_tree(current: &Path, root: &Path, exclude_patterns: &[String]) -> Result<Value> {
    let entries =
        fs::read_dir(current).with_context(|| format!("listing {}", current.display()))?;
    let mut result: Vec<Value> = Vec::new();

    for entry in entries {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        let path = entry.path();
        let is_dir = entry.file_type().map(|t| t.is_dir()).unwrap_or(false);
        let relative = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        if should_exclude(&relative, &name, is_dir, exclude_patterns) {
            continue;
        }

        if is_dir {
            let children = build_tree(&path, root, exclude_patterns)?;
            result.push(json!({
                "name": name,
                "type": "directory",
                "children": children,
            }));
        } else {
            result.push(json!({ "name": name, "type": "file" }));
        }
    }

    Ok(Value::Array(result))
}

/// Match the upstream server's exclude-pattern semantics.
fn should_exclude(relative: &str, name: &str, is_dir: bool, exclude_patterns: &[String]) -> bool {
    exclude_patterns.iter().any(|pattern| {
        if pattern.contains('*') {
            return glob_match(relative, pattern);
        }
        glob_match(relative, pattern)
            || glob_match(relative, &format!("**/{}", pattern))
            || glob_match(relative, &format!("**/{}/**", pattern))
            || (!is_dir && name == pattern)
    })
}

/// Glob match a relative path, treating dot-files as ordinary files.
pub fn glob_match(relative: &str, pattern: &str) -> bool {
    let Ok(pattern) = Pattern::new(pattern) else {
        return false;
    };
    let options = MatchOptions {
        case_sensitive: true,
        require_literal_separator: true,
        require_literal_leading_dot: false,
    };
    pattern.matches_path_with(Path::new(relative), options)
}

/// Move or rename a file. Fails when the destination already exists.
pub fn move_file(source: &Path, destination: &Path) -> Result<()> {
    match fs::symlink_metadata(destination) {
        Ok(_) => Err(anyhow!(
            "Destination already exists: {}",
            destination.display()
        )),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(anyhow!("{}", e)),
    }?;
    fs::rename(source, destination)
        .with_context(|| format!("moving {} to {}", source.display(), destination.display()))
}

/// Recursively search a directory for paths matching a glob.
pub fn search_files(
    root: &Path,
    pattern: &str,
    exclude_patterns: &[String],
) -> Result<Vec<PathBuf>> {
    let mut results = Vec::new();
    for entry in walkdir::WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(e) => e,
            Err(_) => continue,
        };
        let full_path = entry.path().to_path_buf();
        let relative = full_path
            .strip_prefix(root)
            .unwrap_or(&full_path)
            .to_string_lossy()
            .replace('\\', "/");

        if exclude_patterns
            .iter()
            .any(|exclude| glob_match(&relative, exclude))
        {
            continue;
        }

        if glob_match(&relative, pattern) {
            results.push(full_path);
        }
    }
    Ok(results)
}

/// Metadata of a file or directory, formatted like the upstream server.
pub fn get_file_info(path: &Path) -> Result<String> {
    let meta = fs::metadata(path).with_context(|| format!("stat {}", path.display()))?;
    let modified =
        DateTime::<Local>::from(meta.modified().unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let accessed =
        DateTime::<Local>::from(meta.accessed().unwrap_or(std::time::SystemTime::UNIX_EPOCH));
    let created =
        DateTime::<Local>::from(meta.created().unwrap_or(std::time::SystemTime::UNIX_EPOCH));

    let text = format!(
        "size: {}\ncreated: {}\nmodified: {}\naccessed: {}\nisDirectory: {}\nisFile: {}\npermissions: {:03o}",
        meta.len(),
        format_time(&created),
        format_time(&modified),
        format_time(&accessed),
        meta.is_dir(),
        meta.is_file(),
        meta.permissions().mode() & 0o777,
    );
    Ok(text)
}

/// Format a timestamp the way JavaScript's `Date.toString()` does.
fn format_time(time: &DateTime<Local>) -> String {
    time.format("%a %b %d %Y %H:%M:%S GMT%z (%Z)").to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_are_formatted_like_upstream() {
        assert_eq!(format_size(0), "0 B");
        assert_eq!(format_size(512), "512 B");
        assert_eq!(format_size(1024), "1.00 KB");
        assert_eq!(format_size(1048576), "1.00 MB");
        assert_eq!(format_size(1073741824), "1.00 GB");
    }

    #[test]
    fn exact_match_edit() {
        let content = "hello world\nsecond line\n";
        let edits = vec![Edit {
            old_text: "world".into(),
            new_text: "there".into(),
        }];
        let out = apply_edits(content, &edits).unwrap();
        assert_eq!(out, "hello there\nsecond line\n");
    }

    #[test]
    fn whitespace_tolerant_edit() {
        let content = "fn main() {\n    println!(\"x\");\n}\n";
        let edits = vec![Edit {
            old_text: "println!(\"x\");".into(),
            new_text: "println!(\"y\");".into(),
        }];
        let out = apply_edits(content, &edits).unwrap();
        assert!(out.contains("println!(\"y\");"));
        assert!(out.contains("    println!"), "indentation preserved");
    }

    #[test]
    fn missing_match_is_an_error() {
        let edits = vec![Edit {
            old_text: "nope".into(),
            new_text: "yes".into(),
        }];
        let err = apply_edits("abc\n", &edits).unwrap_err();
        assert!(err.to_string().contains("Could not find exact match"));
    }

    #[test]
    fn glob_matches_nested_paths() {
        assert!(glob_match("src/main.rs", "**/*.rs"));
        assert!(glob_match("main.rs", "*.rs"));
        assert!(!glob_match("src/main.rs", "*.rs"));
        assert!(glob_match(".hidden/file.rs", "**/*.rs"));
    }
}
