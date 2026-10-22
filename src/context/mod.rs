use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

/// Gather file context from the current working directory.
/// Returns a string with directory tree + selected file contents.
pub fn gather_cwd_context(max_depth: usize, max_files: usize) -> Result<String> {
    let cwd = std::env::current_dir().context("getting cwd")?;
    let mut out = String::new();
    out.push_str(&format!("# Project context: {}\n\n", cwd.display()));

    // Directory tree
    out.push_str("## Directory structure\n```\n");
    walk_tree(&cwd, &cwd, 0, max_depth, max_files, &mut out)?;
    out.push_str("```\n\n");

    // Key files
    out.push_str("## Key files\n");
    for name in &["README.md", "Cargo.toml", "package.json", "pyproject.toml"] {
        let path = cwd.join(name);
        if path.exists() {
            if let Ok(text) = std::fs::read_to_string(&path) {
                out.push_str(&format!("\n### {}\n```\n{}\n```\n", name, &text[..text.len().min(2000)]));
            }
        }
    }

    Ok(out)
}

fn walk_tree(
    root: &Path,
    current: &Path,
    depth: usize,
    max_depth: usize,
    max_files: usize,
    out: &mut String,
) -> Result<()> {
    if depth > max_depth {
        return Ok(());
    }
    let prefix = "  ".repeat(depth);
    let entries: Vec<_> = std::fs::read_dir(current)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            let name = e.file_name();
            let s = name.to_string_lossy();
            !s.starts_with('.') && !s.starts_with("target") && s != "node_modules" && s != "dist"
        })
        .take(max_files)
        .collect();

    for entry in entries {
        let name = entry.file_name();
        let meta = entry.metadata()?;
        let rel = entry.path().strip_prefix(root).unwrap_or(&entry.path());
        if meta.is_dir() {
            out.push_str(&format!("{}{}/\n", prefix, name.to_string_lossy()));
            walk_tree(root, &entry.path(), depth + 1, max_depth, max_files, out)?;
        } else {
            out.push_str(&format!("{}{}\n", prefix, name.to_string_lossy()));
        }
    }
    Ok(())
}

/// Read a specific file and format it for inclusion in a prompt.
pub fn read_file_for_context(path: &Path) -> Result<String> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading {}", path.display()))?;
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    Ok(format!(
        "## File: {}\n```\n{}\n```\n",
        name,
        &text[..text.len().min(8000)]
    ))
}

/// Parse `@file.rs` or `@/abs/path` references from user input.
pub fn extract_file_refs(input: &str) -> Vec<PathBuf> {
    let mut refs = Vec::new();
    for word in input.split_whitespace() {
        if let Some(path_str) = word.strip_prefix('@') {
            let path = if path_str.starts_with('/') {
                PathBuf::from(path_str)
            } else {
                std::env::current_dir().unwrap_or_default().join(path_str)
            };
            refs.push(path);
        }
    }
    refs
}

/// Strip `@file` references from input for the actual message.
pub fn strip_file_refs(input: &str) -> String {
    input
        .split_whitespace()
        .filter(|w| !w.starts_with('@'))
        .collect::<Vec<_>>()
        .join(" ")
}
