//! Small utilities reused across crates.

use std::path::Path;

/// Truncate a string to a byte limit, respecting char boundaries.
pub fn truncate_bytes(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    match s.char_indices().take_while(|(i, _)| *i < max_bytes).last() {
        Some((idx, c)) => &s[..idx + c.len_utf8()],
        None => "",
    }
}

/// Read a file, but hard-cap at a byte limit.
pub fn read_file_capped(path: &Path, max_bytes: usize) -> anyhow::Result<String> {
    use std::fs;
    let meta = fs::metadata(path)?;
    if meta.len() > max_bytes as u64 {
        let bytes = fs::read(path)?;
        let truncated = truncate_bytes(std::str::from_utf8(&bytes)?, max_bytes);
        Ok(truncated.to_string())
    } else {
        Ok(fs::read_to_string(path)?)
    }
}

/// Extract `@file` references from a query string.
pub fn extract_file_refs(query: &str) -> Vec<String> {
    let mut refs = Vec::new();
    for word in query.split_whitespace() {
        if let Some(stripped) = word.strip_prefix('@') {
            refs.push(stripped.to_string());
        }
    }
    refs
}

/// Strip `@file` references from a query, returning the clean query.
pub fn strip_file_refs(query: &str) -> String {
    query
        .split_whitespace()
        .filter(|w| !w.starts_with('@'))
        .collect::<Vec<_>>()
        .join(" ")
}
