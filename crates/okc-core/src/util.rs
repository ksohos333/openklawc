//! Small utilities reused across crates.

use std::path::Path;

/// Patterns that match conversational filler phrases at the start of lines.
/// These waste tokens and add no value. Used by `filter_filler_phrases`.
const FILLER_PATTERNS: &[&str] = &[
    "let me",
    "Let me",
    "I'll",
    "I will",
    "I am going to",
    "I'm going to",
    "Now I will",
    "Now I'll",
];

/// Remove leading filler phrases from each line of text.
///
/// This acts as a post-processing safety net: even if the prompt's
/// Communication Discipline instruction fails to prevent filler phrases,
/// they are stripped before the text reaches the user.
///
/// # Example
/// ```
/// use okc_core::util::filter_filler_phrases;
/// let clean = filter_filler_phrases("Let me run the tests...\nI'll fix this now.");
/// assert_eq!(clean, "run the tests...\nfix this now.");
/// ```
pub fn filter_filler_phrases(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for line in text.lines() {
        let trimmed = line.trim_start();
        let mut cleaned = trimmed;
        for pattern in FILLER_PATTERNS {
            if let Some(stripped) = cleaned.strip_prefix(pattern) {
                cleaned = stripped.trim_start();
                // Also strip a trailing comma if present
                cleaned = cleaned.strip_prefix(',').unwrap_or(cleaned).trim_start();
                break; // Only strip one pattern per line
            }
        }
        if !result.is_empty() {
            result.push('\n');
        }
        // Preserve original indentation by calculating how much was trimmed
        let leading_spaces = line.len() - trimmed.len();
        result.push_str(&" ".repeat(leading_spaces));
        result.push_str(cleaned);
    }
    result
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_filler_phrases_basic() {
        let input = "Let me run the tests...\nI'll fix this now.";
        let expected = "run the tests...\nfix this now.";
        assert_eq!(filter_filler_phrases(input), expected);
    }

    #[test]
    fn test_filter_filler_phrases_with_comma() {
        // Strips the first filler pattern found; "I will" that follows is not stripped
        // because it's part of a valid sentence, not standalone filler.
        let input = "Let me, check the logs.\nNow I'll, start the server.";
        let expected = "check the logs.\nstart the server.";
        assert_eq!(filter_filler_phrases(input), expected);
    }

    #[test]
    fn test_filter_filler_phrases_passthrough() {
        let input = "Running tests now...\nBuild succeeded.";
        assert_eq!(filter_filler_phrases(input), input);
    }

    #[test]
    fn test_filter_filler_phrases_preserves_indent() {
        let input = "  Let me check this.\n    I'll update the file.";
        let expected = "  check this.\n    update the file.";
        assert_eq!(filter_filler_phrases(input), expected);
    }
}
