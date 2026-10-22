use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Simple keyword-based "RAG" — searches file contents for matching terms.
/// Replace with vector search (sqlite-vss, pgvector, etc.) when ready.
pub struct SimpleRag {
    /// filepath -> (line_number, line_text)
    index: HashMap<PathBuf, Vec<(usize, String)>>,
}

impl SimpleRag {
    pub fn new() -> Self {
        Self {
            index: HashMap::new(),
        }
    }

    /// Index all text files in a directory (recursively, up to depth).
    pub fn index_dir(&mut self, dir: &Path, max_depth: usize) -> Result<()> {
        self.index.clear();
        self.walk_and_index(dir, dir, 0, max_depth)?;
        Ok(())
    }

    fn walk_and_index(&mut self, root: &Path, current: &Path, depth: usize, max_depth: usize) -> Result<()> {
        if depth > max_depth {
            return Ok(());
        }
        let entries = std::fs::read_dir(current).context("reading directory")?;
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            let meta = entry.metadata()?;
            if meta.is_dir() {
                self.walk_and_index(root, &path, depth + 1, max_depth)?;
            } else if Self::is_text_file(&path) {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    let lines: Vec<(usize, String)> = text
                        .lines()
                        .enumerate()
                        .map(|(i, l)| (i + 1, l.to_string()))
                        .collect();
                    self.index.insert(path, lines);
                }
            }
        }
        Ok(())
    }

    fn is_text_file(path: &Path) -> bool {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        matches!(ext, "rs" | "ts" | "js" | "py" | "md" | "txt" | "toml" | "json" | "yaml" | "yml" | "html" | "css" | "sh")
    }

    /// Search for files containing ALL query terms.
    pub fn search(&self, query: &str) -> Vec<SearchResult> {
        let terms: Vec<String> = query
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        let mut results = Vec::new();
        for (path, lines) in &self.index {
            let mut matches = Vec::new();
            for (line_no, line_text) in lines {
                let lower = line_text.to_lowercase();
                if terms.iter().all(|t| lower.contains(t)) {
                    matches.push((*line_no, line_text.clone()));
                }
            }
            if !matches.is_empty() {
                results.push(SearchResult {
                    path: path.clone(),
                    matches,
                });
            }
        }
        // Sort by match count (most relevant first)
        results.sort_by_key(|r| std::cmp::Reverse(r.matches.len()));
        results
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

#[derive(Debug)]
pub struct SearchResult {
    pub path: PathBuf,
    pub matches: Vec<(usize, String)>,
}

impl std::fmt::Display for SearchResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "## {}", self.path.display())?;
        for (line_no, text) in &self.matches {
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                writeln!(f, "  L{}: {}", line_no, &trimmed[..trimmed.len().min(120)])?;
            }
        }
        Ok(())
    }
}
