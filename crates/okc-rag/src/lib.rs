//! openklawc-rag — simple RAG / context indexing.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use regex::Regex;

#[derive(Debug, Clone)]
pub struct Chunk {
    pub file: PathBuf,
    pub text: String,
    pub line_start: usize,
}

pub struct SimpleRag {
    chunks: Vec<Chunk>,
    index: HashMap<String, Vec<usize>>, // word -> chunk indices
}

impl SimpleRag {
    pub fn new() -> Self {
        Self {
            chunks: Vec::new(),
            index: HashMap::new(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn index_dir(&mut self, dir: &Path, max_depth: usize, exts: &[String]) -> Result<()> {
        self.chunks.clear();
        self.index.clear();

        for entry in walkdir::WalkDir::new(dir).max_depth(max_depth) {
            let entry = entry?;
            if !entry.file_type().is_file() {
                continue;
            }
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
            if !exts.iter().any(|e| e == ext) {
                continue;
            }
            if let Ok(text) = std::fs::read_to_string(path) {
                self.index_file(path, &text);
            }
        }
        Ok(())
    }

    fn index_file(&mut self, path: &Path, text: &str) {
        let chunk_size = 20;
        let lines: Vec<&str> = text.lines().collect();
        for (i, window) in lines.windows(chunk_size).enumerate() {
            let chunk = Chunk {
                file: path.to_path_buf(),
                text: window.join("\n"),
                line_start: i + 1,
            };
            let idx = self.chunks.len();
            self.chunks.push(chunk);

            // Very simple word index
            let words: Vec<String> = window
                .join(" ")
                .split_whitespace()
                .map(|w| w.to_lowercase())
                .collect();
            for word in words {
                self.index.entry(word).or_default().push(idx);
            }
        }
    }

    pub fn search(&self, query: &str) -> Vec<&Chunk> {
        let query_words: Vec<String> = query.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
        let mut scores: HashMap<usize, usize> = HashMap::new();

        for word in &query_words {
            if let Some(indices) = self.index.get(word) {
                for &idx in indices {
                    *scores.entry(idx).or_insert(0) += 1;
                }
            }
        }

        let mut scored: Vec<(usize, usize)> = scores.into_iter().collect();
        scored.sort_by(|a, b| b.1.cmp(&a.1));

        scored
            .into_iter()
            .take(10)
            .filter_map(|(idx, _)| self.chunks.get(idx))
            .collect()
    }
}

impl Default for SimpleRag {
    fn default() -> Self {
        Self::new()
    }
}
