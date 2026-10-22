use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;

/// Parsed opencode.jsonc — the subset we care about for the LLM client.
#[derive(Debug, Clone, Deserialize)]
pub struct OpencodeConfig {
    pub model: String,
    #[serde(rename = "small_model")]
    pub small_model: Option<String>,
    pub provider: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderConfig {
    pub npm: String,
    pub name: String,
    pub options: ProviderOptions,
    pub models: HashMap<String, ModelConfig>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProviderOptions {
    #[serde(rename = "baseURL")]
    pub base_url: String,
    #[serde(rename = "apiKey")]
    pub api_key: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    #[serde(default)]
    pub reasoning: bool,
    #[serde(default, rename = "attachment")]
    pub vision: bool,
}

impl OpencodeConfig {
    /// Load from `~/.config/opencode/opencode.jsonc`.
    pub fn load_default() -> Result<Self> {
        let home = std::env::var("HOME").context("HOME not set")?;
        let path = Path::new(&home).join(".config/opencode/opencode.jsonc");
        Self::load_from(&path)
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading {}", path.display()))?;
        let stripped = strip_jsonc_comments(&text);
        let mut cfg: OpencodeConfig = serde_json::from_str(&stripped)
            .with_context(|| format!("parsing JSONC in {}", path.display()))?;

        // Resolve API key placeholders like {file:...} and {env:...}
        for (_, prov) in cfg.provider.iter_mut() {
            prov.options.api_key = resolve_key(&prov.options.api_key)?;
        }

        Ok(cfg)
    }

    /// Get the default model id (provider/model) from the config.
    pub fn default_model_id(&self) -> String {
        self.model.clone()
    }
}

fn strip_comments_only(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '/' {
            match chars.peek() {
                Some('/') => {
                    chars.next();
                    while let Some(ch) = chars.next() {
                        if ch == '\n' {
                            out.push('\n');
                            break;
                        }
                    }
                }
                Some('*') => {
                    chars.next();
                    let mut prev = ' ';
                    while let Some(ch) = chars.next() {
                        if prev == '*' && ch == '/' {
                            break;
                        }
                        prev = ch;
                    }
                }
                _ => out.push(c),
            }
        } else if c == '"' {
            out.push(c);
            while let Some(ch) = chars.next() {
                out.push(ch);
                if ch == '\\' {
                    if let Some(esc) = chars.next() {
                        out.push(esc);
                    }
                } else if ch == '"' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Strip // comments, /* */ comments, and trailing commas from JSONC
/// so that serde_json can parse the result.
fn strip_jsonc_comments(text: &str) -> String {
    let without_comments = strip_comments_only(text);
    strip_trailing_commas(&without_comments)
}

/// Strip trailing commas before ] or }.
fn strip_trailing_commas(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut remove = vec![false; chars.len()];

    let mut i = 0;
    while i < chars.len() {
        if chars[i] == ',' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_whitespace() {
                j += 1;
            }
            if j < chars.len() && (chars[j] == ']' || chars[j] == '}') {
                remove[i] = true;
            }
        }
        i += 1;
    }

    let mut out = String::with_capacity(text.len());
    for (idx, c) in chars.iter().enumerate() {
        if !remove[idx] {
            out.push(*c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_comments_and_trailing_commas() {
        let input = r#"{
  "model": "test", // a comment
  "provider": {
    "local": {
      "options": {
        "baseURL": "http://x",
        "apiKey": "secret",
      }, // trailing comma
    },
  },
}"#;
        let after_comments = strip_comments_only(input);
        let result = strip_trailing_commas(&after_comments);
        let parsed: serde_json::Value = serde_json::from_str(&result).expect("should parse");
        assert_eq!(parsed["model"], "test");
        assert_eq!(parsed["provider"]["local"]["options"]["apiKey"], "secret");
    }

    #[test]
    fn test_load_real_config() {
        let home = std::env::var("HOME").unwrap();
        let path = std::path::Path::new(&home).join(".config/opencode/opencode.jsonc");
        if !path.exists() {
            return;
        }
        let cfg = OpencodeConfig::load_from(&path).expect("should load real config");
        assert!(!cfg.provider.is_empty(), "should have providers");
    }
}

fn resolve_key(key: &str) -> Result<String> {
    if let Some(path) = key.strip_prefix("{file:") {
        let path = path.strip_suffix("}").unwrap_or(path);
        let expanded = shellexpand::tilde(path);
        let contents = std::fs::read_to_string(expanded.as_ref())
            .with_context(|| format!("reading API key from {}", expanded))?;
        Ok(contents.trim().to_string())
    } else if let Some(var) = key.strip_prefix("{env:") {
        let var = var.strip_suffix("}").unwrap_or(var);
        // If env var not set, return the placeholder so the config still loads
        Ok(std::env::var(var).unwrap_or_else(|_| key.to_string()))
    } else {
        Ok(key.to_string())
    }
}
