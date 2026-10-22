// config.rs — load ~/.config/openklawc/config.toml.
//
// Phase 3: just the [mcp.*] section. Other sections ([loader.rules], etc.)
// are read in later phases.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

use crate::mcp::McpConfig;

const DEFAULT_PATH: &str = ".config/openklawc/config.toml";

pub struct AppConfig {
    pub path: PathBuf,
    pub mcp: McpConfig,
}

impl AppConfig {
    /// Load from `~/.config/openklawc/config.toml` by default. The HOME env
    /// var is used; on Windows we'd need USERPROFILE, but openklawc is
    /// Linux/macOS only for now.
    pub fn load_default() -> Result<Self> {
        let home = std::env::var("HOME")
            .context("HOME env var not set; cannot resolve config path")?;
        Self::load_from(&Path::new(&home).join(DEFAULT_PATH))
    }

    pub fn load_from(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config {}", path.display()))?;
        let mcp: McpConfig = toml::from_str(&text)
            .with_context(|| format!("parsing config {}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            mcp,
        })
    }
}
