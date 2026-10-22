//! openklawc-config — loads merged configuration from files and env.
//!
//! Config hierarchy (highest priority first):
//!   1. Environment variables (`OKC_*`)
//!   2. Project-local `.openklawc.toml`
//!   3. User config `~/.config/openklawc/config.toml`
//!   4. Built-in defaults

use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Which model to use by default: `provider/model_id`
    pub model: String,
    /// Optional small/fast model for cheap tasks.
    pub small_model: Option<String>,
    /// LLM provider configurations.
    pub provider: HashMap<String, ProviderConfig>,
    /// MCP server definitions.
    pub mcp: HashMap<String, McpServerConfig>,
    /// UI / TUI settings.
    pub ui: UiConfig,
    /// Autonomy default.
    pub autonomy: AutonomyConfig,
    /// RAG settings.
    pub rag: RagConfig,
    /// Session settings.
    pub session: SessionConfig,
    /// Plugin settings.
    pub plugin: PluginConfig,
    /// Built-in tool enablement.
    pub tools: ToolConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub default_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// If true, use SSE transport instead of stdio.
    #[serde(default)]
    pub sse_url: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiConfig {
    #[serde(default = "default_theme")]
    pub theme: String,
    #[serde(default)]
    pub show_reasoning: bool,
    #[serde(default = "true_fn")]
    pub stream_responses: bool,
    #[serde(default = "true_fn")]
    pub vim_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomyConfig {
    #[serde(default)]
    pub default_level: String, // "ask", "read_only", "yolo", "god"
    #[serde(default)]
    pub yolo_tools: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RagConfig {
    #[serde(default = "default_rag_depth")]
    pub index_depth: usize,
    #[serde(default = "default_rag_exts")]
    pub extensions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionConfig {
    #[serde(default = "true_fn")]
    pub auto_save: bool,
    #[serde(default)]
    pub history_dir: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PluginConfig {
    #[serde(default)]
    pub load_paths: Vec<String>,
    #[serde(default)]
    pub disabled: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolConfig {
    #[serde(default = "true_fn")]
    pub shell: bool,
    #[serde(default = "true_fn")]
    pub file_read: bool,
    #[serde(default = "true_fn")]
    pub file_write: bool,
    #[serde(default = "true_fn")]
    pub git: bool,
    #[serde(default = "true_fn")]
    pub search: bool,
}

fn default_theme() -> String { "gold".to_string() }
fn true_fn() -> bool { true }
fn default_rag_depth() -> usize { 3 }
fn default_rag_exts() -> Vec<String> {
    vec!["rs".into(), "py".into(), "js".into(), "ts".into(), "md".into(), "toml".into()]
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model: "ollama/qwen3".to_string(),
            small_model: None,
            provider: HashMap::new(),
            mcp: HashMap::new(),
            ui: UiConfig {
                theme: default_theme(),
                show_reasoning: true,
                stream_responses: true,
                vim_mode: true,
            },
            autonomy: AutonomyConfig {
                default_level: "read_only".to_string(),
                yolo_tools: vec!["search".into(), "read_file".into(), "list_dir".into()],
            },
            rag: RagConfig {
                index_depth: default_rag_depth(),
                extensions: default_rag_exts(),
            },
            session: SessionConfig {
                auto_save: true,
                history_dir: None,
            },
            plugin: PluginConfig {
                load_paths: Vec::new(),
                disabled: Vec::new(),
            },
            tools: ToolConfig::default(),
        }
    }
}

impl Config {
    /// Load from the default paths.
    pub fn load_default() -> Result<Self> {
        let mut cfg = Self::default();

        // User config
        if let Some(user_cfg) = Self::user_config_path() {
            if user_cfg.exists() {
                cfg = cfg.merge_file(&user_cfg)?;
            }
        }

        // Project-local config
        if let Ok(cwd) = std::env::current_dir() {
            let local = cwd.join(".openklawc.toml");
            if local.exists() {
                cfg = cfg.merge_file(&local)?;
            }
        }

        Ok(cfg)
    }

    /// Load from a specific path.
    pub fn load_file(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        let cfg: Config = toml::from_str(&text)
            .with_context(|| format!("parsing config from {}", path.display()))?;
        Ok(cfg)
    }

    fn user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("openklawc").join("config.toml"))
    }

    fn merge_file(mut self, path: &std::path::Path) -> Result<Self> {
        let other = Self::load_file(path)?;
        if !other.model.is_empty() {
            self.model = other.model;
        }
        if other.small_model.is_some() {
            self.small_model = other.small_model;
        }
        self.provider.extend(other.provider);
        self.mcp.extend(other.mcp);
        self.ui = other.ui;
        self.autonomy = other.autonomy;
        self.rag = other.rag;
        self.session = other.session;
        self.plugin = other.plugin;
        self.tools = other.tools;
        Ok(self)
    }
}
