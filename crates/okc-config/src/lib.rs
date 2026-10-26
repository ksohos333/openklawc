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

/// Built-in default provider (MiniMax, Anthropic-compatible wire format).
const DEFAULT_PROVIDER: &str = "minimax";
/// Built-in default model.
const DEFAULT_MODEL: &str = "MiniMax-M3";
/// MiniMax subscription endpoint (same one opencode uses).
const DEFAULT_MINIMAX_BASE_URL: &str = "https://api.minimax.io/anthropic/v1";

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
    /// Orchestrator / agent loop settings.
    pub orchestrator: OrchestratorConfig,
    /// Agent personas / profiles.
    #[serde(default)]
    pub personas: HashMap<String, PersonaConfig>,
    /// Persona activated when `--persona` is not given.
    #[serde(default)]
    pub default_persona: Option<String>,
    /// Per-model token pricing, used to estimate session cost.
    ///
    /// Keyed by the model id, either bare (`MiniMax-M3`) or qualified
    /// (`minimax/MiniMax-M3`). Models without an entry report an unknown cost.
    #[serde(default)]
    pub pricing: HashMap<String, ModelPricing>,
}

/// Token pricing for one model, in US dollars per million tokens.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ModelPricing {
    /// USD per 1M input (prompt) tokens.
    pub input: f64,
    /// USD per 1M output (completion) tokens.
    pub output: f64,
}

impl ModelPricing {
    /// Look up pricing for a model id, trying the qualified id first
    /// (`provider/model`) and then the bare model name.
    pub fn lookup(pricing: &HashMap<String, ModelPricing>, model: &str) -> Option<ModelPricing> {
        if let Some(p) = pricing.get(model) {
            return Some(*p);
        }
        let bare = model.rsplit('/').next().unwrap_or(model);
        pricing.get(bare).copied()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderConfig {
    /// Which wire format to speak: `anthropic`, `openai` or `ollama`.
    /// When omitted, the format is inferred from `base_url`.
    #[serde(default)]
    pub api: Option<String>,
    pub base_url: String,
    pub api_key: String,
    #[serde(default)]
    pub models: Vec<String>,
    #[serde(default)]
    pub default_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpServerConfig {
    /// Executable to spawn. Empty means "use the bundled Rust server of the
    /// same name" (see `okc_mcp::bundled`).
    #[serde(default)]
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// If true, use SSE transport instead of stdio.
    #[serde(default)]
    pub sse_url: Option<String>,
    /// Set to false to keep a server out of the registry even when it is
    /// bundled. Defaults to true.
    #[serde(default = "true_fn")]
    pub enabled: bool,
}

impl McpServerConfig {
    /// A disabled entry — the server is skipped entirely.
    pub fn disabled() -> Self {
        Self {
            command: String::new(),
            args: Vec::new(),
            env: HashMap::new(),
            sse_url: None,
            enabled: false,
        }
    }
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
    /// User-defined custom tools (shell-based).
    #[serde(default)]
    pub custom: Vec<CustomToolDef>,
}

/// One user-defined custom tool.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CustomToolDef {
    pub name: String,
    pub description: String,
    pub command: String,
    #[serde(default)]
    pub cwd: Option<String>,
    #[serde(default = "default_timeout")]
    pub timeout: u64,
    #[serde(default = "true_fn")]
    pub fail_on_error: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestratorConfig {
    /// How to choose tools: auto | required | none
    #[serde(default = "default_tool_choice")]
    pub tool_choice: String,
    /// Maximum iterations per turn.
    #[serde(default = "default_max_iterations")]
    pub max_iterations: usize,
    /// Whether to execute independent tool calls in parallel.
    #[serde(default = "true_fn")]
    pub parallel_tools: bool,
    /// Number of retries for transient failures.
    #[serde(default = "default_retry_count")]
    pub retry_count: u32,
    /// Base delay in milliseconds for exponential backoff.
    #[serde(default = "default_retry_delay_ms")]
    pub retry_delay_ms: u64,
    /// Whether to summarize long tool results before feeding back.
    #[serde(default = "true_fn")]
    pub summarize_results: bool,
    /// Length threshold (chars) above which tool results are summarized.
    #[serde(default = "default_summarize_threshold")]
    pub summarize_threshold: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            tool_choice: default_tool_choice(),
            max_iterations: default_max_iterations(),
            parallel_tools: true,
            retry_count: default_retry_count(),
            retry_delay_ms: default_retry_delay_ms(),
            summarize_results: true,
            summarize_threshold: default_summarize_threshold(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersonaConfig {
    /// Human-readable description.
    pub description: String,
    /// System prompt override.
    pub system_prompt: String,
    /// Which tools this persona is allowed to use (empty = all).
    #[serde(default)]
    pub allowed_tools: Vec<String>,
    /// Override model for this persona (`provider/model`).
    #[serde(default)]
    pub model: Option<String>,
    /// Temperature for this persona (if supported by provider).
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    /// Maximum iterations for this persona.
    #[serde(default)]
    pub max_iterations: Option<usize>,
    /// Whether this persona can delegate to sub-agents.
    #[serde(default = "true_fn")]
    pub can_delegate: bool,
}

impl Default for PersonaConfig {
    fn default() -> Self {
        Self {
            description: String::new(),
            system_prompt: String::new(),
            allowed_tools: Vec::new(),
            model: None,
            temperature: default_temperature(),
            max_iterations: None,
            can_delegate: true,
        }
    }
}

fn default_timeout() -> u64 {
    30
}
fn default_temperature() -> f32 {
    0.7
}
fn default_tool_choice() -> String {
    "auto".to_string()
}
fn default_max_iterations() -> usize {
    10
}
fn default_retry_count() -> u32 {
    3
}
fn default_retry_delay_ms() -> u64 {
    500
}
fn default_summarize_threshold() -> usize {
    4000
}

fn default_theme() -> String {
    "gold".to_string()
}
fn true_fn() -> bool {
    true
}
fn default_rag_depth() -> usize {
    3
}
fn default_rag_exts() -> Vec<String> {
    vec![
        "rs".into(),
        "py".into(),
        "js".into(),
        "ts".into(),
        "md".into(),
        "toml".into(),
    ]
}

impl Default for Config {
    fn default() -> Self {
        let mut provider = HashMap::new();
        provider.insert(
            DEFAULT_PROVIDER.to_string(),
            ProviderConfig {
                api: Some("anthropic".to_string()),
                base_url: DEFAULT_MINIMAX_BASE_URL.to_string(),
                api_key: "{env:MINIMAX_API_KEY}".to_string(),
                models: vec!["MiniMax-M3".to_string()],
                default_model: Some("MiniMax-M3".to_string()),
            },
        );

        Self {
            model: format!("{}/{}", DEFAULT_PROVIDER, DEFAULT_MODEL),
            small_model: None,
            provider,
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
            orchestrator: OrchestratorConfig::default(),
            personas: HashMap::new(),
            default_persona: None,
            pricing: HashMap::new(),
        }
    }
}

impl Config {
    /// Load from the default paths: built-in defaults, then the user config,
    /// then the project-local `.openklawc.toml`.
    ///
    /// Files are deep-merged as TOML tables (project-local wins) so a local
    /// file that only sets `model` no longer wipes out unrelated user
    /// settings.
    pub fn load_default() -> Result<Self> {
        let mut merged =
            toml::Value::try_from(Self::default()).context("serialising built-in defaults")?;

        if let Some(user_cfg) = Self::user_config_path() {
            if user_cfg.exists() {
                merged = merge_tables(merged, Self::load_toml(&user_cfg)?);
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            let local = cwd.join(".openklawc.toml");
            if local.exists() {
                merged = merge_tables(merged, Self::load_toml(&local)?);
            }
        }

        let mut cfg: Config = merged.try_into().context("parsing merged configuration")?;
        cfg.resolve_secrets();
        Ok(cfg)
    }

    /// Load from a specific path.
    pub fn load_file(path: &std::path::Path) -> Result<Self> {
        let mut cfg: Config = Self::load_toml(path)?
            .try_into()
            .with_context(|| format!("parsing config from {}", path.display()))?;
        cfg.resolve_secrets();
        Ok(cfg)
    }

    fn load_toml(path: &std::path::Path) -> Result<toml::Value> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading config from {}", path.display()))?;
        let value: toml::Value = toml::from_str(&text)
            .with_context(|| format!("parsing config from {}", path.display()))?;
        Ok(value)
    }

    fn user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("openklawc").join("config.toml"))
    }

    /// Replace `{file:path}` / `{env:VAR}` references with real values.
    pub fn resolve_secrets(&mut self) {
        for provider in self.provider.values_mut() {
            provider.api_key = resolve_secret(&provider.api_key);
        }
    }
}

/// Deep-merge two TOML values. Scalars from `overlay` win; tables merge
/// recursively.
fn merge_tables(base: toml::Value, overlay: toml::Value) -> toml::Value {
    match (base, overlay) {
        (toml::Value::Table(mut base), toml::Value::Table(overlay)) => {
            for (key, value) in overlay {
                let merged = match base.remove(&key) {
                    Some(existing) => merge_tables(existing, value),
                    None => value,
                };
                base.insert(key, merged);
            }
            toml::Value::Table(base)
        }
        (_, overlay) => overlay,
    }
}

/// Resolve a secret reference:
///   `{file:/path/to/secret}` — read the file, trim whitespace
///   `{env:VAR_NAME}`         — read the environment variable
/// Anything else is returned unchanged.
fn resolve_secret(raw: &str) -> String {
    let trimmed = raw.trim();

    if let Some(path) = trimmed
        .strip_prefix("{file:")
        .and_then(|rest| rest.strip_suffix('}'))
    {
        let expanded = shellexpand::tilde(path).to_string();
        return match std::fs::read_to_string(&expanded) {
            Ok(value) => value.trim().to_string(),
            Err(e) => {
                tracing::warn!("could not read secret file {}: {}", expanded, e);
                String::new()
            }
        };
    }

    if let Some(var) = trimmed
        .strip_prefix("{env:")
        .and_then(|rest| rest.strip_suffix('}'))
    {
        return std::env::var(var).unwrap_or_default();
    }

    trimmed.to_string()
}
