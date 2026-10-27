//! openklawc-llm — multi-provider LLM abstraction with streaming.
//!
//! Supports:
//!   - OpenAI-compatible APIs (OpenAI, Groq, Z.AI, llama-server, etc.)
//!   - Anthropic Claude
//!   - Local llama.cpp via its OpenAI-compatible endpoint

pub mod native_tools;
pub mod provider;
pub mod streaming;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use okc_config::Config;
use okc_core::{ChatMessage, LlmProvider, ModelCapabilities, ModelRef, StreamTx};

pub use provider::anthropic::AnthropicProvider;
pub use provider::openai::OpenAiProvider;

/// Build an OpenAI-compatible provider from config, honouring the optional
/// `native_tools` override (endpoints that accept `tools` but never emit
/// calls — llama-server without `--jinja` — set it to false).
fn openai_provider_from_config(prov_cfg: &okc_config::ProviderConfig) -> OpenAiProvider {
    let mut provider = OpenAiProvider::new(&prov_cfg.base_url, &prov_cfg.api_key);
    if let Some(enabled) = prov_cfg.native_tools {
        provider = provider.with_native_tools(enabled);
    }
    provider
}

/// Registry of all configured LLM providers and their models.
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    models: Vec<ModelRef>,
    default_model: String,
}

impl Default for ProviderRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            models: Vec::new(),
            default_model: String::new(),
        }
    }

    pub fn from_config(cfg: &Config) -> Self {
        let mut registry = Self::new();
        registry.default_model = cfg.model.clone();

        for (name, prov_cfg) in &cfg.provider {
            let provider: Arc<dyn LlmProvider> = match prov_cfg
                .api
                .as_deref()
                .map(|a| a.to_lowercase())
            {
                Some(ref api) if api == "anthropic" => {
                    Arc::new(AnthropicProvider::new(&prov_cfg.base_url, &prov_cfg.api_key))
                }
                Some(ref api) if api == "openai" || api == "openai-compatible" => {
                    Arc::new(openai_provider_from_config(prov_cfg))
                }
                // No explicit API type: infer it from the base URL.
                _ => {
                    if prov_cfg.base_url.contains("anthropic") {
                        Arc::new(AnthropicProvider::new(&prov_cfg.base_url, &prov_cfg.api_key))
                    } else {
                        Arc::new(openai_provider_from_config(prov_cfg))
                    }
                }
            };

            registry.providers.insert(name.clone(), provider.clone());

            for model_id in &prov_cfg.models {
                let display = format!("{}/{}", name, model_id);
                let capabilities = detect_capabilities(model_id);
                registry.models.push(ModelRef {
                    provider: name.clone(),
                    model_id: model_id.clone(),
                    display_name: display.clone(),
                    capabilities,
                });
            }
        }

        // No providers configured → an empty registry. `okc_app::boot`
        // turns that into a clear error instead of silently guessing an
        // endpoint.
        registry
    }

    pub fn default_id(&self) -> &str {
        &self.default_model
    }

    pub fn set_default(&mut self, id: String) {
        self.default_model = id;
    }

    /// Register a provider and its models directly. Used by embedders and
    /// tests that need to inject a provider without going through config.
    pub fn register(&mut self, name: &str, provider: Arc<dyn LlmProvider>, models: Vec<String>) {
        self.providers.insert(name.to_string(), provider);
        for model_id in models {
            let display = format!("{}/{}", name, model_id);
            self.models.push(ModelRef {
                provider: name.to_string(),
                capabilities: detect_capabilities(&model_id),
                display_name: display,
                model_id,
            });
        }
    }

    pub fn get(&self, qualified_id: &str) -> Option<(Arc<dyn LlmProvider>, String)> {
        let (provider_name, model_id) = qualified_id.split_once('/')?;
        let provider = self.providers.get(provider_name)?.clone();
        Some((provider, model_id.to_string()))
    }

    pub fn list(&self) -> &Vec<ModelRef> {
        &self.models
    }

    pub fn resolve(&self, qualified_id: &str) -> Option<ModelRef> {
        self.models.iter().find(|m| {
            format!("{}/{}", m.provider, m.model_id) == qualified_id
        }).cloned()
    }
}

fn detect_capabilities(model_id: &str) -> ModelCapabilities {
    let id = model_id.to_lowercase();
    let reasoning_models = ["reasoning", "deepseek-r1", "qwen3", "minimax", "glm"];
    let tool_models = ["gpt", "claude", "qwen", "minimax", "glm"];
    ModelCapabilities {
        reasoning: reasoning_models.iter().any(|m| id.contains(m))
            || id.contains("o1")
            || id.contains("o3"),
        vision: id.contains("vision") || id.contains("claude-3") || id.contains("gpt-4o"),
        streaming: true,
        function_calling: tool_models.iter().any(|m| id.contains(m)),
        max_tokens: if id.contains("claude-3-opus") { Some(200_000) } else { Some(128_000) },
    }
}

/// Convenience wrapper for making chat requests.
#[derive(Clone)]
pub struct LlmClient;

impl LlmClient {
    pub fn new() -> Self {
        Self
    }

    pub async fn chat(
        &self,
        provider: &Arc<dyn LlmProvider>,
        model_id: &str,
        messages: Vec<ChatMessage>,
    ) -> Result<String> {
        provider.chat(model_id, messages).await
    }

    pub async fn chat_stream(
        &self,
        provider: &Arc<dyn LlmProvider>,
        model_id: &str,
        messages: Vec<ChatMessage>,
        tx: StreamTx,
    ) -> Result<()> {
        provider.chat_stream(model_id, messages, tx).await
    }
}

impl Default for LlmClient {
    fn default() -> Self {
        Self::new()
    }
}
