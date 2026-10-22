//! openklawc-llm — multi-provider LLM abstraction with streaming.
//!
//! Supports:
//!   - OpenAI-compatible APIs (OpenAI, Groq, etc.)
//!   - Anthropic Claude
//!   - Ollama (local)

pub mod native_tools;
pub mod provider;
pub mod streaming;

use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use okc_config::Config;
use okc_core::{ChatMessage, LlmProvider, ModelCapabilities, ModelRef, StreamTx};
use tokio::sync::RwLock;

pub use provider::anthropic::AnthropicProvider;
pub use provider::ollama::OllamaProvider;
pub use provider::openai::OpenAiProvider;

/// Registry of all configured LLM providers and their models.
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn LlmProvider>>,
    models: Vec<ModelRef>,
    default_model: String,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
            models: Vec::new(),
            default_model: "ollama/qwen3".to_string(),
        }
    }

    pub fn from_config(cfg: &Config) -> Self {
        let mut registry = Self::new();
        registry.default_model = cfg.model.clone();

        for (name, prov_cfg) in &cfg.provider {
            let provider: Arc<dyn LlmProvider> = if prov_cfg.base_url.contains("anthropic") {
                Arc::new(AnthropicProvider::new(&prov_cfg.base_url, &prov_cfg.api_key))
            } else if prov_cfg.base_url.contains("ollama") || prov_cfg.base_url.contains(":11434") {
                Arc::new(OllamaProvider::new(&prov_cfg.base_url))
            } else {
                Arc::new(OpenAiProvider::new(&prov_cfg.base_url, &prov_cfg.api_key))
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

        // If no providers configured, add a default Ollama
        if registry.providers.is_empty() {
            let ollama = Arc::new(OllamaProvider::new("http://localhost:11434"));
            registry.providers.insert("ollama".into(), ollama);
            registry.models.push(ModelRef {
                provider: "ollama".into(),
                model_id: "qwen3".into(),
                display_name: "ollama/qwen3".into(),
                capabilities: detect_capabilities("qwen3"),
            });
        }

        registry
    }

    pub fn default_id(&self) -> &str {
        &self.default_model
    }

    pub fn set_default(&mut self, id: String) {
        self.default_model = id;
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
    ModelCapabilities {
        reasoning: id.contains("o1") || id.contains("o3") || id.contains("reasoning") || id.contains("deepseek-r1") || id.contains("qwen3"),
        vision: id.contains("vision") || id.contains("claude-3") || id.contains("gpt-4o"),
        streaming: true,
        function_calling: id.contains("gpt") || id.contains("claude") || id.contains("qwen"),
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
