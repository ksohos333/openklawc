use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};

use super::provider::{ProviderKind, QualifiedModel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ─── OpenAI-compatible API types ──────────────────────────────────────────

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<ChatMessage>,
    stream: bool,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: Option<ChatMessage>,
}

// ─── Anthropic API types ──────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
    role: Option<String>,
    #[serde(rename = "stop_reason")]
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    ty: String,
    text: Option<String>,
}

pub struct LlmClient {
    http: reqwest::Client,
}

impl LlmClient {
    pub fn new() -> Self {
        Self {
            http: reqwest::Client::new(),
        }
    }

    /// Send a chat completion, dispatching to the correct API format.
    pub async fn chat(&self, model: &QualifiedModel, messages: Vec<ChatMessage>) -> Result<String> {
        match model.kind {
            ProviderKind::Anthropic => self.chat_anthropic(model, messages).await,
            _ => self.chat_openai(model, messages).await,
        }
    }

    // ─── OpenAI-compatible ─────────────────────────────────────────────────

    async fn chat_openai(&self, model: &QualifiedModel, messages: Vec<ChatMessage>) -> Result<String> {
        let req = OpenAIRequest {
            model: model.model_id.clone(),
            messages,
            stream: false,
        };
        let url = format!("{}/chat/completions", model.base_url.trim_end_matches('/'));
        let resp = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", model.api_key))
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let body: OpenAIResponse = resp.json().await.context("parsing OpenAI response")?;
        let text = body
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .map(|m| m.content)
            .unwrap_or_default();
        Ok(text)
    }

    // ─── Anthropic ─────────────────────────────────────────────────────────

    async fn chat_anthropic(&self, model: &QualifiedModel, messages: Vec<ChatMessage>) -> Result<String> {
        // Extract system message if present
        let (system, user_messages): (Option<String>, Vec<AnthropicMessage>) = {
            let sys = messages
                .iter()
                .find(|m| m.role == "system")
                .map(|m| m.content.clone());
            let msgs: Vec<_> = messages
                .iter()
                .filter(|m| m.role != "system")
                .map(|m| AnthropicMessage {
                    role: if m.role == "assistant" { "assistant".to_string() } else { "user".to_string() },
                    content: m.content.clone(),
                })
                .collect();
            (sys, msgs)
        };

        let req = AnthropicRequest {
            model: model.model_id.clone(),
            messages: user_messages,
            system,
            max_tokens: 4096,
        };

        // Anthropic baseURL may already end with /v1; append /messages
        let base = model.base_url.trim_end_matches('/');
        let url = format!("{}/messages", base);

        let resp = self
            .http
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header("x-api-key", &model.api_key)
            .header("anthropic-version", "2023-06-01")
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let body: AnthropicResponse = resp.json().await.context("parsing Anthropic response")?;
        let text: String = body
            .content
            .into_iter()
            .filter(|b| b.ty == "text")
            .filter_map(|b| b.text)
            .collect();
        Ok(text)
    }
}
