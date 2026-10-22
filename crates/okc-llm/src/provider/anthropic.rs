use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::Sender;

use okc_core::{ChatMessage, LlmProvider, Role, StreamTx};

use crate::streaming::{into_sse_stream, SseEvent};

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
}

impl AnthropicProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
        }
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    async fn chat(&self, model: &str, messages: Vec<ChatMessage>) -> Result<String> {
        let (system, user_messages) = extract_system(messages);
        let req = AnthropicRequest {
            model: model.to_string(),
            max_tokens: 8192,
            system,
            messages: user_messages,
            stream: false,
        };
        let url = format!("{}/messages", self.base_url);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).context("bad api key")?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let body: AnthropicResponse = resp.json().await.context("parsing response")?;
        let text = body
            .content
            .into_iter()
            .filter_map(|b| b.text)
            .collect::<String>();
        Ok(text)
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tx: StreamTx,
    ) -> Result<()> {
        let (system, user_messages) = extract_system(messages);
        let req = AnthropicRequest {
            model: model.to_string(),
            max_tokens: 8192,
            system,
            messages: user_messages,
            stream: true,
        };
        let url = format!("{}/messages", self.base_url);
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).context("bad api key")?,
        );
        headers.insert(
            "anthropic-version",
            HeaderValue::from_static("2023-06-01"),
        );

        let resp = self
            .client
            .post(&url)
            .headers(headers)
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let mut stream = into_sse_stream(resp);
        while let Some(event) = stream.next().await {
            let event = event?;
            let chunk: AnthropicStreamEvent = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if let Some(delta) = &chunk.delta {
                if let Some(text) = &delta.text {
                    let _ = tx.send(text.clone()).await;
                }
            }
        }
        Ok(())
    }
}

fn extract_system(messages: Vec<ChatMessage>) -> (Option<String>, Vec<AnthropicMessage>) {
    let sys = messages
        .iter()
        .find(|m| m.role == Role::System)
        .map(|m| m.content.clone());
    let msgs: Vec<_> = messages
        .iter()
        .filter(|m| m.role != Role::System)
        .map(|m| AnthropicMessage {
            role: if m.role == Role::Assistant {
                "assistant".to_string()
            } else {
                "user".to_string()
            },
            content: m.content.clone(),
        })
        .collect();
    (sys, msgs)
}

// ─── Types ───────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    content: Vec<AnthropicContentBlock>,
}

#[derive(Debug, Deserialize)]
struct AnthropicContentBlock {
    #[serde(rename = "type")]
    ty: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    #[serde(rename = "type")]
    ty: String,
    text: Option<String>,
}

use futures::StreamExt;
