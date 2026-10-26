use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, HeaderValue, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use okc_core::{
    ChatMessage, LlmProvider, NativeToolDef, NativeToolResult, Role, StreamTx, TokenUsage,
    TurnOutput, UsageSlot,
};

use crate::native_tools::{parse_anthropic_content, to_anthropic_tool_results, to_anthropic_tools};
use crate::streaming::into_sse_stream;

pub struct AnthropicProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    usage: UsageSlot,
}

impl AnthropicProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            usage: UsageSlot::default(),
        }
    }

    fn build_headers(&self) -> Result<HeaderMap> {
        let mut headers = HeaderMap::new();
        headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
        headers.insert(
            "x-api-key",
            HeaderValue::from_str(&self.api_key).context("bad api key")?,
        );
        headers.insert("anthropic-version", HeaderValue::from_static("2023-06-01"));
        Ok(headers)
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

    fn supports_native_tools(&self) -> bool {
        true
    }

    fn last_usage(&self) -> Option<TokenUsage> {
        self.usage.get()
    }

    async fn chat(&self, model: &str, messages: Vec<ChatMessage>) -> Result<String> {
        let (system, user_messages) = extract_system(messages);
        let req = AnthropicRequest {
            model: model.to_string(),
            max_tokens: 8192,
            system,
            messages: user_messages,
            stream: false,
            tools: None,
        };
        let url = format!("{}/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.build_headers()?)
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
        if let Some(u) = &body.usage {
            self.usage
                .store(TokenUsage::new(u.input_tokens, u.output_tokens));
        }
        let text = body
            .content
            .iter()
            .filter_map(|b| {
                b.get("text")
                    .and_then(|t| t.as_str())
                    .map(|s| s.to_string())
            })
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
            tools: None,
        };
        self.do_stream(req, tx).await
    }

    // ─── Native tools ───────────────────────────────────────────────────────

    async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<NativeToolDef>,
    ) -> Result<TurnOutput> {
        let (system, user_messages) = extract_system(messages);
        let req = AnthropicRequest {
            model: model.to_string(),
            max_tokens: 8192,
            system,
            messages: user_messages,
            stream: false,
            tools: Some(to_anthropic_tools(&tools)),
        };
        let url = format!("{}/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.build_headers()?)
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
        if let Some(u) = &body.usage {
            self.usage
                .store(TokenUsage::new(u.input_tokens, u.output_tokens));
        }
        Ok(parse_anthropic_content(&body.content))
    }

    async fn chat_stream_with_tools(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<NativeToolDef>,
        tx: StreamTx,
    ) -> Result<TurnOutput> {
        // Anthropic tool streaming is complex (content_block events).
        // For now, fall back to non-streaming for tool calls.
        let result = self.chat_with_tools(model, messages, tools).await?;
        // Still send text chunks to the UI
        let _ = tx.send(result.text.clone()).await;
        Ok(result)
    }
}

impl AnthropicProvider {
    async fn do_stream(&self, req: AnthropicRequest, tx: StreamTx) -> Result<()> {
        let url = format!("{}/messages", self.base_url);
        let resp = self
            .client
            .post(&url)
            .headers(self.build_headers()?)
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
        .map(|m| {
            let role = if m.role == Role::Assistant {
                "assistant".to_string()
            } else {
                "user".to_string()
            };

            // Structured tool results → user message with `tool_result` blocks.
            if let Some(results) = &m.tool_results {
                let blocks = to_anthropic_tool_results(results);
                return AnthropicMessage {
                    role,
                    content: serde_json::json!(blocks),
                };
            }

            // Structured tool calls → assistant message with `tool_use` blocks.
            if let Some(calls) = &m.tool_calls {
                let mut blocks: Vec<Value> = Vec::new();
                if !m.content.trim().is_empty() {
                    blocks.push(serde_json::json!({ "type": "text", "text": m.content }));
                }
                for call in calls {
                    blocks.push(serde_json::json!({
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }));
                }
                return AnthropicMessage {
                    role,
                    content: serde_json::json!(blocks),
                };
            }

            // Plain text message. Legacy sessions may have serialised content
            // blocks into the string, so keep that path working.
            let content = if m.content.trim().starts_with('[') {
                serde_json::from_str(&m.content).unwrap_or_else(|_| serde_json::json!(m.content))
            } else {
                serde_json::json!(m.content)
            };

            AnthropicMessage { role, content }
        })
        .collect();
    (sys, msgs)
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<AnthropicMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AnthropicMessage {
    role: String,
    content: Value, // Can be String or array of content blocks
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<Value>,
    #[serde(default)]
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    /// Event discriminator ("message_start", "content_block_delta", …).
    #[allow(dead_code)]
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    delta: Option<AnthropicDelta>,
}

#[derive(Debug, Deserialize)]
struct AnthropicDelta {
    /// Delta discriminator ("text_delta", "input_json_delta", …).
    #[allow(dead_code)]
    #[serde(rename = "type")]
    ty: Option<String>,
    text: Option<String>,
}

use futures::StreamExt;

// ═══════════════════════════════════════════════════════════════════════════════
// PUBLIC: Build tool result messages for Anthropic
// ═══════════════════════════════════════════════════════════════════════════════

/// Build an Anthropic user message containing tool_result blocks.
pub fn build_anthropic_tool_result_message(results: &[NativeToolResult]) -> AnthropicMessage {
    let blocks = to_anthropic_tool_results(results);
    AnthropicMessage {
        role: "user".to_string(),
        content: serde_json::json!(blocks),
    }
}
