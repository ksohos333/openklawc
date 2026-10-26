use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use okc_core::{
    ChatMessage, LlmProvider, NativeToolDef, StreamTx, TokenUsage, TurnOutput, UsageSlot,
};

use crate::native_tools::{parse_openai_message, to_openai_tools};
use crate::streaming::into_sse_stream;

pub struct OpenAiProvider {
    client: reqwest::Client,
    base_url: String,
    api_key: String,
    usage: UsageSlot,
}

impl OpenAiProvider {
    pub fn new(base_url: &str, api_key: &str) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.to_string(),
            usage: UsageSlot::default(),
        }
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    fn name(&self) -> &str {
        "openai"
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
        let req = OpenAiRequest {
            model: model.to_string(),
            messages: to_openai_messages(messages),
            stream: false,
            tools: None,
        };
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let body: OpenAiResponse = resp.json().await.context("parsing response")?;
        if let Some(u) = &body.usage {
            self.usage
                .store(TokenUsage::new(u.prompt_tokens, u.completion_tokens));
        }
        let text = body
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .and_then(|m| m.content)
            .unwrap_or_default();
        Ok(text)
    }

    async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tx: StreamTx,
    ) -> Result<()> {
        let req = OpenAiRequest {
            model: model.to_string(),
            messages: to_openai_messages(messages),
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
        let req = OpenAiRequest {
            model: model.to_string(),
            messages: to_openai_messages(messages),
            stream: false,
            tools: Some(to_openai_tools(&tools)),
        };
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
            .json(&req)
            .send()
            .await
            .with_context(|| format!("POST {}", url))?;

        let status = resp.status();
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("{} from {}: {}", status, url, text);
        }

        let body: OpenAiResponse = resp.json().await.context("parsing response")?;
        if let Some(u) = &body.usage {
            self.usage
                .store(TokenUsage::new(u.prompt_tokens, u.completion_tokens));
        }
        let message = body
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message)
            .map(|m| serde_json::to_value(m).unwrap_or_default())
            .unwrap_or_default();

        Ok(parse_openai_message(&message))
    }

    async fn chat_stream_with_tools(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<NativeToolDef>,
        tx: StreamTx,
    ) -> Result<TurnOutput> {
        let req = OpenAiRequest {
            model: model.to_string(),
            messages: to_openai_messages(messages),
            stream: true,
            tools: Some(to_openai_tools(&tools)),
        };
        // Streaming with tools: we need to accumulate the full response to parse tool_calls.
        // OpenAI streams partial tool_calls. For now, we collect and parse at the end.
        let (collector_tx, mut collector_rx) = tokio::sync::mpsc::channel::<String>(256);
        let stream_task = self.do_stream(req, collector_tx);

        let mut full = String::new();
        while let Some(chunk) = collector_rx.recv().await {
            let _ = tx.send(chunk.clone()).await;
            full.push_str(&chunk);
        }
        stream_task.await?;

        // Re-parse the accumulated response as a full completion
        // This is a workaround; OpenAI tool streaming is complex.
        // Better: use non-streaming for tool calls.
        Ok(TurnOutput {
            text: full,
            tool_calls: Vec::new(),
            usage: self.last_usage(),
        })
    }
}

impl OpenAiProvider {
    async fn do_stream(&self, req: OpenAiRequest, tx: StreamTx) -> Result<()> {
        let url = format!("{}/chat/completions", self.base_url);
        let resp = self
            .client
            .post(&url)
            .header(CONTENT_TYPE, "application/json")
            .header(AUTHORIZATION, format!("Bearer {}", self.api_key))
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
            if event.data == "[DONE]" {
                break;
            }
            let chunk: OpenAiStreamChunk = match serde_json::from_str(&event.data) {
                Ok(c) => c,
                Err(_) => continue,
            };
            if let Some(choice) = chunk.choices.first() {
                if let Some(delta) = &choice.delta.content {
                    let _ = tx.send(delta.clone()).await;
                }
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct OpenAiRequest {
    model: String,
    messages: Vec<OpenAiMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Value>,
}

#[derive(Debug, Serialize, Deserialize, Clone, Default)]
struct OpenAiMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

/// Translate our message history into OpenAI's wire format.
///
/// Tool results become one `role: "tool"` message per result (carrying the
/// originating `tool_call_id`), and assistant tool calls are rebuilt as the
/// native `tool_calls` array so the API can match them to their results.
fn to_openai_messages(messages: Vec<ChatMessage>) -> Vec<OpenAiMessage> {
    let mut out = Vec::new();

    for msg in messages {
        if let Some(results) = msg.tool_results {
            for result in results {
                out.push(OpenAiMessage {
                    role: "tool".to_string(),
                    content: Some(if result.is_error {
                        format!("Error: {}", result.content)
                    } else {
                        result.content
                    }),
                    tool_calls: Vec::new(),
                    tool_call_id: Some(result.call_id),
                });
            }
            continue;
        }

        let mut tool_calls = Vec::new();
        if let Some(calls) = msg.tool_calls {
            for call in calls {
                let arguments = match call.arguments.as_str() {
                    Some(raw) => raw.to_string(),
                    None => call.arguments.to_string(),
                };
                tool_calls.push(serde_json::json!({
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.name,
                        "arguments": arguments,
                    }
                }));
            }
        }

        out.push(OpenAiMessage {
            role: msg.role.to_string(),
            content: Some(msg.content).filter(|s| !s.is_empty()),
            tool_calls,
            tool_call_id: msg.name,
        });
    }

    out
}

#[derive(Debug, Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    #[serde(default)]
    usage: Option<OpenAiUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiUsage {
    #[serde(default)]
    prompt_tokens: u64,
    #[serde(default)]
    completion_tokens: u64,
}

#[derive(Debug, Deserialize)]
struct OpenAiChoice {
    message: Option<OpenAiMessage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Vec<OpenAiStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiMessage,
}

use futures::StreamExt;
