//! Integration test: boot the application core with a mock provider and no
//! MCP servers, run one turn end to end, and observe the Done update.

use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use okc_app::{App, AppOptions};
use okc_core::{AutonomyLevel, ChatMessage, LlmProvider, StreamTx};
use okc_llm::ProviderRegistry;
use okc_reasoning::AgentUpdate;

struct EchoProvider;

#[async_trait]
impl LlmProvider for EchoProvider {
    fn name(&self) -> &str {
        "mock"
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    async fn chat(&self, _model: &str, _messages: Vec<ChatMessage>) -> Result<String> {
        Ok("echo: hello".to_string())
    }

    async fn chat_stream(
        &self,
        _model: &str,
        _messages: Vec<ChatMessage>,
        _tx: StreamTx,
    ) -> Result<()> {
        Ok(())
    }
}

#[tokio::test]
async fn boots_and_runs_one_turn_with_mock_provider() {
    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock",
        Arc::new(EchoProvider),
        vec!["echo-1".to_string()],
    );

    let (mut app, boot) = App::boot_with(
        AppOptions {
            model: Some("mock/echo-1".to_string()),
            autonomy: Some(AutonomyLevel::Yolo),
            persona: None,
            start_mcp: false,
        },
        Some(registry),
    )
    .await
    .expect("boot must succeed");

    assert_eq!(boot.model, "mock/echo-1");
    assert_eq!(boot.mcp_tools, 0);
    assert!(boot.mcp_failures.is_empty());

    let mut rx = app.submit("hello", Vec::new()).await;
    let mut done_text: Option<String> = None;
    while let Some(update) = rx.recv().await {
        if let AgentUpdate::Done { text, .. } = update {
            done_text = Some(text);
        }
    }
    let text = done_text.expect("turn must finish with Done");
    assert!(
        text.contains("echo"),
        "unexpected final text: {:?}",
        text
    );
}

#[tokio::test]
async fn unknown_explicit_model_is_a_boot_error() {
    let mut registry = ProviderRegistry::new();
    registry.register("mock", Arc::new(EchoProvider), vec!["echo-1".to_string()]);

    let result = App::boot_with(
        AppOptions {
            model: Some("nope/nothing".to_string()),
            start_mcp: false,
            ..Default::default()
        },
        Some(registry),
    )
    .await;

    assert!(result.is_err(), "explicit unknown model must error at boot");
}
