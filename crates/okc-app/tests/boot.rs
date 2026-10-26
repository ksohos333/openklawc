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
            audit_db: None,
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

#[tokio::test]
async fn policy_is_active_at_the_booted_executor_seam() {    let mut registry = ProviderRegistry::new();
    registry.register("mock", Arc::new(EchoProvider), vec!["echo-1".to_string()]);

    // God mode: the most permissive autonomy. Policy must still hold.
    let (app, _boot) = App::boot_with(
        AppOptions {
            model: Some("mock/echo-1".to_string()),
            autonomy: Some(AutonomyLevel::God),
            start_mcp: false,
            ..Default::default()
        },
        Some(registry),
    )
    .await
    .expect("boot");

    use okc_core::ToolCall;

    // Protected path — denied even in god mode.
    let call = ToolCall {
        id: okc_core::Id::new(),
        name: "write_file".to_string(),
        arguments: serde_json::json!({"path": "~/.ssh/authorized_keys", "content": "x"}),
    };
    let out = app.tools.execute_tool(&call).await.unwrap();
    assert!(
        !out.success,
        "protected path must be denied even in god mode"
    );
    assert!(out.error.unwrap().contains("protected"));

    // Sandbox — outside allowed roots denied.
    let call = ToolCall {
        id: okc_core::Id::new(),
        name: "read_file".to_string(),
        arguments: serde_json::json!({"path": "/etc/passwd"}),
    };
    let out = app.tools.execute_tool(&call).await.unwrap();
    assert!(!out.success, "out-of-root read must be denied");
    assert!(out.error.unwrap().contains("allowed roots"));
}

#[tokio::test]
async fn audit_trail_records_events_for_the_bound_session() {
    let dir = std::env::temp_dir().join(format!("okc-app-audit-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let audit_db = dir.join("audit.db");
    let _ = std::fs::remove_file(&audit_db);

    let mut registry = ProviderRegistry::new();
    registry.register("mock", Arc::new(EchoProvider), vec!["echo-1".to_string()]);

    let (app, _boot) = App::boot_with(
        AppOptions {
            model: Some("mock/echo-1".to_string()),
            start_mcp: false,
            audit_db: Some(audit_db.clone()),
            ..Default::default()
        },
        Some(registry),
    )
    .await
    .expect("boot");

    // An event on the bus must land in the audit trail.
    app.event_bus.publish(okc_core::Event::UserMessage {
        session_id: okc_core::Id("sess-audit-1".to_string()),
        content: "audit probe".to_string(),
    });

    // The subscriber runs on its own thread — poll briefly.
    let mut events = Vec::new();
    for _ in 0..40 {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let store = okc_session::SessionStore::open(&audit_db).expect("open audit db");
        events = store.audit_for_session("sess-audit-1", 10).unwrap();
        if !events.is_empty() {
            break;
        }
    }
    assert!(
        !events.is_empty(),
        "audit subscriber must record the event"
    );
    assert_eq!(events[0].kind, "UserMessage");

    let _ = std::fs::remove_file(&audit_db);
    let _ = std::fs::remove_dir(&dir);
}
