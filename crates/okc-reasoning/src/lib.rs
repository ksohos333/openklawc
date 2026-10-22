//! openklawc-reasoning — the autonomy engine.
//!
//! Inspired by Hermes (Nous Research): the agent loop uses three phases:
//!   1. PLAN — determine what needs to be done
//!   2. EXECUTE — run tools, collect observations
//!   3. REFLECT — evaluate results, decide next step or respond
//!
//! The reasoning is ALWAYS visible to the user (unlike hidden CoT).
//! The autonomy level controls whether confirmation is required.

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use okc_core::{
    AutonomyLevel, ChatMessage, Event, EventBus, ReasoningKind, ReasoningStep, Role,
    ToolCall, ToolExecutor, ToolOutcome, TurnPhase, TurnState,
};
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

pub mod prompt;

/// The autonomy engine that drives the agent loop.
pub struct AutonomyEngine {
    pub level: AutonomyLevel,
    event_bus: EventBus,
}

impl AutonomyEngine {
    pub fn new(level: AutonomyLevel, event_bus: EventBus) -> Self {
        Self { level, event_bus }
    }

    /// Start a new turn with a user message.
    pub async fn run_turn(
        &self,
        user_message: &str,
        tools: &Arc<dyn ToolExecutor>,
        messages: &mut Vec<ChatMessage>,
    ) -> Result<String> {
        let mut turn = TurnState::new();
        turn.phase = TurnPhase::Planning;

        // 1. PLAN
        let step = ReasoningStep {
            timestamp: Utc::now(),
            kind: ReasoningKind::Plan,
            content: format!("User asked: {}", user_message),
            tool_call: None,
            tool_result: None,
        };
        self.publish_reasoning(&step);
        turn.reasoning_log.push(step);

        // For this v0.2 implementation, we do a simplified loop:
        // We let the LLM decide if it wants to use tools by including tool schemas in the system prompt.
        // In a full implementation, we'd parse function calls from the LLM response.

        // For now, return a placeholder indicating the engine is running
        turn.phase = TurnPhase::Responding;
        Ok("Autonomy engine is active. In a full implementation, this would parse tool calls from the LLM response, execute them with autonomy checks, and build a reply.".to_string())
    }

    /// Check if a tool call needs user confirmation based on autonomy level.
    pub fn needs_confirm(&self, tool_name: &str) -> bool {
        let is_mutation = matches!(tool_name, "write_file" | "shell" | "git");
        self.level.needs_confirm(tool_name, is_mutation)
    }

    /// Execute a tool call, respecting autonomy.
    pub async fn execute_with_consent(
        &self,
        call: &ToolCall,
        tools: &Arc<dyn ToolExecutor>,
    ) -> Result<ToolOutcome> {
        if self.needs_confirm(&call.name) {
            // In the TUI, this would pause and ask the user.
            // For now, we log and proceed (simulating yolo).
            warn!("TOOL REQUIRES CONFIRM (autonomy={:?}): {} {:?}", self.level, call.name, call.arguments);
        }

        let step = ReasoningStep {
            timestamp: Utc::now(),
            kind: ReasoningKind::Execute,
            content: format!("Executing {} ...", call.name),
            tool_call: Some(call.clone()),
            tool_result: None,
        };
        self.publish_reasoning(&step);

        let result = tools.execute_tool(call).await?;

        let step = ReasoningStep {
            timestamp: Utc::now(),
            kind: if result.success { ReasoningKind::Observe } else { ReasoningKind::Error },
            content: format!("Result: {}", result.result),
            tool_call: Some(call.clone()),
            tool_result: Some(result.clone()),
        };
        self.publish_reasoning(&step);

        Ok(result)
    }

    fn publish_reasoning(&self, step: &ReasoningStep) {
        self.event_bus.publish(Event::Reasoning {
            session_id: okc_core::Id::new(),
            step: step.clone(),
        });
    }
}
