//! openklawc-reasoning — the full agent loop with native tool support.
//!
//! Hermes-inspired architecture: plan → execute → reflect → respond.
//! Reasoning is ALWAYS visible. Autonomy controls confirmation.
//!
//! Native tool calling is supported for OpenAI and Anthropic providers.
//! Fallback to text-based JSON parsing for all other providers.

pub mod orchestrator;
pub mod prompt;

use std::sync::Arc;

use okc_core::{
    AutonomyLevel, ChatMessage, EventBus, ReasoningStep, ToolExecutor,
};
use okc_llm::LlmClient;
use tokio::sync::mpsc;

use crate::orchestrator::Orchestrator;

const DEFAULT_MAX_ITERATIONS: usize = 10;

/// All updates an agent turn can produce. Sent to the TUI via mpsc.
#[derive(Debug, Clone)]
pub enum AgentUpdate {
    /// Streaming text chunk from the LLM.
    TextChunk(String),
    /// A reasoning / planning step.
    Reasoning { kind: String, content: String },
    /// A tool was invoked.
    ToolCall { name: String, arguments: serde_json::Value },
    /// A tool returned a result.
    ToolResult { name: String, success: bool, result: serde_json::Value, elapsed_ms: u64 },
    /// Turn complete.
    Done { text: String, messages: Vec<ChatMessage> },
    /// Turn failed.
    Error(String),
}

/// The agent. Owns the LLM client, tools, and autonomy level.
pub struct Agent {
    pub llm_client: LlmClient,
    pub provider: Arc<dyn okc_core::LlmProvider>,
    pub model_id: String,
    pub tools: Arc<dyn ToolExecutor>,
    pub autonomy: AutonomyLevel,
    pub event_bus: EventBus,
    pub max_iterations: usize,
    pub streaming: bool,
    pub reasoning_log: Vec<ReasoningStep>,
    pub update_tx: Option<mpsc::Sender<AgentUpdate>>,
}

impl Agent {
    pub fn new(
        llm_client: LlmClient,
        provider: Arc<dyn okc_core::LlmProvider>,
        model_id: String,
        tools: Arc<dyn ToolExecutor>,
        autonomy: AutonomyLevel,
        event_bus: EventBus,
    ) -> Self {
        Self {
            llm_client,
            provider,
            model_id,
            tools,
            autonomy,
            event_bus,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            streaming: true,
            reasoning_log: Vec::new(),
            update_tx: None,
        }
    }

    /// Run one complete turn.
    pub async fn run_turn(&mut self, user_input: &str, messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        let mut orch = Orchestrator {
            llm_client: self.llm_client.clone(),
            provider: self.provider.clone(),
            model_id: self.model_id.clone(),
            tools: self.tools.clone(),
            autonomy: self.autonomy,
            event_bus: self.event_bus.clone(),
            max_iterations: self.max_iterations,
            update_tx: self.update_tx.clone(),
            reasoning_log: Vec::new(),
        };

        let new_messages = orch.run_turn(user_input, messages).await;

        // Sync reasoning log back
        self.reasoning_log = orch.reasoning_log;
        new_messages
    }
}
