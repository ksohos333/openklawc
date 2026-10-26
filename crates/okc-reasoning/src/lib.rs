//! openklawc-reasoning — the full agent loop with native tool support.
//!
//! Hermes-inspired architecture: plan → execute → reflect → respond.
//! Reasoning is ALWAYS visible. Autonomy controls confirmation.
//!
//! Native tool calling is supported for OpenAI and Anthropic providers.
//! Fallback to text-based JSON parsing for all other providers.

pub mod context;
pub mod delegate;
pub mod orchestrator;
pub mod persona;
pub mod prompt;
pub mod workflow;

use std::sync::Arc;

use okc_core::{AutonomyLevel, ChatMessage, EventBus, ReasoningStep, ToolExecutor};
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
    ToolCall {
        name: String,
        arguments: serde_json::Value,
    },
    /// A tool returned a result.
    ToolResult {
        name: String,
        success: bool,
        result: serde_json::Value,
        elapsed_ms: u64,
    },
    /// Turn complete.
    Done {
        text: String,
        messages: Vec<ChatMessage>,
        /// Token usage for the turn, when the provider reported it.
        usage: Option<okc_core::TokenUsage>,
    },
    /// Turn failed.
    Error(String),
}

/// A tool call waiting for the user's approval.
///
/// The agent sends one of these over `Agent::confirm_tx` and waits for the
/// matching response. `respond` carries the answer: `true` allows the call,
/// `false` denies it.
#[derive(Debug)]
pub struct ConfirmRequest {
    /// Provider-assigned call id, echoed back in the tool result.
    pub id: String,
    /// Tool name.
    pub name: String,
    /// Tool arguments, for display.
    pub arguments: serde_json::Value,
    /// Send `true` to allow the call, `false` to deny it.
    pub respond: tokio::sync::oneshot::Sender<bool>,
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
    /// Optional fast decision layer (Laya / Ollaya) for plan validation.
    pub decision_orchestrator: Option<okc_decision::DecisionOrchestrator>,
    /// Runtime orchestrator configuration.
    pub config: crate::orchestrator::OrchestratorConfig,
    /// Active persona (if any).
    pub persona: Option<crate::persona::Persona>,
    /// Channel used to ask the user to approve a tool call.
    ///
    /// When `None` there is nobody to ask, so calls that need approval are
    /// denied rather than silently executed.
    pub confirm_tx: Option<mpsc::Sender<ConfirmRequest>>,
    /// Extra tool names classified as mutating (custom tools, configured
    /// MCP tools). Extends `okc_core::is_mutation_tool`, so these require
    /// approval outside `yolo`/`god`.
    pub mutation_extra: std::collections::HashSet<String>,
    /// Session this agent is bound to — used to attribute audit events
    /// (user messages, reasoning steps) to a session.
    pub session_id: Option<String>,
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
            decision_orchestrator: None,
            config: crate::orchestrator::OrchestratorConfig::default(),
            persona: None,
            confirm_tx: None,
            mutation_extra: std::collections::HashSet::new(),
            session_id: None,
        }
    }

    /// Attach a decision orchestrator for fast plan validation.
    pub fn with_decision_layer(mut self, orchestrator: okc_decision::DecisionOrchestrator) -> Self {
        self.decision_orchestrator = Some(orchestrator);
        self
    }

    /// Apply runtime configuration.
    pub fn with_config(mut self, config: crate::orchestrator::OrchestratorConfig) -> Self {
        self.max_iterations = config.max_iterations;
        self.config = config;
        self
    }

    /// Set an active persona.
    pub fn with_persona(mut self, persona: crate::persona::Persona) -> Self {
        self.persona = Some(persona);
        self
    }

    /// Extend the mutation classification with extra tool names (custom
    /// tools, configured MCP tools). These then require approval outside
    /// `yolo`/`god` exactly like the built-in mutating tools.
    pub fn with_mutation_extra(mut self, names: std::collections::HashSet<String>) -> Self {
        self.mutation_extra = names;
        self
    }

    /// Produce an independent copy of this agent for a background turn.
    ///
    /// The copy shares the provider, tool executor and event bus (all
    /// reference counted) but owns its own reasoning log, so a turn running in
    /// a spawned task cannot corrupt the interactive agent's state.
    pub fn fork(&self) -> Self {
        Self {
            llm_client: self.llm_client.clone(),
            provider: self.provider.clone(),
            model_id: self.model_id.clone(),
            tools: self.tools.clone(),
            autonomy: self.autonomy,
            event_bus: self.event_bus.clone(),
            max_iterations: self.max_iterations,
            streaming: self.streaming,
            reasoning_log: Vec::new(),
            update_tx: self.update_tx.clone(),
            decision_orchestrator: self.decision_orchestrator.clone(),
            config: self.config.clone(),
            persona: self.persona.clone(),
            confirm_tx: self.confirm_tx.clone(),
            mutation_extra: self.mutation_extra.clone(),
            session_id: self.session_id.clone(),
        }
    }

    /// Run one complete turn.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        messages: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
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
            backtrack_count: 0,
            max_backtracks: 3,
            decision_orchestrator: self.decision_orchestrator.clone(),
            config: self.config.clone(),
            persona: self.persona.clone(),
            confirm_tx: self.confirm_tx.clone(),
            mutation_extra: self.mutation_extra.clone(),
            session_id: self.session_id.clone(),
        };

        let new_messages = orch.run_turn(user_input, messages).await;

        // Sync reasoning log back
        self.reasoning_log = orch.reasoning_log;
        new_messages
    }
}
