//! openklawc-reasoning — the full agent loop with tool call parsing.
//!
//! Hermes-inspired architecture: plan → execute → reflect → respond.
//! Reasoning is ALWAYS visible. Autonomy controls confirmation.

use std::fmt::Write;
use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use okc_core::{
    AutonomyLevel, ChatMessage, Event, EventBus, Id, ReasoningKind, ReasoningStep,
    Role, ToolCall, ToolExecutor, ToolOutcome, TurnPhase, TurnState,
};
use okc_llm::LlmClient;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

pub mod prompt;

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
    /// Turn complete. Contains the final assistant text and the updated conversation history.
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
    /// If set, all progress updates (text, reasoning, tool events) are sent here.
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

    fn emit(&self, update: AgentUpdate) {
        if let Some(ref tx) = self.update_tx {
            let _ = tx.try_send(update);
        }
    }

    /// Run one complete turn. `messages` is the conversation history (will be mutated).
    pub async fn run_turn(&mut self, user_input: &str, mut messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
        self.reasoning_log.clear();
        let mut turn = TurnState::new();
        turn.phase = TurnPhase::Planning;

        // Append user message
        messages.push(ChatMessage {
            role: Role::User,
            content: user_input.to_string(),
            name: None,
        });

        self.log_reasoning(ReasoningKind::Plan, format!("User: {}", user_input));

        // Inject / refresh system prompt at the front
        let system_prompt = self.build_system_prompt().await;
        if messages.is_empty() || messages[0].role != Role::System {
            messages.insert(0, ChatMessage {
                role: Role::System,
                content: system_prompt,
                name: None,
            });
        } else {
            messages[0].content = system_prompt;
        }

        let mut final_text = String::new();

        for iteration in 0..self.max_iterations {
            debug!("agent iteration {}/{}", iteration + 1, self.max_iterations);

            // ─── Call LLM ───────────────────────────────────────────────────
            let llm_response = match self.chat_with_messages(&messages).await {
                Ok(text) => text,
                Err(e) => {
                    warn!("LLM call failed: {}", e);
                    self.emit(AgentUpdate::Error(format!("LLM error: {}", e)));
                    break;
                }
            };

            // ─── Parse for tool calls ───────────────────────────────────────
            let (clean_text, tool_calls) = okc_core::parse_tool_calls(&llm_response);

            debug!(
                "parsed: {} chars text, {} tool call(s)",
                clean_text.len(),
                tool_calls.len()
            );

            if tool_calls.is_empty() {
                // No tools — this is the final response
                final_text = clean_text;
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    content: final_text.clone(),
                    name: None,
                });
                self.log_reasoning(ReasoningKind::Reflect, "No further tools needed — responding.".to_string());
                break;
            }

            // ─── Tool calls detected ────────────────────────────────────────
            // Record what the model asked for
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: llm_response.clone(),
                name: None,
            });

            self.log_reasoning(
                ReasoningKind::Decide,
                format!("Calling tools: {}", tool_calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")),
            );

            // Execute each tool
            for call in &tool_calls {
                self.emit(AgentUpdate::ToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });

                let outcome = match self.execute_tool_call(call).await {
                    Ok(o) => o,
                    Err(e) => {
                        warn!("Tool '{}' dispatch error: {}", call.name, e);
                        ToolOutcome {
                            call_id: call.id.clone(),
                            success: false,
                            result: serde_json::Value::Null,
                            error: Some(e.to_string()),
                            elapsed_ms: 0,
                        }
                    }
                };

                self.emit(AgentUpdate::ToolResult {
                    name: call.name.clone(),
                    success: outcome.success,
                    result: outcome.result.clone(),
                    elapsed_ms: outcome.elapsed_ms,
                });

                // Add tool result to conversation history (so LLM can see it)
                let result_summary = serde_json::to_string(&outcome.result).unwrap_or_default();
                messages.push(ChatMessage {
                    role: Role::Tool,
                    content: format!(
                        "Tool '{}' {} ({}ms): {}",
                        call.name,
                        if outcome.success { "succeeded" } else { "failed" },
                        outcome.elapsed_ms,
                        outcome.error.as_deref().unwrap_or(&result_summary)
                    ),
                    name: Some(call.name.clone()),
                });

                let kind = if outcome.success { ReasoningKind::Observe } else { ReasoningKind::Error };
                self.log_reasoning(kind, format!("{} → {:?}", call.name, outcome.result));
            }

            turn.phase = TurnPhase::Reflecting;
        }

        if final_text.is_empty() {
            final_text = format!(
                "Reached max iterations ({}). Summary of actions:\n",
                self.max_iterations
            );
            for step in &self.reasoning_log {
                let _ = writeln!(final_text, "- [{:?}] {}", step.kind, step.content);
            }
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: final_text.clone(),
                name: None,
            });
        }

        turn.phase = TurnPhase::Idle;
        self.emit(AgentUpdate::Done { text: final_text, messages: messages.clone() });
        messages
    }

    async fn execute_tool_call(&self, call: &ToolCall) -> Result<ToolOutcome> {
        let is_mutation = is_mutation_tool(&call.name);

        if self.autonomy.needs_confirm(&call.name, is_mutation) {
            info!(
                "TOOL REQUIRES CONFIRM (autonomy={:?}): {} {:?}",
                self.autonomy, call.name, call.arguments
            );
        }

        let start = std::time::Instant::now();
        let outcome = self.tools.execute_tool(call).await?;
        let elapsed = start.elapsed().as_millis() as u64;

        let enriched = ToolOutcome {
            call_id: call.id.clone(),
            success: outcome.success,
            result: outcome.result,
            error: outcome.error,
            elapsed_ms: elapsed,
        };

        self.publish(Event::ToolResult {
            session_id: Id::new(),
            result: enriched.clone(),
        });

        Ok(enriched)
    }

    /// Send messages to the LLM, streaming if enabled.
    async fn chat_with_messages(&self, messages: &[ChatMessage]) -> Result<String> {
        if self.streaming {
            let (tx, mut rx) = mpsc::channel::<String>(128);
            let client = self.llm_client.clone();
            let provider = self.provider.clone();
            let model_id = self.model_id.clone();
            let msgs = messages.to_vec();

            tokio::spawn(async move {
                let _ = client.chat_stream(&provider, &model_id, msgs, tx).await;
            });

            let mut full = String::new();
            while let Some(chunk) = rx.recv().await {
                full.push_str(&chunk);
                self.emit(AgentUpdate::TextChunk(chunk));
            }
            Ok(full)
        } else {
            let text = self
                .llm_client
                .chat(&self.provider, &self.model_id, messages.to_vec())
                .await?;
            self.emit(AgentUpdate::TextChunk(text.clone()));
            Ok(text)
        }
    }

    async fn build_system_prompt(&self) -> String {
        let mut prompt = String::new();
        prompt.push_str(
            "You are openklawc, an autonomous coding agent running in a Rust TUI.\n\n\
             TOOL USAGE:\n\
             When you need a tool, output ONE standalone JSON block per line:\n\
             {\"tool\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\n\
             Do NOT wrap tool calls in markdown code fences.\n\
             After each tool result you will be prompted again.\n\
             Stop calling tools when you can answer the user.\n\n"
        );

        match self.tools.list_tools().await {
            Ok(tools) if !tools.is_empty() => {
                prompt.push_str("Available tools:\n");
                for tool in tools {
                    prompt.push_str(&format!("\n### {}\n{}", tool.name, tool.description));
                    let schema = serde_json::to_string(&tool.parameters).unwrap_or_default();
                    prompt.push_str(&format!("\n  Schema: {}\n", schema));
                }
            }
            _ => {}
        }

        if let Ok(cwd) = std::env::current_dir() {
            prompt.push_str(&format!("\n\nCurrent working directory: {}\n", cwd.display()));
        }

        prompt.push_str("\nRespond concisely. Think step by step.");
        prompt
    }

    fn log_reasoning(&mut self, kind: ReasoningKind, content: String) {
        let step = ReasoningStep {
            timestamp: Utc::now(),
            kind,
            content: content.clone(),
            tool_call: None,
            tool_result: None,
        };
        self.reasoning_log.push(step.clone());
        self.emit(AgentUpdate::Reasoning {
            kind: format!("{:?}", kind).to_lowercase(),
            content,
        });
        self.publish(Event::Reasoning {
            session_id: Id::new(),
            step,
        });
    }

    fn publish(&self, event: Event) {
        self.event_bus.publish(event);
    }
}

fn is_mutation_tool(name: &str) -> bool {
    matches!(name, "write_file" | "shell" | "git")
}
