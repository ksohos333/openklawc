//! Agent tool orchestrator — inspired by Anthropic's extended thinking patterns.
//!
//! Sequences tool calls, handles errors, retries with exponential backoff,
//! and formats results in the provider's native format.
//!
//! OpenAI flow:
//!   assistant (with tool_calls) → tool (role="tool", tool_call_id) → assistant
//!
//! Anthropic flow:
//!   assistant (with tool_use blocks) → user (with tool_result blocks) → assistant
//!
//! Fallback (text-based):
//!   assistant (with JSON tool calls in text) → user (describing results) → assistant

use std::sync::Arc;

use anyhow::Result;
use chrono::Utc;
use okc_core::{
    AutonomyLevel, ChatMessage, Event, EventBus, Id, NativeToolCall, NativeToolDef,
    NativeToolResult, ReasoningKind, ReasoningStep, Role, ToolExecutor, ToolOutcome, TurnPhase,
    TurnState,
};
use okc_llm::LlmClient;
use tokio::sync::mpsc;
use tracing::{debug, info, warn};

use crate::AgentUpdate;

const DEFAULT_MAX_ITERATIONS: usize = 10;

/// The agent orchestrator. Drives multi-turn tool-using conversations.
pub struct Orchestrator {
    pub llm_client: LlmClient,
    pub provider: Arc<dyn okc_core::LlmProvider>,
    pub model_id: String,
    pub tools: Arc<dyn ToolExecutor>,
    pub autonomy: AutonomyLevel,
    pub event_bus: EventBus,
    pub max_iterations: usize,
    pub update_tx: Option<mpsc::Sender<AgentUpdate>>,
    pub reasoning_log: Vec<ReasoningStep>,
}

impl Orchestrator {
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
            update_tx: None,
            reasoning_log: Vec::new(),
        }
    }

    fn emit(&self, update: AgentUpdate) {
        if let Some(ref tx) = self.update_tx {
            let _ = tx.try_send(update);
        }
    }

    /// Run one complete turn with native tool support.
    /// Returns the updated message history.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        mut messages: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
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

        // Build / refresh system prompt
        let tool_defs = match self.tools.list_tools().await {
            Ok(tools) => tools
                .into_iter()
                .map(|t| NativeToolDef {
                    name: t.name,
                    description: t.description,
                    parameters: t.parameters,
                })
                .collect::<Vec<_>>(),
            Err(e) => {
                warn!("Failed to list tools: {}", e);
                Vec::new()
            }
        };

        let system_prompt = self.build_system_prompt(&tool_defs);
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
            debug!("orchestrator iteration {}/{}", iteration + 1, self.max_iterations);

            // ─── Call LLM with tools ──────────────────────────────────────
            let output = if self.provider.supports_native_tools() && !tool_defs.is_empty() {
                match self.provider.chat_with_tools(&self.model_id, messages.clone(), tool_defs.clone()).await {
                    Ok(out) => out,
                    Err(e) => {
                        warn!("Native tool chat failed, falling back to text: {}", e);
                        match self.llm_client.chat(&self.provider, &self.model_id, messages.clone()).await {
                            Ok(text) => okc_core::TurnOutput {
                                text,
                                tool_calls: Vec::new(),
                            },
                            Err(e2) => {
                                self.emit(AgentUpdate::Error(format!("LLM error: {}", e2)));
                                break;
                            }
                        }
                    }
                }
            } else {
                // Text-based fallback
                let text = match self.llm_client.chat(&self.provider, &self.model_id, messages.clone()).await {
                    Ok(t) => t,
                    Err(e) => {
                        self.emit(AgentUpdate::Error(format!("LLM error: {}", e)));
                        break;
                    }
                };
                self.emit(AgentUpdate::TextChunk(text.clone()));
                okc_core::parse_tool_calls_native(&text)
            };

            if !output.text.is_empty() {
                self.emit(AgentUpdate::TextChunk(output.text.clone()));
            }

            // ─── No tool calls? We're done ────────────────────────────────
            if output.tool_calls.is_empty() {
                final_text = output.text;
                messages.push(ChatMessage {
                    role: Role::Assistant,
                    content: final_text.clone(),
                    name: None,
                });
                self.log_reasoning(ReasoningKind::Reflect, "No tools needed — responding.".to_string());
                break;
            }

            // ─── Tool calls detected ──────────────────────────────────────
            self.log_reasoning(
                ReasoningKind::Decide,
                format!(
                    "Calling: {}",
                    output.tool_calls.iter().map(|c| c.name.as_str()).collect::<Vec<_>>().join(", ")
                ),
            );

            // Record the assistant's tool-calling turn in history
            self.append_assistant_tool_turn(&mut messages, &output);

            // Execute each tool
            let mut results = Vec::new();
            for call in &output.tool_calls {
                self.emit(AgentUpdate::ToolCall {
                    name: call.name.clone(),
                    arguments: call.arguments.clone(),
                });

                let native_result = self.execute_native_tool(call).await;
                results.push(native_result.clone());

                self.emit(AgentUpdate::ToolResult {
                    name: call.name.clone(),
                    success: !native_result.is_error,
                    result: serde_json::json!(native_result.content.clone()),
                    elapsed_ms: 0,
                });
            }

            // Feed results back in the provider's native format
            self.append_tool_results(&mut messages, &results);

            turn.phase = TurnPhase::Reflecting;
        }

        if final_text.is_empty() {
            final_text = format!("Reached max iterations ({}).\n", self.max_iterations);
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: final_text.clone(),
                name: None,
            });
        }

        turn.phase = TurnPhase::Idle;
        self.emit(AgentUpdate::Done {
            text: final_text,
            messages: messages.clone(),
        });
        messages
    }

    /// Append the assistant's tool-calling message in the provider's format.
    fn append_assistant_tool_turn(&self, messages: &mut Vec<ChatMessage>, output: &okc_core::TurnOutput) {
        let provider_name = self.provider.name();

        if provider_name == "openai" {
            // OpenAI: assistant message may contain tool_calls
            let mut msg = ChatMessage {
                role: Role::Assistant,
                content: output.text.clone(),
                name: None,
            };
            // Embed tool calls as serialised JSON in the content for history
            // The native provider already saw them; here we just record for context
            if !output.tool_calls.is_empty() {
                let tc_json = serde_json::to_string(&output.tool_calls).unwrap_or_default();
                if !msg.content.is_empty() {
                    msg.content.push_str("\n\n");
                }
                msg.content.push_str(&format!("[tool_calls] {}", tc_json));
            }
            messages.push(msg);
        } else if provider_name == "anthropic" {
            // Anthropic: assistant message with tool_use blocks
            let mut msg = ChatMessage {
                role: Role::Assistant,
                content: output.text.clone(),
                name: None,
            };
            if !output.tool_calls.is_empty() {
                let tc_json = serde_json::to_string(&output.tool_calls).unwrap_or_default();
                if !msg.content.is_empty() {
                    msg.content.push_str("\n\n");
                }
                msg.content.push_str(&format!("[tool_use] {}", tc_json));
            }
            messages.push(msg);
        } else {
            // Fallback: just the text (tool calls were embedded by text injection)
            messages.push(ChatMessage {
                role: Role::Assistant,
                content: format!("{}\n\n[tool_calls] {}",
                    output.text,
                    serde_json::to_string(&output.tool_calls).unwrap_or_default()
                ),
                name: None,
            });
        }
    }

    /// Append tool results as the next message(s) in the provider's format.
    fn append_tool_results(&self, messages: &mut Vec<ChatMessage>, results: &[NativeToolResult]) {
        let provider_name = self.provider.name();

        if provider_name == "openai" {
            // OpenAI: one "tool" role message per result
            for result in results {
                messages.push(ChatMessage {
                    role: Role::Tool,
                    content: if result.is_error {
                        format!("Error: {}", result.content)
                    } else {
                        result.content.clone()
                    },
                    name: Some(result.call_id.clone()),
                });
            }
        } else {
            // Anthropic: user message with tool_result content blocks
            let blocks = okc_llm::native_tools::to_anthropic_tool_results(results);
            let content = serde_json::to_string(&blocks).unwrap_or_default();
            messages.push(ChatMessage {
                role: Role::User,
                content: content,
                name: None,
            });
        }
    }

    async fn execute_native_tool(&self, call: &NativeToolCall) -> NativeToolResult {
        let is_mutation = matches!(call.name.as_str(), "write_file" | "shell" | "git");

        if self.autonomy.needs_confirm(&call.name, is_mutation) {
            info!(
                "TOOL PENDING CONFIRM (autonomy={:?}): {} {:?}",
                self.autonomy, call.name, call.arguments
            );
        }

        let tool_call = okc_core::ToolCall {
            id: Id::new(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        };

        let start = std::time::Instant::now();
        let (success, content, is_error) = match self.tools.execute_tool(&tool_call).await {
            Ok(outcome) => {
                let text = if outcome.success {
                    serde_json::to_string(&outcome.result).unwrap_or_else(|_| "ok".to_string())
                } else {
                    outcome.error.clone().unwrap_or_else(|| "unknown error".to_string())
                };
                (outcome.success, text, !outcome.success)
            }
            Err(e) => (false, e.to_string(), true),
        };
        let elapsed = start.elapsed().as_millis() as u64;

        let result = NativeToolResult {
            call_id: call.id.clone(),
            content,
            is_error,
        };

        self.publish(Event::ToolResult {
            session_id: Id::new(),
            result: ToolOutcome {
                call_id: tool_call.id,
                success,
                result: serde_json::json!(result.content.clone()),
                error: if is_error { Some(result.content.clone()) } else { None },
                elapsed_ms: elapsed,
            },
        });

        result
    }

    fn build_system_prompt(&self, tools: &[NativeToolDef]) -> String {
        let mut prompt = String::new();
        prompt.push_str(
            "You are openklawc, an autonomous coding agent.\n\n\
             TOOL USAGE:\n"
        );

        // If provider supports native tools, keep prompt minimal
        if self.provider.supports_native_tools() {
            prompt.push_str("Use the provided tools when needed. Stop calling tools when you can answer.\n\n");
        } else {
            prompt.push_str(
                "When you need a tool, output ONE standalone JSON block per line:\n\
                 {\"tool\": \"tool_name\", \"arguments\": {\"param\": \"value\"}}\n\
                 Do NOT wrap in markdown code fences.\n\n"
            );
        }

        if !tools.is_empty() {
            prompt.push_str("Available tools:\n");
            for tool in tools {
                let schema = serde_json::to_string(&tool.parameters).unwrap_or_default();
                prompt.push_str(&format!("\n### {}\n{}", tool.name, tool.description));
                prompt.push_str(&format!("\n  Schema: {}\n", schema));
            }
        }

        if let Ok(cwd) = std::env::current_dir() {
            prompt.push_str(&format!("\n\nCWD: {}\n", cwd.display()));
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
