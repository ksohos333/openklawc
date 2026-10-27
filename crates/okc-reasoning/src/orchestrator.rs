//! Agent tool orchestrator — REASON-enhanced with decision metadata extraction
//! and confidence-driven backtracking.
//!
//! Architecture: plan → analyze → evaluate → decide → execute → reflect → respond.
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

use chrono::Utc;
use futures::future::join_all;
use okc_core::{
    AutonomyLevel, ChatMessage, DecisionMetadata, Event, EventBus, Id, NativeToolCall,
    NativeToolDef, NativeToolResult, ReasoningKind, ReasoningStep, Role, TokenUsage, ToolExecutor,
    ToolOutcome, ToolSchema, TurnPhase, TurnState,
};
use okc_llm::LlmClient;
use tokio::sync::mpsc;
use tokio::time::{sleep, Duration};
use tracing::{debug, warn};

use crate::prompt::{build_agent_prompt, extract_decision_metadata, format_tool_descriptions};
use crate::{AgentUpdate, ConfirmRequest};

const DEFAULT_MAX_ITERATIONS: usize = 10;

/// Confidence threshold below which the agent should prefer read-only tools.
fn confidence_below_threshold(confidence: &str) -> bool {
    confidence == "low"
}

/// The result of asking the user to approve a tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ConfirmOutcome {
    Approved,
    Denied,
    /// Nobody was available to ask.
    Unavailable,
}

/// How the model is allowed to use tools. Shared type from `okc_core` so
/// providers and config speak the same enum.
pub use okc_core::ToolChoice;

/// Configuration for the orchestrator's behavior.
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    pub tool_choice: ToolChoice,
    pub max_iterations: usize,
    pub parallel_tools: bool,
    pub retry_count: u32,
    pub retry_delay_ms: u64,
    pub summarize_results: bool,
    pub summarize_threshold: usize,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            tool_choice: ToolChoice::Auto,
            max_iterations: DEFAULT_MAX_ITERATIONS,
            parallel_tools: true,
            retry_count: 3,
            retry_delay_ms: 500,
            summarize_results: true,
            summarize_threshold: 4000,
        }
    }
}

impl From<&okc_config::OrchestratorConfig> for OrchestratorConfig {
    fn from(cfg: &okc_config::OrchestratorConfig) -> Self {
        Self {
            tool_choice: ToolChoice::parse(&cfg.tool_choice),
            max_iterations: cfg.max_iterations,
            parallel_tools: cfg.parallel_tools,
            retry_count: cfg.retry_count,
            retry_delay_ms: cfg.retry_delay_ms,
            summarize_results: cfg.summarize_results,
            summarize_threshold: cfg.summarize_threshold,
        }
    }
}

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
    /// Tracks how many backtracks have occurred this turn (prevents infinite loops).
    pub backtrack_count: usize,
    /// Maximum backtracks allowed per turn.
    pub max_backtracks: usize,
    /// Optional fast decision layer (Laya / Ollaya) for plan validation.
    pub decision_orchestrator: Option<okc_decision::DecisionOrchestrator>,
    /// Runtime configuration for execution behavior.
    pub config: OrchestratorConfig,
    /// Active persona (if any).
    pub persona: Option<crate::persona::Persona>,
    /// Channel used to ask the user to approve a tool call.
    pub confirm_tx: Option<tokio::sync::mpsc::Sender<ConfirmRequest>>,
    /// Extra tool names classified as mutating (extends
    /// `okc_core::is_mutation_tool`).
    pub mutation_extra: std::collections::HashSet<String>,
    /// Session this turn belongs to — audit events carry it so the trail
    /// can be queried per session.
    pub session_id: Option<String>,
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
            backtrack_count: 0,
            max_backtracks: 3,
            decision_orchestrator: None,
            config: OrchestratorConfig::default(),
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
    pub fn with_config(mut self, config: OrchestratorConfig) -> Self {
        self.max_iterations = config.max_iterations;
        self.config = config;
        self
    }

    fn emit(&self, update: AgentUpdate) {
        if let Some(ref tx) = self.update_tx {
            let _ = tx.try_send(update);
        }
    }

    /// Run one complete turn with native tool support, decision extraction,
    /// confidence checking, and backtracking.
    /// Returns the updated message history.
    pub async fn run_turn(
        &mut self,
        user_input: &str,
        mut messages: Vec<ChatMessage>,
    ) -> Vec<ChatMessage> {
        self.reasoning_log.clear();
        self.backtrack_count = 0;
        let mut turn = TurnState::new();
        turn.phase = TurnPhase::Planning;

        // Append user message
        messages.push(ChatMessage::user(user_input));

        // Record the request in the audit trail (when bound to a session).
        if self.session_id.is_some() {
            self.publish(Event::UserMessage {
                session_id: self.session_id(),
                content: user_input.to_string(),
            });
        }

        self.log_reasoning(ReasoningKind::Plan, format!("User: {}", user_input), None);

        // Apply persona overrides
        let max_iterations = self
            .persona
            .as_ref()
            .and_then(|p| p.max_iterations)
            .unwrap_or(self.max_iterations);

        // Build / refresh system prompt with REASON framework
        let all_tool_schemas = match self.tools.list_tools().await {
            Ok(tools) => tools,
            Err(e) => {
                warn!("Failed to list tools: {}", e);
                Vec::new()
            }
        };

        // Filter tools by persona whitelist
        let tool_schemas: Vec<ToolSchema> = if let Some(ref persona) = self.persona {
            all_tool_schemas
                .into_iter()
                .filter(|t| persona.allows_tool(&t.name))
                .collect()
        } else {
            all_tool_schemas
        };

        let tool_defs: Vec<NativeToolDef> = tool_schemas
            .iter()
            .map(|t| NativeToolDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            })
            .collect();

        let tool_descriptions = format_tool_descriptions(&tool_schemas);
        let cwd = std::env::current_dir()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_else(|_| String::from("."));
        let autonomy_str = format!("{:?}", self.autonomy).to_lowercase();

        // Detect and inject project context
        let project_context =
            if let Ok(ctx) = crate::context::detect_project(std::path::Path::new(&cwd)) {
                let formatted = crate::context::format_context(&ctx);
                self.log_reasoning(
                    ReasoningKind::Observe,
                    format!(
                        "Detected {:?} project at {}",
                        ctx.project_type,
                        ctx.root.display()
                    ),
                    None,
                );
                formatted
            } else {
                String::from("No project detected.")
            };

        let mut system_prompt = build_agent_prompt(
            &tool_descriptions,
            &cwd,
            &autonomy_str,
            Some(&project_context),
        );

        // Inject persona system prompt
        if let Some(ref persona) = self.persona {
            if !persona.system_prompt.is_empty() {
                system_prompt.push_str("\n\n# Persona\n\n");
                system_prompt.push_str(&persona.system_prompt);
            }
            if let Some(ref name) = persona.name {
                self.log_reasoning(
                    ReasoningKind::Plan,
                    format!("Using persona: {}", name),
                    None,
                );
            }
        }

        // Claim verification: instruct only when the anticonf observation
        // tool is actually loaded — no tool, no nagging.
        if tool_schemas
            .iter()
            .any(|t| t.name == "note_observation")
        {
            system_prompt.push_str(&crate::prompt::claim_verification_prompt());
        }

        if messages.is_empty() || messages[0].role != Role::System {
            messages.insert(0, ChatMessage::system(system_prompt));
        } else {
            messages[0].content = system_prompt;
        }

        let mut final_text = String::new();
        let mut total_usage = TokenUsage::default();
        let mut usage_reported = false;

        for iteration in 0..max_iterations {
            debug!(
                "orchestrator iteration {}/{}",
                iteration + 1,
                self.max_iterations
            );

            // ─── Call LLM with tools ──────────────────────────────────────
            let output = if self.provider.supports_native_tools() && !tool_defs.is_empty() {
                match self
                    .provider
                    .chat_with_tools(
                        &self.model_id,
                        messages.clone(),
                        tool_defs.clone(),
                        self.config.tool_choice,
                    )
                    .await
                {
                    Ok(out) => out,
                    Err(e) => {
                        warn!("Native tool chat failed, falling back to text: {}", e);
                        match self
                            .llm_client
                            .chat(&self.provider, &self.model_id, messages.clone())
                            .await
                        {
                            Ok(text) => okc_core::TurnOutput {
                                text,
                                tool_calls: Vec::new(),
                                usage: None,
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
                let text = match self
                    .llm_client
                    .chat(&self.provider, &self.model_id, messages.clone())
                    .await
                {
                    Ok(t) => t,
                    Err(e) => {
                        self.emit(AgentUpdate::Error(format!("LLM error: {}", e)));
                        break;
                    }
                };
                let filtered = okc_core::util::filter_filler_phrases(&text);
                // The response text is emitted once, below, from the cleaned
                // output. Emitting it here as well showed text-fallback
                // users (llama.cpp and friends) the reply twice — the first
                // copy still containing the raw tool JSON.
                okc_core::parse_tool_calls_native(&filtered)
            };

            // Providers expose token counts for the call they just made.
            if let Some(u) = self.provider.last_usage() {
                total_usage.merge(u);
                usage_reported = true;
            }

            if !output.text.is_empty() {
                let filtered = okc_core::util::filter_filler_phrases(&output.text);
                self.emit(AgentUpdate::TextChunk(filtered));
            }

            // ─── Extract decision metadata from reasoning text ────────────
            let decision_meta = extract_decision_metadata(&output.text);
            let mut enriched_meta = decision_meta.clone();
            enriched_meta.iteration = iteration;

            // Log the analytical reasoning step
            self.log_reasoning(
                ReasoningKind::Analyze,
                output.text.clone(),
                Some(enriched_meta.clone()),
            );

            // ─── No tool calls? We're done ────────────────────────────────
            if output.tool_calls.is_empty() {
                final_text = okc_core::util::filter_filler_phrases(&output.text);
                messages.push(ChatMessage::assistant(final_text.clone()));
                self.log_reasoning(
                    ReasoningKind::Reflect,
                    "No tools needed — responding.".to_string(),
                    None,
                );
                break;
            }

            // ─── Confidence check ─────────────────────────────────────────
            if let Some(ref confidence) = decision_meta.confidence {
                if confidence_below_threshold(confidence)
                    && self.backtrack_count < self.max_backtracks
                {
                    self.log_reasoning(
                        ReasoningKind::Evaluate,
                        format!(
                            "Low confidence ({}). Reconsidering approach before executing tools.",
                            confidence
                        ),
                        Some(DecisionMetadata {
                            confidence: Some(confidence.clone()),
                            is_backtrack: true,
                            iteration,
                            ..Default::default()
                        }),
                    );
                    // Ask the model to reconsider by inserting a reflection prompt
                    messages.push(ChatMessage::user(
                        "Your previous confidence was low. Please re-evaluate your options and choose a safer approach. Consider using read-only tools first.".to_string(),
                    ));
                    self.backtrack_count += 1;
                    continue; // Re-run the loop with the new prompt
                }
            }

            // ─── Decision layer validation (Laya) ────────────────────────
            if let Some(ref decision_orchestrator) = self.decision_orchestrator {
                let proposed_tools: Vec<String> =
                    output.tool_calls.iter().map(|c| c.name.clone()).collect();

                match decision_orchestrator
                    .validate_plan(user_input, &proposed_tools)
                    .await
                {
                    Ok(validation) => {
                        self.log_reasoning(
                            ReasoningKind::Evaluate,
                            format!(
                                "Decision layer: {} (confidence: {:.2})",
                                validation.explanation, validation.confidence
                            ),
                            Some(DecisionMetadata {
                                confidence: Some(format!("{:.2}", validation.confidence)),
                                chosen_approach: Some(if validation.proceed {
                                    "proceed".to_string()
                                } else {
                                    "halt".to_string()
                                }),
                                iteration,
                                ..Default::default()
                            }),
                        );

                        if !validation.proceed && self.backtrack_count < self.max_backtracks {
                            self.backtrack_count += 1;
                            let suggestion = validation.suggestion.unwrap_or_else(|| {
                                "The decision layer flagged risks with this plan. Please reconsider."
                                    .to_string()
                            });

                            self.log_reasoning(
                                ReasoningKind::Backtrack,
                                format!(
                                    "Decision layer blocked plan (attempt {}/{}). {}",
                                    self.backtrack_count, self.max_backtracks, suggestion
                                ),
                                Some(DecisionMetadata {
                                    is_backtrack: true,
                                    iteration,
                                    ..Default::default()
                                }),
                            );

                            messages.push(ChatMessage::user(suggestion));
                            continue; // Re-run the loop
                        }
                    }
                    Err(e) => {
                        warn!(
                            "Decision layer validation failed: {}. Proceeding anyway.",
                            e
                        );
                    }
                }
            }

            // ─── Tool calls detected ──────────────────────────────────────
            self.log_reasoning(
                ReasoningKind::Decide,
                format!(
                    "Calling: {}",
                    output
                        .tool_calls
                        .iter()
                        .map(|c| c.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                Some(DecisionMetadata {
                    confidence: decision_meta.confidence.clone(),
                    chosen_approach: decision_meta.chosen_approach.clone(),
                    alternatives: decision_meta.alternatives.clone(),
                    key_tradeoff: decision_meta.key_tradeoff.clone(),
                    iteration,
                    ..Default::default()
                }),
            );

            // Record the assistant's tool-calling turn in history
            self.append_assistant_tool_turn(&mut messages, &output);

            // ─── Execute tools (parallel if configured) ─────────────────
            let results = if self.config.parallel_tools && output.tool_calls.len() > 1 {
                self.execute_tools_parallel(&output.tool_calls).await
            } else {
                self.execute_tools_sequential(&output.tool_calls).await
            };
            let any_failed = results.iter().any(|r| r.is_error);

            // Feed results back in the provider's native format FIRST.
            // Anthropic-family APIs reject a tool_use that is not followed
            // by its tool_result ("tool call result does not follow tool
            // call", error 2013), so results must land in the history even
            // when the turn backtracks.
            self.append_tool_results(&mut messages, &results).await;

            // ─── Backtracking on failure ──────────────────────────────────
            if any_failed && self.backtrack_count < self.max_backtracks {
                self.backtrack_count += 1;
                let failed_names: Vec<_> = output
                    .tool_calls
                    .iter()
                    .zip(&results)
                    .filter(|(_, r)| r.is_error)
                    .map(|(c, _)| c.name.clone())
                    .collect();

                self.log_reasoning(
                    ReasoningKind::Backtrack,
                    format!(
                        "Tools failed: {}. Backtracking (attempt {}/{}).",
                        failed_names.join(", "),
                        self.backtrack_count,
                        self.max_backtracks
                    ),
                    Some(DecisionMetadata {
                        is_backtrack: true,
                        iteration,
                        ..Default::default()
                    }),
                );

                // Append a user message prompting reconsideration
                messages.push(ChatMessage::user(format!(
                    "The following tool calls failed: {}. Please reconsider your approach. Try an alternative method or gather more information with read-only tools before mutating anything.",
                    failed_names.join(", ")
                )));

                turn.phase = TurnPhase::Reflecting;
                continue; // Loop again with the backtrack prompt
            }

            turn.phase = TurnPhase::Reflecting;
        }

        if final_text.is_empty() {
            final_text = format!("Reached max iterations ({}).\n", self.max_iterations);
            messages.push(ChatMessage::assistant(final_text.clone()));
        }

        turn.phase = TurnPhase::Idle;
        let filtered_final = okc_core::util::filter_filler_phrases(&final_text);
        self.emit(AgentUpdate::Done {
            text: filtered_final,
            messages: messages.clone(),
            usage: if usage_reported {
                Some(total_usage)
            } else {
                None
            },
        });
        messages
    }

    /// Append the assistant's tool-calling message to the history.
    fn append_assistant_tool_turn(
        &self,
        messages: &mut Vec<ChatMessage>,
        output: &okc_core::TurnOutput,
    ) {
        let mut msg = ChatMessage::assistant(output.text.clone());
        if !output.tool_calls.is_empty() {
            msg.tool_calls = Some(output.tool_calls.clone());
        }
        messages.push(msg);
    }

    /// Append tool results as the next message, in the provider's own format.
    /// Optionally summarizes long results first.
    async fn append_tool_results(
        &self,
        messages: &mut Vec<ChatMessage>,
        results: &[NativeToolResult],
    ) {
        let mut processed = results.to_vec();

        if self.config.summarize_results {
            for (i, result) in processed.iter_mut().enumerate() {
                if result.content.len() > self.config.summarize_threshold && !result.is_error {
                    debug!(
                        "Summarizing tool result {} ({} chars)",
                        i,
                        result.content.len()
                    );
                    match self.summarize_tool_result(result).await {
                        Ok(summary) => {
                            result.content = format!(
                                "[SUMMARIZED — original {} chars]\n{}",
                                result.content.len(),
                                summary
                            );
                        }
                        Err(e) => {
                            warn!("Failed to summarize tool result: {}. Keeping original.", e);
                        }
                    }
                }
            }
        }

        messages.push(ChatMessage::tool_results(processed));
    }

    /// Ask the LLM to summarize a long tool result.
    async fn summarize_tool_result(&self, result: &NativeToolResult) -> anyhow::Result<String> {
        let prompt = format!(
            "Summarize the following tool output concisely. Keep the key facts, numbers, and actionable information. Discard noise and redundant detail.\n\n```\n{}\n```",
            &result.content[..result.content.len().min(8000)] // safety cap
        );

        let messages = vec![
            ChatMessage::system(
                "You are a concise summarizer. Output only the summary, no preamble.",
            ),
            ChatMessage::user(prompt),
        ];

        let summary = self
            .llm_client
            .chat(&self.provider, &self.model_id, messages)
            .await?;
        Ok(summary.trim().to_string())
    }

    /// Execute tools sequentially (one at a time).
    async fn execute_tools_sequential(&self, calls: &[NativeToolCall]) -> Vec<NativeToolResult> {
        let mut results = Vec::with_capacity(calls.len());
        for call in calls {
            self.emit(AgentUpdate::ToolCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });

            let native_result = self.execute_native_tool_with_retry(call).await;

            self.emit(AgentUpdate::ToolResult {
                name: call.name.clone(),
                success: !native_result.is_error,
                result: serde_json::json!(native_result.content.clone()),
                elapsed_ms: 0,
            });

            results.push(native_result);
        }
        results
    }

    /// Execute tools in parallel (concurrent, no ordering guarantees).
    async fn execute_tools_parallel(&self, calls: &[NativeToolCall]) -> Vec<NativeToolResult> {
        // Emit all tool-call notifications first so the UI sees them immediately
        for call in calls {
            self.emit(AgentUpdate::ToolCall {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            });
        }

        let futures: Vec<_> = calls
            .iter()
            .map(|call| self.execute_native_tool_with_retry(call))
            .collect();

        let results = join_all(futures).await;

        // Emit results as they complete
        for (call, result) in calls.iter().zip(&results) {
            self.emit(AgentUpdate::ToolResult {
                name: call.name.clone(),
                success: !result.is_error,
                result: serde_json::json!(result.content.clone()),
                elapsed_ms: 0,
            });
        }

        results
    }

    /// Execute a single tool with exponential-backoff retry for transient failures.
    ///
    /// The approval prompt happens *before* the retry loop: a declined call is
    /// a decision, not a transient failure, so it must not be re-asked.
    async fn execute_native_tool_with_retry(&self, call: &NativeToolCall) -> NativeToolResult {
        let is_mutation = okc_core::is_mutation_tool(&call.name)
            || self.mutation_extra.contains(&call.name);
        if self.autonomy.needs_confirm(&call.name, is_mutation) {
            match self.await_confirmation(call).await {
                ConfirmOutcome::Approved => {}
                ConfirmOutcome::Denied => {
                    let msg = format!(
                        "Tool '{}' was not approved — the user declined the call.",
                        call.name
                    );
                    warn!("{}", msg);
                    return NativeToolResult {
                        call_id: call.id.clone(),
                        content: msg,
                        is_error: true,
                    };
                }
                ConfirmOutcome::Unavailable => {
                    let msg = format!(
                        "Tool '{}' requires confirmation but no interactive session is available — \
                         denied. Use --autonomy yolo to allow it unattended.",
                        call.name
                    );
                    warn!("{}", msg);
                    return NativeToolResult {
                        call_id: call.id.clone(),
                        content: msg,
                        is_error: true,
                    };
                }
            }
        }

        let mut last_error = String::new();

        for attempt in 0..=self.config.retry_count {
            if attempt > 0 {
                let delay = self.config.retry_delay_ms * 2_u64.pow(attempt - 1);
                debug!(
                    "Retrying tool '{}' (attempt {}/{}) after {}ms",
                    call.name, attempt, self.config.retry_count, delay
                );
                sleep(Duration::from_millis(delay)).await;
            }

            let result = self.execute_native_tool(call).await;
            if !result.is_error {
                return result;
            }
            last_error = result.content.clone();
        }

        // All retries exhausted
        NativeToolResult {
            call_id: call.id.clone(),
            content: format!(
                "Failed after {} retries. Last error: {}",
                self.config.retry_count, last_error
            ),
            is_error: true,
        }
    }

    /// Ask the user to approve a tool call and wait for the answer.
    async fn await_confirmation(&self, call: &NativeToolCall) -> ConfirmOutcome {
        let Some(tx) = &self.confirm_tx else {
            return ConfirmOutcome::Unavailable;
        };
        let (respond, rx) = tokio::sync::oneshot::channel();
        let request = ConfirmRequest {
            id: call.id.clone(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
            respond,
        };
        if tx.send(request).await.is_err() {
            return ConfirmOutcome::Unavailable;
        }
        match rx.await {
            Ok(true) => ConfirmOutcome::Approved,
            Ok(false) => ConfirmOutcome::Denied,
            Err(_) => ConfirmOutcome::Unavailable,
        }
    }

    async fn execute_native_tool(&self, call: &NativeToolCall) -> NativeToolResult {
        let tool_call = okc_core::ToolCall {
            id: Id::new(),
            name: call.name.clone(),
            arguments: call.arguments.clone(),
        };

        // Record the call in the audit trail (when bound to a session).
        if self.session_id.is_some() {
            self.publish(Event::ToolCall {
                session_id: self.session_id(),
                call: tool_call.clone(),
            });
        }

        let start = std::time::Instant::now();
        let (success, content, is_error) = match self.tools.execute_tool(&tool_call).await {
            Ok(outcome) => {
                let text = if outcome.success {
                    serde_json::to_string(&outcome.result).unwrap_or_else(|_| "ok".to_string())
                } else {
                    outcome
                        .error
                        .clone()
                        .unwrap_or_else(|| "unknown error".to_string())
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
            session_id: self.session_id(),
            result: ToolOutcome {
                call_id: tool_call.id,
                success,
                result: serde_json::json!(result.content.clone()),
                error: if is_error {
                    Some(result.content.clone())
                } else {
                    None
                },
                elapsed_ms: elapsed,
            },
        });

        result
    }

    fn log_reasoning(
        &mut self,
        kind: ReasoningKind,
        content: String,
        decision: Option<DecisionMetadata>,
    ) {
        let step = ReasoningStep {
            timestamp: Utc::now(),
            kind,
            content: content.clone(),
            tool_call: None,
            tool_result: None,
            decision: decision.unwrap_or_default(),
        };
        self.reasoning_log.push(step.clone());
        self.emit(AgentUpdate::Reasoning {
            kind: format!("{:?}", kind).to_lowercase(),
            content,
        });
        self.publish(Event::Reasoning {
            session_id: self.session_id(),
            step,
        });
    }

    fn publish(&self, event: Event) {
        self.event_bus.publish(event);
    }

    /// The session this turn belongs to, or a throwaway id when the agent
    /// is not bound to a session.
    fn session_id(&self) -> Id {
        self.session_id.clone().map(Id).unwrap_or_else(Id::new)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use okc_core::{
        ChatMessage, LlmProvider, StreamTx, ToolCall, ToolExecutor, ToolOutcome, ToolSchema,
    };

    /// A provider that never gets called — these tests exercise the tool gate.
    struct NullProvider;

    #[async_trait]
    impl LlmProvider for NullProvider {
        fn name(&self) -> &str {
            "null"
        }
        fn supports_streaming(&self) -> bool {
            false
        }
        async fn chat(&self, _model: &str, _messages: Vec<ChatMessage>) -> anyhow::Result<String> {
            Ok(String::new())
        }
        async fn chat_stream(
            &self,
            _model: &str,
            _messages: Vec<ChatMessage>,
            _tx: StreamTx,
        ) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Records the tool calls it received.
    struct RecordingTools {
        calls: std::sync::Mutex<Vec<String>>,
    }

    impl RecordingTools {
        fn new() -> Self {
            Self {
                calls: std::sync::Mutex::new(Vec::new()),
            }
        }
        fn calls(&self) -> Vec<String> {
            self.calls.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl ToolExecutor for RecordingTools {
        async fn list_tools(&self) -> anyhow::Result<Vec<ToolSchema>> {
            Ok(Vec::new())
        }
        async fn execute_tool(&self, call: &ToolCall) -> anyhow::Result<ToolOutcome> {
            self.calls.lock().unwrap().push(call.name.clone());
            Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: true,
                result: serde_json::json!("ok"),
                error: None,
                elapsed_ms: 0,
            })
        }
    }

    fn orchestrator(tools: Arc<dyn ToolExecutor>, autonomy: AutonomyLevel) -> Orchestrator {
        Orchestrator::new(
            LlmClient::new(),
            Arc::new(NullProvider),
            "test".to_string(),
            tools,
            autonomy,
            EventBus::new(8),
        )
    }

    fn call(name: &str) -> NativeToolCall {
        NativeToolCall {
            id: "call-1".to_string(),
            name: name.to_string(),
            arguments: serde_json::json!({}),
        }
    }

    #[tokio::test]
    async fn mutation_is_denied_without_a_confirmer() {
        let recorder = Arc::new(RecordingTools::new());
        let tools: Arc<dyn ToolExecutor> = recorder.clone();
        let orch = orchestrator(tools, AutonomyLevel::ReadOnly);

        let result = orch
            .execute_native_tool_with_retry(&call("write_file"))
            .await;

        assert!(result.is_error, "the tool should have been denied");
        assert!(result.content.contains("no interactive session"));
        assert!(recorder.calls().is_empty(), "the tool must not run");
    }

    #[tokio::test]
    async fn read_only_tool_runs_without_approval() {
        let recorder = Arc::new(RecordingTools::new());
        let tools: Arc<dyn ToolExecutor> = recorder.clone();
        let orch = orchestrator(tools, AutonomyLevel::ReadOnly);

        let result = orch
            .execute_native_tool_with_retry(&call("read_file"))
            .await;

        assert!(!result.is_error);
        assert_eq!(recorder.calls(), vec!["read_file".to_string()]);
    }

    #[tokio::test]
    async fn declined_call_never_reaches_the_tool() {
        let recorder = Arc::new(RecordingTools::new());
        let tools: Arc<dyn ToolExecutor> = recorder.clone();
        let mut orch = orchestrator(tools, AutonomyLevel::ReadOnly);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<ConfirmRequest>(4);
        orch.confirm_tx = Some(tx);

        // Answer "no" to the first request.
        let approver = tokio::spawn(async move {
            if let Some(request) = rx.recv().await {
                let _ = request.respond.send(false);
            }
        });

        let result = orch
            .execute_native_tool_with_retry(&call("write_file"))
            .await;
        approver.await.unwrap();

        assert!(result.is_error);
        assert!(result.content.contains("declined"));
        assert!(recorder.calls().is_empty(), "the tool must not run");
    }

    #[tokio::test]
    async fn approved_call_reaches_the_tool_once() {
        let recorder = Arc::new(RecordingTools::new());
        let tools: Arc<dyn ToolExecutor> = recorder.clone();
        let mut orch = orchestrator(tools, AutonomyLevel::ReadOnly);

        let (tx, mut rx) = tokio::sync::mpsc::channel::<ConfirmRequest>(4);
        orch.confirm_tx = Some(tx);

        let approver = tokio::spawn(async move {
            if let Some(request) = rx.recv().await {
                let _ = request.respond.send(true);
            }
        });

        let result = orch
            .execute_native_tool_with_retry(&call("write_file"))
            .await;
        approver.await.unwrap();

        assert!(!result.is_error);
        assert_eq!(
            recorder.calls(),
            vec!["write_file".to_string()],
            "the tool must run exactly once"
        );
    }

    #[tokio::test]
    async fn yolo_skips_the_prompt_entirely() {
        let recorder = Arc::new(RecordingTools::new());
        let tools: Arc<dyn ToolExecutor> = recorder.clone();
        let orch = orchestrator(tools, AutonomyLevel::Yolo);

        let result = orch
            .execute_native_tool_with_retry(&call("write_file"))
            .await;

        assert!(!result.is_error);
        assert_eq!(recorder.calls(), vec!["write_file".to_string()]);
    }

    #[tokio::test]
    async fn test_confidence_parsing_from_text() {
        let text = r#"
        <analysis>
        Decision: Option 1 because it is simpler
        Confidence: low — the API might have changed
        KeyTradeoff: simplicity vs correctness
        </analysis>
        "#;
        let meta = extract_decision_metadata(text);
        assert_eq!(meta.confidence, Some("low".to_string()));
        assert_eq!(
            meta.chosen_approach,
            Some("Option 1 because it is simpler".to_string())
        );
        assert_eq!(
            meta.key_tradeoff,
            Some("simplicity vs correctness".to_string())
        );
    }

    #[tokio::test]
    async fn test_backtrack_detection() {
        let text = "Backtracking: test failed. Trying alternative approach.";
        let meta = extract_decision_metadata(text);
        assert!(meta.is_backtrack);
    }
}
