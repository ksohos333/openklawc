//! openklawc-core — foundational types, traits, and the event bus.
//!
//! Everything in openklawc builds on `okc-core`. It defines:
//! - The agent loop state machine (`AgentLoop`, `Turn`)
//! - Message and event types (`Event`, `TurnEvent`)
//! - Tool abstractions (`Tool`, `ToolCall`, `ToolResult`)
//! - The reasoning / autonomy types (`ReasoningStep`, `AutonomyLevel`)

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::{broadcast, mpsc};

pub mod util;

// ═══════════════════════════════════════════════════════════════════════════════
// IDENTIFIERS
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Id(pub String);

impl Id {
    pub fn new() -> Self {
        Self(uuid::Uuid::new_v4().to_string())
    }
}

impl fmt::Display for Id {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl Default for Id {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EVENT BUS
// ═══════════════════════════════════════════════════════════════════════════════

/// Global event type. All subsystems produce and consume these.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Event {
    /// A user sent a message.
    UserMessage { session_id: Id, content: String },
    /// An assistant produced a message (or delta).
    AssistantMessage { session_id: Id, delta: MessageDelta },
    /// A reasoning step is visible to the user.
    Reasoning { session_id: Id, step: ReasoningStep },
    /// A tool was called.
    ToolCall { session_id: Id, call: ToolCall },
    /// A tool returned a result.
    ToolResult { session_id: Id, result: ToolOutcome },
    /// The autonomy level changed.
    AutonomyChanged { level: AutonomyLevel },
    /// A file was mutated (created, updated, deleted).
    FileMutation { path: String, kind: MutationKind },
    /// A command was executed (shell, git, etc.).
    CommandExecuted { command: String, stdout: String, stderr: String, exit_code: Option<i32> },
    /// Session lifecycle.
    SessionCreated { session_id: Id, name: String },
    SessionLoaded { session_id: Id },
    SessionSaved { session_id: Id },
    /// Errors.
    Error { message: String },
    /// Raw log line.
    Log { level: LogLevel, message: String },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LogLevel {
    Trace,
    Debug,
    Info,
    Warn,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MessageDelta {
    Start { message_id: Id },
    Content { text: String },
    Done,
    Error { message: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MutationKind {
    Created,
    Updated,
    Deleted,
}

/// Central event bus. Clone this to subscribe or publish.
#[derive(Clone)]
pub struct EventBus {
    tx: broadcast::Sender<Event>,
}

impl EventBus {
    pub fn new(capacity: usize) -> Self {
        let (tx, _) = broadcast::channel(capacity);
        Self { tx }
    }

    pub fn publish(&self, event: Event) {
        let _ = self.tx.send(event);
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Event> {
        self.tx.subscribe()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MESSAGES (LLM-facing)
// ═══════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

impl fmt::Display for Role {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Role::System => write!(f, "system"),
            Role::User => write!(f, "user"),
            Role::Assistant => write!(f, "assistant"),
            Role::Tool => write!(f, "tool"),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOOL SYSTEM
// ═══════════════════════════════════════════════════════════════════════════════

/// Schema for a tool parameter.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    pub parameters: Value, // JSON Schema object
}

/// A concrete tool call requested by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: Id,
    pub name: String,
    pub arguments: Value,
}

/// The outcome of a tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutcome {
    pub call_id: Id,
    pub success: bool,
    pub result: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    pub elapsed_ms: u64,
}

/// Trait for anything that can execute tools.
#[async_trait]
pub trait ToolExecutor: Send + Sync {
    /// List available tools.
    async fn list_tools(&self) -> anyhow::Result<Vec<ToolSchema>>;
    /// Execute one tool call.
    async fn execute_tool(&self, call: &ToolCall) -> anyhow::Result<ToolOutcome>;
}

// ═══════════════════════════════════════════════════════════════════════════════
// REASONING / AUTONOMY
// ═══════════════════════════════════════════════════════════════════════════════

/// The user's chosen autonomy level. This controls how much the agent
/// can do without explicit confirmation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum AutonomyLevel {
    /// Ask before EVERY tool call and command.
    #[default]
    Ask,
    /// YOLO mode: execute read-only tools silently, ask for mutations.
    ReadOnly,
    /// YOLO++: execute everything, log what happened.
    Yolo,
    /// God mode: execute everything, don't even log (not recommended).
    God,
}

impl AutonomyLevel {
    pub fn needs_confirm(&self, tool_name: &str, is_mutation: bool) -> bool {
        match self {
            AutonomyLevel::Ask => true,
            AutonomyLevel::ReadOnly => is_mutation,
            AutonomyLevel::Yolo | AutonomyLevel::God => false,
        }
    }

    pub fn can_read(&self) -> bool {
        true // All levels can read
    }

    pub fn can_mutate(&self) -> bool {
        matches!(self, AutonomyLevel::Yolo | AutonomyLevel::God)
    }
}

/// One visible reasoning step. The TUI renders these inline so the user
/// sees the agent's "thought process".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningStep {
    pub timestamp: DateTime<Utc>,
    pub kind: ReasoningKind,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call: Option<ToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_result: Option<ToolOutcome>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReasoningKind {
    Plan,       // "I need to search for X, then edit Y"
    Observe,    // "The search returned ..."
    Reflect,    // "That didn't work, trying Z instead"
    Decide,     // "I will run `cargo test`"
    Execute,    // Actually running something
    Error,      // Something went wrong
}

// ═══════════════════════════════════════════════════════════════════════════════
// AGENT LOOP
// ═══════════════════════════════════════════════════════════════════════════════

/// The agent's turn state machine — Hermes-inspired.
///
/// Each turn goes through:
///   Plan → (observe context, search, etc.)
///   → Decide → Execute tools
///   → Reflect → Respond
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnPhase {
    Idle,
    Planning,
    Observing,
    Executing,
    Reflecting,
    Responding,
    WaitingForUser, // blocked on confirmation
}

/// Mutable state for one active turn.
pub struct TurnState {
    pub phase: TurnPhase,
    pub plan: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    pub tool_results: Vec<ToolOutcome>,
    pub reasoning_log: Vec<ReasoningStep>,
    pub started_at: DateTime<Utc>,
}

impl TurnState {
    pub fn new() -> Self {
        Self {
            phase: TurnPhase::Idle,
            plan: None,
            tool_calls: Vec::new(),
            tool_results: Vec::new(),
            reasoning_log: Vec::new(),
            started_at: Utc::now(),
        }
    }
}

impl Default for TurnState {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// SESSION
// ═══════════════════════════════════════════════════════════════════════════════

/// One conversation session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Id,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub messages: Vec<ChatMessage>,
    pub metadata: HashMap<String, Value>,
    pub cwd: String,
}

impl Session {
    pub fn new(name: impl Into<String>, cwd: impl Into<String>) -> Self {
        let now = Utc::now();
        Self {
            id: Id::new(),
            name: name.into(),
            created_at: now,
            updated_at: now,
            messages: Vec::new(),
            metadata: HashMap::new(),
            cwd: cwd.into(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// MODEL / PROVIDER DESCRIPTOR
// ═══════════════════════════════════════════════════════════════════════════════

/// A qualified model reference: `provider/model_id`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelRef {
    pub provider: String,
    pub model_id: String,
    pub display_name: String,
    pub capabilities: ModelCapabilities,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ModelCapabilities {
    pub reasoning: bool,
    pub vision: bool,
    pub streaming: bool,
    pub function_calling: bool,
    pub max_tokens: Option<u32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// NATIVE TOOL TYPES (Autarky)
// ═══════════════════════════════════════════════════════════════════════════════

/// One tool made available to the LLM in our native format.
/// Providers translate this to their wire format at the boundary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeToolDef {
    pub name: String,
    pub description: String,
    /// JSON Schema object describing parameters.
    pub parameters: Value,
}

/// A tool call emitted by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeToolCall {
    /// Provider-generated call ID (must echo back in result).
    pub id: String,
    pub name: String,
    pub arguments: Value,
}

/// The result of executing one tool call, fed back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NativeToolResult {
    /// Must match the `id` from the corresponding `NativeToolCall`.
    pub call_id: String,
    /// Text content the model will see.
    pub content: String,
    /// If true, the model receives this as an error.
    #[serde(default)]
    pub is_error: bool,
}

/// What an LLM turn produces: either text, tool calls, or both.
#[derive(Debug, Clone, Default)]
pub struct TurnOutput {
    pub text: String,
    pub tool_calls: Vec<NativeToolCall>,
}

impl TurnOutput {
    pub fn is_tool_call(&self) -> bool {
        !self.tool_calls.is_empty()
    }
    pub fn is_empty(&self) -> bool {
        self.text.is_empty() && self.tool_calls.is_empty()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// STREAMING
// ═══════════════════════════════════════════════════════════════════════════════

pub type StreamTx = mpsc::Sender<String>;
pub type StreamRx = mpsc::Receiver<String>;

/// Trait for LLM providers.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    fn name(&self) -> &str;
    fn supports_streaming(&self) -> bool;

    /// Basic chat without tools (text-only fallback).
    async fn chat(&self, model: &str, messages: Vec<ChatMessage>) -> anyhow::Result<String>;

    /// Basic streaming chat without tools.
    async fn chat_stream(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tx: StreamTx,
    ) -> anyhow::Result<()>;

    // ─── Native tool calling (optional, default = not supported) ───────────

    fn supports_native_tools(&self) -> bool {
        false
    }

    /// Chat with native tool support. Returns TurnOutput (text + tool_calls).
    /// Default implementation falls back to text injection + text parsing.
    async fn chat_with_tools(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<NativeToolDef>,
    ) -> anyhow::Result<TurnOutput> {
        let mut msgs = messages;
        if let Some(sys) = msgs.iter_mut().find(|m| m.role == Role::System) {
            sys.content = inject_tools_into_prompt(&sys.content, &tools);
        }
        let text = self.chat(model, msgs).await?;
        Ok(parse_tool_calls_native(&text))
    }

    /// Streaming chat with native tool support.
    /// Default implementation falls back to text-only stream.
    async fn chat_stream_with_tools(
        &self,
        model: &str,
        messages: Vec<ChatMessage>,
        tools: Vec<NativeToolDef>,
        tx: StreamTx,
    ) -> anyhow::Result<TurnOutput> {
        let mut msgs = messages;
        if let Some(sys) = msgs.iter_mut().find(|m| m.role == Role::System) {
            sys.content = inject_tools_into_prompt(&sys.content, &tools);
        }
        self.chat_stream(model, msgs, tx).await?;
        // Streaming doesn't return TurnOutput directly; caller must re-parse.
        // This is the fallback path.
        Ok(TurnOutput::default())
    }
}

fn inject_tools_into_prompt(base: &str, tools: &[NativeToolDef]) -> String {
    let mut prompt = base.to_string();
    if !tools.is_empty() {
        prompt.push_str("\n\nYou have access to the following tools. ");
        prompt.push_str("When you need to use a tool, output ONE standalone JSON block per line:\n");
        prompt.push_str(r#"{"tool": "tool_name", "arguments": {"param": "value"}}"#);
        prompt.push('\n');
        prompt.push_str("Do NOT wrap tool calls in markdown code fences.\n\n");
        prompt.push_str("Available tools:\n");
        for tool in tools {
            let schema = serde_json::to_string(&tool.parameters).unwrap_or_default();
            prompt.push_str(&format!("\n### {}\n{}", tool.name, tool.description));
            prompt.push_str(&format!("\n  Parameters: {}\n", schema));
        }
    }
    prompt
}

pub fn parse_tool_calls_native(text: &str) -> TurnOutput {
    let (clean_text, calls) = parse_tool_calls(text);
    TurnOutput {
        text: clean_text,
        tool_calls: calls
            .into_iter()
            .map(|tc| NativeToolCall {
                id: tc.id.to_string(),
                name: tc.name,
                arguments: tc.arguments,
            })
            .collect(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TOOL CALL PARSING
// ═══════════════════════════════════════════════════════════════════════════════

/// Parse tool calls embedded in an LLM text response.
/// 
/// Detects several formats:
///   1. OpenAI function call JSON: {"name":"tool","arguments":{...}}
///   2. Anthropic tool use: {"tool":"name", "arguments":{...}}
///   3. Code fence: ```json\n{"tool":"name",...}\n```
///
/// Returns (cleaned_response_text, tool_calls)
pub fn parse_tool_calls(text: &str) -> (String, Vec<ToolCall>) {
    let mut tool_calls = Vec::new();
    let mut cleaned = text.to_string();

    // Extract fenced JSON blocks with "tool" or "name" + "arguments"
    // Try fenced first
    let fence_re = regex::Regex::new(r"```(?:json)?\s*\n(.*?)\n```").unwrap();
    for cap in fence_re.captures_iter(text) {
        let json_str = &cap[1];
        if let Some(call) = try_parse_tool_json(json_str) {
            tool_calls.push(call);
            cleaned = cleaned.replace(&cap[0], "");
        }
    }

    // Try inline JSON objects
    let inline_re = regex::Regex::new(r"\{(?:[^{}]|\{[^{}]*\})*\}").unwrap();
    for cap in inline_re.captures_iter(text) {
        let json_str = &cap[0];
        if let Some(call) = try_parse_tool_json(json_str) {
            // Don't duplicate
            if !tool_calls.iter().any(|c: &ToolCall| c.id.0 == call.id.0) {
                tool_calls.push(call);
                cleaned = cleaned.replace(&cap[0], "");
            }
        }
    }

    (cleaned.trim().to_string(), tool_calls)
}

fn try_parse_tool_json(json_str: &str) -> Option<ToolCall> {
    let value: serde_json::Value = serde_json::from_str(json_str).ok()?;

    // Format 1: {"tool": "name", "arguments": {...}}
    if let Some(tool) = value.get("tool").and_then(|v| v.as_str()) {
        let args = value.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
        return Some(ToolCall {
            id: Id::new(),
            name: tool.to_string(),
            arguments: args,
        });
    }

    // Format 2: {"name": "name", "arguments": {...}} (OpenAI style)
    if let Some(name) = value.get("name").and_then(|v| v.as_str()) {
        let args = value.get("arguments").cloned().unwrap_or(serde_json::Value::Null);
        return Some(ToolCall {
            id: Id::new(),
            name: name.to_string(),
            arguments: args,
        });
    }

    // Format 3: {"function": {"name": "...", "arguments": "..."}} (OpenAI chat completion)
    if let Some(func) = value.get("function") {
        if let Some(name) = func.get("name").and_then(|v| v.as_str()) {
            let args_str = func.get("arguments").and_then(|v| v.as_str()).unwrap_or("{}");
            let args = serde_json::from_str(args_str).unwrap_or(serde_json::Value::Null);
            return Some(ToolCall {
                id: Id::new(),
                name: name.to_string(),
                arguments: args,
            });
        }
    }

    None
}

// ═══════════════════════════════════════════════════════════════════════════════
// SIMPLE SHARED TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Result of a shell command execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShellResult {
    pub stdout: String,
    pub stderr: String,
    pub exit_code: Option<i32>,
}

/// File change kind for diffs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FileChangeKind {
    Add,
    Remove,
    Modify,
    Rename,
}
