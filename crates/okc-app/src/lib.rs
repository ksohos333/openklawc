//! okc-app — the openklawc application core.
//!
//! Everything a front-end (TUI, headless, future daemon) needs to run an
//! openklawc session, behind one type: [`App`]. Front-ends render
//! [`AgentUpdate`]s; this crate owns bootstrap, model resolution, tool and
//! MCP assembly, the agent, RAG indexing, session lifecycle and persistence.
//!
//! The approval contract: a front-end that can ask a human attaches a
//! confirm channel via [`App::attach_approval`]. A front-end that cannot
//! (headless) attaches nothing, and every tool call that needs confirmation
//! is then **denied** — explicitly, by construction.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use tokio::sync::mpsc;
use tracing::warn;

use okc_config::{Config, ModelPricing};
use okc_core::{AutonomyLevel, ChatMessage, Event, EventBus, TokenUsage, ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use okc_llm::{LlmClient, ProviderRegistry};
use okc_mcp::McpRegistry;
use okc_policy::PolicyExecutor;
use okc_rag::SimpleRag;
use okc_reasoning::{Agent, AgentUpdate, ConfirmRequest};
use okc_session::{SessionConfig, SessionStore};
use okc_tool::{BuiltInTools, CompositeExecutor, CustomTools};

/// Options for [`App::boot`].
pub struct AppOptions {
    /// Model override (`provider/model`), e.g. from `-m` or `OKC_MODEL`.
    pub model: Option<String>,
    /// Autonomy override, e.g. from `--autonomy` or `OKC_AUTONOMY`.
    pub autonomy: Option<AutonomyLevel>,
    /// Persona name, e.g. from `--persona`.
    pub persona: Option<String>,
    /// Start MCP servers (false for `--no-mcp`).
    pub start_mcp: bool,
    /// Where the audit trail is written. `None` uses the session database
    /// next to the app's data; tests point this at a temp file.
    pub audit_db: Option<std::path::PathBuf>,
}

impl Default for AppOptions {
    fn default() -> Self {
        Self {
            model: None,
            autonomy: None,
            persona: None,
            start_mcp: true,
            audit_db: None,
        }
    }
}

/// What [`App::boot`] found on the way up.
pub struct BootInfo {
    /// Number of MCP tools registered.
    pub mcp_tools: usize,
    /// MCP servers that failed to start (`name: error`), for display.
    pub mcp_failures: Vec<String>,
    /// The resolved qualified model id (`provider/model`).
    pub model: String,
}

/// The openklawc application core.
pub struct App {
    pub cfg: Config,
    pub event_bus: EventBus,
    pub llm_registry: ProviderRegistry,
    pub llm_client: LlmClient,
    /// Qualified model id (`provider/model`) currently in use.
    pub model_id: String,
    pub tools: Arc<dyn ToolExecutor>,
    /// Number of MCP tools registered at boot.
    pub mcp_tool_count: usize,
    /// The template agent; forked per turn by [`App::submit`].
    pub agent: Agent,
    pub rag: SimpleRag,
    pub session_store: Option<SessionStore>,
}

impl App {
    /// Bootstrap the application core from default config paths.
    pub async fn boot(opts: AppOptions) -> Result<(App, BootInfo)> {
        Self::boot_with(opts, None).await
    }

    /// Bootstrap with an optional provider-registry override (tests,
    /// embedders). When given, `[provider]` config tables are ignored.
    pub async fn boot_with(
        opts: AppOptions,
        registry: Option<ProviderRegistry>,
    ) -> Result<(App, BootInfo)> {
        let cfg = Config::load_default().unwrap_or_default();
        let audit_db = opts
            .audit_db
            .clone()
            .unwrap_or_else(SessionStore::default_path);

        // Environment overrides — highest precedence, above config files.
        let model_req = std::env::var("OKC_MODEL").ok().or(opts.model);
        let autonomy_req = std::env::var("OKC_AUTONOMY")
            .ok()
            .map(|s| parse_autonomy(&s))
            .or(opts.autonomy);

        let event_bus = EventBus::new(256);
        let llm_registry =
            registry.unwrap_or_else(|| ProviderRegistry::from_config(&cfg));
        let llm_client = LlmClient::new();

        let model_id = resolve_model(&llm_registry, &cfg, model_req.as_deref())?;
        let (provider, model) = llm_registry
            .get(&model_id)
            .ok_or_else(|| anyhow!("model '{}' disappeared during boot", model_id))?;

        // ─── Tools: built-in (filtered by [tools] flags) + custom + MCP ──
        let mut composite = CompositeExecutor::new();
        composite.add(Arc::new(BuiltInTools::with_enabled(&cfg.tools)));
        composite.add(Arc::new(CustomTools::from_config(&cfg)));

        let mut mcp_tool_count = 0;
        let mut mcp_failures = Vec::new();
        if opts.start_mcp {
            match McpRegistry::start_with_report(&cfg.mcp).await {
                Ok((registry, failures)) => {
                    mcp_tool_count = registry.all_tools().len();
                    mcp_failures = failures;
                    composite.add(Arc::new(registry));
                }
                Err(e) => mcp_failures.push(format!("registry: {}", e)),
            }
        }
        // Persona (built once — its tool restriction also feeds policy).
        let persona = build_persona(opts.persona.as_deref(), &cfg);
        let persona_allowed = persona
            .as_ref()
            .map(|p| p.allowed_tools.clone())
            .filter(|s| !s.is_empty());

        // Extra mutation classification: config-listed tools plus every
        // custom tool that is not explicitly vouched for with
        // `mutating = false`. These require approval outside yolo/god.
        let mut mutation_extra: std::collections::HashSet<String> =
            cfg.policy.mutating_tools.iter().cloned().collect();
        for tool in &cfg.tools.custom {
            if tool.mutating {
                mutation_extra.insert(tool.name.clone());
            }
        }

        // Policy inside the timeout: verdicts return instantly, inner calls
        // stay bounded. Policy outranks autonomy — a deny is a deny, even
        // in `god`.
        let policy = PolicyExecutor::new(
            Arc::new(composite),
            &cfg.policy,
            persona_allowed,
            event_bus.clone(),
        );
        let tools: Arc<dyn ToolExecutor> = Arc::new(TimeoutExecutor {
            inner: Arc::new(policy),
            timeout: Duration::from_millis(cfg.orchestrator.call_timeout_ms),
        });

        let autonomy =
            autonomy_req.unwrap_or_else(|| parse_autonomy(&cfg.autonomy.default_level));

        let mut agent = Agent::new(
            llm_client.clone(),
            provider,
            model,
            tools.clone(),
            autonomy,
            event_bus.clone(),
        )
        .with_config((&cfg.orchestrator).into())
        .with_mutation_extra(mutation_extra);

        if let Some(p) = persona {
            agent = agent.with_persona(p);
        }

        // ─── RAG index of the working directory ───────────────────────────
        let mut rag = SimpleRag::new();
        if let Ok(cwd) = std::env::current_dir() {
            let _ = rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions);
        }

        // ─── Sessions ─────────────────────────────────────────────────────
        let session_store = SessionStore::new().ok();

        // ─── Audit trail ─────────────────────────────────────────────────
        // Every event on the bus is written to the session database: tool
        // calls and results, file mutations, command executions, user
        // messages, reasoning steps. The subscriber runs on its own thread
        // so a slow disk never blocks the agent loop.
        {
            let mut rx = event_bus.subscribe();
            std::thread::Builder::new()
                .name("okc-audit".to_string())
                .spawn(move || {
                    let Ok(store) = SessionStore::open(&audit_db) else {
                        return;
                    };
                    while let Ok(event) = rx.blocking_recv() {
                        let ts = chrono::Utc::now().to_rfc3339();
                        match &event {
                            Event::Reasoning { session_id, step } => {
                                let _ = store.append_reasoning_step(&session_id.0, step);
                            }
                            Event::UserMessage {
                                session_id,
                                content,
                            } => {
                                let _ = store.append_audit(
                                    &ts,
                                    Some(&session_id.0),
                                    "UserMessage",
                                    &serde_json::json!({ "content": content }),
                                );
                            }
                            other => {
                                let (kind, session_id, payload) = flatten_event(other);
                                let _ =
                                    store.append_audit(&ts, session_id.as_deref(), kind, &payload);
                            }
                        }
                    }
                })
                .ok();
        }

        Ok((
            App {
                cfg,
                event_bus,
                llm_registry,
                llm_client,
                model_id: model_id.clone(),
                tools,
                mcp_tool_count,
                agent,
                rag,
                session_store,
            },
            BootInfo {
                mcp_tools: mcp_tool_count,
                mcp_failures,
                model: model_id,
            },
        ))
    }

    /// Switch the active model. Returns false for unknown ids.
    pub fn set_model(&mut self, qualified: &str) -> bool {
        match self.llm_registry.get(qualified) {
            Some((provider, model)) => {
                self.agent.provider = provider;
                self.agent.model_id = model;
                self.model_id = qualified.to_string();
                true
            }
            None => false,
        }
    }

    /// Change the autonomy level and announce it on the event bus.
    pub fn set_autonomy(&mut self, level: AutonomyLevel) {
        self.agent.autonomy = level;
        self.event_bus
            .publish(okc_core::Event::AutonomyChanged { level });
    }

    /// Bind the agent to a session so its user messages and reasoning steps
    /// land in that session's audit trail.
    pub fn bind_session(&mut self, session_id: &str) {
        self.agent.session_id = Some(session_id.to_string());
    }

    /// Recent audit events for a session, newest first.
    pub fn audit_trail(&self, session_id: &str, limit: usize) -> Vec<okc_session::AuditEvent> {
        self.session_store
            .as_ref()
            .and_then(|s| s.audit_for_session(session_id, limit).ok())
            .unwrap_or_default()
    }

    /// Reasoning steps for a session, oldest first — the decision trail.
    pub fn reasoning_trail(
        &self,
        session_id: &str,
        limit: usize,
    ) -> Vec<okc_session::ReasoningRow> {
        self.session_store
            .as_ref()
            .and_then(|s| s.reasoning_for_session(session_id, limit).ok())
            .unwrap_or_default()
    }

    /// Attach an approval channel (the TUI's confirm queue). Without one,
    /// any tool call that needs confirmation is denied — the explicit
    /// contract for headless runs.
    pub fn attach_approval(&mut self, tx: mpsc::Sender<ConfirmRequest>) {
        self.agent.confirm_tx = Some(tx);
    }

    /// Create a new session; returns its id, or None when the store is down.
    pub fn create_session(&mut self, title: &str, mode: &str) -> Option<String> {
        let store = self.session_store.as_mut()?;
        let cwd = std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().to_string());
        let provider = self.model_id.split('/').next().map(|s| s.to_string());
        let config = SessionConfig {
            agent_mode: Some(mode.to_string()),
            ..Default::default()
        };
        store
            .create_with_config(
                title,
                Some(&self.model_id),
                provider.as_deref(),
                cwd.as_deref(),
                &config,
            )
            .ok()
    }

    /// Persist a finished turn into the session store.
    pub fn persist_turn(
        &mut self,
        session_id: &str,
        persisted_count: &mut usize,
        messages: &[ChatMessage],
        usage: Option<TokenUsage>,
    ) {
        let Some(store) = self.session_store.as_mut() else {
            return;
        };
        match store.sync_messages(session_id, messages, *persisted_count) {
            Ok(stored) => *persisted_count = stored,
            Err(e) => warn!("failed to persist messages: {}", e),
        }
        if let Ok(mut config) = store.get_config(session_id) {
            if let Some(u) = usage {
                config.tokens_in += u.prompt;
                config.tokens_out += u.completion;
            }
            config.cost_usd = cost_for(
                &self.cfg.pricing,
                &self.model_id,
                TokenUsage::new(config.tokens_in, config.tokens_out),
            );
            if let Err(e) = store.set_config(session_id, &config) {
                warn!("failed to update session config: {}", e);
            }
        }
    }

    /// Total tokens recorded for a session (in + out), 0 when unknown.
    pub fn session_tokens(&self, session_id: &str) -> u64 {
        self.session_store
            .as_ref()
            .and_then(|s| s.get_config(session_id).ok())
            .map(|c| c.tokens_in + c.tokens_out)
            .unwrap_or(0)
    }

    /// Run one agent turn. Enriches `@file` references, forks the template
    /// agent, spawns the turn, and returns the update stream to render.
    /// The turn's final message list arrives in `AgentUpdate::Done`.
    pub async fn submit(
        &mut self,
        input: &str,
        messages: Vec<ChatMessage>,
    ) -> mpsc::Receiver<AgentUpdate> {
        let file_refs = okc_core::util::extract_file_refs(input);
        let clean = okc_core::util::strip_file_refs(input);
        let mut enriched = clean;
        for path in &file_refs {
            if let Ok(content) = tokio::fs::read_to_string(path).await {
                enriched.push_str(&format!("\n\n--- @{} ---\n{}", path, content));
            }
        }

        let (tx, rx) = mpsc::channel::<AgentUpdate>(256);
        let mut task_agent = self.agent.fork();
        task_agent.update_tx = Some(tx);
        tokio::spawn(async move {
            let _ = task_agent.run_turn(&enriched, messages).await;
        });
        rx
    }

    /// Recent sessions, newest first.
    pub fn list_sessions(&self, limit: usize) -> Vec<okc_session::SessionMeta> {
        self.session_store
            .as_ref()
            .and_then(|s| s.list(limit).ok())
            .unwrap_or_default()
    }

    /// Full-text session search.
    pub fn search_sessions(
        &self,
        query: &str,
        limit: usize,
    ) -> Vec<okc_session::SearchResult> {
        self.session_store
            .as_ref()
            .and_then(|s| s.search(query, limit).ok())
            .unwrap_or_default()
    }

    /// Stored messages of a session (for resume).
    pub fn get_messages(&self, id: &str) -> Option<Vec<okc_session::PersistentMessage>> {
        self.session_store.as_ref()?.get_messages(id).ok()
    }

    /// Verified / unverified claim counts for a session, based on the
    /// `note_observation` calls in its audit trail.
    pub fn verification_counts(&self, session_id: &str) -> (usize, usize) {
        let events = self.audit_trail(session_id, 500);
        count_verifications(&events)
    }

    /// Delete a session. Returns false when the store is down or the id is
    /// unknown.
    pub fn delete_session(&mut self, id: &str) -> bool {
        match self.session_store.as_mut() {
            Some(store) => store.delete(id).is_ok(),
            None => false,
        }
    }
}

/// Count verified / unverified claims among `note_observation` audit
/// payloads. An observation is verified when its `source_kind` says so —
/// the agent's self-report, cross-checked against the sources it passed.
pub fn count_verifications(events: &[okc_session::AuditEvent]) -> (usize, usize) {
    let mut verified = 0;
    let mut unverified = 0;
    for event in events {
        if event.kind != "ToolCall" {
            continue;
        }
        let Some(call) = event.payload.get("ToolCall").and_then(|t| t.get("call")) else {
            continue;
        };
        if call.get("name").and_then(|n| n.as_str()) != Some("note_observation") {
            continue;
        }
        let kind = call
            .get("arguments")
            .and_then(|a| a.get("source_kind"))
            .and_then(|k| k.as_str())
            .unwrap_or("unspecified")
            .to_lowercase();
        if kind.contains("unverified") {
            unverified += 1;
        } else if kind.contains("verified") {
            verified += 1;
        } else {
            unverified += 1;
        }
    }
    (verified, unverified)
}

/// Split an event into (kind, session id, payload) for the audit table.
fn flatten_event(event: &Event) -> (&'static str, Option<String>, serde_json::Value) {
    let kind = match event {
        Event::UserMessage { .. } => "UserMessage",
        Event::AssistantMessage { .. } => "AssistantMessage",
        Event::Reasoning { .. } => "Reasoning",
        Event::ToolCall { .. } => "ToolCall",
        Event::ToolResult { .. } => "ToolResult",
        Event::AutonomyChanged { .. } => "AutonomyChanged",
        Event::FileMutation { .. } => "FileMutation",
        Event::CommandExecuted { .. } => "CommandExecuted",
        Event::SessionCreated { .. } => "SessionCreated",
        Event::SessionLoaded { .. } => "SessionLoaded",
        Event::SessionSaved { .. } => "SessionSaved",
        Event::Error { .. } => "Error",
        Event::Log { .. } => "Log",
    };
    let session_id = match event {
        Event::UserMessage { session_id, .. }
        | Event::AssistantMessage { session_id, .. }
        | Event::Reasoning { session_id, .. }
        | Event::ToolCall { session_id, .. }
        | Event::ToolResult { session_id, .. }
        | Event::SessionCreated { session_id, .. }
        | Event::SessionLoaded { session_id, .. }
        | Event::SessionSaved { session_id, .. } => Some(session_id.0.clone()),
        _ => None,
    };
    (
        kind,
        session_id,
        serde_json::to_value(event).unwrap_or(serde_json::Value::Null),
    )
}

/// Resolve which model to use.
///
/// - An explicit request (`-m` / `OKC_MODEL`) that cannot be resolved is an
///   error: the caller's intent was clear.
/// - The configured default falls back to the first registered model.
/// - An empty registry is an error — there is nothing to talk to.
fn resolve_model(
    registry: &ProviderRegistry,
    cfg: &Config,
    requested: Option<&str>,
) -> Result<String> {
    if let Some(q) = requested {
        if registry.get(q).is_some() {
            return Ok(q.to_string());
        }
        return Err(anyhow!(
            "unknown model '{}' — run /models inside the TUI to list available models",
            q
        ));
    }
    if registry.get(&cfg.model).is_some() {
        return Ok(cfg.model.clone());
    }
    if let Some(first) = registry.list().first() {
        let q = format!("{}/{}", first.provider, first.model_id);
        warn!(
            "configured model '{}' not found — falling back to '{}'",
            cfg.model, q
        );
        return Ok(q);
    }
    Err(anyhow!(
        "no providers configured — add a [provider.<name>] table to \
         ~/.config/openklawc/config.toml or .openklawc.toml"
    ))
}

/// Wraps a [`ToolExecutor`] with a hard per-call timeout.
struct TimeoutExecutor {
    inner: Arc<dyn ToolExecutor>,
    timeout: Duration,
}

#[async_trait::async_trait]
impl ToolExecutor for TimeoutExecutor {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        self.inner.list_tools().await
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        match tokio::time::timeout(self.timeout, self.inner.execute_tool(call)).await {
            Ok(result) => result,
            Err(_) => Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: false,
                result: serde_json::Value::Null,
                error: Some(format!(
                    "tool '{}' timed out after {}ms",
                    call.name,
                    self.timeout.as_millis()
                )),
                elapsed_ms: self.timeout.as_millis() as u64,
            }),
        }
    }
}

/// Parse an autonomy level name; unknown names degrade to `ReadOnly`.
pub fn parse_autonomy(s: &str) -> AutonomyLevel {
    match s.to_lowercase().as_str() {
        "ask" => AutonomyLevel::Ask,
        "read_only" | "readonly" => AutonomyLevel::ReadOnly,
        "yolo" => AutonomyLevel::Yolo,
        "god" => AutonomyLevel::God,
        _ => AutonomyLevel::ReadOnly,
    }
}

/// Build a persona from an explicit request or the configured default.
pub fn build_persona(
    requested: Option<&str>,
    cfg: &Config,
) -> Option<okc_reasoning::persona::Persona> {
    let name = requested.or(cfg.default_persona.as_deref())?;

    // Config-defined personas first.
    if let Some(pc) = cfg.personas.get(name) {
        return Some(okc_reasoning::persona::Persona::from_config(name, pc));
    }

    // Fall back to built-in personas.
    okc_reasoning::persona::builtin_persona_prompt(name).map(|prompt| {
        okc_reasoning::persona::Persona {
            name: Some(name.to_string()),
            system_prompt: prompt,
            allowed_tools: std::collections::HashSet::new(),
            model_override: None,
            temperature: None,
            max_iterations: None,
            can_delegate: true,
        }
    })
}

/// Derive a chat title from the first user message.
pub fn title_from(text: &str) -> String {
    let one_line: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if one_line.is_empty() {
        return "New chat".to_string();
    }
    let truncated: String = one_line.chars().take(60).collect();
    if one_line.chars().count() > 60 {
        format!("{}…", truncated)
    } else {
        truncated
    }
}

/// Cost in USD for the given usage, when the model has a configured price.
pub fn cost_for(
    pricing: &std::collections::HashMap<String, ModelPricing>,
    model: &str,
    usage: TokenUsage,
) -> Option<f64> {
    let price = ModelPricing::lookup(pricing, model)?;
    Some(
        (usage.prompt as f64 / 1_000_000.0) * price.input
            + (usage.completion as f64 / 1_000_000.0) * price.output,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn title_from_flattens_and_truncates() {
        assert_eq!(title_from("hello   world"), "hello world");
        assert_eq!(title_from(""), "New chat");
        let long = "x".repeat(100);
        let title = title_from(&long);
        assert_eq!(title.chars().count(), 61);
        assert!(title.ends_with('…'));
    }

    #[test]
    fn cost_is_unknown_without_a_price() {
        let pricing = std::collections::HashMap::new();
        assert_eq!(
            cost_for(&pricing, "minimax/MiniMax-M3", TokenUsage::new(1000, 1000)),
            None
        );
    }

    #[test]
    fn cost_uses_the_configured_price() {
        let mut pricing = std::collections::HashMap::new();
        pricing.insert(
            "MiniMax-M3".to_string(),
            ModelPricing {
                input: 1.0,
                output: 2.0,
            },
        );
        let cost = cost_for(
            &pricing,
            "minimax/MiniMax-M3",
            TokenUsage::new(1_000_000, 1_000_000),
        )
        .unwrap();
        assert!((cost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn unknown_autonomy_degrades_to_read_only() {
        assert_eq!(parse_autonomy("yolo"), AutonomyLevel::Yolo);
        assert_eq!(parse_autonomy("READONLY"), AutonomyLevel::ReadOnly);
        assert_eq!(parse_autonomy("nonsense"), AutonomyLevel::ReadOnly);
    }

    #[test]
    fn tool_config_defaults_to_everything_enabled() {
        let cfg = okc_config::ToolConfig::default();
        assert!(cfg.shell && cfg.file_read && cfg.file_write && cfg.git && cfg.search);
    }

    #[test]
    fn flatten_event_extracts_kind_and_session() {
        let event = Event::ToolResult {
            session_id: okc_core::Id("s1".to_string()),
            result: okc_core::ToolOutcome {
                call_id: okc_core::Id("c1".to_string()),
                success: true,
                result: serde_json::json!({"ok": true}),
                error: None,
                elapsed_ms: 3,
            },
        };
        let (kind, session_id, payload) = flatten_event(&event);
        assert_eq!(kind, "ToolResult");
        assert_eq!(session_id.as_deref(), Some("s1"));
        assert!(payload.get("ToolResult").is_some(), "payload: {}", payload);
    }

    #[test]
    fn flatten_event_handles_session_less_events() {
        let event = Event::AutonomyChanged {
            level: AutonomyLevel::Yolo,
        };
        let (kind, session_id, _) = flatten_event(&event);
        assert_eq!(kind, "AutonomyChanged");
        assert!(session_id.is_none());
    }

    fn audit_event(kind: &str, payload: serde_json::Value) -> okc_session::AuditEvent {
        okc_session::AuditEvent {
            ts: "t".to_string(),
            kind: kind.to_string(),
            payload,
        }
    }

    #[test]
    fn count_verifications_splits_by_source_kind() {
        let call = |kind: &str| {
            audit_event(
                "ToolCall",
                serde_json::json!({
                    "ToolCall": { "call": {
                        "name": "note_observation",
                        "arguments": { "observation": "x", "source_url": "u", "source_kind": kind }
                    }}
                }),
            )
        };
        let other = audit_event(
            "ToolCall",
            serde_json::json!({
                "ToolCall": { "call": { "name": "read_file", "arguments": { "path": "a" } } }
            }),
        );
        let events = vec![
            call("verified"),
            call("unverified"),
            call("web-verified"),
            call("model-claim"),
            other,
        ];
        let (ok, bad) = count_verifications(&events);
        assert_eq!(ok, 2, "verified + web-verified");
        assert_eq!(bad, 2, "unverified + unspecified kind");
    }
}
