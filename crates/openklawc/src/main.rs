//! openklawc — autonomous Rust coding agent.
//!
//! Architecture inspired by:
//!   - opencode (feature-rich AI dev tool)
//!   - openclaude (Rust TUI patterns)
//!   - hermes (Nous Research agent loop — plan → execute → reflect)

use std::io::{self, stdout};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Result};
use crossterm::{
    event::{self, Event},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;
use tracing::{info, warn};

use okc_config::{Config, ModelPricing};
use okc_core::{AutonomyLevel, ChatMessage, EventBus, TokenUsage};
use okc_llm::{LlmClient, ProviderRegistry};
use okc_mcp::McpRegistry;
use okc_rag::SimpleRag;
use okc_reasoning::{Agent, AgentUpdate, ConfirmRequest};
use okc_session::{SessionConfig, SessionStore};
use okc_tool::{BuiltInTools, CompositeExecutor, CustomTools};
use okc_tui::{App, AppEvent, ChatSession, PickerItem, Theme};

/// Command-line arguments.
#[derive(Debug, Default)]
struct Args {
    /// Run a single turn headlessly with this prompt instead of the TUI.
    prompt: Option<String>,
    /// Override the model (`provider/model`).
    model: Option<String>,
    /// Override the autonomy level (`ask|read_only|yolo|god`).
    autonomy: Option<String>,
    /// Activate a persona (`architect|reviewer|tester|security_auditor`).
    persona: Option<String>,
    /// Don't start MCP servers.
    no_mcp: bool,
    help: bool,
}

impl Args {
    fn parse(mut it: impl Iterator<Item = String>) -> Self {
        let mut args = Args::default();
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-p" | "--prompt" => args.prompt = it.next(),
                "-m" | "--model" => args.model = it.next(),
                "--autonomy" => args.autonomy = it.next(),
                "--persona" => args.persona = it.next(),
                "--no-mcp" => args.no_mcp = true,
                "-h" | "--help" => args.help = true,
                other => {
                    if args.prompt.is_none() {
                        args.prompt = Some(other.to_string());
                    }
                }
            }
        }
        args
    }

    fn usage() -> &'static str {
        "openklawc — autonomous Rust coding agent\n\n\
         Usage:\n  \
         openklawc                      start the TUI\n  \
         openklawc -p \"<prompt>\"       run one turn headlessly and exit\n\n\
         Options:\n  \
         -p, --prompt <text>   prompt for a single headless turn\n  \
         -m, --model <id>      model as provider/model (e.g. minimax/MiniMax-M3)\n      \
         --autonomy <level>    ask | read_only | yolo | god\n      \
         --persona <name>      architect | reviewer | tester | security_auditor\n      \
         --no-mcp              don't start MCP servers\n      \
         -h, --help            show this help\n\n\
         Autonomy:\n\
         read_only (default) mutating tools need approval — in the TUI you are\n\
         asked, headless runs deny them. Use --autonomy yolo to allow them.\n\n\
         MCP servers:\n  \
         The bundled Rust servers (filesystem, brave-search, playwright, vision, local-brain,\n  \
         laya) start automatically. A [mcp.<name>] entry in .openklawc.toml or\n  \
         ~/.config/openklawc/config.toml overrides one; set enabled = false to switch it off.\n"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let args = Args::parse(std::env::args().skip(1));

    if args.help {
        print!("{}", Args::usage());
        return Ok(());
    }

    if let Some(prompt) = args.prompt.clone() {
        return run_headless(&args, prompt).await;
    }

    info!("openklawc starting...");

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = run_app(&mut terminal, &args).await;

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Run a single turn without the TUI. Used for scripting and for verifying
/// the provider/tool wiring end to end.
async fn run_headless(args: &Args, prompt: String) -> Result<()> {
    let cfg = Config::load_default().unwrap_or_default();
    let event_bus = EventBus::new(256);

    let registry = ProviderRegistry::from_config(&cfg);
    let model_id = args.model.clone().unwrap_or_else(|| cfg.model.clone());
    let (provider, model) = registry.get(&model_id).ok_or_else(|| {
        anyhow!(
            "unknown model '{}' — run /models inside the TUI to list available models",
            model_id
        )
    })?;

    // ─── Tools (built-in + custom + MCP) ───────────────────────────────────
    let mut composite = CompositeExecutor::new();
    composite.add(Arc::new(BuiltInTools::new()));
    composite.add(Arc::new(CustomTools::from_config(&cfg)));
    if !args.no_mcp {
        match McpRegistry::start(&cfg.mcp).await {
            Ok(r) => {
                eprintln!("[mcp] {} tools registered", r.all_tools().len());
                composite.add(Arc::new(r));
            }
            Err(e) => eprintln!("[mcp] startup failed: {}", e),
        }
    }
    let tools: Arc<dyn okc_core::ToolExecutor> = Arc::new(composite);

    let level = args
        .autonomy
        .clone()
        .unwrap_or_else(|| cfg.autonomy.default_level.clone());
    let autonomy = parse_autonomy(&level);

    let mut agent = Agent::new(
        LlmClient::new(),
        provider,
        model,
        tools,
        autonomy,
        event_bus.clone(),
    );

    if let Some(persona) = build_persona(args, &cfg) {
        agent = agent.with_persona(persona);
    }

    let (tx, mut rx) = mpsc::channel::<AgentUpdate>(256);
    agent.update_tx = Some(tx);

    // Headless runs are saved too, so they show up in the chat picker.
    let mut session_store = SessionStore::new().ok();
    let mut session_id: Option<String> = None;

    let failed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let printer_failed = failed.clone();
    let printer = tokio::spawn(async move {
        let mut turn_usage: Option<TokenUsage> = None;
        while let Some(update) = rx.recv().await {
            match update {
                AgentUpdate::TextChunk(chunk) => print!("{}", chunk),
                AgentUpdate::Reasoning { kind, content } => {
                    eprintln!("[{}] {}", kind, content)
                }
                AgentUpdate::ToolCall { name, arguments } => {
                    eprintln!(
                        "[tool] {} {}",
                        name,
                        serde_json::to_string(&arguments).unwrap_or_default()
                    )
                }
                AgentUpdate::ToolResult {
                    name,
                    success,
                    result,
                    elapsed_ms,
                } => {
                    let status = if success { "ok" } else { "err" };
                    let body = serde_json::to_string(&result).unwrap_or_default();
                    eprintln!(
                        "[{}] {} ({}ms) {}",
                        status,
                        name,
                        elapsed_ms,
                        truncate(&body, 200)
                    );
                }
                AgentUpdate::Error(e) => {
                    eprintln!("[error] {}", e);
                    printer_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                AgentUpdate::Done { usage, .. } => {
                    turn_usage = usage;
                }
            }
        }
        turn_usage
    });

    eprintln!("[model] {}", model_id);
    let final_messages = agent.run_turn(&prompt, Vec::new()).await;

    // Dropping the sender closes the channel so the printer can finish.
    agent.update_tx = None;
    let turn_usage = printer.await.unwrap_or(None);
    println!();

    // Persist the run.
    if let Some(store) = session_store.as_mut() {
        let cwd = std::env::current_dir()
            .ok()
            .map(|p| p.to_string_lossy().to_string());
        let provider = model_id.split('/').next().map(|s| s.to_string());
        let session_cfg = SessionConfig {
            agent_mode: Some("build".to_string()),
            ..Default::default()
        };
        match store.create_with_config(
            &title_from(&prompt),
            Some(&model_id),
            provider.as_deref(),
            cwd.as_deref(),
            &session_cfg,
        ) {
            Ok(id) => {
                if let Err(e) = store.sync_messages(&id, &final_messages, 0) {
                    warn!("failed to persist headless session: {}", e);
                }
                if let Some(u) = turn_usage {
                    if let Ok(mut session_cfg) = store.get_config(&id) {
                        session_cfg.tokens_in += u.prompt;
                        session_cfg.tokens_out += u.completion;
                        session_cfg.cost_usd = cost_for(
                            &cfg.pricing,
                            &model_id,
                            TokenUsage::new(session_cfg.tokens_in, session_cfg.tokens_out),
                        );
                        let _ = store.set_config(&id, &session_cfg);
                    }
                }
                session_id = Some(id);
            }
            Err(e) => warn!("failed to create headless session: {}", e),
        }
    }
    if let Some(id) = &session_id {
        eprintln!("[session] {}", &id[..8.min(id.len())]);
    }

    if failed.load(std::sync::atomic::Ordering::Relaxed) {
        std::process::exit(1);
    }
    Ok(())
}

async fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, args: &Args) -> Result<()> {
    // ─── Config ─────────────────────────────────────────────────────────────
    let cfg = Config::load_default().unwrap_or_default();
    let event_bus = EventBus::new(256);

    // ─── LLM ────────────────────────────────────────────────────────────────
    let llm_registry = ProviderRegistry::from_config(&cfg);
    let llm_client = LlmClient::new();
    let mut current_model = cfg.model.clone();

    let (default_provider, default_model_id) =
        llm_registry.get(&current_model).unwrap_or_else(|| {
            let first = llm_registry.list().first().cloned();
            let p = first
                .as_ref()
                .and_then(|m| llm_registry.get(&format!("{}/{}", m.provider, m.model_id)));
            p.unwrap_or_else(|| {
                let ollama = Arc::new(okc_llm::OllamaProvider::new("http://localhost:11434"));
                (ollama, "qwen3".to_string())
            })
        });

    // ─── MCP ────────────────────────────────────────────────────────────────
    let mcp_registry = if args.no_mcp {
        None
    } else {
        match McpRegistry::start(&cfg.mcp).await {
            Ok(r) => {
                info!("MCP registry: {} tools", r.all_tools().len());
                Some(r)
            }
            Err(e) => {
                warn!("MCP startup failed: {}", e);
                None
            }
        }
    };

    // ─── Composite Tools (builtin + custom + MCP) ───────────────────────────
    let mut composite = CompositeExecutor::new();
    composite.add(Arc::new(BuiltInTools::new()));
    composite.add(Arc::new(CustomTools::from_config(&cfg)));
    let mcp_tool_count = mcp_registry
        .as_ref()
        .map(|r| r.all_tools().len())
        .unwrap_or(0);
    let tools: Arc<dyn okc_core::ToolExecutor> = if let Some(mcp) = mcp_registry {
        info!("MCP tools: {} registered", mcp.all_tools().len());
        composite.add(Arc::new(mcp));
        Arc::new(composite)
    } else {
        Arc::new(composite)
    };

    // ─── Agent ──────────────────────────────────────────────────────────────
    // The CLI flag wins over the configured default.
    let autonomy = parse_autonomy(
        args.autonomy
            .as_deref()
            .unwrap_or(&cfg.autonomy.default_level),
    );
    let mut agent = Agent::new(
        llm_client.clone(),
        default_provider,
        default_model_id,
        tools,
        autonomy,
        event_bus.clone(),
    );
    agent.streaming = cfg.ui.stream_responses;

    if let Some(persona) = build_persona(args, &cfg) {
        agent = agent.with_persona(persona);
    }

    // ─── RAG ────────────────────────────────────────────────────────────────
    let mut rag = SimpleRag::new();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions);
    }

    // ─── Session ────────────────────────────────────────────────────────────
    let mut session_store = SessionStore::new().ok();
    // How many entries of `messages` are already written to the database.
    let mut persisted_count: usize = 0;

    // ─── Tool approval channel ──────────────────────────────────────────────
    let (confirm_tx, mut confirm_rx) = mpsc::channel::<ConfirmRequest>(32);
    agent.confirm_tx = Some(confirm_tx.clone());

    // ─── TUI ────────────────────────────────────────────────────────────────
    let mut app = App::new();
    app.model = current_model.clone();
    app.autonomy = agent.autonomy;
    app.mcp_count = mcp_tool_count;
    app.show_reasoning = cfg.ui.show_reasoning;
    app.theme = Theme::from_name(&cfg.ui.theme);

    // ─── Conversation history ───────────────────────────────────────────────
    let mut messages: Vec<ChatMessage> = Vec::new();

    // ─── Agent ↔ TUI channel ────────────────────────────────────────────────
    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentUpdate>(256);

    // ─── Main loop state ────────────────────────────────────────────────────
    let mut agent_running = false;
    let mut pending_confirms: std::collections::VecDeque<ConfirmRequest> =
        std::collections::VecDeque::new();

    loop {
        // ─── Drain agent updates ──────────────────────────────────────────
        while let Ok(update) = agent_rx.try_recv() {
            match update {
                AgentUpdate::TextChunk(chunk) => {
                    // Append to the last assistant message
                    if let Some(okc_tui::app::ScrollEntry::Assistant(text)) =
                        app.scrollback.last_mut()
                    {
                        if text == "..." {
                            *text = chunk;
                        } else {
                            text.push_str(&chunk);
                        }
                    }
                }
                AgentUpdate::Reasoning { kind: _, content } => {
                    app.reasoning_steps.push_back(okc_core::ReasoningStep {
                        timestamp: chrono::Utc::now(),
                        kind: okc_core::ReasoningKind::Reflect, // simplified mapping
                        content,
                        tool_call: None,
                        tool_result: None,
                        decision: okc_core::DecisionMetadata::default(),
                    });
                }
                AgentUpdate::ToolCall { name, arguments } => {
                    app.add_system(format!(
                        "[tool call] {} {}",
                        name,
                        serde_json::to_string(&arguments).unwrap_or_default()
                    ));
                }
                AgentUpdate::ToolResult {
                    name,
                    success,
                    result,
                    elapsed_ms,
                } => {
                    let prefix = if success { "[ok]" } else { "[err]" };
                    let result_str = serde_json::to_string(&result).unwrap_or_default();
                    app.add_system(format!(
                        "{} {} ({}ms) → {}",
                        prefix,
                        name,
                        elapsed_ms,
                        &result_str[..result_str.len().min(200)]
                    ));
                }
                AgentUpdate::Done {
                    text,
                    messages: new_messages,
                    usage,
                } => {
                    agent_running = false;
                    messages = new_messages;
                    // Ensure final text is shown
                    app.update_last_assistant(text);

                    persist_turn(
                        session_store.as_mut(),
                        &mut persisted_count,
                        &mut app,
                        &messages,
                        &current_model,
                        &cfg.pricing,
                        usage,
                    );
                }
                AgentUpdate::Error(e) => {
                    agent_running = false;
                    app.add_error(format!("Agent error: {}", e));
                }
            }
        }

        // ─── Pending tool approvals ──────────────────────────────────────
        while let Ok(request) = confirm_rx.try_recv() {
            pending_confirms.push_back(request);
        }
        if app.pending_confirm.is_none() {
            if let Some(request) = pending_confirms.front() {
                app.pending_confirm = Some(okc_tui::PendingConfirm {
                    name: request.name.clone(),
                    arguments: request
                        .arguments
                        .to_string()
                        .chars()
                        .take(160)
                        .collect::<String>(),
                });
            }
        }

        // ─── Render ───────────────────────────────────────────────────────
        terminal.draw(|frame| {
            app.render(frame, frame.area());
        })?;

        // ─── Poll keyboard ────────────────────────────────────────────────
        if !event::poll(Duration::from_millis(50))? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            match app.handle_key(key) {
                Some(AppEvent::Quit) => break,
                Some(AppEvent::Redraw) => continue,
                Some(AppEvent::Submit(query)) => {
                    if agent_running {
                        app.add_system("Agent is already running. Wait.");
                        continue;
                    }

                    let file_refs = okc_core::util::extract_file_refs(&query);
                    let clean_query = okc_core::util::strip_file_refs(&query);

                    // Build enriched query with file context
                    let mut enriched = clean_query.clone();
                    for path in &file_refs {
                        if let Ok(content) = tokio::fs::read_to_string(path).await {
                            enriched.push_str(&format!("\n\n--- @{} ---\n{}", path, content));
                        }
                    }

                    // Update provider if model changed
                    if let Some((provider, model_id)) = llm_registry.get(&current_model) {
                        agent.provider = provider;
                        agent.model_id = model_id;
                    }

                    // Hook up the update channel
                    agent.update_tx = Some(agent_tx.clone());

                    // Start a session the first time the user talks, so the
                    // chat picker has something to show.
                    if app.current_session_id.is_none() {
                        start_session(
                            session_store.as_mut(),
                            &mut app,
                            &mut persisted_count,
                            &title_from(&clean_query),
                            &current_model,
                        );
                    }

                    // Show placeholder
                    app.add_assistant("...");
                    agent_running = true;

                    // Clone what we need for the async task
                    let mut task_agent = agent.fork();
                    task_agent.update_tx = Some(agent_tx.clone());
                    let msgs = messages.clone();

                    tokio::spawn(async move {
                        let _new_messages = task_agent.run_turn(&enriched, msgs).await;
                    });
                }
                Some(AppEvent::ListModels) => {
                    let items: Vec<PickerItem> = llm_registry
                        .list()
                        .iter()
                        .map(|m| {
                            let qid = format!("{}/{}", m.provider, m.model_id);
                            let mut tags = Vec::new();
                            if m.capabilities.reasoning {
                                tags.push("reasoning");
                            }
                            if m.capabilities.vision {
                                tags.push("vision");
                            }
                            if m.capabilities.function_calling {
                                tags.push("tools");
                            }
                            let subtitle = if tags.is_empty() {
                                None
                            } else {
                                Some(format!("[{}]", tags.join(", ")))
                            };
                            PickerItem {
                                id: qid.clone(),
                                title: qid,
                                subtitle,
                            }
                        })
                        .collect();
                    app.set_overlay_items(items);
                }
                Some(AppEvent::ListSessions) => {
                    let items = if let Some(ref store) = session_store {
                        match store.list(50) {
                            Ok(sessions) => sessions
                                .into_iter()
                                .map(|s| PickerItem {
                                    id: s.id.clone(),
                                    title: s.name.clone(),
                                    subtitle: Some(format!(
                                        "{} msgs • {} • {}",
                                        s.message_count,
                                        s.model.as_deref().unwrap_or("no model"),
                                        s.updated_at.format("%Y-%m-%d %H:%M")
                                    )),
                                })
                                .collect(),
                            Err(e) => {
                                app.add_error(format!("Failed to list sessions: {}", e));
                                Vec::new()
                            }
                        }
                    } else {
                        app.add_error("Session store unavailable");
                        Vec::new()
                    };
                    app.set_overlay_items(items);
                }
                Some(AppEvent::SwitchModel(id)) => {
                    if llm_registry.get(&id).is_some() {
                        current_model = id.clone();
                        app.model = id;
                        app.add_system(format!("Switched to {}", current_model));
                    } else {
                        app.add_error(format!("Unknown model: {}", id));
                    }
                }
                Some(AppEvent::ResumeSession(id)) => {
                    if let Some(ref store) = session_store {
                        match store.get_messages(&id) {
                            Ok(persisted) => {
                                messages = persisted
                                    .iter()
                                    .filter(|pm| pm.role != "system")
                                    .map(|pm| match pm.role.as_str() {
                                        "assistant" => ChatMessage::assistant(pm.content.clone()),
                                        _ => ChatMessage::user(pm.content.clone()),
                                    })
                                    .collect();
                                // Rebuild the visible transcript too — otherwise
                                // the model has the history but the user does not.
                                let transcript: Vec<(String, String)> = persisted
                                    .iter()
                                    .map(|pm| (pm.role.clone(), pm.content.clone()))
                                    .collect();
                                app.set_transcript(&transcript);
                                app.current_session_id = Some(id.clone());
                                persisted_count = persisted.len();
                                app.token_count = 0;
                                if let Ok(config) = store.get_config(&id) {
                                    app.token_count =
                                        (config.tokens_in + config.tokens_out) as usize;
                                }
                                app.add_system(format!(
                                    "Resumed session {}",
                                    &id[..8.min(id.len())]
                                ));
                            }
                            Err(e) => app.add_error(format!("Failed to resume: {}", e)),
                        }
                    }
                }
                Some(AppEvent::SetAutonomy(level)) => {
                    agent.autonomy = level;
                }
                Some(AppEvent::OpenChatPicker) => {
                    let sessions = match session_store.as_ref() {
                        Some(store) => load_chat_sessions(store),
                        None => {
                            app.add_error("Session store unavailable");
                            Vec::new()
                        }
                    };
                    app.chat_picker.set_sessions(sessions);
                }
                Some(AppEvent::CreateSession(mode)) => {
                    let name = format!("New {} chat", mode.label().to_lowercase());
                    match start_session(
                        session_store.as_mut(),
                        &mut app,
                        &mut persisted_count,
                        &name,
                        &current_model,
                    ) {
                        Some(id) => {
                            messages.clear();
                            app.clear_transcript();
                            app.token_count = 0;
                            app.chat_picker.notice = Some(format!(
                                "Started {} chat {}",
                                mode.label().to_lowercase(),
                                &id[..8.min(id.len())]
                            ));
                            app.add_system(format!(
                                "Started {} chat {}",
                                mode.label().to_lowercase(),
                                &id[..8.min(id.len())]
                            ));
                        }
                        None => {
                            app.chat_picker.notice =
                                Some("Could not create the session".to_string());
                            app.add_error("Could not create the session");
                        }
                    }
                    // Refresh the list so the new chat shows up.
                    if let Some(store) = session_store.as_ref() {
                        app.chat_picker.set_sessions(load_chat_sessions(store));
                    }
                }
                Some(AppEvent::DeleteSession(id)) => {
                    let deleted = match session_store.as_mut() {
                        Some(store) => store.delete(&id).is_ok(),
                        None => false,
                    };
                    if deleted {
                        if app.current_session_id.as_deref() == Some(id.as_str()) {
                            app.current_session_id = None;
                            messages.clear();
                            persisted_count = 0;
                            app.token_count = 0;
                        }
                        app.chat_picker.notice =
                            Some(format!("Deleted chat {}", &id[..8.min(id.len())]));
                        app.add_system(format!("Deleted chat {}", &id[..8.min(id.len())]));
                    } else {
                        app.chat_picker.notice = Some("Could not delete the chat".to_string());
                        app.add_error("Could not delete the chat");
                    }
                    if let Some(store) = session_store.as_ref() {
                        app.chat_picker.set_sessions(load_chat_sessions(store));
                    }
                }
                Some(AppEvent::ConfirmTool(approved)) => {
                    if let Some(request) = pending_confirms.pop_front() {
                        let _ = request.respond.send(approved);
                    }
                    app.pending_confirm = None;
                }
                Some(AppEvent::SearchHistory(query)) => {
                    let items = if let Some(ref store) = session_store {
                        match store.search(&query, 20) {
                            Ok(results) => results
                                .into_iter()
                                .map(|r| PickerItem {
                                    id: r.meta.id.clone(),
                                    title: r.meta.name.clone(),
                                    subtitle: Some(format!(
                                        "{} msgs • {}",
                                        r.meta.message_count,
                                        r.meta.updated_at.format("%Y-%m-%d %H:%M")
                                    )),
                                })
                                .collect(),
                            Err(e) => {
                                app.add_error(format!("Search failed: {}", e));
                                Vec::new()
                            }
                        }
                    } else {
                        app.add_error("Session store unavailable");
                        Vec::new()
                    };
                    app.set_overlay_items(items);
                }
                None => {}
            }
        }
    }

    Ok(())
}

/// Truncate to at most `max` bytes without splitting a UTF-8 character.
fn truncate(text: &str, max: usize) -> &str {
    if text.len() <= max {
        return text;
    }
    let mut end = max;
    while end > 0 && !text.is_char_boundary(end) {
        end -= 1;
    }
    &text[..end]
}

/// Derive a chat title from the first user message.
fn title_from(text: &str) -> String {
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

/// Create a new session and point the app at it.
fn start_session(
    store: Option<&mut SessionStore>,
    app: &mut App,
    persisted_count: &mut usize,
    title: &str,
    model: &str,
) -> Option<String> {
    let store = store?;
    let cwd = std::env::current_dir()
        .ok()
        .map(|p| p.to_string_lossy().to_string());
    let provider = model.split('/').next().map(|s| s.to_string());
    let config = SessionConfig {
        agent_mode: Some(app.chat_picker.mode.as_str().to_string()),
        ..Default::default()
    };
    let id = store
        .create_with_config(
            title,
            Some(model),
            provider.as_deref(),
            cwd.as_deref(),
            &config,
        )
        .ok()?;
    app.current_session_id = Some(id.clone());
    *persisted_count = 0;
    Some(id)
}

/// Write the messages produced by a turn into the session database.
fn persist_turn(
    store: Option<&mut SessionStore>,
    persisted_count: &mut usize,
    app: &mut App,
    messages: &[ChatMessage],
    model: &str,
    pricing: &std::collections::HashMap<String, ModelPricing>,
    usage: Option<TokenUsage>,
) {
    let Some(store) = store else { return };
    let Some(session_id) = app.current_session_id.clone() else {
        return;
    };

    match store.sync_messages(&session_id, messages, *persisted_count) {
        Ok(stored) => *persisted_count = stored,
        Err(e) => tracing::warn!("failed to persist messages: {}", e),
    }

    // Fold the token counters into the session config.
    if let Ok(mut config) = store.get_config(&session_id) {
        if let Some(u) = usage {
            config.tokens_in += u.prompt;
            config.tokens_out += u.completion;
        }
        config.cost_usd = cost_for(
            pricing,
            model,
            TokenUsage::new(config.tokens_in, config.tokens_out),
        );
        if let Err(e) = store.set_config(&session_id, &config) {
            tracing::warn!("failed to update session config: {}", e);
        }
        app.token_count = (config.tokens_in + config.tokens_out) as usize;
    }
}

/// Cost in USD for the given usage, when the model has a configured price.
fn cost_for(
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

/// Build the chat-picker rows from the session store.
fn load_chat_sessions(store: &SessionStore) -> Vec<ChatSession> {
    let Ok(sessions) = store.list(200) else {
        return Vec::new();
    };
    sessions
        .into_iter()
        .map(|s| ChatSession {
            id: s.id.clone(),
            title: s.config.title.clone().unwrap_or_else(|| s.name.clone()),
            agent_mode: s.config.agent_mode.clone().unwrap_or_default(),
            model: s.model.clone(),
            provider: s.provider.clone(),
            cwd: s.cwd.clone(),
            updated_at: s.updated_at.format("%Y-%m-%d %H:%M").to_string(),
            created_at: s.created_at.format("%Y-%m-%d %H:%M").to_string(),
            tokens_in: s.config.tokens_in,
            tokens_out: s.config.tokens_out,
            cost_usd: s.config.cost_usd,
            message_count: s.message_count,
        })
        .collect()
}

fn parse_autonomy(s: &str) -> AutonomyLevel {
    match s.to_lowercase().as_str() {
        "ask" => AutonomyLevel::Ask,
        "read_only" | "readonly" => AutonomyLevel::ReadOnly,
        "yolo" => AutonomyLevel::Yolo,
        "god" => AutonomyLevel::God,
        _ => AutonomyLevel::ReadOnly,
    }
}

/// Build a persona from CLI args or config.
fn build_persona(args: &Args, cfg: &Config) -> Option<okc_reasoning::persona::Persona> {
    let persona_name = args.persona.as_ref().or(cfg.default_persona.as_ref())?;

    // Check config-defined personas first
    if let Some(pc) = cfg.personas.get(persona_name) {
        return Some(okc_reasoning::persona::Persona::from_config(
            persona_name,
            pc,
        ));
    }

    // Fall back to built-in personas
    okc_reasoning::persona::builtin_persona_prompt(persona_name).map(|prompt| {
        okc_reasoning::persona::Persona {
            name: Some(persona_name.clone()),
            system_prompt: prompt,
            allowed_tools: std::collections::HashSet::new(),
            model_override: None,
            temperature: None,
            max_iterations: None,
            can_delegate: true,
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_never_splits_a_character() {
        // "é" is two bytes; cutting at 1 would panic on a naive slice.
        let text = "éé";
        assert_eq!(truncate(text, 1), "");
        assert_eq!(truncate(text, 2), "é");
        assert_eq!(truncate(text, 99), "éé");
        assert_eq!(truncate("abc", 2), "ab");
    }

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
        // 1M input at $1 and 1M output at $2.
        let cost = cost_for(
            &pricing,
            "minimax/MiniMax-M3",
            TokenUsage::new(1_000_000, 1_000_000),
        )
        .unwrap();
        assert!((cost - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn tool_mutation_classifier_covers_the_mcp_filesystem_tools() {
        for name in [
            "write_file",
            "edit_file",
            "create_directory",
            "move_file",
            "shell",
            "git",
        ] {
            assert!(okc_core::is_mutation_tool(name), "{} should mutate", name);
        }
        for name in ["read_file", "list_directory", "search_files", "web_search"] {
            assert!(
                !okc_core::is_mutation_tool(name),
                "{} should not mutate",
                name
            );
        }
    }
}
