//! openklawc — autonomous Rust coding agent.
//!
//! Architecture inspired by:
//!   - opencode (feature-rich AI dev tool)
//!   - openclaude (Rust TUI patterns)
//!   - hermes (Nous Research agent loop — plan → execute → reflect)

use std::io::{self, stdout};
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{backend::CrosstermBackend, Terminal};
use tokio::sync::mpsc;
use tracing::{info, warn};

use okc_config::Config;
use okc_core::{AutonomyLevel, ChatMessage, EventBus};
use okc_llm::{LlmClient, ProviderRegistry};
use okc_mcp::McpRegistry;
use okc_rag::SimpleRag;
use okc_reasoning::{Agent, AgentUpdate};
use okc_session::SessionStore;
use okc_tool::{BuiltInTools, CompositeExecutor};
use okc_tui::{App, AppEvent};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    info!("openklawc starting...");

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = run_app(&mut terminal).await;

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

async fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>) -> Result<()> {
    // ─── Config ─────────────────────────────────────────────────────────────
    let cfg = Config::load_default().unwrap_or_default();
    let event_bus = EventBus::new(256);

    // ─── LLM ────────────────────────────────────────────────────────────────
    let llm_registry = ProviderRegistry::from_config(&cfg);
    let llm_client = LlmClient::new();
    let mut current_model = cfg.model.clone();

    let (default_provider, default_model_id) = llm_registry
        .get(&current_model)
        .unwrap_or_else(|| {
            let first = llm_registry.list().first().cloned();
            let p = first.as_ref().and_then(|m| {
                llm_registry.get(&format!("{}/{}", m.provider, m.model_id))
            });
            p.unwrap_or_else(|| {
                let ollama = Arc::new(okc_llm::OllamaProvider::new("http://localhost:11434"));
                (ollama, "qwen3".to_string())
            })
        });

    // ─── MCP ────────────────────────────────────────────────────────────────
    let mcp_registry = if cfg.mcp.is_empty() {
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

    // ─── Composite Tools (builtin + MCP) ────────────────────────────────────
    let mut composite = CompositeExecutor::new();
    composite.add(Arc::new(BuiltInTools::new()));
    let mcp_tool_count = mcp_registry.as_ref().map(|r| r.all_tools().len()).unwrap_or(0);
    let tools: Arc<dyn okc_core::ToolExecutor> = if let Some(mcp) = mcp_registry {
        info!("MCP tools: {} registered", mcp.all_tools().len());
        composite.add(Arc::new(mcp));
        Arc::new(composite)
    } else {
        Arc::new(composite)
    };

    // ─── Agent ──────────────────────────────────────────────────────────────
    let mut agent = Agent::new(
        llm_client.clone(),
        default_provider,
        default_model_id,
        tools,
        parse_autonomy(&cfg.autonomy.default_level),
        event_bus.clone(),
    );
    agent.streaming = cfg.ui.stream_responses;

    // ─── RAG ────────────────────────────────────────────────────────────────
    let mut rag = SimpleRag::new();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions);
    }

    // ─── Session ────────────────────────────────────────────────────────────
    let _session_store = SessionStore::new().ok();

    // ─── TUI ────────────────────────────────────────────────────────────────
    let mut app = App::new();
    app.model = current_model.clone();
    app.autonomy = agent.autonomy;
    app.mcp_count = mcp_tool_count;
    app.show_reasoning = cfg.ui.show_reasoning;

    // ─── Conversation history ───────────────────────────────────────────────
    let mut messages: Vec<ChatMessage> = Vec::new();

    // ─── Agent ↔ TUI channel ────────────────────────────────────────────────
    let (agent_tx, mut agent_rx) = mpsc::channel::<AgentUpdate>(256);

    // ─── Main loop state ────────────────────────────────────────────────────
    let mut agent_running = false;

    loop {
        // ─── Drain agent updates ──────────────────────────────────────────
        while let Ok(update) = agent_rx.try_recv() {
            match update {
                AgentUpdate::TextChunk(chunk) => {
                    // Append to the last assistant message
                    if let Some(last) = app.scrollback.last_mut() {
                        if let okc_tui::app::ScrollEntry::Assistant(ref mut text) = last {
                            if text == "..." {
                                *text = chunk;
                            } else {
                                text.push_str(&chunk);
                            }
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
                    });
                }
                AgentUpdate::ToolCall { name, arguments } => {
                    app.add_system(format!(
                        "[tool call] {} {}",
                        name,
                        serde_json::to_string(&arguments).unwrap_or_default()
                    ));
                }
                AgentUpdate::ToolResult { name, success, result, elapsed_ms } => {
                    let prefix = if success { "[ok]" } else { "[err]" };
                    let result_str = serde_json::to_string(&result).unwrap_or_default();
                    app.add_system(format!(
                        "{} {} ({}ms) → {}",
                        prefix, name, elapsed_ms,
                        &result_str[..result_str.len().min(200)]
                    ));
                }
                AgentUpdate::Done { text, messages: new_messages } => {
                    agent_running = false;
                    messages = new_messages;
                    // Ensure final text is shown
                    app.update_last_assistant(text);
                }
                AgentUpdate::Error(e) => {
                    agent_running = false;
                    app.add_error(format!("Agent error: {}", e));
                }
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

                    // Show placeholder
                    app.add_assistant("...");
                    agent_running = true;

                    // Clone what we need for the async task
                    let mut task_agent = Agent {
                        llm_client: llm_client.clone(),
                        provider: agent.provider.clone(),
                        model_id: agent.model_id.clone(),
                        tools: agent.tools.clone(),
                        autonomy: agent.autonomy,
                        event_bus: event_bus.clone(),
                        max_iterations: agent.max_iterations,
                        streaming: agent.streaming,
                        reasoning_log: Vec::new(),
                        update_tx: Some(agent_tx.clone()),
                    };
                    let msgs = messages.clone();
                    let _tx = agent_tx.clone();

                    tokio::spawn(async move {
                        let _new_messages = task_agent.run_turn(&enriched, msgs).await;
                        // The Done event is already emitted inside run_turn before returning.
                        // But we need to make sure messages get back to the main loop.
                        // Actually, Done already contains messages. The main loop handles it.
                        // However, if run_turn crashes before emitting Done, we need a fallback:
                    });
                }
                Some(AppEvent::Command(cmd)) => {
                    handle_command(
                        &cmd,
                        &mut app,
                        &llm_registry,
                        &mut current_model,
                        &mut agent,
                        &mut rag,
                        &cfg,
                    );
                }
                None => {}
            }
        }
    }

    Ok(())
}

fn handle_command(
    cmd: &str,
    app: &mut App,
    registry: &ProviderRegistry,
    current_model: &mut String,
    agent: &mut Agent,
    rag: &mut SimpleRag,
    cfg: &Config,
) {
    let parts: Vec<&str> = cmd.split_whitespace().collect();
    if parts.is_empty() {
        return;
    }

    match parts[0] {
        "/models" => {
            let mut out = String::new();
            out.push_str(&format!("models ({} available):\n", registry.list().len()));
            for model in registry.list() {
                let qid = format!("{}/{}", model.provider, model.model_id);
                let marker = if qid == *current_model { "→ " } else { "  " };
                let mut tags = Vec::new();
                if model.capabilities.reasoning { tags.push("reasoning"); }
                if model.capabilities.vision { tags.push("vision"); }
                if model.capabilities.function_calling { tags.push("tools"); }
                let tag_str = if tags.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", tags.join(", "))
                };
                out.push_str(&format!("{}{}{}\n", marker, qid, tag_str));
            }
            app.add_system(out);
        }
        "/model" => {
            if parts.len() < 2 {
                app.add_error("usage: /model <provider>/<model_id>");
            } else {
                let id = parts[1];
                if registry.get(id).is_none() {
                    app.add_error("no such model (use /models)");
                } else {
                    *current_model = id.to_string();
                    app.model = id.to_string();
                    app.add_system(format!("model -> {}", id));
                }
            }
        }
        "/autonomy" => {
            if parts.len() < 2 {
                app.add_system(format!("current: {:?}", app.autonomy));
            } else {
                let level = parse_autonomy(parts[1]);
                app.autonomy = level;
                agent.autonomy = level;
                app.add_system(format!("autonomy -> {:?}", level));
            }
        }
        "/rag" => {
            if parts.len() < 2 {
                app.add_error("usage: /rag <query>");
            } else {
                let query = parts[1..].join(" ");
                let results = rag.search(&query);
                if results.is_empty() {
                    app.add_system("No RAG matches.");
                } else {
                    let mut out = String::from("RAG results:\n");
                    for r in results.iter().take(5) {
                        out.push_str(&format!(
                            "{}:{}\n{:.100}\n\n",
                            r.file.display(),
                            r.line_start,
                            r.text
                        ));
                    }
                    app.add_system(out);
                }
            }
        }
        "/rag-index" => {
            if let Ok(cwd) = std::env::current_dir() {
                match rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions) {
                    Ok(()) => app.add_system("RAG index rebuilt."),
                    Err(e) => app.add_error(format!("RAG failed: {}", e)),
                }
            }
        }
        "/clear" => {
            app.scrollback.clear();
            app.add_system("Cleared.");
        }
        "/help" | "/h" => {
            app.add_system(
                "Commands:\n\
                 /models          — list models\n\
                 /model <id>      — switch model\n\
                 /autonomy <lvl>  — ask|read_only|yolo|god\n\
                 /rag <query>     — search context\n\
                 /rag-index       — reindex cwd\n\
                 /clear           — clear chat\n\
                 /help            — this message"
            );
        }
        _ => {
            app.add_error(format!("Unknown: {}", parts[0]));
        }
    }
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
