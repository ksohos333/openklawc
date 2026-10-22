//! openklawc — autonomous Rust coding agent.
//!
//! Architecture inspired by:
//!   - opencode (feature-rich AI dev tool)
//!   - openclaude (Rust TUI patterns)
//!   - hermes (Nous Research agent loop — plan → execute → reflect)
//!
//! Autonomy levels:
//!   ask       — confirm every tool call
//!   read_only — auto-execute reads, confirm mutations
//!   yolo      — execute everything, log visibly
//!   god       — execute everything silently (not recommended)

use std::io::{self, stdout};
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
use okc_core::{AutonomyLevel, ChatMessage, EventBus, Role};
use okc_llm::{LlmClient, ProviderRegistry};
use okc_mcp::McpRegistry;
use okc_rag::SimpleRag;
use okc_reasoning::AutonomyEngine;
use okc_session::SessionStore;
use okc_tool::BuiltInTools;
use okc_tui::{App, AppEvent};

#[tokio::main]
async fn main() -> Result<()> {
    // Setup tracing
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
    // ─── Configuration ──────────────────────────────────────────────────────
    let cfg = Config::load_default().unwrap_or_default();
    let event_bus = EventBus::new(256);

    // ─── LLM ────────────────────────────────────────────────────────────────
    let llm_registry = ProviderRegistry::from_config(&cfg);
    let llm_client = LlmClient::new();
    let mut current_model = cfg.model.clone();

    // ─── MCP ────────────────────────────────────────────────────────────────
    let mcp_registry = if cfg.mcp.is_empty() {
        None
    } else {
        match McpRegistry::start(&cfg.mcp).await {
            Ok(r) => {
                info!("MCP registry started with {} tools", r.all_tools().len());
                Some(r)
            }
            Err(e) => {
                warn!("MCP startup failed: {}", e);
                None
            }
        }
    };

    // ─── Built-in Tools ─────────────────────────────────────────────────────
    let builtin_tools = std::sync::Arc::new(BuiltInTools::new());

    // ─── RAG ────────────────────────────────────────────────────────────────
    let mut rag = SimpleRag::new();
    if let Ok(cwd) = std::env::current_dir() {
        let _ = rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions);
    }

    // ─── Session ────────────────────────────────────────────────────────────
    let _session_store = SessionStore::new().ok();

    // ─── Autonomy Engine ────────────────────────────────────────────────────
    let autonomy = parse_autonomy(&cfg.autonomy.default_level);
    let _autonomy_engine = AutonomyEngine::new(autonomy, event_bus.clone());

    // ─── TUI State ──────────────────────────────────────────────────────────
    let mut app = App::new();
    app.model = current_model.clone();
    app.autonomy = autonomy;
    app.mcp_count = mcp_registry.as_ref().map(|r| r.all_tools().len()).unwrap_or(0);
    app.show_reasoning = cfg.ui.show_reasoning;

    // ─── Message history ────────────────────────────────────────────────────
    let mut messages: Vec<ChatMessage> = Vec::new();

    // ─── Streaming channel ──────────────────────────────────────────────────
    let (stream_tx, mut stream_rx) = mpsc::channel::<String>(128);

    // ─── Main event loop ────────────────────────────────────────────────────
    let mut streaming = false;
    let mut stream_buffer = String::new();

    loop {
        terminal.draw(|frame| {
            app.render(frame, frame.area());
        })?;

        // Check for stream chunks
        if streaming {
            match tokio::time::timeout(Duration::from_millis(50), stream_rx.recv()).await {
                Ok(Some(chunk)) => {
                    stream_buffer.push_str(&chunk);
                    app.update_last_assistant(stream_buffer.clone());
                }
                Ok(None) => {
                    streaming = false;
                    app.scroll_to_bottom();
                }
                Err(_) => {
                    // Timeout — just redraw
                }
            }
        }

        // Poll for input
        if !event::poll(Duration::from_millis(50))? {
            continue;
        }

        if let Event::Key(key) = event::read()? {
            match app.handle_key(key) {
                Some(AppEvent::Quit) => break,
                Some(AppEvent::Submit(query)) => {
                    let file_refs = okc_core::util::extract_file_refs(&query);
                    let clean_query = okc_core::util::strip_file_refs(&query);

                    // Build context
                    let mut system = build_system_prompt(&cfg, mcp_registry.as_ref(), &builtin_tools);
                    for path in &file_refs {
                        if let Ok(content) = tokio::fs::read_to_string(path).await {
                            system.push_str(&format!("\n\n--- {} ---\n{}", path, content));
                        }
                    }

                    messages.push(ChatMessage {
                        role: Role::System,
                        content: system,
                        name: None,
                    });
                    messages.push(ChatMessage {
                        role: Role::User,
                        content: clean_query.clone(),
                        name: None,
                    });

                    app.add_assistant("...");
                    streaming = true;
                    stream_buffer.clear();

                    // Spawn LLM request
                    let provider = llm_registry.get(&current_model).map(|(p, _)| p);
                    if let Some(provider) = provider {
                        let model_id = current_model.split_once('/').map(|(_, m)| m.to_string()).unwrap_or_default();
                        let msgs = messages.clone();
                        let tx = stream_tx.clone();
                        let llm_client = llm_client.clone();
                        tokio::spawn(async move {
                            if provider.supports_streaming() {
                                let _ = llm_client.chat_stream(&provider, &model_id, msgs, tx).await;
                            } else {
                                match llm_client.chat(&provider, &model_id, msgs).await {
                                    Ok(text) => {
                                        let _ = tx.send(text).await;
                                    }
                                    Err(e) => {
                                        let _ = tx.send(format!("Error: {}", e)).await;
                                    }
                                }
                            }
                        });
                    } else {
                        app.update_last_assistant(format!("Model '{}' not found", current_model));
                        streaming = false;
                    }
                }
                Some(AppEvent::Command(cmd)) => {
                    handle_command(&cmd, &mut app, &llm_registry, &mut current_model, &mut rag, &cfg);
                }
                None => {}
            }
        }
    }

    Ok(())
}

fn build_system_prompt(
    _cfg: &Config,
    mcp: Option<&McpRegistry>,
    _builtin: &BuiltInTools,
) -> String {
    let mut prompt = String::from(
        "You are openklawc, an autonomous coding assistant running in a Rust TUI.\n\
         You have access to tools. When you need to use a tool, output JSON like:\n\
         {\"tool\": \"tool_name\", \"arguments\": {...}}\n\
         After tool results, reflect and continue.\n\n"
    );

    if let Some(mcp) = mcp {
        if !mcp.all_tools().is_empty() {
            prompt.push_str("MCP tools:\n");
            for (_, tool) in mcp.all_tools() {
                prompt.push_str(&format!("- {}: {}\n", tool.name, tool.description));
            }
        }
    }

    prompt.push_str("\nBuilt-in tools:\n");
    prompt.push_str("- read_file: Read file contents\n");
    prompt.push_str("- write_file: Write content to a file\n");
    prompt.push_str("- list_dir: List directory contents\n");
    prompt.push_str("- search_files: Search files by pattern\n");
    prompt.push_str("- shell: Run shell commands\n");
    prompt.push_str("- git: Run git commands\n");
    prompt.push_str("- view_cwd: Show current working directory\n");

    prompt.push_str("\nRespond concisely. Think step by step.");
    prompt
}

fn handle_command(
    cmd: &str,
    app: &mut App,
    registry: &ProviderRegistry,
    current_model: &mut String,
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
                app.add_system(format!("current autonomy: {:?}", app.autonomy));
            } else {
                app.autonomy = parse_autonomy(parts[1]);
                app.add_system(format!("autonomy -> {:?}", app.autonomy));
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
                        out.push_str(&format!("{}:{}\n{:.100}\n\n",
                            r.file.display(), r.line_start, r.text));
                    }
                    app.add_system(out);
                }
            }
        }
        "/rag-index" => {
            if let Ok(cwd) = std::env::current_dir() {
                match rag.index_dir(&cwd, cfg.rag.index_depth, &cfg.rag.extensions) {
                    Ok(()) => app.add_system("RAG index rebuilt."),
                    Err(e) => app.add_error(format!("RAG index failed: {}", e)),
                }
            }
        }
        "/help" | "/h" => {
            app.add_system(
                "Commands:\n\
                 /models          — list available models\n\
                 /model <id>      — switch model\n\
                 /autonomy <lvl>  — ask|read_only|yolo|god\n\
                 /rag <query>     — search indexed context\n\
                 /rag-index       — reindex cwd\n\
                 /quit            — exit\n\
                 /help            — this message"
            );
        }
        "/quit" | "/q" => {
            // Handled by normal-mode 'q' or we could set a flag
        }
        _ => {
            app.add_error(format!("Unknown command: {}", parts[0]));
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
