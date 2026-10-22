// openklawc — Phase 4: LLM client + file context + RAG
//
// Keybindings:
//   Normal: i insert, q quit, j/k scroll, `/` commands, `:` commands
//   Insert: Ctrl-a/e, Ctrl-k/u/w, Enter submit
//   Esc: toggle mode / cancel
//   Ctrl-C: force quit

mod commands;
mod config;
mod context;
mod llm;
mod models;
mod mcp;
mod rag;
mod ui;

use std::io::{self, stdout};
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use ratatui::{backend::CrosstermBackend, Terminal};
use serde_json::json;

use commands::{parse as parse_command, run as run_command, Command, CommandOutput};
use config::AppConfig;
use context::{extract_file_refs, gather_cwd_context, read_file_for_context, strip_file_refs};
use llm::{LlmClient, ChatMessage, ProviderRegistry as LlmProviderRegistry};
use models::Registry as Models;
use mcp::Registry;
use ui::chat::{Chat, Mode, ScrollEntry};

struct AppState {
    chat: Chat,
    registry: Option<Registry>,
    models: Models,
    current_model: String,
    llm_client: LlmClient,
    llm_registry: LlmProviderRegistry,
    rag: rag::SimpleRag,
}

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<()> {
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    // Load opencode config for LLM providers
    let opencode_cfg = match llm::config::OpencodeConfig::load_default() {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("ERROR loading opencode config: {e:#}");
            eprintln!("Falling back to empty providers. Run with ~/.config/opencode/opencode.jsonc present.");
            llm::config::OpencodeConfig {
                model: "local/Qwen3-14B-Q4_K_M".to_string(),
                small_model: None,
                provider: std::collections::HashMap::new(),
            }
        }
    };
    let llm_registry = LlmProviderRegistry::from_config(&opencode_cfg);
    let llm_client = LlmClient::new();
    let current_model = llm_registry.default_id().to_string();

    // Load openklawc config for MCPs
    let registry = match AppConfig::load_default() {
        Ok(cfg) => match Registry::start(&cfg.mcp).await {
            Ok(r) => Some(r),
            Err(e) => {
                eprintln!("warning: MCP startup failed ({e:#}); running in stub mode");
                None
            }
        },
        Err(e) => {
            eprintln!("warning: no openklawc config ({e:#}); MCPs disabled");
            None
        }
    };

    // Initialize simple RAG
    let mut rag = rag::SimpleRag::new();
    let _ = rag.index_dir(&std::env::current_dir().unwrap_or_default(), 3);

    let mut state = AppState {
        chat: Chat::new(),
        registry,
        models: Models::default_registry(),
        current_model,
        llm_client,
        llm_registry,
        rag,
    };

    let result = run(&mut terminal, &mut state).await;

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;
    result
}

#[derive(Debug, PartialEq, Eq)]
enum KeyOutcome {
    Continue,
    Quit,
}

async fn run(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
) -> Result<()> {
    loop {
        terminal.draw(|frame| {
            state.chat.render(frame, frame.size());
        })?;

        if !event::poll(Duration::from_millis(100))? {
            continue;
        }
        let evt = event::read()?;
        if let Event::Key(key) = evt {
            if handle_key(state, key).await == KeyOutcome::Quit {
                return Ok(());
            }
        }
    }
}

async fn handle_key(state: &mut AppState, key: KeyEvent) -> KeyOutcome {
    if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
        return KeyOutcome::Quit;
    }

    if state.chat.command.active {
        return handle_command_key(state, key).await;
    }

    match state.chat.mode {
        Mode::Normal => handle_normal(state, key),
        Mode::Insert => handle_insert(state, key).await,
    }
}

fn handle_normal(state: &mut AppState, key: KeyEvent) -> KeyOutcome {
    match key.code {
        KeyCode::Char('q') => KeyOutcome::Quit,
        KeyCode::Char('i') | KeyCode::Char('a') | KeyCode::Char('A') | KeyCode::Char('o') => {
            state.chat.mode = Mode::Insert;
            KeyOutcome::Continue
        }
        KeyCode::Char('/') => {
            state.chat.command.active = true;
            state.chat.command.input.clear();
            state.chat.command.cursor = 0;
            state.chat.command.input.insert(0, '/');
            state.chat.command.cursor = 1;
            KeyOutcome::Continue
        }
        KeyCode::Char('j') | KeyCode::Down => {
            state.chat.scroll_down();
            KeyOutcome::Continue
        }
        KeyCode::Char('k') | KeyCode::Up => {
            state.chat.scroll_up();
            KeyOutcome::Continue
        }
        _ => KeyOutcome::Continue,
    }
}

async fn handle_insert(state: &mut AppState, key: KeyEvent) -> KeyOutcome {
    if key.code == KeyCode::Esc {
        state.chat.mode = Mode::Normal;
        return KeyOutcome::Continue;
    }

    if key.modifiers.contains(KeyModifiers::CONTROL) {
        match key.code {
            KeyCode::Char('a') => { state.chat.input_home(); }
            KeyCode::Char('e') => { state.chat.input_end(); }
            KeyCode::Char('b') => { state.chat.input_left(); }
            KeyCode::Char('f') => { state.chat.input_right(); }
            KeyCode::Char('k') => { state.chat.input_kill_to_end(); }
            KeyCode::Char('u') => { state.chat.input_kill_to_start(); }
            KeyCode::Char('w') => { state.chat.input_kill_word(); }
            _ => {}
        }
        return KeyOutcome::Continue;
    }

    match key.code {
        KeyCode::Enter => {
            let query = state.chat.submit();
            if !query.is_empty() {
                handle_submit(state, &query).await;
            }
        }
        KeyCode::Backspace => { state.chat.input_backspace(); }
        KeyCode::Delete => { state.chat.input_delete_forward(); }
        KeyCode::Left => { state.chat.input_left(); }
        KeyCode::Right => { state.chat.input_right(); }
        KeyCode::Home => { state.chat.input_home(); }
        KeyCode::End => { state.chat.input_end(); }
        KeyCode::Char(c) => { state.chat.input_insert(c); }
        _ => {}
    }
    KeyOutcome::Continue
}

/// Send user query to the configured LLM with optional file context.
async fn handle_submit(state: &mut AppState, query: &str) {
    // Extract @file references
    let file_refs = extract_file_refs(query);
    let clean_query = strip_file_refs(query);

    // Build messages
    let mut messages: Vec<ChatMessage> = Vec::new();

    // System prompt with file context
    let mut system = String::from("You are a helpful coding assistant. ");
    system.push_str("You have access to the current project context. ");
    system.push_str("Respond concisely but thoroughly.");

    // Attach referenced files
    for path in &file_refs {
        if let Ok(content) = read_file_for_context(path) {
            system.push_str("\n\n");
            system.push_str(&content);
        }
    }

    // If no file refs, include CWD context summary
    if file_refs.is_empty() {
        if let Ok(ctx) = gather_cwd_context(2, 20) {
            system.push_str("\n\n## Current project context:\n");
            // Truncate to avoid blowing context window
            let summary: String = ctx.lines().take(50).collect::<Vec<_>>().join("\n");
            system.push_str(&summary);
        }
    }

    messages.push(ChatMessage {
        role: "system".to_string(),
        content: system,
    });
    messages.push(ChatMessage {
        role: "user".to_string(),
        content: clean_query.to_string(),
    });

    // Add "thinking..." placeholder
    state.chat.scrollback.push(ScrollEntry::Assistant("...".to_string()));
    let assistant_idx = state.chat.scrollback.len() - 1;

    // Send to LLM
    if let Some(model) = state.llm_registry.get(&state.current_model) {
        match state.llm_client.chat(model, messages).await {
            Ok(response) => {
                state.chat.scrollback[assistant_idx] = ScrollEntry::Assistant(response);
            }
            Err(e) => {
                state.chat.scrollback[assistant_idx] = ScrollEntry::System(format!("LLM error: {e}"));
            }
        }
    } else {
        state.chat.scrollback[assistant_idx] = ScrollEntry::System(
            format!("Model '{}' not found (use /models)", state.current_model)
        );
    }
}

async fn handle_command_key(state: &mut AppState, key: KeyEvent) -> KeyOutcome {
    match key.code {
        KeyCode::Esc => {
            state.chat.command.active = false;
            state.chat.command.input.clear();
            state.chat.command.cursor = 0;
            KeyOutcome::Continue
        }
        KeyCode::Enter => {
            let raw = state.chat.command.input.clone();
            state.chat.command.active = false;
            state.chat.command.input.clear();
            state.chat.command.cursor = 0;
            return dispatch_command(state, &raw).await;
        }
        KeyCode::Backspace => {
            if state.chat.command.cursor > 0 {
                let prev = state.chat.command.input[..state.chat.command.cursor]
                    .char_indices()
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                state.chat.command.input.remove(prev);
                state.chat.command.cursor = prev;
            }
            KeyOutcome::Continue
        }
        KeyCode::Left => {
            if state.chat.command.cursor > 0 {
                state.chat.command.cursor = state.chat.command.input[..state.chat.command.cursor]
                    .char_indices()
                    .last()
                    .map(|(i, _)| i)
                    .unwrap_or(0);
            }
            KeyOutcome::Continue
        }
        KeyCode::Right => {
            if state.chat.command.cursor < state.chat.command.input.len() {
                state.chat.command.cursor = state.chat.command.input[state.chat.command.cursor..]
                    .char_indices()
                    .nth(1)
                    .map(|(i, _)| state.chat.command.cursor + i)
                    .unwrap_or(state.chat.command.input.len());
            }
            KeyOutcome::Continue
        }
        KeyCode::Char(c) => {
            state.chat.command.input.insert(state.chat.command.cursor, c);
            state.chat.command.cursor += c.len_utf8();
            KeyOutcome::Continue
        }
        _ => KeyOutcome::Continue,
    }
}

async fn dispatch_command(state: &mut AppState, raw: &str) -> KeyOutcome {
    let cmd = parse_command(raw);
    let mut outputs = Vec::new();

    // Override model commands to use LLM registry
    match &cmd {
        Command::Models => {
            let mut out = String::new();
            out.push_str(&format!("models ({} available, current: {}):\n",
                state.llm_registry.list().len(), state.current_model));
            let mut current_provider = "";
            for (qid, model) in state.llm_registry.list() {
                if model.provider != current_provider {
                    current_provider = &model.provider;
                    out.push_str(&format!("\n  [{}]\n", current_provider));
                }
                let mut tags = Vec::new();
                if model.reasoning { tags.push("reasoning"); }
                if model.vision { tags.push("vision"); }
                let tag_str = if tags.is_empty() {
                    String::new()
                } else {
                    format!(" [{}]", tags.join(", "))
                };
                let marker = if qid == &state.current_model { "→ " } else { "  " };
                out.push_str(&format!("{}{}{}  ({})\n", marker, model.display_name, tag_str, qid));
            }
            outputs.push(CommandOutput::Result(out));
        }
        Command::Model(qid) => {
            if qid.is_empty() {
                outputs.push(CommandOutput::Error("usage: /model <provider>/<model_id>".to_string()));
            } else if state.llm_registry.get(qid).is_none() {
                outputs.push(CommandOutput::Error("no such model (use /models)".to_string()));
            } else {
                state.current_model = qid.to_string();
                outputs.push(CommandOutput::Info(format!("model -> {}", qid)));
            }
        }
        _ => {
            outputs = run_command(&cmd, &state.models, &mut state.current_model);
        }
    }

    // Handle LLM-specific commands
    match &cmd {
        Command::Rag(query) => {
            if state.rag.is_empty() {
                outputs.push(CommandOutput::Info("RAG not indexed. Run /rag-index first.".to_string()));
            } else {
                let results = state.rag.search(query);
                if results.is_empty() {
                    outputs.push(CommandOutput::Info("No matches found.".to_string()));
                } else {
                    let mut out = String::from("RAG results:\n");
                    for r in results.iter().take(5) {
                        out.push_str(&format!("{r}\n"));
                    }
                    outputs.push(CommandOutput::Result(out));
                }
            }
        }
        Command::RagIndex => {
            let cwd = std::env::current_dir().unwrap_or_default();
            match state.rag.index_dir(&cwd, 3) {
                Ok(()) => outputs.push(CommandOutput::Info("RAG index rebuilt.".to_string())),
                Err(e) => outputs.push(CommandOutput::Error(format!("RAG index failed: {e}"))),
            }
        }
        _ => {}
    }

    // Bridge commands (MCP)
    match &cmd {
        Command::Bridge => {
            if let Some(reg) = state.registry.as_mut() {
                if let Some(client) = reg.get_mut("anticonf") {
                    match client.call_tool("bridge_read", json!({"n": 20})).await {
                        Ok(value) => {
                            outputs.push(CommandOutput::Result(format_bridge_messages(&value)));
                        }
                        Err(e) => {
                            outputs.push(CommandOutput::Error(format!("bridge_read failed: {e}")));
                        }
                    }
                } else {
                    outputs.push(CommandOutput::Error("anticonf MCP not loaded".to_string()));
                }
            } else {
                outputs.push(CommandOutput::Error("(no MCPs loaded)".to_string()));
            }
        }
        Command::BridgePost { source, kind, body } => {
            if let Some(reg) = state.registry.as_mut() {
                if let Some(client) = reg.get_mut("anticonf") {
                    match client.call_tool("bridge_post", json!({"source": source, "kind": kind, "body": body})).await {
                        Ok(value) => {
                            let id = value.get("id").and_then(|v| v.as_str()).unwrap_or("?");
                            outputs.push(CommandOutput::Info(format!("posted: {id}")));
                        }
                        Err(e) => {
                            outputs.push(CommandOutput::Error(format!("bridge_post failed: {e}")));
                        }
                    }
                } else {
                    outputs.push(CommandOutput::Error("anticonf MCP not loaded".to_string()));
                }
            } else {
                outputs.push(CommandOutput::Error("(no MCPs loaded)".to_string()));
            }
        }
        _ => {}
    }

    for out in outputs {
        match out {
            CommandOutput::Error(s) => state.chat.scrollback.push(ScrollEntry::System(format!("! {}", s))),
            CommandOutput::Info(s) if s == "__clear__" => {
                state.chat.scrollback.clear();
                state.chat.scroll_offset = 0;
            }
            CommandOutput::Info(s) if s == "__quit__" => return KeyOutcome::Quit,
            CommandOutput::Info(s) => state.chat.scrollback.push(ScrollEntry::System(s)),
            CommandOutput::Result(s) => state.chat.scrollback.push(ScrollEntry::Assistant(s)),
        }
    }

    if matches!(cmd, Command::Quit) {
        return KeyOutcome::Quit;
    }
    KeyOutcome::Continue
}

fn format_bridge_messages(value: &serde_json::Value) -> String {
    let mut s = String::new();
    let messages = value.get("messages").and_then(|m| m.as_array());
    match messages {
        Some(arr) if arr.is_empty() => {
            s.push_str("bridge: (no messages yet)\n");
        }
        Some(arr) => {
            s.push_str(&format!("bridge: {} recent message(s):\n", arr.len()));
            for m in arr {
                let source = m.get("source").and_then(|t| t.as_str()).unwrap_or("?");
                let kind = m.get("kind").and_then(|t| t.as_str()).unwrap_or("?");
                let body = m.get("body").and_then(|t| t.as_str()).unwrap_or("");
                s.push_str(&format!("  [{}] {}: {}\n", source, kind, body));
            }
        }
        None => {
            s.push_str("bridge: (no messages yet)\n");
        }
    }
    s
}
