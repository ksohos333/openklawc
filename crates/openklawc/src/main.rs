//! openklawc — autonomous Rust coding agent.
//!
//! Thin binary: parse arguments, pick a front-end (TUI or headless), and
//! drive the application core in `okc-app`. Bootstrap, tool assembly, model
//! resolution, session lifecycle and persistence all live there.

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

use okc_app::{parse_autonomy, title_from, App as CoreApp, AppOptions};
use okc_core::{ChatMessage, TokenUsage};
use okc_reasoning::{AgentUpdate, ConfirmRequest};
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
         asked, headless runs deny them. Use --autonomy yolo to allow them unattended.\n\n\
         Environment:\n  \
         OKC_MODEL, OKC_AUTONOMY   override model / autonomy (beat every config file)\n\n\
         MCP servers:\n  \
         The bundled Rust servers (filesystem, brave-search, playwright, vision, local-brain,\n  \
         laya) start automatically. A [mcp.<name>] entry in .openklawc.toml or\n  \
         ~/.config/openklawc/config.toml overrides one; set enabled = false to switch it off.\n"
    }

    fn options(&self) -> AppOptions {
        AppOptions {
            model: self.model.clone(),
            autonomy: self.autonomy.clone().map(|a| parse_autonomy(&a)),
            persona: self.persona.clone(),
            start_mcp: !self.no_mcp,
            audit_db: None,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Structured JSON logs on stderr — consumable by CI, the daemon and
    // log shippers.
    tracing_subscriber::fmt().json().init();

    let args = Args::parse(std::env::args().skip(1));

    if args.help {
        print!("{}", Args::usage());
        return Ok(());
    }

    if let Some(prompt) = args.prompt.clone() {
        return run_headless(&args, prompt).await;
    }

    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;
    let mut terminal = Terminal::new(CrosstermBackend::new(stdout()))?;

    let result = run_app(&mut terminal, &args).await;

    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    result
}

/// Headless front-end: one turn, printed to stdout/stderr, then exit.
async fn run_headless(args: &Args, prompt: String) -> Result<()> {
    let (mut core, boot) = CoreApp::boot(args.options()).await?;

    for failure in &boot.mcp_failures {
        eprintln!("[mcp] failed to start: {}", failure);
    }
    eprintln!("[mcp] {} tools registered", boot.mcp_tools);
    for warning in &boot.provider_warnings {
        eprintln!("[warn] {}", warning);
    }
    eprintln!("[model] {}", boot.model);

    // Headless runs never attach an approval channel, so any tool call that
    // needs confirmation is denied — by construction, not by accident.
    let session_id = core.create_session(&title_from(&prompt), "build");
    if let Some(id) = &session_id {
        core.bind_session(id);
    }
    let mut persisted: usize = 0;

    let failed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let done_store: Arc<std::sync::Mutex<Option<(Vec<ChatMessage>, Option<TokenUsage>)>>> =
        Arc::new(std::sync::Mutex::new(None));

    let rx = core.submit(&prompt, Vec::new()).await;

    let printer_failed = failed.clone();
    let printer_done = done_store.clone();
    let printer = tokio::spawn(async move {
        let mut rx = rx;
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
                    eprintln!("[{}] {} ({}ms) {}", status, name, elapsed_ms, truncate(&body, 200));
                }
                AgentUpdate::Error(e) => {
                    eprintln!("[error] {}", e);
                    printer_failed.store(true, std::sync::atomic::Ordering::Relaxed);
                }
                AgentUpdate::Done { messages, usage, .. } => {
                    *printer_done.lock().unwrap() = Some((messages, usage));
                }
            }
        }
    });
    printer.await.ok();
    println!();

    // Persist the run.
    if let Some(id) = &session_id {
        if let Some((messages, usage)) = done_store.lock().unwrap().take() {
            core.persist_turn(id, &mut persisted, &messages, usage);
        }
        eprintln!("[session] {}", &id[..8.min(id.len())]);
    }

    if failed.load(std::sync::atomic::Ordering::Relaxed) {
        std::process::exit(1);
    }
    Ok(())
}

/// TUI front-end: render loop plus keyboard handling. All engine work goes
/// through the `okc-app` core.
async fn run_app(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, args: &Args) -> Result<()> {
    let (mut core, boot) = CoreApp::boot(args.options()).await?;

    // The TUI can ask a human: attach the approval channel.
    let (confirm_tx, mut confirm_rx) = mpsc::channel::<ConfirmRequest>(32);
    core.attach_approval(confirm_tx);

    let mut app = App::new();
    app.model = boot.model.clone();
    app.autonomy = core.agent.autonomy;
    app.mcp_count = boot.mcp_tools;
    app.show_reasoning = core.cfg.ui.show_reasoning;
    app.theme = Theme::from_name(&core.cfg.ui.theme);
    for failure in &boot.mcp_failures {
        app.add_system(format!("[mcp] failed to start: {}", failure));
    }
    for warning in &boot.provider_warnings {
        app.add_system(format!("[warn] {}", warning));
    }

    let mut messages: Vec<ChatMessage> = Vec::new();
    let mut persisted_count: usize = 0;
    let mut agent_running = false;
    let mut rx: Option<mpsc::Receiver<AgentUpdate>> = None;
    let mut pending_confirms: std::collections::VecDeque<ConfirmRequest> =
        std::collections::VecDeque::new();

    loop {
        // ─── Drain agent updates ──────────────────────────────────────
        if let Some(stream) = rx.as_mut() {
            loop {
                match stream.try_recv() {
                    Ok(update) => match update {
                        AgentUpdate::TextChunk(chunk) => {
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
                            let result_str =
                                serde_json::to_string(&result).unwrap_or_default();
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
                            app.update_last_assistant(text);
                            if let Some(sid) = app.current_session_id.clone() {
                                core.persist_turn(&sid, &mut persisted_count, &messages, usage);
                                app.token_count = core.session_tokens(&sid) as usize;
                                app.verify_badge = Some(core.verification_counts(&sid));
                            }
                        }
                        AgentUpdate::Error(e) => {
                            agent_running = false;
                            app.add_error(format!("Agent error: {}", e));
                        }
                    },
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => {
                        // Turn finished (Done already drained) or the task
                        // died; either way nothing more arrives.
                        rx = None;
                        agent_running = false;
                        break;
                    }
                }
            }
        }

        // ─── Pending tool approvals ───────────────────────────────────
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

        // ─── Render ───────────────────────────────────────────────────
        terminal.draw(|frame| {
            app.render(frame, frame.area());
        })?;

        // ─── Poll keyboard ────────────────────────────────────────────
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

                    // Start a session the first time the user talks, so the
                    // chat picker has something to show.
                    if app.current_session_id.is_none() {
                        let title = title_from(&query);
                        let mode = app.chat_picker.mode.as_str().to_string();
                        if let Some(id) = core.create_session(&title, &mode) {
                            app.current_session_id = Some(id.clone());
                            core.bind_session(&id);
                            persisted_count = 0;
                            app.verify_badge = None;
                        }
                    }

                    app.add_assistant("...");
                    agent_running = true;
                    rx = Some(core.submit(&query, messages.clone()).await);
                }
                Some(AppEvent::ListModels) => {
                    let items: Vec<PickerItem> = core
                        .llm_registry
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
                    let items = core
                        .list_sessions(50)
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
                        .collect();
                    app.set_overlay_items(items);
                }
                Some(AppEvent::SwitchModel(id)) => {
                    if core.set_model(&id) {
                        app.model = id.clone();
                        app.add_system(format!("Switched to {}", id));
                    } else {
                        app.add_error(format!("Unknown model: {}", id));
                    }
                }
                Some(AppEvent::ResumeSession(id)) => {
                    match core.get_messages(&id) {
                        Some(persisted) => {
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
                            core.bind_session(&id);
                            persisted_count = persisted.len();
                            app.token_count = core.session_tokens(&id) as usize;
                            app.verify_badge = Some(core.verification_counts(&id));
                            app.add_system(format!(
                                "Resumed session {}",
                                &id[..8.min(id.len())]
                            ));
                        }
                        None => app.add_error("Failed to resume: session not found"),
                    }
                }
                Some(AppEvent::SetAutonomy(level)) => {
                    core.set_autonomy(level);
                }
                Some(AppEvent::OpenChatPicker) => {
                    app.chat_picker.set_sessions(load_chat_sessions(&core));
                }
                Some(AppEvent::CreateSession(mode)) => {
                    let name = format!("New {} chat", mode.label().to_lowercase());
                    match core.create_session(&name, mode.as_str()) {
                        Some(id) => {
                            messages.clear();
                            app.clear_transcript();
                            app.token_count = 0;
                            app.verify_badge = None;
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
                    app.chat_picker.set_sessions(load_chat_sessions(&core));
                }
                Some(AppEvent::DeleteSession(id)) => {
                    let deleted = core.delete_session(&id);
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
                    app.chat_picker.set_sessions(load_chat_sessions(&core));
                }
                Some(AppEvent::ConfirmTool(approved)) => {
                    // Take ownership, then answer. A failed send means the
                    // agent side is already gone — nothing to notify.
                    if let Some(request) = pending_confirms.pop_front() {
                        let _ = request.respond.send(approved);
                    }
                    app.pending_confirm = None;
                }
                Some(AppEvent::SearchHistory(query)) => {
                    let items = core
                        .search_sessions(&query, 20)
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
                        .collect();
                    app.set_overlay_items(items);
                }
                Some(AppEvent::AuditTrail) => {
                    match app.current_session_id.clone() {
                        Some(sid) => {
                            let steps = core.reasoning_trail(&sid, 25);
                            let events = core.audit_trail(&sid, 25);
                            if steps.is_empty() && events.is_empty() {
                                app.add_system("No audit trail for this session yet.");
                            }
                            for step in steps {
                                let decision = step
                                    .decision_json
                                    .as_deref()
                                    .and_then(|d| serde_json::from_str::<serde_json::Value>(d).ok())
                                    .and_then(|v| {
                                        v.get("confidence")
                                            .and_then(|c| c.as_str())
                                            .map(|c| format!(" confidence={}", c))
                                    })
                                    .unwrap_or_default();
                                let content: String = step.content.chars().take(120).collect();
                                app.add_system(format!(
                                    "[reason] {} {}{}",
                                    step.kind, content, decision
                                ));
                            }
                            for event in events {
                                app.add_system(format!("[audit] {} {}", event.kind, event.payload));
                            }
                        }
                        None => app.add_error("No session yet — start a chat first."),
                    }
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

/// Build the chat-picker rows from the core's session list.
fn load_chat_sessions(core: &CoreApp) -> Vec<ChatSession> {
    core.list_sessions(200)
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
