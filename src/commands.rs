// commands.rs — `/` palette command dispatch.

use crate::models::Registry;

#[derive(Debug, Clone)]
pub enum CommandOutput {
    Info(String),
    Result(String),
    Error(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Help,
    Models,
    Model(String),
    Clear,
    Quit,
    Theme,
    Bridge,
    BridgePost { source: String, kind: String, body: String },
    Rag(String),
    RagIndex,
    Unknown(String),
}

pub fn parse(input: &str) -> Command {
    let s = input.trim();
    let s = s.strip_prefix('/').unwrap_or(s);
    let mut parts = s.splitn(2, char::is_whitespace);
    let head = parts.next().unwrap_or("").to_lowercase();
    let rest = parts.next().unwrap_or("").trim().to_string();
    if head == "models" { return Command::Models; }
    if head == "model" { return Command::Model(rest); }
    if head == "clear" { return Command::Clear; }
    if head == "quit" || head == "exit" || head == "q" { return Command::Quit; }
    if head == "theme" { return Command::Theme; }
    if head == "help" || head == "" { return Command::Help; }
    if head == "rag" {
        return Command::Rag(rest);
    }
    if head == "rag-index" {
        return Command::RagIndex;
    }
    if head == "bridge" {
        if rest.is_empty() {
            return Command::Bridge;
        }
        let mut parts = rest.splitn(3, char::is_whitespace);
        let sub = parts.next().unwrap_or("");
        if sub == "post" {
            let source = parts.next().unwrap_or("").to_string();
            let body = parts.next().unwrap_or("").to_string();
            if !source.is_empty() && !body.is_empty() {
                return Command::BridgePost { source, kind: "note".to_string(), body };
            }
        }
        return Command::Unknown(input.to_string());
    }
    Command::Unknown(input.to_string())
}

pub fn run(cmd: &Command, models: &Registry, current_model: &mut String) -> Vec<CommandOutput> {
    match cmd {
        Command::Help => vec![CommandOutput::Info(help_text())],
        Command::Models => vec![CommandOutput::Result(render_models(models))],
        Command::Model(qid) => {
            if qid.is_empty() {
                return vec![CommandOutput::Error("usage: /model <provider>/<model_id>".to_string())];
            }
            if models.get(qid).is_none() {
                return vec![CommandOutput::Error("no such model (use /models)".to_string())];
            }
            *current_model = qid.to_string();
            vec![CommandOutput::Info("model -> ".to_string() + qid)]
        }
        Command::Clear => vec![CommandOutput::Info("__clear__".to_string())],
        Command::Quit => vec![CommandOutput::Info("__quit__".to_string())],
        Command::Theme => vec![CommandOutput::Info("theme switch (placeholder)".to_string())],
        Command::Bridge => vec![CommandOutput::Info("__bridge_read__".to_string())],
        Command::BridgePost { source: _, .. } => vec![CommandOutput::Info("__bridge_post__".to_string())],
        Command::Rag(_) => vec![CommandOutput::Info("__rag__".to_string())],
        Command::RagIndex => vec![CommandOutput::Info("__rag_index__".to_string())],
        Command::Unknown(_) => vec![CommandOutput::Error("unknown command (try /help)".to_string())],
    }
}

fn help_text() -> String {
    let mut s = String::from("commands (Phase 3.5):\n");
    s.push_str("  /help                    show this\n");
    s.push_str("  /models                  list all available models\n");
    s.push_str("  /model <id>              set active model\n");
    s.push_str("  /rag <query>             search indexed codebase (keyword RAG)\n");
    s.push_str("  /rag-index               rebuild RAG index from cwd\n");
    s.push_str("  /bridge                  read recent opencode<->openklawc bridge messages\n");
    s.push_str("  /bridge post <src> <body>   post a message to the bridge\n");
    s.push_str("  /clear                   clear scrollback\n");
    s.push_str("  /quit                    exit openklawc (also: q in normal mode)\n");
    s.push_str("  /theme                   (placeholder) switch theme\n");
    s.push_str("\n");
    s.push_str("tips:\n");
    s.push_str("  @file.rs in a message attaches that file as context\n");
    s
}

fn render_models(models: &Registry) -> String {
    let mut s = String::new();
    s.push_str(&format!("models ({} available, current: {}):\n",
                         models.iter().count(), models.default_id()));
    let mut current_provider = "";
    for m in models.iter() {
        if m.provider != current_provider {
            current_provider = m.provider;
            s.push_str(&format!("\n  [{}]\n", current_provider));
        }
        let mut tags = Vec::new();
        if m.reasoning { tags.push("reasoning"); }
        if m.vision { tags.push("vision"); }
        if let Some(c) = m.context { tags.push(c); }
        let tag_str = if tags.is_empty() {
            String::new()
        } else {
            format!("  ({})", tags.join(", "))
        };
        s.push_str(&format!("    {:<60}{}\n", m.display_name, tag_str));
        s.push_str(&format!("      {}\n", m.qualified_id()));
    }
    s
}
