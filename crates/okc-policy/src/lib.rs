//! okc-policy — enforceable rules at the tool-executor seam.
//!
//! Policy composes with, and outranks, autonomy: autonomy decides when to
//! ASK (the existing confirm channel, fed by the mutation classification);
//! policy decides what is PERMITTED at all. A policy deny stays denied in
//! every autonomy level, including `god`.
//!
//! Rules enforced here:
//! 1. **Persona restriction** — when the active persona restricts its tools,
//!    a call outside that set is denied at run time (prompt-level hiding
//!    alone could be ignored by the model).
//! 2. **Protected paths** — path globs (`.env`, `*.key`, `~/.ssh/**`, …)
//!    that are never writable/readable, checked in path-like arguments and
//!    in shell command strings.
//! 3. **Root sandbox** — file tools may only touch paths under a configured
//!    root (`$CWD` plus explicit additions). Everything outside is denied.
//!
//! The executor also publishes the previously-orphaned
//! [`Event::FileMutation`] and [`Event::CommandExecuted`] events, which the
//! audit trail (P3) consumes.

use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use glob::Pattern;
use okc_config::PolicyConfig;
use okc_core::{Event, EventBus, MutationKind, ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use tracing::warn;

/// Argument keys treated as path-like when checking policy rules.
const PATH_ARG_KEYS: &[&str] = &[
    "path",
    "file",
    "filename",
    "source",
    "src",
    "destination",
    "dest",
    "target",
    "to",
    "from",
];

/// Compiled policy rules.
pub struct Policy {
    protected: Vec<Pattern>,
    allowed_roots: Vec<PathBuf>,
}

impl Policy {
    /// Compile rules from config. Invalid patterns are warned about and
    /// skipped rather than failing boot.
    pub fn compile(cfg: &PolicyConfig) -> Self {
        let protected = cfg
            .protected_paths
            .iter()
            .filter_map(|raw| {
                let expanded = shellexpand::tilde(raw).to_string();
                match Pattern::new(&expanded) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        warn!("invalid protected_paths pattern '{}': {}", raw, e);
                        None
                    }
                }
            })
            .collect();

        let allowed_roots = cfg
            .allowed_roots
            .iter()
            .map(|raw| {
                if raw == "$CWD" {
                    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
                } else {
                    PathBuf::from(shellexpand::tilde(raw).to_string())
                }
            })
            .map(normalize)
            .collect();

        Self {
            protected,
            allowed_roots,
        }
    }

    /// True when the path matches a protected glob (checked against the
    /// full path and the file name).
    pub fn is_protected(&self, raw: &str) -> bool {
        let path = absolutize(raw);
        let name = path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        self.protected.iter().any(|p| {
            p.matches_path(&path) || (!name.is_empty() && p.matches(&name))
        })
    }

    /// True when the path sits under an allowed root. An empty root list
    /// means "no sandbox".
    pub fn within_roots(&self, raw: &str) -> bool {
        if self.allowed_roots.is_empty() {
            return true;
        }
        let path = absolutize(raw);
        self.allowed_roots.iter().any(|root| path.starts_with(root))
    }
}

/// Expand `~`, make relative paths absolute against the cwd, and normalize
/// `.`/`..` lexically (no filesystem access — targets may not exist yet).
fn absolutize(raw: &str) -> PathBuf {
    let expanded = shellexpand::tilde(raw).to_string();
    let p = PathBuf::from(expanded);
    let absolute = if p.is_absolute() {
        p
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(p)
    };
    normalize(absolute)
}

fn normalize(p: PathBuf) -> PathBuf {
    let mut out = PathBuf::new();
    for comp in p.components() {
        match comp {
            std::path::Component::CurDir => {}
            std::path::Component::ParentDir => {
                out.pop();
            }
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// Tool executor that enforces policy rules around an inner executor.
pub struct PolicyExecutor {
    inner: Arc<dyn ToolExecutor>,
    policy: Policy,
    /// When the active persona restricts its tools: the allow-set. `None`
    /// means no restriction.
    persona_allowed: Option<HashSet<String>>,
    event_bus: EventBus,
}

impl PolicyExecutor {
    pub fn new(
        inner: Arc<dyn ToolExecutor>,
        cfg: &PolicyConfig,
        persona_allowed: Option<HashSet<String>>,
        event_bus: EventBus,
    ) -> Self {
        Self {
            inner,
            policy: Policy::compile(cfg),
            persona_allowed,
            event_bus,
        }
    }

    /// Returns `Some(reason)` when the call is denied.
    fn check(&self, call: &ToolCall) -> Option<String> {
        // 1. Persona restriction (runtime enforcement, not just prompt
        //    hiding).
        if let Some(allowed) = &self.persona_allowed {
            if !allowed.contains(&call.name) {
                return Some(format!(
                    "policy: tool '{}' is not in this persona's allowed tools",
                    call.name
                ));
            }
        }

        // 2. Protected paths in path-like arguments…
        for key in PATH_ARG_KEYS {
            if let Some(path) = call.arguments.get(*key).and_then(|v| v.as_str()) {
                if self.policy.is_protected(path) {
                    return Some(format!("policy: '{}' is a protected path", path));
                }
            }
        }
        // …and in shell command strings (token-scan).
        if let Some(cmd) = call.arguments.get("command").and_then(|v| v.as_str()) {
            for token in cmd.split_whitespace() {
                let cleaned = token.trim_matches(|c| c == '"' || c == '\'' || c == ';' || c == '&');
                if self.policy.is_protected(cleaned) {
                    return Some(format!(
                        "policy: command touches protected path '{}'",
                        cleaned
                    ));
                }
            }
        }

        // 3. Root sandbox for any call carrying path-like arguments.
        for key in PATH_ARG_KEYS {
            if let Some(path) = call.arguments.get(*key).and_then(|v| v.as_str()) {
                if !self.policy.within_roots(path) {
                    return Some(format!(
                        "policy: '{}' is outside the allowed roots — add it to \
                         [policy] allowed_roots if this is intended",
                        path
                    ));
                }
            }
        }

        None
    }

    /// Publish the previously-orphaned mutation/command events for the
    /// audit trail (P3). Best-effort: unparseable results publish what is
    /// known.
    fn publish_events(&self, call: &ToolCall, outcome: &ToolOutcome) {
        if !outcome.success {
            return;
        }
        match call.name.as_str() {
            "write_file" | "edit_file" | "create_directory" | "move_file" => {
                let path = PATH_ARG_KEYS
                    .iter()
                    .find_map(|k| call.arguments.get(*k).and_then(|v| v.as_str()))
                    .unwrap_or("")
                    .to_string();
                let kind = if call.name == "create_directory" {
                    MutationKind::Created
                } else {
                    MutationKind::Updated
                };
                self.event_bus.publish(Event::FileMutation { path, kind });
            }
            "shell" | "git" => {
                let command = call
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| call.name.clone());
                let shell: Option<okc_core::ShellResult> =
                    serde_json::from_value(outcome.result.clone()).ok();
                self.event_bus.publish(Event::CommandExecuted {
                    command,
                    stdout: shell.as_ref().map(|s| s.stdout.clone()).unwrap_or_default(),
                    stderr: shell.as_ref().map(|s| s.stderr.clone()).unwrap_or_default(),
                    exit_code: shell.as_ref().and_then(|s| s.exit_code),
                });
            }
            _ => {}
        }
    }
}

#[async_trait]
impl ToolExecutor for PolicyExecutor {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        self.inner.list_tools().await
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        if let Some(reason) = self.check(call) {
            return Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: false,
                result: serde_json::Value::Null,
                error: Some(reason),
                elapsed_ms: 0,
            });
        }
        let outcome = self.inner.execute_tool(call).await?;
        self.publish_events(call, &outcome);
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// Inner executor that always succeeds — policy is the thing under test.
    struct OkTools;

    #[async_trait]
    impl ToolExecutor for OkTools {
        async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
            Ok(vec![])
        }

        async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
            Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: true,
                result: json!({"ok": true}),
                error: None,
                elapsed_ms: 1,
            })
        }
    }

    fn executor(persona: Option<HashSet<String>>) -> (PolicyExecutor, EventBus) {
        let bus = EventBus::new(64);
        let ex = PolicyExecutor::new(
            Arc::new(OkTools),
            &PolicyConfig::default(),
            persona,
            bus.clone(),
        );
        (ex, bus)
    }

    fn call(name: &str, args: serde_json::Value) -> ToolCall {
        ToolCall {
            id: okc_core::Id::new(),
            name: name.to_string(),
            arguments: args,
        }
    }

    async fn run(ex: &PolicyExecutor, name: &str, args: serde_json::Value) -> ToolOutcome {
        ex.execute_tool(&call(name, args)).await.unwrap()
    }

    #[tokio::test]
    async fn protected_paths_are_denied_in_any_autonomy() {
        let (ex, _) = executor(None);
        for path in [
            "~/.ssh/id_rsa",
            "~/.ssh/config",
            "~/.config/opencode/keys/minimax-sub.key",
            ".env",
            "./config/.env",
            "/tmp/whatever/server.key",
            "secrets.yaml",
            "prod.pem",
        ] {
            let out = run(&ex, "write_file", json!({"path": path, "content": "x"})).await;
            assert!(!out.success, "'{}' must be denied", path);
            assert!(
                out.error.unwrap().contains("protected"),
                "deny reason must name the rule"
            );
        }
    }

    #[tokio::test]
    async fn shell_commands_touching_protected_paths_are_denied() {
        let (ex, _) = executor(None);
        let out = run(
            &ex,
            "shell",
            json!({"command": "cat ~/.ssh/id_rsa >> /tmp/stolen"}),
        )
        .await;
        assert!(!out.success);
    }

    #[tokio::test]
    async fn paths_outside_allowed_roots_are_denied() {
        let (ex, _) = executor(None);
        let out = run(&ex, "read_file", json!({"path": "/etc/passwd"})).await;
        assert!(!out.success);
        assert!(out.error.unwrap().contains("allowed roots"));
    }

    #[tokio::test]
    async fn paths_inside_cwd_pass_through() {
        let (ex, _) = executor(None);
        let out = run(&ex, "write_file", json!({"path": "src/main.rs", "content": "x"})).await;
        assert!(out.success);
        let out = run(&ex, "read_file", json!({"path": "./notes.md"})).await;
        assert!(out.success);
    }

    #[tokio::test]
    async fn empty_roots_mean_no_sandbox() {
        let cfg = PolicyConfig {
            allowed_roots: vec![],
            ..PolicyConfig::default()
        };
        let ex = PolicyExecutor::new(Arc::new(OkTools), &cfg, None, EventBus::new(64));
        let out = run(&ex, "read_file", json!({"path": "/etc/passwd"})).await;
        assert!(out.success);
    }

    #[tokio::test]
    async fn persona_restrictions_are_enforced_at_runtime() {
        let allowed: HashSet<String> = ["read_file", "list_dir"].iter().map(|s| s.to_string()).collect();
        let (ex, _) = executor(Some(allowed));

        let ok = run(&ex, "read_file", json!({"path": "src/main.rs"})).await;
        assert!(ok.success);

        let denied = run(&ex, "write_file", json!({"path": "src/main.rs", "content": "x"})).await;
        assert!(!denied.success);
        assert!(denied.error.unwrap().contains("persona"));
    }

    #[tokio::test]
    async fn successful_mutations_publish_events() {
        let (ex, bus) = executor(None);
        let mut rx = bus.subscribe();

        run(&ex, "write_file", json!({"path": "src/main.rs", "content": "x"})).await;

        match rx.try_recv() {
            Ok(Event::FileMutation { path, kind }) => {
                assert!(path.ends_with("src/main.rs"), "path was {:?}", path);
                assert!(matches!(kind, MutationKind::Updated));
            }
            other => panic!("expected FileMutation event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn commands_publish_execution_events() {
        let (ex, bus) = executor(None);
        let mut rx = bus.subscribe();

        run(&ex, "shell", json!({"command": "echo hi"})).await;

        match rx.try_recv() {
            Ok(Event::CommandExecuted { command, .. }) => assert_eq!(command, "echo hi"),
            other => panic!("expected CommandExecuted event, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn policy_denies_do_not_publish_events() {
        let (ex, bus) = executor(None);
        let mut rx = bus.subscribe();

        let out = run(&ex, "write_file", json!({"path": ".env", "content": "x"})).await;
        assert!(!out.success);
        assert!(rx.try_recv().is_err(), "denied call must not publish");
    }

    #[tokio::test]
    async fn normalization_blocks_dotdot_escapes() {
        let (ex, _) = executor(None);
        let out = run(
            &ex,
            "write_file",
            json!({"path": "../../etc/passwd", "content": "x"}),
        )
        .await;
        assert!(!out.success, "'../..' escape must be caught by the sandbox");
    }
}
