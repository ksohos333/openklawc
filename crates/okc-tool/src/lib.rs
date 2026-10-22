//! openklawc-tool — built-in tools (shell, file, git, search, etc.).
//!
//! These are the "native" tools that don't require an MCP server.
//! They form the "yolo" capability of openklawc.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use okc_core::{ChatMessage, ShellResult, ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use serde_json::{json, Value};
use tokio::process::Command;
use tracing::{debug, info, warn};

pub mod composite;
pub mod fs;
pub mod shell;

pub use composite::CompositeExecutor;

/// Built-in tool registry.
pub struct BuiltInTools {
    schemas: HashMap<String, ToolSchema>,
}

impl BuiltInTools {
    pub fn new() -> Self {
        let mut schemas = HashMap::new();

        schemas.insert(
            "read_file".to_string(),
            ToolSchema {
                name: "read_file".to_string(),
                description: "Read the contents of a file.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "File path to read" },
                        "offset": { "type": "integer", "description": "Line offset (0-based)" },
                        "limit": { "type": "integer", "description": "Max lines to read" }
                    },
                    "required": ["path"]
                }),
            },
        );

        schemas.insert(
            "write_file".to_string(),
            ToolSchema {
                name: "write_file".to_string(),
                description: "Write content to a file. Creates or overwrites.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string" },
                        "content": { "type": "string" }
                    },
                    "required": ["path", "content"]
                }),
            },
        );

        schemas.insert(
            "list_dir".to_string(),
            ToolSchema {
                name: "list_dir".to_string(),
                description: "List files in a directory.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory path" },
                        "recursive": { "type": "boolean" }
                    },
                    "required": ["path"]
                }),
            },
        );

        schemas.insert(
            "search_files".to_string(),
            ToolSchema {
                name: "search_files".to_string(),
                description: "Search for a pattern in files.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regex or substring" },
                        "path": { "type": "string", "description": "Directory to search" },
                        "glob": { "type": "string", "description": "File glob pattern, e.g. '*.rs'" }
                    },
                    "required": ["pattern", "path"]
                }),
            },
        );

        schemas.insert(
            "shell".to_string(),
            ToolSchema {
                name: "shell".to_string(),
                description: "Run a shell command. Use with caution.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "Shell command to run" },
                        "timeout": { "type": "integer", "description": "Timeout in seconds (default 30)" },
                        "cwd": { "type": "string", "description": "Working directory" }
                    },
                    "required": ["command"]
                }),
            },
        );

        schemas.insert(
            "git".to_string(),
            ToolSchema {
                name: "git".to_string(),
                description: "Run a git command.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Git arguments, e.g. ['status', '--short']"
                        },
                        "cwd": { "type": "string" }
                    },
                    "required": ["args"]
                }),
            },
        );

        schemas.insert(
            "view_cwd".to_string(),
            ToolSchema {
                name: "view_cwd".to_string(),
                description: "Show current working directory and project structure summary.".to_string(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "depth": { "type": "integer", "description": "How many levels deep to show" }
                    }
                }),
            },
        );

        Self { schemas }
    }

    fn is_mutation(name: &str) -> bool {
        matches!(name, "write_file" | "shell" | "git")
    }
}

#[async_trait]
impl ToolExecutor for BuiltInTools {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(self.schemas.values().cloned().collect())
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        let start = std::time::Instant::now();
        let result = match call.name.as_str() {
            "read_file" => tool_read_file(&call.arguments).await,
            "write_file" => tool_write_file(&call.arguments).await,
            "list_dir" => tool_list_dir(&call.arguments).await,
            "search_files" => tool_search_files(&call.arguments).await,
            "shell" => tool_shell(&call.arguments).await,
            "git" => tool_git(&call.arguments).await,
            "view_cwd" => tool_view_cwd(&call.arguments).await,
            _ => Err(anyhow::anyhow!("Unknown built-in tool: {}", call.name)),
        };

        let elapsed = start.elapsed().as_millis() as u64;
        match result {
            Ok(value) => Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: true,
                result: value,
                error: None,
                elapsed_ms: elapsed,
            }),
            Err(e) => Ok(ToolOutcome {
                call_id: call.id.clone(),
                success: false,
                result: Value::Null,
                error: Some(e.to_string()),
                elapsed_ms: elapsed,
            }),
        }
    }
}

// ─── Tool implementations ────────────────────────────────────────────────────

async fn tool_read_file(args: &Value) -> Result<Value> {
    let path = args["path"].as_str().ok_or_else(|| anyhow::anyhow!("path required"))?;
    let path = shellexpand::tilde(path);
    let text = tokio::fs::read_to_string(path.as_ref()).await?;

    let offset = args["offset"].as_u64().unwrap_or(0) as usize;
    let limit = args["limit"].as_u64().unwrap_or(200) as usize;

    let lines: Vec<&str> = text.lines().skip(offset).take(limit).collect();
    Ok(json!({
        "content": lines.join("\n"),
        "offset": offset,
        "total_lines": text.lines().count(),
        "truncated": lines.len() < text.lines().count(),
    }))
}

async fn tool_write_file(args: &Value) -> Result<Value> {
    let path = args["path"].as_str().ok_or_else(|| anyhow::anyhow!("path required"))?;
    let content = args["content"].as_str().ok_or_else(|| anyhow::anyhow!("content required"))?;
    let path = shellexpand::tilde(path);

    if let Some(parent) = Path::new(path.as_ref()).parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(path.as_ref(), content).await?;
    Ok(json!({ "written": true, "path": path.as_ref() }))
}

fn tool_list_dir_sync(args: &Value) -> Result<Value> {
    let path = args["path"].as_str().unwrap_or(".");
    let recursive = args["recursive"].as_bool().unwrap_or(false);
    let path = shellexpand::tilde(path);

    let mut entries = Vec::new();
    let read_dir = std::fs::read_dir(path.as_ref())?;

    for entry in read_dir {
        let entry = entry?;
        let meta = entry.metadata()?;
        let name = entry.file_name().to_string_lossy().to_string();
        let is_dir = meta.is_dir();
        entries.push(json!({
            "name": name,
            "is_dir": is_dir,
            "size": meta.len(),
        }));

        if recursive && is_dir {
            let sub = json!({ "path": entry.path().to_string_lossy().to_string(), "recursive": true });
            if let Ok(sub_result) = tool_list_dir_sync(&sub) {
                if let Some(sub_entries) = sub_result["entries"].as_array() {
                    for se in sub_entries {
                        entries.push(se.clone());
                    }
                }
            }
        }
    }

    Ok(json!({ "entries": entries }))
}

async fn tool_list_dir(args: &Value) -> Result<Value> {
    // Run the sync version in a blocking task to avoid recursive async
    let args = args.clone();
    tokio::task::spawn_blocking(move || tool_list_dir_sync(&args))
        .await
        .map_err(|e| anyhow::anyhow!("task join error: {}", e))?
}

async fn tool_search_files(args: &Value) -> Result<Value> {
    let pattern = args["pattern"].as_str().ok_or_else(|| anyhow::anyhow!("pattern required"))?;
    let path = args["path"].as_str().unwrap_or(".");
    let path = shellexpand::tilde(path);
    let glob_pat = args["glob"].as_str().unwrap_or("*");

    let mut matches = Vec::new();
    let regex = regex::Regex::new(pattern).ok();

    for entry in walkdir::WalkDir::new(path.as_ref()).max_depth(5) {
        let entry = entry?;
        if !entry.file_type().is_file() {
            continue;
        }
        let name = entry.file_name().to_string_lossy();
        if !glob::Pattern::new(glob_pat)?.matches(&name) {
            continue;
        }

        let text = match tokio::fs::read_to_string(entry.path()).await {
            Ok(t) => t,
            Err(_) => continue,
        };

        for (i, line) in text.lines().enumerate() {
            let matched = if let Some(re) = &regex {
                re.is_match(line)
            } else {
                line.contains(pattern)
            };
            if matched {
                matches.push(json!({
                    "file": entry.path().to_string_lossy().to_string(),
                    "line": i + 1,
                    "text": line.chars().take(200).collect::<String>(),
                }));
                if matches.len() >= 50 {
                    break;
                }
            }
        }
        if matches.len() >= 50 {
            break;
        }
    }

    Ok(json!({ "matches": matches, "count": matches.len() }))
}

async fn tool_shell(args: &Value) -> Result<Value> {
    let cmd = args["command"].as_str().ok_or_else(|| anyhow::anyhow!("command required"))?;
    let timeout_secs = args["timeout"].as_u64().unwrap_or(30);
    let cwd = args["cwd"].as_str().map(|s| shellexpand::tilde(s).to_string());

    info!("shell exec: {}", cmd);

    let mut command = Command::new("sh");
    command.arg("-c").arg(cmd);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        command.output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("shell command timed out after {}s", timeout_secs))??;

    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(json!({
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": output.status.code(),
    }))
}

async fn tool_git(args: &Value) -> Result<Value> {
    let git_args: Vec<String> = args["args"]
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("args required"))?
        .iter()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();

    let cwd = args["cwd"].as_str().map(|s| shellexpand::tilde(s).to_string());

    let mut command = Command::new("git");
    command.args(&git_args);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }

    let output = command.output().await?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();

    Ok(json!({
        "stdout": stdout,
        "stderr": stderr,
        "exit_code": output.status.code(),
    }))
}

async fn tool_view_cwd(args: &Value) -> Result<Value> {
    let depth = args["depth"].as_u64().unwrap_or(2) as usize;
    let cwd = std::env::current_dir()?;

    let mut tree = Vec::new();
    for entry in walkdir::WalkDir::new(&cwd).max_depth(depth) {
        let entry = entry?;
        let rel = entry.path().strip_prefix(&cwd)?.to_string_lossy().to_string();
        tree.push(json!({
            "path": rel,
            "is_dir": entry.file_type().is_dir(),
        }));
    }

    Ok(json!({
        "cwd": cwd.to_string_lossy().to_string(),
        "tree": tree,
    }))
}
