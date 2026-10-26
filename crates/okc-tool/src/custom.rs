//! Custom tools defined in configuration — no Rust code required.
//!
//! Users can define ad-hoc tools in `config.toml` under `[tools.custom]`:
//!
//! ```toml
//! [[tools.custom]]
//! name = "npm_test"
//! description = "Run the test suite with npm."
//! command = "npm test"
//! cwd = "."               # optional
//! timeout = 60            # optional, seconds
//!
//! [[tools.custom]]
//! name = "cargo_check"
//! description = "Run cargo check and return errors."
//! command = "cargo check --message-format=short"
//! ```

use std::collections::HashMap;

use anyhow::Result;
use async_trait::async_trait;
use okc_core::{ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use serde_json::{json, Value};
use tokio::process::Command;

/// Executor for user-defined custom tools.
pub struct CustomTools {
    tools: HashMap<String, okc_config::CustomToolDef>,
}

impl Default for CustomTools {
    fn default() -> Self {
        Self::new()
    }
}

impl CustomTools {
    pub fn new() -> Self {
        Self {
            tools: HashMap::new(),
        }
    }

    pub fn add(&mut self, tool: okc_config::CustomToolDef) {
        self.tools.insert(tool.name.clone(), tool);
    }

    pub fn from_config(cfg: &okc_config::Config) -> Self {
        let mut exec = Self::new();
        for tool in &cfg.tools.custom {
            exec.add(tool.clone());
        }
        exec
    }
}

#[async_trait]
impl ToolExecutor for CustomTools {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(self
            .tools
            .values()
            .map(|t| ToolSchema {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: json!({
                    "type": "object",
                    "properties": {
                        "args": { "type": "string", "description": "Optional arguments to pass to the command" }
                    }
                }),
            })
            .collect())
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        let def = self
            .tools
            .get(&call.name)
            .ok_or_else(|| anyhow::anyhow!("unknown custom tool: {}", call.name))?;

        let start = std::time::Instant::now();

        // Substitute any arguments into the command string
        let mut command_str = def.command.clone();
        if let Some(obj) = call.arguments.as_object() {
            for (key, value) in obj {
                let placeholder = format!("{{{}}}", key);
                let replacement = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                command_str = command_str.replace(&placeholder, &replacement);
            }
        }

        let mut cmd = Command::new("sh");
        cmd.arg("-c").arg(&command_str);

        let cwd = def
            .cwd
            .as_ref()
            .map(|p| shellexpand::tilde(p).to_string())
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .map(|p| p.to_string_lossy().to_string())
            });

        if let Some(ref dir) = cwd {
            cmd.current_dir(dir);
        }

        let output =
            tokio::time::timeout(std::time::Duration::from_secs(def.timeout), cmd.output())
                .await
                .map_err(|_| {
                    anyhow::anyhow!(
                        "custom tool '{}' timed out after {}s",
                        def.name,
                        def.timeout
                    )
                })??;

        let elapsed = start.elapsed().as_millis() as u64;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let success = output.status.success() || !def.fail_on_error;

        let result = json!({
            "stdout": stdout.trim(),
            "stderr": stderr.trim(),
            "exit_code": output.status.code(),
            "command": command_str,
        });

        Ok(ToolOutcome {
            call_id: call.id.clone(),
            success,
            result,
            error: if success {
                None
            } else {
                Some(stderr.to_string())
            },
            elapsed_ms: elapsed,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_custom_tool_echo() {
        let mut custom = CustomTools::new();
        custom.add(okc_config::CustomToolDef {
            name: "echo".to_string(),
            description: "Echo a message".to_string(),
            command: "echo 'hello world'".to_string(),
            cwd: None,
            timeout: 5,
            fail_on_error: true,
        });

        let schema = custom.list_tools().await.unwrap();
        assert_eq!(schema.len(), 1);
        assert_eq!(schema[0].name, "echo");

        let outcome = custom
            .execute_tool(&okc_core::ToolCall {
                id: okc_core::Id::new(),
                name: "echo".to_string(),
                arguments: json!({}),
            })
            .await
            .unwrap();

        assert!(outcome.success);
        assert!(outcome.result.as_object().unwrap()["stdout"]
            .as_str()
            .unwrap()
            .contains("hello world"));
    }
}
