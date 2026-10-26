use std::collections::HashMap;
use std::sync::Arc;

use anyhow::Result;
use async_trait::async_trait;
use okc_config::McpServerConfig;
use okc_core::{ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::client::McpClient;

/// Registry that holds multiple MCP clients and multiplexes tool calls.
pub struct Registry {
    clients: HashMap<String, Arc<Mutex<McpClient>>>,
    all_tools: Vec<(String, ToolSchema)>, // (server_name, tool)
}

impl Registry {
    /// Start every configured MCP server.
    ///
    /// Bundled servers (see [`crate::bundled`]) that the configuration does
    /// not mention are launched automatically; a configuration entry always
    /// wins and may disable a bundled server.
    pub async fn start(mcp_configs: &HashMap<String, McpServerConfig>) -> Result<Self> {
        let resolved = crate::bundled::resolve_configs(mcp_configs);
        let mut clients = HashMap::new();
        let mut all_tools = Vec::new();

        for (name, cfg) in &resolved {
            if !cfg.enabled {
                continue;
            }
            if cfg.command.trim().is_empty() {
                warn!("MCP '{}' has no command configured — skipping", name);
                continue;
            }
            let args: Vec<String> = cfg.args.clone();
            let env: Vec<(String, String)> = cfg
                .env
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            match McpClient::spawn_with_env(name, &cfg.command, &args, &env).await {
                Ok(client) => {
                    let tools = client.tools().to_vec();
                    for tool in &tools {
                        all_tools.push((name.to_string(), tool.clone()));
                    }
                    clients.insert(name.to_string(), Arc::new(Mutex::new(client)));
                    info!("MCP '{}' connected with {} tools", name, tools.len());
                }
                Err(e) => {
                    warn!("Failed to start MCP '{}': {}", name, e);
                }
            }
        }

        Ok(Self { clients, all_tools })
    }

    pub fn all_tools(&self) -> &[(String, ToolSchema)] {
        &self.all_tools
    }

    pub fn find_server_for_tool(&self, tool_name: &str) -> Option<&str> {
        self.all_tools
            .iter()
            .find(|(_, t)| t.name == tool_name)
            .map(|(server, _)| server.as_str())
    }

    pub async fn call_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        let server_name = self
            .find_server_for_tool(&call.name)
            .ok_or_else(|| anyhow::anyhow!("No MCP server provides tool '{}'", call.name))?;

        let client = self
            .clients
            .get(server_name)
            .ok_or_else(|| anyhow::anyhow!("MCP client '{}' not found", server_name))?;

        let mut guard = client.lock().await;
        let start = std::time::Instant::now();

        match guard
            .call_tool_raw(&call.name, call.arguments.clone())
            .await
        {
            Ok(result) => {
                let elapsed = start.elapsed().as_millis() as u64;
                Ok(ToolOutcome {
                    call_id: call.id.clone(),
                    success: true,
                    result,
                    error: None,
                    elapsed_ms: elapsed,
                })
            }
            Err(e) => {
                let elapsed = start.elapsed().as_millis() as u64;
                Ok(ToolOutcome {
                    call_id: call.id.clone(),
                    success: false,
                    result: serde_json::Value::Null,
                    error: Some(e.to_string()),
                    elapsed_ms: elapsed,
                })
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        self.clients.is_empty()
    }
}

#[async_trait]
impl ToolExecutor for Registry {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(self.all_tools.iter().map(|(_, t)| t.clone()).collect())
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        self.call_tool(call).await
    }
}
