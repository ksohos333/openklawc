use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Context, Result};
use okc_config::McpServerConfig;
use okc_core::{Id, ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use tokio::sync::Mutex;
use tracing::{info, warn};

use crate::client::McpClient;

/// Registry that holds multiple MCP clients and multiplexes tool calls.
pub struct Registry {
    clients: HashMap<String, Arc<Mutex<McpClient>>>,
    all_tools: Vec<(String, ToolSchema)>, // (server_name, tool)
}

impl Registry {
    pub async fn start(mcp_configs: &HashMap<String, McpServerConfig>) -> Result<Self> {
        let mut clients = HashMap::new();
        let mut all_tools = Vec::new();

        for (name, cfg) in mcp_configs {
            let args: Vec<String> = cfg.args.clone();
            match McpClient::spawn(name, &cfg.command, &args).await {
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

        match guard.call_tool_raw(&call.name, call.arguments.clone()).await {
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
