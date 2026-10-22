use std::process::Stdio;

use anyhow::{anyhow, Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};
use tracing::{debug, error, info, warn};

use okc_core::{ToolCall, ToolExecutor, ToolOutcome, ToolSchema};

/// One MCP client connected over stdio.
pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    server_name: String,
    tools: Vec<ToolSchema>,
}

#[derive(Serialize)]
struct JsonRpcRequest<'a> {
    jsonrpc: &'static str,
    id: u64,
    method: &'a str,
    params: &'a Value,
}

#[derive(Deserialize)]
struct JsonRpcResponse {
    id: u64,
    result: Option<Value>,
    error: Option<JsonRpcError>,
}

#[derive(Deserialize, Debug)]
struct JsonRpcError {
    code: i64,
    message: String,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
pub struct McpTool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpClient {
    pub async fn spawn(
        server_name: &str,
        command: &str,
        args: &[String],
    ) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to spawn MCP '{}'", server_name))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP '{}' has no stdin", server_name))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP '{}' has no stdout", server_name))?;

        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            server_name: server_name.to_string(),
            tools: Vec::new(),
        };

        client.initialize().await.with_context(|| {
            format!("MCP '{}' initialize handshake failed", server_name)
        })?;

        // List tools immediately so we know what's available
        match client.list_tools_raw().await {
            Ok(tools) => {
                info!("MCP '{}' registered {} tools", server_name, tools.len());
                client.tools = tools
                    .into_iter()
                    .map(|t| ToolSchema {
                        name: t.name,
                        description: t.description,
                        parameters: t.input_schema,
                    })
                    .collect();
            }
            Err(e) => {
                warn!("MCP '{}' tools/list failed: {}", server_name, e);
            }
        }

        Ok(client)
    }

    async fn initialize(&mut self) -> Result<()> {
        let params = serde_json::json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "clientInfo": {
                "name": "openklawc",
                "version": env!("CARGO_PKG_VERSION"),
            },
        });
        self.send("initialize", &params).await?;
        self.send_no_reply("notifications/initialized", &Value::Null)
            .await?;
        Ok(())
    }

    async fn list_tools_raw(&mut self) -> Result<Vec<McpTool>> {
        let resp = self.send("tools/list", &Value::Null).await?;
        let tools_value = resp
            .get("tools")
            .cloned()
            .ok_or_else(|| anyhow!("tools/list response missing 'tools' key"))?;
        let tools: Vec<McpTool> = serde_json::from_value(tools_value)
            .with_context(|| "could not parse tools/list response")?;
        Ok(tools)
    }

    pub async fn call_tool_raw(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let resp = self.send("tools/call", &params).await?;
        Ok(resp)
    }

    pub fn tools(&self) -> &[ToolSchema] {
        &self.tools
    }

    async fn send(&mut self, method: &str, params: &Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        let line = serde_json::to_string(&req)? + "\n";
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        debug!("MCP → {} {}", self.server_name, method);

        let mut response_line = String::new();
        self.stdout.read_line(&mut response_line).await?;

        if response_line.trim().is_empty() {
            anyhow::bail!("MCP '{}' returned empty response for {}", self.server_name, method);
        }

        let resp: JsonRpcResponse = serde_json::from_str(&response_line)
            .with_context(|| format!("parsing JSON-RPC response for {}", method))?;

        if let Some(err) = resp.error {
            anyhow::bail!("MCP '{}' error in {}: {} (code {})",
                self.server_name, method, err.message, err.code);
        }

        resp.result.ok_or_else(|| anyhow!("MCP '{}' response missing result", self.server_name))
    }

    async fn send_no_reply(&mut self, method: &str, params: &Value) -> Result<()> {
        let req = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let line = req.to_string() + "\n";
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.flush().await?;
        Ok(())
    }
}

#[async_trait]
impl ToolExecutor for McpClient {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(self.tools.clone())
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        // NOTE: This requires &mut self but the trait uses &self.
        // In practice, the registry wraps this in an Arc<Mutex<>> or similar.
        // For now, we return an error indicating this needs to be called through the registry.
        anyhow::bail!("McpClient::execute_tool requires mutable access — use McpRegistry instead")
    }
}
