// mcp/client.rs — stdio JSON-RPC client for one MCP server.
//
// Spawns a child process (e.g., a Python interpreter running an mcp_server.py),
// sends newline-delimited JSON-RPC 2.0 messages on its stdin, reads responses
// from stdout. Async via tokio. One client per MCP, held in the registry.
//
// Wire protocol: MCP uses JSON-RPC 2.0 over stdio. Each request is one line
// of JSON; each response is one line of JSON. No newlines allowed inside
// any string field (we use compact serialization).

use std::process::Stdio;

use anyhow::{anyhow, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, ChildStdout, Command};

pub struct McpClient {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
    next_id: u64,
    server_name: String,
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

#[derive(Deserialize, Debug, Clone)]
pub struct Tool {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "inputSchema")]
    pub input_schema: Value,
}

impl McpClient {
    /// Spawn a child process and run the MCP `initialize` handshake.
    /// After this returns, the client is ready to call `tools/list` and `tools/call`.
    pub async fn spawn(
        server_name: &str,
        command: &str,
        args: &[&str],
    ) -> Result<Self> {
        let mut child = Command::new(command)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("failed to spawn MCP '{server_name}': {command}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow!("MCP '{server_name}' has no stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("MCP '{server_name}' has no stdout"))?;

        let mut client = Self {
            child,
            stdin,
            stdout: BufReader::new(stdout),
            next_id: 1,
            server_name: server_name.to_string(),
        };

        client.initialize().await.with_context(|| {
            format!("MCP '{server_name}' initialize handshake failed")
        })?;
        Ok(client)
    }

    /// Send `initialize` and read the response. Per JSON-RPC + MCP, this is
    /// a 2-step handshake: send initialize, then send `notifications/initialized`.
    /// We do both.
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
        // `notifications/initialized` is fire-and-forget; per JSON-RPC 2.0 it
        // has no `id` and expects no response. We still write a line so the
        // child has something to read past the buffered input.
        self.send_no_reply("notifications/initialized", &Value::Null)
            .await?;
        Ok(())
    }

    /// List the tools this MCP exposes. Used at startup to populate the
    /// registry and (later) for command palette tab-completion.
    pub async fn list_tools(&mut self) -> Result<Vec<Tool>> {
        let resp = self.send("tools/list", &Value::Null).await?;
        let tools_value = resp
            .get("tools")
            .cloned()
            .ok_or_else(|| anyhow!("tools/list response missing 'tools' key"))?;
        let tools: Vec<Tool> = serde_json::from_value(tools_value)
            .with_context(|| "could not parse tools/list response")?;
        Ok(tools)
    }

    /// Call a tool by name with a JSON-Value arguments object. Returns the
    /// raw `result` from the MCP response.
    pub async fn call_tool(&mut self, name: &str, arguments: Value) -> Result<Value> {
        let params = serde_json::json!({
            "name": name,
            "arguments": arguments,
        });
        let resp = self.send("tools/call", &params).await?;
        Ok(resp)
    }

    /// Send a JSON-RPC request, read one line of response, parse it.
    async fn send(&mut self, method: &str, params: &Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;
        let req = JsonRpcRequest {
            jsonrpc: "2.0",
            id,
            method,
            params,
        };
        let line = serde_json::to_string(&req)
            .with_context(|| format!("serializing request for {method}"))?;
        self.write_line(&line).await?;
        self.read_response(id).await
    }

    /// Send a notification (no `id` field, expects no response). Used for
    /// `notifications/initialized` per the MCP handshake.
    async fn send_no_reply(&mut self, method: &str, params: &Value) -> Result<()> {
        let body = serde_json::json!({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        });
        let line = serde_json::to_string(&body)?;
        self.write_line(&line).await
    }

    async fn write_line(&mut self, line: &str) -> Result<()> {
        self.stdin.write_all(line.as_bytes()).await?;
        self.stdin.write_all(b"\n").await?;
        self.stdin.flush().await?;
        Ok(())
    }

    /// Read one JSON line. Since multiple requests can be in flight on one
    /// stdin, we match by `id` — skip any non-matching lines (the MCP spec
    /// permits server-initiated notifications on stdout, which we don't use
    /// yet but should be ready for).
    async fn read_response(&mut self, expected_id: u64) -> Result<Value> {
        let mut line = String::new();
        loop {
            line.clear();
            let n = self
                .stdout
                .read_line(&mut line)
                .await
                .with_context(|| {
                    format!("MCP '{}' stdout closed", self.server_name)
                })?;
            if n == 0 {
                return Err(anyhow!(
                    "MCP '{}' closed stdout before sending response id={}",
                    self.server_name,
                    expected_id
                ));
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let resp: JsonRpcResponse = serde_json::from_str(trimmed)
                .with_context(|| {
                    format!(
                        "MCP '{}' sent non-JSON line: {}",
                        self.server_name,
                        trimmed.chars().take(80).collect::<String>()
                    )
                })?;
            if resp.id != expected_id {
                // Server-initiated message we don't handle. Skip and keep reading.
                continue;
            }
            if let Some(err) = resp.error {
                return Err(anyhow!(
                    "MCP '{}' tool/method error (code {}): {}",
                    self.server_name,
                    err.code,
                    err.message
                ));
            }
            return resp.result.ok_or_else(|| {
                anyhow!(
                    "MCP '{}' response id={} has neither result nor error",
                    self.server_name,
                    expected_id
                )
            });
        }
    }
}
