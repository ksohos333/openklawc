//! MCP server framework — run a stdio JSON-RPC MCP server.
//!
//! Usage:
//! ```rust,no_run
//! use okc_mcp::server::McpServer;
//! use serde_json::json;
//!
//! # async fn example() -> anyhow::Result<()> {
//! let mut server = McpServer::new("my_server");
//! let schema = json!({"type": "object", "properties": {}});
//! server.add_tool(
//!     "greet",
//!     "Say hello",
//!     schema,
//!     Box::new(|args| Box::pin(async move { Ok(json!("hello")) })),
//! );
//! server.run().await?;
//! # Ok(())
//! # }
//! ```

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use serde::Deserialize;
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::client::McpTool;

pub type ToolHandler = Box<
    dyn Fn(Value) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Value>> + Send>>
        + Send
        + Sync,
>;

pub struct McpServer {
    name: String,
    version: String,
    tools: Vec<McpTool>,
    handlers: HashMap<String, ToolHandler>,
}

impl McpServer {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            tools: Vec::new(),
            handlers: HashMap::new(),
        }
    }

    pub fn add_tool(
        &mut self,
        name: impl Into<String>,
        description: impl Into<String>,
        input_schema: Value,
        handler: ToolHandler,
    ) {
        let name = name.into();
        self.tools.push(McpTool {
            name: name.clone(),
            description: description.into(),
            input_schema,
        });
        self.handlers.insert(name, handler);
    }

    pub async fn run(&self) -> Result<()> {
        let stdin = tokio::io::stdin();
        let mut stdout = tokio::io::stdout();
        let mut reader = BufReader::new(stdin);
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader.read_line(&mut line).await?;
            if n == 0 {
                break; // EOF
            }
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            let req: JsonRpcRequest = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(e) => {
                    let resp = serde_json::json!({
                        "jsonrpc": "2.0",
                        "id": null,
                        "error": { "code": -32700, "message": format!("parse error: {}", e) }
                    });
                    Self::write_line(&mut stdout, &resp.to_string()).await?;
                    continue;
                }
            };

            let response = self.handle_request(req).await;
            if let Some(resp) = response {
                Self::write_line(&mut stdout, &resp).await?;
            }
        }

        Ok(())
    }

    async fn handle_request(&self, req: JsonRpcRequest) -> Option<String> {
        // Notifications (no id) don't get responses
        let is_notification = req.id.is_none();

        let result: Result<Value> = match req.method.as_str() {
            "initialize" => Ok(serde_json::json!({
                "protocolVersion": "2024-11-05",
                "serverInfo": { "name": self.name, "version": self.version },
                "capabilities": { "tools": {} }
            })),
            "notifications/initialized" => {
                return None; // No response for notifications
            }
            "tools/list" => Ok(serde_json::json!({ "tools": self.tools })),
            "tools/call" => self.handle_tool_call(req.params).await,
            _ => Err(anyhow!("unknown method: {}", req.method)),
        };

        if is_notification {
            return None;
        }

        let response = match result {
            Ok(value) => serde_json::json!({
                "jsonrpc": "2.0",
                "id": req.id,
                "result": value,
            }),
            Err(e) => serde_json::json!({
                "jsonrpc": "2.0",
                "id": req.id,
                "error": { "code": -32603, "message": e.to_string() },
            }),
        };

        Some(response.to_string())
    }

    async fn handle_tool_call(&self, params: Option<Value>) -> Result<Value> {
        let params = params.ok_or_else(|| anyhow!("missing params"))?;
        let name = params
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow!("missing tool name"))?;
        let arguments = params.get("arguments").cloned().unwrap_or(Value::Null);

        let handler = self
            .handlers
            .get(name)
            .ok_or_else(|| anyhow!("unknown tool: {}", name))?;

        let result = handler(arguments).await?;

        // A handler may return:
        //   * a string            → one text content block,
        //   * an MCP-shaped object → passed through unchanged (e.g. image or
        //     audio blocks),
        //   * anything else       → serialised into one text content block.
        let value = if let Some(text) = result.as_str() {
            serde_json::json!({ "content": [{ "type": "text", "text": text }] })
        } else if result.get("content").map(|c| c.is_array()).unwrap_or(false) {
            result
        } else {
            serde_json::json!({ "content": [{ "type": "text", "text": result.to_string() }] })
        };

        Ok(value)
    }

    async fn write_line(stdout: &mut tokio::io::Stdout, text: &str) -> Result<()> {
        stdout.write_all(text.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    /// Protocol version tag; required by the spec but not acted on.
    #[allow(dead_code)]
    jsonrpc: String,
    id: Option<serde_json::Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}
