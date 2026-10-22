// mcp/registry.rs — manage one client per configured MCP server.
//
// Reads the `[mcp.*]` sections of `~/.config/openklawc/config.toml`, spawns
// each as a child process, runs the JSON-RPC initialize handshake, and holds
// the resulting clients in a `HashMap<name, McpClient>`.
//
// For v0.1, callers grab a client by name and call its tools. The registry
// itself doesn't auto-route — that's done in the chat handler (main.rs).

use std::collections::HashMap;

use anyhow::{Context, Result};
use serde::Deserialize;
use serde_json::Value;

use super::client::{McpClient, Tool};

#[derive(Debug, Clone, Deserialize, Default)]
pub struct McpConfig {
    #[serde(default)]
    pub mcp: HashMap<String, McpEntry>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct McpEntry {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
}

pub struct Registry {
    clients: HashMap<String, McpClient>,
    tools: HashMap<String, Vec<Tool>>,
}

impl Registry {
    /// Spawn each configured MCP from an already-parsed `McpConfig`,
    /// run `initialize`, and cache the resulting `McpClient` per server.
    /// Returns the registry ready for `tools/call`.
    pub async fn start(config: &McpConfig) -> Result<Self> {
        let mut clients: HashMap<String, McpClient> = HashMap::new();
        let mut tools: HashMap<String, Vec<Tool>> = HashMap::new();

        for (name, entry) in &config.mcp {
            let arg_refs: Vec<&str> = entry.args.iter().map(String::as_str).collect();
            let mut client = McpClient::spawn(name, &entry.command, &arg_refs)
                .await
                .with_context(|| format!("starting MCP '{name}'"))?;
            let ts = client
                .list_tools()
                .await
                .with_context(|| format!("listing tools for MCP '{name}'"))?;
            tools.insert(name.clone(), ts);
            clients.insert(name.clone(), client);
        }

        Ok(Self { clients, tools })
    }

    /// Look up a client by config name (e.g., "web_research", "anticonf").
    pub fn get_mut(&mut self, name: &str) -> Option<&mut McpClient> {
        self.clients.get_mut(name)
    }

    /// List the tool names known to a particular MCP. Useful for the
    /// command palette / status bar.
    pub fn tools_for(&self, name: &str) -> &[Tool] {
        self.tools.get(name).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// All configured MCP names.
    pub fn server_names(&self) -> Vec<String> {
        self.clients.keys().cloned().collect()
    }
}
