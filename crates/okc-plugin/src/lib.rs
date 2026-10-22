//! openklawc-plugin — plugin system stub.
//!
//! Future: load WASM or dynamic lib plugins that extend tools / TUI / LLM providers.

use anyhow::Result;
use async_trait::async_trait;
use okc_core::{ToolCall, ToolExecutor, ToolOutcome, ToolSchema};

pub struct PluginManager;

impl PluginManager {
    pub fn new() -> Self {
        Self
    }

    pub async fn load_plugins(&self, _paths: &[String]) -> Result<()> {
        // Stub for future plugin loading
        Ok(())
    }
}

impl Default for PluginManager {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ToolExecutor for PluginManager {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        Ok(Vec::new())
    }

    async fn execute_tool(&self, _call: &ToolCall) -> Result<ToolOutcome> {
        anyhow::bail!("No plugins loaded")
    }
}
