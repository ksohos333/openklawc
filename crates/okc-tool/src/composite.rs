//! Composite tool executor that combines built-in + MCP tools.

use std::sync::Arc;

use anyhow::{Context, Result};
use async_trait::async_trait;
use okc_core::{ToolCall, ToolExecutor, ToolOutcome, ToolSchema};
use tracing::{debug, warn};

/// Tries each inner executor in order until one can handle the tool call.
pub struct CompositeExecutor {
    executors: Vec<Arc<dyn ToolExecutor>>,
}

impl CompositeExecutor {
    pub fn new() -> Self {
        Self {
            executors: Vec::new(),
        }
    }

    pub fn add(&mut self, executor: Arc<dyn ToolExecutor>) {
        self.executors.push(executor);
    }
}

#[async_trait]
impl ToolExecutor for CompositeExecutor {
    async fn list_tools(&self) -> Result<Vec<ToolSchema>> {
        let mut all = Vec::new();
        for exe in &self.executors {
            match exe.list_tools().await {
                Ok(tools) => {
                    for tool in tools {
                        // avoid duplicates by name
                        if !all.iter().any(|t: &ToolSchema| t.name == tool.name) {
                            all.push(tool);
                        }
                    }
                }
                Err(e) => {
                    warn!("Failed to list tools from one executor: {}", e);
                }
            }
        }
        Ok(all)
    }

    async fn execute_tool(&self, call: &ToolCall) -> Result<ToolOutcome> {
        // First, find which executor has this tool in its catalog
        let mut found_in: Option<usize> = None;
        for (i, exe) in self.executors.iter().enumerate() {
            if let Ok(tools) = exe.list_tools().await {
                if tools.iter().any(|t| t.name == call.name) {
                    found_in = Some(i);
                    break;
                }
            }
        }

        if let Some(idx) = found_in {
            debug!("Routing tool '{}' to executor {}", call.name, idx);
            self.executors[idx].execute_tool(call).await
        } else {
            // Try each executor anyway (some might not report all tools, e.g. MCP)
            for (i, exe) in self.executors.iter().enumerate() {
                match exe.execute_tool(call).await {
                    Ok(outcome) => return Ok(outcome),
                    Err(e) => {
                        debug!("Executor {} failed for '{}': {}", i, call.name, e);
                    }
                }
            }
            anyhow::bail!("No executor could handle tool '{}'", call.name)
        }
    }
}

impl Default for CompositeExecutor {
    fn default() -> Self {
        Self::new()
    }
}
