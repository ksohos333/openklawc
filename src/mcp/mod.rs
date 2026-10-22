// mcp/mod.rs — Model Context Protocol client. Spawns child processes that speak
// JSON-RPC over stdio and exposes their tools to the rest of openklawc.

pub mod client;
pub mod registry;

pub use client::{McpClient, Tool};
pub use registry::{McpConfig, McpEntry, Registry};
