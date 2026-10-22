//! openklawc-mcp — Model Context Protocol client (stdio + SSE).

pub mod client;
pub mod registry;
pub mod server;

pub use client::McpClient;
pub use registry::Registry as McpRegistry;
pub use server::McpServer;
