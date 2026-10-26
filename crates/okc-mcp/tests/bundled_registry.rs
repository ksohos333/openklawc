//! End-to-end checks for the bundled MCP registry.
//!
//! The test starts the bundled filesystem server through the registry, the
//! same way openklawc does, and verifies that the configured arguments reach
//! the server. It is skipped when the server binary has not been built.

use std::collections::HashMap;
use std::path::PathBuf;

use okc_config::McpServerConfig;
use okc_core::{ToolCall, ToolExecutor};
use serde_json::json;

/// Find a bundled binary next to the test executable or in the target dir.
fn find_binary(name: &str) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            candidates.push(parent.join(name));
            if let Some(grandparent) = parent.parent() {
                candidates.push(grandparent.join(name));
            }
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("target/debug").join(name));
        candidates.push(cwd.join("target/release").join(name));
    }
    candidates.into_iter().find(|path| path.is_file())
}

#[tokio::test]
async fn registry_starts_a_configured_server_and_passes_its_arguments() {
    let Some(binary) = find_binary("mcp-filesystem") else {
        eprintln!("mcp-filesystem is not built — skipping");
        return;
    };

    let allowed = std::env::temp_dir().join("openklawc-registry-test");
    let _ = std::fs::create_dir_all(&allowed);

    let mut configs = HashMap::new();
    configs.insert(
        "filesystem".to_string(),
        McpServerConfig {
            command: binary.to_string_lossy().to_string(),
            args: vec![allowed.to_string_lossy().to_string()],
            env: HashMap::new(),
            sse_url: None,
            enabled: true,
        },
    );

    let registry = okc_mcp::McpRegistry::start(&configs)
        .await
        .expect("registry starts");
    assert!(
        registry
            .all_tools()
            .iter()
            .any(|(server, _)| server == "filesystem"),
        "the filesystem server registered no tools"
    );

    let call = ToolCall {
        id: okc_core::Id("test-1".to_string()),
        name: "list_allowed_directories".to_string(),
        arguments: json!({}),
    };
    let outcome = registry
        .execute_tool(&call)
        .await
        .expect("tool call succeeds");
    assert!(outcome.success, "tool call failed: {:?}", outcome.error);

    let text = outcome.result["content"][0]["text"]
        .as_str()
        .unwrap_or_default()
        .to_string();
    assert!(
        text.contains(&allowed.to_string_lossy().to_string()),
        "the configured argument never reached the server: {}",
        text
    );

    let _ = std::fs::remove_dir_all(&allowed);
}

#[tokio::test]
async fn registry_skips_a_disabled_server() {
    let mut configs = HashMap::new();
    configs.insert("filesystem".to_string(), McpServerConfig::disabled());

    let registry = okc_mcp::McpRegistry::start(&configs)
        .await
        .expect("registry starts");
    assert!(
        registry.all_tools().is_empty(),
        "a disabled server was registered anyway"
    );
}
