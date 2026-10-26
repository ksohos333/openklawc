//! Bundled MCP servers.
//!
//! openklawc ships Rust re-implementations of the MCP servers that opencode
//! exposes out of the box. They are launched automatically when the user
//! configuration does not mention them; a `[mcp.<name>]` entry in the config
//! file always wins and may disable the server entirely.
//!
//! Lookup order for a bundled binary:
//!   1. `$OPENCLAWC_MCP_BIN_DIR`
//!   2. next to the running `openklawc` executable
//!   3. `../bin`, `../libexec` relative to the executable
//!   4. `<cwd>/target/release`, `<cwd>/target/debug`
//!   5. `$PATH`

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use okc_config::McpServerConfig;
use tracing::{debug, info, warn};

/// One MCP server that ships with openklawc.
#[derive(Debug, Clone, Copy)]
pub struct BundledSpec {
    /// Registry name — matches the `[mcp.<name>]` config table.
    pub name: &'static str,
    /// Executable name.
    pub binary: &'static str,
    /// Short description used in diagnostics.
    pub description: &'static str,
    /// True when the server needs a secret or an external service and will
    /// fail its calls until one is provided.
    pub needs_secret: bool,
}

/// The bundled catalogue.
pub const BUNDLED_SERVERS: &[BundledSpec] = &[
    BundledSpec {
        name: "filesystem",
        binary: "mcp-filesystem",
        description: "sandboxed file read/write/edit/list/search",
        needs_secret: false,
    },
    BundledSpec {
        name: "brave-search",
        binary: "mcp-brave-search",
        description: "Brave Search web + local search",
        needs_secret: true,
    },
    BundledSpec {
        name: "playwright",
        binary: "mcp-playwright",
        description: "browser automation over the Chrome DevTools Protocol",
        needs_secret: false,
    },
    BundledSpec {
        name: "vision",
        binary: "mcp-vision",
        description: "image understanding and audio transcription",
        needs_secret: false,
    },
    BundledSpec {
        name: "local-brain",
        binary: "mcp-local-brain",
        description: "RAG memory over the local-brain HTTP service",
        needs_secret: false,
    },
    BundledSpec {
        name: "laya",
        binary: "mcp-laya",
        description: "Laya decision oracle (route / decide / status)",
        needs_secret: false,
    },
];

/// Look up a bundled spec by registry name.
pub fn spec(name: &str) -> Option<&'static BundledSpec> {
    BUNDLED_SERVERS.iter().find(|s| s.name == name)
}

/// Locate a bundled binary on disk.
pub fn resolve_binary(binary: &str) -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("OPENCLAWC_MCP_BIN_DIR") {
        let candidate = Path::new(&dir).join(binary);
        if candidate.is_file() {
            return Some(candidate);
        }
    }

    if let Ok(exe) = std::env::current_exe() {
        if let Some(parent) = exe.parent() {
            let relatives = [
                parent.join(binary),
                parent.join("../bin").join(binary),
                parent.join("../libexec").join(binary),
            ];
            for candidate in relatives {
                if candidate.is_file() {
                    return Some(normalize(candidate));
                }
            }
        }
    }

    if let Ok(cwd) = std::env::current_dir() {
        for sub in ["target/release", "target/debug"] {
            let candidate = cwd.join(sub).join(binary);
            if candidate.is_file() {
                return Some(normalize(candidate));
            }
        }
    }

    if let Some(paths) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&paths) {
            let candidate = dir.join(binary);
            if candidate.is_file() {
                return Some(candidate);
            }
        }
    }

    None
}

/// Canonicalise when possible so the spawned process keeps a stable path.
fn normalize(path: PathBuf) -> PathBuf {
    path.canonicalize().unwrap_or(path)
}

/// Merge the user configuration with the bundled catalogue.
///
/// * a user entry for a bundled name replaces the bundled entry entirely
///   (`command` set → full override, `command` empty → bundled binary with
///   the user's args/env),
/// * `enabled = false` removes the server from the registry,
/// * bundled servers without a user entry are added automatically.
pub fn resolve_configs(
    user: &HashMap<String, McpServerConfig>,
) -> HashMap<String, McpServerConfig> {
    let mut merged = user.clone();

    for spec in BUNDLED_SERVERS {
        match merged.get(spec.name) {
            Some(entry) if !entry.enabled => {
                info!("MCP '{}' disabled by configuration", spec.name);
                merged.remove(spec.name);
                continue;
            }
            Some(entry) if !entry.command.trim().is_empty() => {
                debug!(
                    "MCP '{}' overridden by configuration (command={})",
                    spec.name, entry.command
                );
                continue;
            }
            Some(_) => {
                // User supplied args/env but no command — use the bundled
                // binary with their overrides.
                match resolve_binary(spec.binary) {
                    Some(path) => {
                        let entry = merged.get_mut(spec.name).expect("entry present");
                        entry.command = path.to_string_lossy().to_string();
                    }
                    None => {
                        warn!(
                            "bundled MCP '{}' binary '{}' not found — build with `cargo build --release`",
                            spec.name, spec.binary
                        );
                        merged.remove(spec.name);
                    }
                }
            }
            None => match resolve_binary(spec.binary) {
                Some(path) => {
                    let entry = McpServerConfig {
                        command: path.to_string_lossy().to_string(),
                        args: Vec::new(),
                        env: HashMap::new(),
                        sse_url: None,
                        enabled: true,
                    };
                    merged.insert(spec.name.to_string(), entry);
                }
                None => {
                    warn!(
                        "bundled MCP '{}' binary '{}' not found — skipping (build with `cargo build --release`)",
                        spec.name, spec.binary
                    );
                }
            },
        }
    }

    merged
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_server_is_removed() {
        let mut user = HashMap::new();
        user.insert("filesystem".to_string(), McpServerConfig::disabled());
        let merged = resolve_configs(&user);
        assert!(!merged.contains_key("filesystem"));
    }

    #[test]
    fn explicit_command_is_kept() {
        let mut user = HashMap::new();
        user.insert(
            "filesystem".to_string(),
            McpServerConfig {
                command: "/usr/bin/my-filesystem".to_string(),
                args: vec!["/tmp".to_string()],
                env: HashMap::new(),
                sse_url: None,
                enabled: true,
            },
        );
        let merged = resolve_configs(&user);
        assert_eq!(
            merged.get("filesystem").map(|c| c.command.as_str()),
            Some("/usr/bin/my-filesystem")
        );
    }
}
