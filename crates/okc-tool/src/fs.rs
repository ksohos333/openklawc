//! Filesystem helpers for built-in tools.

use std::path::Path;

/// Check if a path is inside the current working directory or explicitly allowed.
pub fn is_path_allowed(path: &Path) -> bool {
    if let Ok(cwd) = std::env::current_dir() {
        if path.starts_with(&cwd) {
            return true;
        }
    }
    // Allow home dir and known safe paths
    if let Some(home) = dirs::home_dir() {
        if path.starts_with(&home) {
            return true;
        }
    }
    false
}
