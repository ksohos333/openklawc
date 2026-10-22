//! openklawc-git — git operations.

use std::path::Path;

use anyhow::{Context, Result};
use tokio::process::Command;

pub struct GitRepo {
    pub root: std::path::PathBuf,
}

impl GitRepo {
    pub async fn discover(cwd: &Path) -> Result<Option<Self>> {
        let output = Command::new("git")
            .args(["rev-parse", "--show-toplevel"])
            .current_dir(cwd)
            .output()
            .await?;
        if output.status.success() {
            let root = String::from_utf8_lossy(&output.stdout).trim().to_string();
            Ok(Some(Self {
                root: root.into(),
            }))
        } else {
            Ok(None)
        }
    }

    pub async fn status(&self) -> Result<String> {
        let output = Command::new("git")
            .args(["status", "--short"])
            .current_dir(&self.root)
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub async fn diff(&self) -> Result<String> {
        let output = Command::new("git")
            .args(["diff"])
            .current_dir(&self.root)
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }

    pub async fn log(&self, n: usize) -> Result<String> {
        let output = Command::new("git")
            .args(["log", &format!("-{}", n), "--oneline"])
            .current_dir(&self.root)
            .output()
            .await?;
        Ok(String::from_utf8_lossy(&output.stdout).to_string())
    }
}
