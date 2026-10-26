//! Shell execution helpers.

use std::process::Stdio;

use tokio::process::Command;

/// Run a command with a timeout, returning structured output.
pub async fn run(
    cmd: &str,
    cwd: Option<&str>,
    timeout_secs: u64,
) -> anyhow::Result<okc_core::ShellResult> {
    let mut command = Command::new("sh");
    command.arg("-c").arg(cmd);
    if let Some(cwd) = cwd {
        command.current_dir(cwd);
    }
    command.stdout(Stdio::piped()).stderr(Stdio::piped());

    let output = tokio::time::timeout(
        std::time::Duration::from_secs(timeout_secs),
        command.output(),
    )
    .await
    .map_err(|_| anyhow::anyhow!("timeout after {}s", timeout_secs))??;

    Ok(okc_core::ShellResult {
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
        exit_code: output.status.code(),
    })
}
