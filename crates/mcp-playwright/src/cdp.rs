//! Chrome DevTools Protocol client.
//!
//! A small, dependency-light CDP client: it connects to the browser's
//! WebSocket endpoint, issues commands and buffers the events that arrive in
//! between (console output, page errors).

use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use futures::{SinkExt, StreamExt};
use serde_json::{json, Value};
use tokio::net::TcpStream;
use tokio_tungstenite::tungstenite::Message;
use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

/// One buffered console message.
#[derive(Debug, Clone)]
pub struct ConsoleMessage {
    pub level: String,
    pub text: String,
}

/// A page target exposed by the browser.
#[derive(Debug, Clone)]
pub struct Target {
    pub id: String,
    pub title: String,
    pub url: String,
    pub web_socket_url: String,
}

/// A live CDP session attached to one page target.
pub struct CdpSession {
    socket: WebSocketStream<MaybeTlsStream<TcpStream>>,
    next_id: u64,
    pub console: Vec<ConsoleMessage>,
}

impl CdpSession {
    /// Attach to a target's WebSocket endpoint.
    pub async fn connect(web_socket_url: &str) -> Result<Self> {
        let (socket, _response) = tokio_tungstenite::connect_async(web_socket_url)
            .await
            .with_context(|| format!("connecting to {}", web_socket_url))?;
        Ok(Self {
            socket,
            next_id: 1,
            console: Vec::new(),
        })
    }

    /// Issue a CDP command and wait for its response.
    pub async fn send(&mut self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id;
        self.next_id += 1;

        let request = json!({
            "id": id,
            "method": method,
            "params": params,
        });
        self.socket
            .send(Message::Text(request.to_string().into()))
            .await
            .context("sending CDP command")?;

        loop {
            let message =
                self.socket.next().await.ok_or_else(|| {
                    anyhow!("CDP connection closed while waiting for {}", method)
                })??;

            let text = match message {
                Message::Text(text) => text.to_string(),
                Message::Binary(bytes) => String::from_utf8_lossy(&bytes).to_string(),
                Message::Close(_) => {
                    return Err(anyhow!(
                        "CDP connection closed while waiting for {}",
                        method
                    ))
                }
                _ => continue,
            };

            let Ok(value) = serde_json::from_str::<Value>(&text) else {
                continue;
            };

            if value.get("id").and_then(|v| v.as_u64()) == Some(id) {
                if let Some(error) = value.get("error") {
                    let message = error
                        .get("message")
                        .and_then(|m| m.as_str())
                        .unwrap_or("unknown CDP error");
                    return Err(anyhow!("CDP error in {}: {}", method, message));
                }
                return Ok(value.get("result").cloned().unwrap_or(Value::Null));
            }

            self.record_event(&value);
        }
    }

    /// Buffer the events we care about.
    fn record_event(&mut self, event: &Value) {
        let method = event.get("method").and_then(|m| m.as_str()).unwrap_or("");
        let params = event.get("params").cloned().unwrap_or(Value::Null);

        match method {
            "Runtime.consoleAPICalled" => {
                let level = params
                    .get("type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("log")
                    .to_string();
                let text = params
                    .get("args")
                    .and_then(|v| v.as_array())
                    .map(|args| {
                        args.iter()
                            .map(|arg| {
                                arg.get("value")
                                    .map(|v| v.to_string())
                                    .or_else(|| {
                                        arg.get("description")
                                            .and_then(|v| v.as_str())
                                            .map(String::from)
                                    })
                                    .unwrap_or_default()
                            })
                            .collect::<Vec<_>>()
                            .join(" ")
                    })
                    .unwrap_or_default();
                self.console.push(ConsoleMessage { level, text });
            }
            "Runtime.exceptionThrown" => {
                let text = params
                    .get("exceptionDetails")
                    .and_then(|d| d.get("text"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("exception");
                self.console.push(ConsoleMessage {
                    level: "error".to_string(),
                    text: text.to_string(),
                });
            }
            "Log.entryAdded" => {
                let entry = params.get("entry").cloned().unwrap_or(Value::Null);
                let level = entry
                    .get("level")
                    .and_then(|v| v.as_str())
                    .unwrap_or("info")
                    .to_string();
                let text = entry
                    .get("text")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                self.console.push(ConsoleMessage { level, text });
            }
            _ => {}
        }
    }
}

/// The browser's HTTP endpoint, e.g. `http://127.0.0.1:9222`.
pub fn endpoint() -> String {
    std::env::var("PLAYWRIGHT_CDP_ENDPOINT")
        .unwrap_or_else(|_| "http://127.0.0.1:9222".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// List the page targets the browser exposes.
pub async fn list_targets(http: &reqwest::Client) -> Result<Vec<Target>> {
    let url = format!("{}/json/list", endpoint());
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?
        .error_for_status()
        .with_context(|| format!("listing targets at {}", url))?;

    let items: Vec<Value> = response.json().await.context("parsing target list")?;

    Ok(items
        .iter()
        .filter(|item| {
            item.get("type")
                .and_then(|t| t.as_str())
                .map(|t| t == "page")
                .unwrap_or(false)
        })
        .map(|item| Target {
            id: item
                .get("id")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            title: item
                .get("title")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            url: item
                .get("url")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
            web_socket_url: item
                .get("webSocketDebuggerUrl")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .to_string(),
        })
        .collect())
}

/// Open a new page target and return it.
pub async fn new_target(http: &reqwest::Client, url: &str) -> Result<Target> {
    let endpoint_url = format!("{}/json/new?{}", endpoint(), url);
    let mut response = http
        .put(&endpoint_url)
        .send()
        .await
        .with_context(|| format!("requesting {}", endpoint_url))?;

    if !response.status().is_success() {
        // Older Chrome builds only accept GET here.
        response = http
            .get(&endpoint_url)
            .send()
            .await
            .with_context(|| format!("requesting {}", endpoint_url))?;
    }
    response = response.error_for_status()?;
    let item: Value = response.json().await.context("parsing new target")?;

    Ok(Target {
        id: item
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        title: item
            .get("title")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        url: item
            .get("url")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
        web_socket_url: item
            .get("webSocketDebuggerUrl")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_string(),
    })
}

/// Close a target by id.
pub async fn close_target(http: &reqwest::Client, target_id: &str) -> Result<()> {
    let url = format!("{}/json/close/{}", endpoint(), target_id);
    http.get(&url)
        .send()
        .await
        .with_context(|| format!("closing {}", target_id))?;
    Ok(())
}

/// Wait until the browser answers its version endpoint.
pub async fn wait_for_browser(http: &reqwest::Client, timeout: Duration) -> Result<()> {
    let deadline = std::time::Instant::now() + timeout;
    let url = format!("{}/json/version", endpoint());
    loop {
        if let Ok(response) = http.get(&url).send().await {
            if response.status().is_success() {
                return Ok(());
            }
        }
        if std::time::Instant::now() >= deadline {
            anyhow::bail!("no browser answered at {} within {:?}", endpoint(), timeout);
        }
        tokio::time::sleep(Duration::from_millis(200)).await;
    }
}

/// Find a Chromium binary to launch, if one is installed.
pub fn find_chromium() -> Option<std::path::PathBuf> {
    if let Ok(explicit) = std::env::var("PLAYWRIGHT_CHROMIUM") {
        let path = std::path::PathBuf::from(explicit);
        if path.is_file() {
            return Some(path);
        }
    }
    let cache = dirs::cache_dir()?.join("ms-playwright");
    let mut candidates: Vec<std::path::PathBuf> = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&cache) {
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.starts_with("chromium") {
                candidates.push(entry.path().join("chrome-linux64").join("chrome"));
                candidates.push(entry.path().join("chrome-linux").join("chrome"));
                candidates.push(
                    entry
                        .path()
                        .join("chrome-headless-shell-linux64")
                        .join("chrome-headless-shell"),
                );
            }
        }
    }
    candidates.into_iter().find(|path| path.is_file())
}

/// Launch a Chromium with a remote debugging port.
pub fn launch_chromium(binary: &std::path::Path, port: u16) -> Result<tokio::process::Child> {
    let mut command = tokio::process::Command::new(binary);
    command
        .arg(format!("--remote-debugging-port={}", port))
        .arg("--user-data-dir=/tmp/opencode/openklawc-chrome-profile")
        .arg("--no-first-run")
        .arg("--no-default-browser-check")
        .arg("--disable-gpu")
        .arg("--headless=new")
        .arg("about:blank")
        .kill_on_drop(true)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null());
    command
        .spawn()
        .with_context(|| format!("launching {}", binary.display()))
}
