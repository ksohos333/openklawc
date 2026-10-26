//! mcp-playwright — Rust MCP server driving a browser over the Chrome
//! DevTools Protocol.
//!
//! A Rust port of the `@playwright/mcp` server, scoped to the tools that make
//! sense over raw CDP. The server attaches to a browser that is already
//! listening on `PLAYWRIGHT_CDP_ENDPOINT` (default http://127.0.0.1:9222),
//! or launches a local Chromium when one is installed and nothing answers
//! there.
//!
//! Tools:
//!   browser_navigate(url)
//!   browser_snapshot()
//!   browser_click(element)
//!   browser_type(text, element)
//!   browser_fill_form(fields[])
//!   browser_select_option(element, value)
//!   browser_press_key(key)
//!   browser_evaluate(function)
//!   browser_find(text)
//!   browser_wait_for(text?, time?)
//!   browser_take_screenshot(filename?, fullPage?, element?)
//!   browser_console_messages(level?)
//!   browser_tabs(action, index?, url?)
//!   browser_get_config()
//!   browser_close()

mod cdp;
mod scripts;

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use serde_json::{json, Value};
use tokio::sync::Mutex;

use cdp::{CdpSession, Target};

/// Everything the tools share.
struct Browser {
    http: reqwest::Client,
    session: Mutex<Option<CdpSession>>,
    target: Mutex<Option<Target>>,
    #[allow(dead_code)]
    launched: Mutex<Option<tokio::process::Child>>,
}

impl Browser {
    fn new() -> Result<Self> {
        Ok(Self {
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(30))
                .build()?,
            session: Mutex::new(None),
            target: Mutex::new(None),
            launched: Mutex::new(None),
        })
    }

    /// Make sure a browser is running and a page is attached.
    async fn ensure_browser(&self) -> Result<()> {
        if self.session.lock().await.is_some() {
            return Ok(());
        }

        if cdp::wait_for_browser(&self.http, Duration::from_millis(600))
            .await
            .is_err()
        {
            self.launch_browser().await?;
            cdp::wait_for_browser(&self.http, Duration::from_secs(20)).await?;
        }

        let targets = cdp::list_targets(&self.http).await?;
        let target = targets
            .into_iter()
            .next()
            .ok_or_else(|| anyhow!("the browser exposes no page target"))?;
        let mut session = CdpSession::connect(&target.web_socket_url).await?;
        session.send("Page.enable", json!({})).await.ok();
        session.send("Runtime.enable", json!({})).await.ok();
        session.send("Log.enable", json!({})).await.ok();

        *self.target.lock().await = Some(target);
        *self.session.lock().await = Some(session);
        Ok(())
    }

    /// Launch a local Chromium when nothing answers on the CDP endpoint.
    async fn launch_browser(&self) -> Result<()> {
        let Some(binary) = cdp::find_chromium() else {
            anyhow::bail!(
                "no browser at {} and no Chromium binary found — start Chrome with \
                 --remote-debugging-port or set PLAYWRIGHT_CHROMIUM",
                cdp::endpoint()
            );
        };
        let port = cdp::endpoint()
            .rsplit(':')
            .next()
            .and_then(|p| p.trim_end_matches('/').parse::<u16>().ok())
            .unwrap_or(9222);
        tracing::info!("launching {}", binary.display());
        let child = cdp::launch_chromium(&binary, port)?;
        *self.launched.lock().await = Some(child);
        Ok(())
    }

    /// Evaluate an expression in the page and return its value.
    async fn evaluate(&self, expression: &str) -> Result<Value> {
        let mut guard = self.session.lock().await;
        if guard.is_none() {
            drop(guard);
            self.ensure_browser().await?;
            guard = self.session.lock().await;
        }
        let session = guard
            .as_mut()
            .ok_or_else(|| anyhow!("no browser session attached"))?;

        let result = session
            .send(
                "Runtime.evaluate",
                json!({
                    "expression": expression,
                    "returnByValue": true,
                    "awaitPromise": true,
                    "userGesture": true,
                }),
            )
            .await?;

        if let Some(exception) = result.get("exceptionDetails") {
            let message = exception
                .get("text")
                .and_then(|t| t.as_str())
                .unwrap_or("evaluation failed");
            anyhow::bail!("page error: {}", message);
        }

        Ok(result
            .get("result")
            .and_then(|r| r.get("value"))
            .cloned()
            .unwrap_or(Value::Null))
    }

    /// Attach to a specific target.
    async fn switch_to(&self, target: Target) -> Result<()> {
        let mut session = CdpSession::connect(&target.web_socket_url).await?;
        session.send("Page.enable", json!({})).await.ok();
        session.send("Runtime.enable", json!({})).await.ok();
        session.send("Log.enable", json!({})).await.ok();
        *self.target.lock().await = Some(target);
        *self.session.lock().await = Some(session);
        Ok(())
    }

    /// The target the session is attached to.
    async fn current_target(&self) -> Option<Target> {
        self.target.lock().await.clone()
    }
}

/// Read a required string argument.
fn need_str(args: &Value, key: &str) -> Result<String> {
    args[key]
        .as_str()
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("{} required", key))
}

/// Normalise a URL: add the scheme when it is missing.
fn normalize_url(url: &str) -> String {
    let trimmed = url.trim();
    if trimmed.starts_with("http://")
        || trimmed.starts_with("https://")
        || trimmed.starts_with("about:")
    {
        return trimmed.to_string();
    }
    if trimmed.starts_with("file://") || trimmed.starts_with("data:") {
        return trimmed.to_string();
    }
    format!("https://{}", trimmed)
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    let browser = Arc::new(Browser::new()?);
    let mut server = okc_mcp::McpServer::new("playwright");

    // ─── browser_navigate ─────────────────────────────────────────────────
    let nav = browser.clone();
    server.add_tool(
        "browser_navigate",
        "Navigate the current tab to a URL and wait for the page to load.",
        json!({
            "type": "object",
            "properties": { "url": { "type": "string" } },
            "required": ["url"]
        }),
        Box::new(move |args| {
            let browser = nav.clone();
            Box::pin(async move {
                let url = need_str(&args, "url")?;
                browser.ensure_browser().await?;

                let mut guard = browser.session.lock().await;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("no browser session attached"))?;
                session
                    .send("Page.navigate", json!({ "url": normalize_url(&url) }))
                    .await?;
                drop(guard);

                // Give the page a moment, then report where we ended up.
                tokio::time::sleep(Duration::from_millis(600)).await;
                let info = browser
                    .evaluate("({ title: document.title, url: location.href })")
                    .await?;
                Ok(json!({
                    "url": info.get("url").cloned().unwrap_or(Value::Null),
                    "title": info.get("title").cloned().unwrap_or(Value::Null),
                }))
            })
        }),
    );

    // ─── browser_snapshot ─────────────────────────────────────────────────
    let snap = browser.clone();
    server.add_tool(
        "browser_snapshot",
        "Capture a text snapshot of the page. Interactive elements get a ref like [e3] that \
         browser_click and browser_type accept.",
        json!({ "type": "object", "properties": {} }),
        Box::new(move |_args| {
            let browser = snap.clone();
            Box::pin(async move {
                browser.ensure_browser().await?;
                let info = browser.evaluate(&scripts::snapshot()).await?;
                Ok(json!({
                    "title": info.get("title").cloned().unwrap_or(Value::Null),
                    "url": info.get("url").cloned().unwrap_or(Value::Null),
                    "outline": info.get("outline").cloned().unwrap_or(Value::Null),
                }))
            })
        }),
    );

    // ─── browser_click ────────────────────────────────────────────────────
    let click = browser.clone();
    server.add_tool(
        "browser_click",
        "Click an element. The element may be a snapshot ref (e3), a CSS selector, or visible text.",
        json!({
            "type": "object",
            "properties": { "element": { "type": "string" } },
            "required": ["element"]
        }),
        Box::new(move |args| {
            let browser = click.clone();
            Box::pin(async move {
                let element = need_str(&args, "element")?;
                browser.ensure_browser().await?;
                let described = browser.evaluate(&scripts::describe(&element)).await?;

                if described.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                    anyhow::bail!(
                        "{}",
                        described
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("element not found")
                    );
                }

                let x = described.get("x").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let y = described.get("y").and_then(|v| v.as_f64()).unwrap_or(0.0);

                let mut guard = browser.session.lock().await;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("no browser session attached"))?;
                for event_type in ["mousePressed", "mouseReleased"] {
                    session
                        .send(
                            "Input.dispatchMouseEvent",
                            json!({
                                "type": event_type,
                                "x": x,
                                "y": y,
                                "button": "left",
                                "clickCount": 1,
                                "buttons": if event_type == "mousePressed" { 1 } else { 0 },
                            }),
                        )
                        .await?;
                }

                Ok(json!({
                    "clicked": described.get("text").cloned().unwrap_or(Value::Null),
                    "tag": described.get("tag").cloned().unwrap_or(Value::Null),
                }))
            })
        }),
    );

    // ─── browser_type ─────────────────────────────────────────────────────
    let type_tool = browser.clone();
    server.add_tool(
        "browser_type",
        "Type text into an element. The element may be a snapshot ref (e3), a CSS selector, or \
         visible text.",
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "element": { "type": "string" }
            },
            "required": ["text", "element"]
        }),
        Box::new(move |args| {
            let browser = type_tool.clone();
            Box::pin(async move {
                let text = need_str(&args, "text")?;
                let element = need_str(&args, "element")?;
                browser.ensure_browser().await?;
                let focused = browser.evaluate(&scripts::focus(&element)).await?;
                if focused.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                    anyhow::bail!(
                        "{}",
                        focused
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("element not found")
                    );
                }

                let mut guard = browser.session.lock().await;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("no browser session attached"))?;
                session
                    .send("Input.insertText", json!({ "text": text }))
                    .await?;
                Ok(json!({ "typed": text }))
            })
        }),
    );

    // ─── browser_fill_form ────────────────────────────────────────────────
    let fill = browser.clone();
    server.add_tool(
        "browser_fill_form",
        "Fill several fields at once. Each field names an element and the value to type into it.",
        json!({
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "element": { "type": "string" },
                            "value": { "type": "string" }
                        },
                        "required": ["element", "value"]
                    }
                }
            },
            "required": ["fields"]
        }),
        Box::new(move |args| {
            let browser = fill.clone();
            Box::pin(async move {
                let fields = args["fields"]
                    .as_array()
                    .ok_or_else(|| anyhow!("fields required"))?;
                browser.ensure_browser().await?;

                let mut filled = Vec::new();
                for field in fields {
                    let element = need_str(field, "element")?;
                    let value = need_str(field, "value")?;
                    let focused = browser.evaluate(&scripts::focus(&element)).await?;
                    if focused.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                        anyhow::bail!(
                            "{}",
                            focused
                                .get("error")
                                .and_then(|v| v.as_str())
                                .unwrap_or("element not found")
                        );
                    }
                    let mut guard = browser.session.lock().await;
                    let session = guard
                        .as_mut()
                        .ok_or_else(|| anyhow!("no browser session attached"))?;
                    session
                        .send("Input.insertText", json!({ "text": value }))
                        .await?;
                    drop(guard);
                    filled.push(json!({ "element": element, "value": value }));
                }
                Ok(json!({ "filled": filled }))
            })
        }),
    );

    // ─── browser_select_option ────────────────────────────────────────────
    let select = browser.clone();
    server.add_tool(
        "browser_select_option",
        "Choose an option in a `<select>` element.",
        json!({
            "type": "object",
            "properties": {
                "element": { "type": "string" },
                "value": { "type": "string" }
            },
            "required": ["element", "value"]
        }),
        Box::new(move |args| {
            let browser = select.clone();
            Box::pin(async move {
                let element = need_str(&args, "element")?;
                let value = need_str(&args, "value")?;
                browser.ensure_browser().await?;
                let result = browser
                    .evaluate(&scripts::select_option(&element, &value))
                    .await?;
                if result.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                    anyhow::bail!(
                        "{}",
                        result
                            .get("error")
                            .and_then(|v| v.as_str())
                            .unwrap_or("option not found")
                    );
                }
                Ok(json!({
                    "value": result.get("value").cloned().unwrap_or(Value::Null),
                    "text": result.get("text").cloned().unwrap_or(Value::Null),
                }))
            })
        }),
    );

    // ─── browser_press_key ────────────────────────────────────────────────
    let press = browser.clone();
    server.add_tool(
        "browser_press_key",
        "Press a key in the page: Enter, Escape, Tab, Backspace, ArrowDown, or a single character.",
        json!({
            "type": "object",
            "properties": { "key": { "type": "string" } },
            "required": ["key"]
        }),
        Box::new(move |args| {
            let browser = press.clone();
            Box::pin(async move {
                let key = need_str(&args, "key")?;
                browser.ensure_browser().await?;

                let (windows_key, code, key_code) = match key.as_str() {
                    "Enter" => ("Enter", "Enter", 13),
                    "Escape" => ("Escape", "Escape", 27),
                    "Tab" => ("Tab", "Tab", 9),
                    "Backspace" => ("Backspace", "Backspace", 8),
                    "Delete" => ("Delete", "Delete", 46),
                    "ArrowUp" => ("ArrowUp", "ArrowUp", 38),
                    "ArrowDown" => ("ArrowDown", "ArrowDown", 40),
                    "ArrowLeft" => ("ArrowLeft", "ArrowLeft", 37),
                    "ArrowRight" => ("ArrowRight", "ArrowRight", 39),
                    other => {
                        let ch = other.chars().next().unwrap_or('\0');
                        (other, "", ch as u32)
                    }
                };

                let mut guard = browser.session.lock().await;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("no browser session attached"))?;
                session
                    .send(
                        "Input.dispatchKeyEvent",
                        json!({
                            "type": "keyDown",
                            "key": windows_key,
                            "code": code,
                            "windowsVirtualKeyCode": key_code,
                            "nativeVirtualKeyCode": key_code,
                            "text": if key_code < 32 { "" } else { &key },
                        }),
                    )
                    .await?;
                session
                    .send(
                        "Input.dispatchKeyEvent",
                        json!({
                            "type": "keyUp",
                            "key": windows_key,
                            "code": code,
                            "windowsVirtualKeyCode": key_code,
                            "nativeVirtualKeyCode": key_code,
                        }),
                    )
                    .await?;
                Ok(json!({ "pressed": key }))
            })
        }),
    );

    // ─── browser_evaluate ─────────────────────────────────────────────────
    let evaluate = browser.clone();
    server.add_tool(
        "browser_evaluate",
        "Run JavaScript in the page and return the result. The function is evaluated as written.",
        json!({
            "type": "object",
            "properties": { "function": { "type": "string" } },
            "required": ["function"]
        }),
        Box::new(move |args| {
            let browser = evaluate.clone();
            Box::pin(async move {
                let function = need_str(&args, "function")?;
                browser.ensure_browser().await?;
                let value = browser.evaluate(&function).await?;
                Ok(json!({ "result": value }))
            })
        }),
    );

    // ─── browser_find ─────────────────────────────────────────────────────
    let find = browser.clone();
    server.add_tool(
        "browser_find",
        "Find text on the page and report where it appears.",
        json!({
            "type": "object",
            "properties": { "text": { "type": "string" } },
            "required": ["text"]
        }),
        Box::new(move |args| {
            let browser = find.clone();
            Box::pin(async move {
                let text = need_str(&args, "text")?;
                browser.ensure_browser().await?;
                let result = browser.evaluate(&scripts::find_text(&text)).await?;
                Ok(json!({
                    "count": result.get("count").cloned().unwrap_or(json!(0)),
                    "hits": result.get("hits").cloned().unwrap_or(json!([])),
                }))
            })
        }),
    );

    // ─── browser_wait_for ─────────────────────────────────────────────────
    let wait = browser.clone();
    server.add_tool(
        "browser_wait_for",
        "Wait for text to appear on the page, or simply wait a number of seconds.",
        json!({
            "type": "object",
            "properties": {
                "text": { "type": "string" },
                "time": { "type": "number", "description": "seconds to wait" }
            }
        }),
        Box::new(move |args| {
            let browser = wait.clone();
            Box::pin(async move {
                browser.ensure_browser().await?;
                let text = args["text"].as_str().filter(|t| !t.is_empty());
                let seconds = args["time"].as_f64();

                if let Some(seconds) = seconds {
                    tokio::time::sleep(Duration::from_secs_f64(seconds.max(0.0))).await;
                    return Ok(json!({ "waited": seconds }));
                }

                let Some(text) = text else {
                    anyhow::bail!("either text or time is required");
                };

                for _ in 0..50 {
                    let result = browser.evaluate(&scripts::wait_for_text(text)).await?;
                    if result.get("found").and_then(|v| v.as_bool()) == Some(true) {
                        return Ok(json!({ "found": true }));
                    }
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                anyhow::bail!("text not found within 10s: {}", text)
            })
        }),
    );

    // ─── browser_take_screenshot ──────────────────────────────────────────
    let screenshot = browser.clone();
    server.add_tool(
        "browser_take_screenshot",
        "Capture a PNG screenshot of the page. With a filename the image is written to disk; \
         otherwise it is returned inline.",
        json!({
            "type": "object",
            "properties": {
                "filename": { "type": "string" },
                "fullPage": { "type": "boolean", "default": false },
                "element": { "type": "string" }
            }
        }),
        Box::new(move |args| {
            let browser = screenshot.clone();
            Box::pin(async move {
                browser.ensure_browser().await?;
                let filename = args["filename"].as_str().filter(|f| !f.is_empty());
                let full_page = args["fullPage"].as_bool().unwrap_or(false);
                let element = args["element"].as_str().filter(|e| !e.is_empty());

                let mut clip = Value::Null;
                if let Some(element) = element {
                    let described = browser.evaluate(&scripts::describe(element)).await?;
                    if described.get("ok").and_then(|v| v.as_bool()) != Some(true) {
                        anyhow::bail!(
                            "{}",
                            described
                                .get("error")
                                .and_then(|v| v.as_str())
                                .unwrap_or("element not found")
                        );
                    }
                    clip = json!({
                        "x": described.get("x").cloned().unwrap_or(json!(0.0)),
                        "y": described.get("y").cloned().unwrap_or(json!(0.0)),
                        "width": described.get("width").cloned().unwrap_or(json!(0.0)),
                        "height": described.get("height").cloned().unwrap_or(json!(0.0)),
                        "scale": 1,
                    });
                }

                let mut params = json!({ "format": "png" });
                if full_page {
                    params["captureBeyondViewport"] = json!(true);
                }
                if !clip.is_null() {
                    params["clip"] = clip;
                }

                let mut guard = browser.session.lock().await;
                let session = guard
                    .as_mut()
                    .ok_or_else(|| anyhow!("no browser session attached"))?;
                let result = session.send("Page.captureScreenshot", params).await?;
                drop(guard);

                let data = result
                    .get("data")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| anyhow!("screenshot failed"))?
                    .to_string();

                if let Some(filename) = filename {
                    let path = PathBuf::from(filename);
                    if let Some(parent) = path.parent() {
                        if !parent.as_os_str().is_empty() {
                            std::fs::create_dir_all(parent).ok();
                        }
                    }
                    let bytes = base64::engine::general_purpose::STANDARD
                        .decode(&data)
                        .context("decoding screenshot")?;
                    std::fs::write(&path, bytes)
                        .with_context(|| format!("writing {}", path.display()))?;
                    Ok(json!({ "saved": path.to_string_lossy() }))
                } else {
                    Ok(json!({
                        "content": [
                            { "type": "image", "data": data, "mimeType": "image/png" }
                        ]
                    }))
                }
            })
        }),
    );

    // ─── browser_console_messages ─────────────────────────────────────────
    let console = browser.clone();
    server.add_tool(
        "browser_console_messages",
        "Return the console output and page errors collected since the session attached.",
        json!({
            "type": "object",
            "properties": {
                "level": { "type": "string", "enum": ["debug", "info", "log", "warning", "error"] }
            }
        }),
        Box::new(move |args| {
            let browser = console.clone();
            Box::pin(async move {
                browser.ensure_browser().await?;
                let level = args["level"].as_str();
                let mut guard = browser.session.lock().await;
                let messages = guard
                    .as_mut()
                    .map(|session| session.console.clone())
                    .unwrap_or_default();
                let filtered: Vec<Value> = messages
                    .into_iter()
                    .filter(|m| match level {
                        Some("warning") => m.level == "warning" || m.level == "warn",
                        Some(level) => m.level == level,
                        None => true,
                    })
                    .map(|m| json!({ "level": m.level, "text": m.text }))
                    .collect();
                Ok(json!({ "messages": filtered }))
            })
        }),
    );

    // ─── browser_tabs ─────────────────────────────────────────────────────
    let tabs = browser.clone();
    server.add_tool(
        "browser_tabs",
        "List the browser's tabs, open a new one, switch to one, or close one.",
        json!({
            "type": "object",
            "properties": {
                "action": { "type": "string", "enum": ["list", "new", "select", "close"], "default": "list" },
                "index": { "type": "integer" },
                "url": { "type": "string" }
            }
        }),
        Box::new(move |args| {
            let browser = tabs.clone();
            Box::pin(async move {
                browser.ensure_browser().await?;
                let action = args["action"].as_str().unwrap_or("list");

                match action {
                    "list" => {
                        let targets = cdp::list_targets(&browser.http).await?;
                        let current = browser.current_target().await;
                        let items: Vec<Value> = targets
                            .iter()
                            .map(|t| {
                                json!({
                                    "id": t.id,
                                    "title": t.title,
                                    "url": t.url,
                                    "active": current.as_ref().map(|c| c.id == t.id).unwrap_or(false),
                                })
                            })
                            .collect();
                        Ok(json!({ "tabs": items }))
                    }
                    "new" => {
                        let url = args["url"].as_str().unwrap_or("about:blank");
                        let target = cdp::new_target(&browser.http, url).await?;
                        browser.switch_to(target.clone()).await?;
                        Ok(json!({ "opened": target.url, "id": target.id }))
                    }
                    "select" => {
                        let index = args["index"]
                            .as_u64()
                            .ok_or_else(|| anyhow!("index required for select"))? as usize;
                        let targets = cdp::list_targets(&browser.http).await?;
                        let target = targets
                            .get(index)
                            .cloned()
                            .ok_or_else(|| anyhow!("no tab at index {}", index))?;
                        browser.switch_to(target.clone()).await?;
                        Ok(json!({ "selected": target.url, "id": target.id }))
                    }
                    "close" => {
                        let index = args["index"]
                            .as_u64()
                            .ok_or_else(|| anyhow!("index required for close"))? as usize;
                        let targets = cdp::list_targets(&browser.http).await?;
                        let target = targets
                            .get(index)
                            .cloned()
                            .ok_or_else(|| anyhow!("no tab at index {}", index))?;
                        cdp::close_target(&browser.http, &target.id).await?;
                        *browser.session.lock().await = None;
                        *browser.target.lock().await = None;
                        Ok(json!({ "closed": target.url }))
                    }
                    other => anyhow::bail!("unknown action: {}", other),
                }
            })
        }),
    );

    // ─── browser_get_config ───────────────────────────────────────────────
    let config = browser.clone();
    server.add_tool(
        "browser_get_config",
        "Report the CDP endpoint and the tab the session is attached to.",
        json!({ "type": "object", "properties": {} }),
        Box::new(move |_args| {
            let browser = config.clone();
            Box::pin(async move {
                let current = browser.current_target().await;
                Ok(json!({
                    "cdp_endpoint": cdp::endpoint(),
                    "tab": current.map(|t| json!({"id": t.id, "title": t.title, "url": t.url})),
                }))
            })
        }),
    );

    // ─── browser_close ────────────────────────────────────────────────────
    let close = browser.clone();
    server.add_tool(
        "browser_close",
        "Detach from the current tab and close it.",
        json!({ "type": "object", "properties": {} }),
        Box::new(move |_args| {
            let browser = close.clone();
            Box::pin(async move {
                let current = browser.current_target().await;
                if let Some(target) = current {
                    cdp::close_target(&browser.http, &target.id).await.ok();
                }
                *browser.session.lock().await = None;
                *browser.target.lock().await = None;
                Ok(json!({ "closed": true }))
            })
        }),
    );

    server.run().await
}
