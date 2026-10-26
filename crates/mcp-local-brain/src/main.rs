//! mcp-local-brain — Rust MCP server for the local-brain RAG service.
//!
//! A Rust port of `~/.local/share/local-brain/mcp_rag_server.py`.
//!
//! Tools:
//!   search_memory(query, k=5)
//!   recent_context(session_id, n=10)
//!   index_memory(source, content, session_id?, role?)
//!
//! The service URL comes from `LOCAL_BRAIN_URL` and defaults to
//! `http://127.0.0.1:9876`.

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde_json::{json, Value};

/// Base URL of the local-brain HTTP service.
fn base_url() -> String {
    std::env::var("LOCAL_BRAIN_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:9876".to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Read a string field, returning `None` when it is absent or not a string.
fn opt_str(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string())
}

/// Format one search hit.
fn format_hit(hit: &Value) -> String {
    let source = hit
        .get("source")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let title = hit
        .get("metadata")
        .and_then(|m| m.get("title"))
        .and_then(|v| v.as_str())
        .unwrap_or("untitled");
    let score = hit
        .get("score")
        .and_then(|v| v.as_f64())
        .or_else(|| hit.get("vector_score").and_then(|v| v.as_f64()))
        .unwrap_or(0.0);
    let content = hit
        .get("content")
        .and_then(|v| v.as_str())
        .or_else(|| hit.get("text").and_then(|v| v.as_str()))
        .unwrap_or("");

    format!(
        "[Source: {} | Title: {} | Score: {:.3}]\n{}",
        source,
        title,
        score,
        preview(content, 800)
    )
}

/// Format one recent turn.
fn format_turn(turn: &Value) -> String {
    let role = turn
        .get("role")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let content = turn
        .get("content")
        .and_then(|v| v.as_str())
        .or_else(|| turn.get("text").and_then(|v| v.as_str()))
        .unwrap_or("");
    format!("[{}]\n{}", role, preview(content, 600))
}

/// Truncate a preview the way the upstream server does.
fn preview(content: &str, limit: usize) -> String {
    if content.chars().count() > limit {
        let head: String = content.chars().take(limit).collect();
        format!("{}...", head)
    } else {
        content.to_string()
    }
}

/// Search the RAG store.
async fn search_memory(client: &Client, query: &str, k: usize) -> Result<String> {
    let url = format!(
        "{}/api/rag/search?q={}&k={}",
        base_url(),
        urlencoding(query),
        k
    );
    let data = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?
        .error_for_status()
        .with_context(|| format!("local-brain search failed for {}", url))?
        .json::<Value>()
        .await
        .context("parsing local-brain search response")?;

    let results = match &data {
        Value::Array(items) => items.clone(),
        _ => data
            .get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default(),
    };

    if results.is_empty() {
        return Ok("No relevant memories found.".to_string());
    }
    Ok(results
        .iter()
        .map(format_hit)
        .collect::<Vec<_>>()
        .join("\n\n---\n\n"))
}

/// Fetch the most recent turns of a session.
async fn recent_context(client: &Client, session_id: &str, n: usize) -> Result<String> {
    let url = format!(
        "{}/api/rag/recent?session_id={}&n={}",
        base_url(),
        urlencoding(session_id),
        n
    );
    let data = client
        .get(&url)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?
        .error_for_status()
        .with_context(|| format!("local-brain recent failed for {}", url))?
        .json::<Value>()
        .await
        .context("parsing local-brain recent response")?;

    let turns = match &data {
        Value::Array(items) => items.clone(),
        _ => data
            .get("results")
            .or_else(|| data.get("turns"))
            .or_else(|| data.get("items"))
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default(),
    };

    if turns.is_empty() {
        return Ok(format!(
            "No recent context found for session {}.",
            session_id
        ));
    }
    Ok(turns
        .iter()
        .map(format_turn)
        .collect::<Vec<_>>()
        .join("\n\n---\n\n"))
}

/// Index a piece of content into the RAG store.
async fn index_memory(
    client: &Client,
    source: &str,
    content: &str,
    session_id: Option<&str>,
    role: Option<&str>,
) -> Result<String> {
    let url = format!("{}/api/rag/index", base_url());
    let payload = json!({
        "source": source,
        "source_id": session_id.unwrap_or(""),
        "content": content,
        "session_id": session_id,
        "role": role,
        "metadata": {},
    });

    let response = client
        .post(&url)
        .json(&payload)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?;

    let body = response.text().await.unwrap_or_default();
    let data: Value = serde_json::from_str(&body).unwrap_or(Value::Null);

    if let Some(error) = data.get("error").and_then(|v| v.as_str()) {
        return Ok(format!("Error indexing: {}", error));
    }
    let chunks = data.get("chunks").and_then(|v| v.as_i64()).unwrap_or(0);
    Ok(format!(
        "Indexed successfully. Created {} chunk(s).",
        chunks
    ))
}

/// Percent-encode a query string value.
fn urlencoding(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for byte in input.as_bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(*byte as char)
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{:02X}", byte)),
        }
    }
    out
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

    let client = Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .build()?;

    let mut server = okc_mcp::McpServer::new("local-brain");

    let search_client = client.clone();
    server.add_tool(
        "search_memory",
        "Search the local RAG memory for relevant information across all indexed sources \
         (notes, documents, past conversations). Returns the most relevant passages with \
         their source, title and similarity score.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "k": { "type": "integer", "default": 5 }
            },
            "required": ["query"]
        }),
        Box::new(move |args| {
            let client = search_client.clone();
            Box::pin(async move {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| anyhow!("query required"))?;
                let k = args["k"].as_u64().unwrap_or(5) as usize;
                Ok(json!(search_memory(&client, query, k).await?))
            })
        }),
    );

    let recent_client = client.clone();
    server.add_tool(
        "recent_context",
        "Get the most recent indexed turns of a session.",
        json!({
            "type": "object",
            "properties": {
                "session_id": { "type": "string" },
                "n": { "type": "integer", "default": 10 }
            },
            "required": ["session_id"]
        }),
        Box::new(move |args| {
            let client = recent_client.clone();
            Box::pin(async move {
                let session_id = args["session_id"]
                    .as_str()
                    .ok_or_else(|| anyhow!("session_id required"))?;
                let n = args["n"].as_u64().unwrap_or(10) as usize;
                Ok(json!(recent_context(&client, session_id, n).await?))
            })
        }),
    );

    let index_client = client.clone();
    server.add_tool(
        "index_memory",
        "Index a piece of content into the local RAG memory so it can be recalled later.",
        json!({
            "type": "object",
            "properties": {
                "source": { "type": "string" },
                "content": { "type": "string" },
                "session_id": { "type": "string" },
                "role": { "type": "string" }
            },
            "required": ["source", "content"]
        }),
        Box::new(move |args| {
            let client = index_client.clone();
            Box::pin(async move {
                let source = args["source"]
                    .as_str()
                    .ok_or_else(|| anyhow!("source required"))?;
                let content = args["content"]
                    .as_str()
                    .ok_or_else(|| anyhow!("content required"))?;
                let session_id = opt_str(&args, "session_id");
                let role = opt_str(&args, "role");
                Ok(json!(
                    index_memory(
                        &client,
                        source,
                        content,
                        session_id.as_deref(),
                        role.as_deref(),
                    )
                    .await?
                ))
            })
        }),
    );

    server.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hits_are_formatted_with_metadata() {
        let hit = json!({
            "source": "opencode",
            "metadata": {"title": "Test review"},
            "score": 0.75,
            "content": "hello"
        });
        let text = format_hit(&hit);
        assert!(text.contains("[Source: opencode | Title: Test review | Score: 0.750]"));
        assert!(text.contains("hello"));
    }

    #[test]
    fn previews_are_truncated() {
        let hit = json!({"content": "x".repeat(2000)});
        let text = format_hit(&hit);
        assert_eq!(text.lines().count(), 2);
        assert_eq!(text.lines().nth(1).unwrap().chars().count(), 803);
        assert!(text.lines().nth(1).unwrap().ends_with("..."));
    }

    #[test]
    fn short_previews_are_untouched() {
        let hit = json!({"content": "short"});
        let text = format_hit(&hit);
        assert!(text.ends_with("short"));
    }
}
