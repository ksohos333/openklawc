//! mcp-anticonf — Anti-confabulation MCP server.
//!
//! Tools:
//!   verify_claim(claim, source_urls=[])
//!   note_observation(observation, source_url, source_kind)
//!   list_observations(limit=20)

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::fs;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut server = okc_mcp::McpServer::new("anticonf");

    // ─── verify_claim ─────────────────────────────────────────────────────
    server.add_tool(
        "verify_claim",
        "Fetch sources for a claim and return raw observed facts. No verdict.",
        json!({
            "type": "object",
            "properties": {
                "claim": { "type": "string" },
                "source_urls": { "type": "array", "items": { "type": "string" } }
            },
            "required": ["claim"]
        }),
        Box::new(|args| {
            Box::pin(async move {
                let claim = args["claim"]
                    .as_str()
                    .ok_or_else(|| anyhow!("claim required"))?;
                let urls: Vec<String> = args["source_urls"]
                    .as_array()
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    })
                    .unwrap_or_default();
                let result = verify_claim(claim, urls).await;
                Ok(json!(result))
            })
        }),
    );

    // ─── note_observation ─────────────────────────────────────────────────
    server.add_tool(
        "note_observation",
        "Log an observed fact with provenance. Returns fact_id.",
        json!({
            "type": "object",
            "properties": {
                "observation": { "type": "string" },
                "source_url": { "type": "string" },
                "source_kind": { "type": "string" }
            },
            "required": ["observation", "source_url"]
        }),
        Box::new(|args| {
            Box::pin(async move {
                let obs = args["observation"]
                    .as_str()
                    .ok_or_else(|| anyhow!("observation required"))?;
                let url = args["source_url"]
                    .as_str()
                    .ok_or_else(|| anyhow!("source_url required"))?;
                let kind = args["source_kind"].as_str().unwrap_or("unspecified");
                let result = note_observation(obs, url, kind).await;
                Ok(json!(result))
            })
        }),
    );

    // ─── list_observations ────────────────────────────────────────────────
    server.add_tool(
        "list_observations",
        "List recent observations with provenance.",
        json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "default": 20 }
            }
        }),
        Box::new(|args| {
            Box::pin(async move {
                let limit = args["limit"].as_u64().unwrap_or(20) as usize;
                let result = list_observations(limit).await;
                Ok(json!(result))
            })
        }),
    );

    server.run().await
}

// ══════════════════════════════════════════════════════════════════════════
// DATA DIRECTORY
// ══════════════════════════════════════════════════════════════════════════

fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("anticonf")
        .join("data")
}

fn observations_path() -> PathBuf {
    data_dir().join("observations.jsonl")
}

async fn ensure_data_dir() -> Result<()> {
    fs::create_dir_all(data_dir()).await?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════
// VERIFY CLAIM
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct VerifyResult {
    claim: String,
    sources_checked: Vec<SourceCheck>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct SourceCheck {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    observed_facts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn verify_claim(claim: &str, source_urls: Vec<String>) -> VerifyResult {
    let mut sources = Vec::new();
    let mut urls = source_urls;

    // If no URLs provided, search for sources
    let mut search_titles: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();
    if urls.is_empty() {
        let search = search_ddg(claim).await;
        for item in search.into_iter().take(5) {
            search_titles.insert(item.url.clone(), item.title);
            urls.push(item.url);
        }
    }

    for url in urls {
        let check = match fetch_and_extract(&url).await {
            Ok((title, text)) => SourceCheck {
                url: url.clone(),
                title: Some(title.unwrap_or_else(|| "no title".into())),
                observed_facts: Some(text.chars().take(2000).collect()),
                error: None,
            },
            Err(e) => SourceCheck {
                url: url.clone(),
                title: search_titles.get(&url).cloned(),
                observed_facts: None,
                error: Some(e.to_string()),
            },
        };
        sources.push(check);
    }

    VerifyResult {
        claim: claim.to_string(),
        sources_checked: sources,
        error: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════
// OBSERVATIONS
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize)]
struct Observation {
    fact_id: String,
    timestamp: String,
    observation: String,
    source_url: String,
    source_kind: String,
}

async fn note_observation(obs: &str, url: &str, kind: &str) -> serde_json::Value {
    let _ = ensure_data_dir().await;
    let id = uuid::Uuid::new_v4().to_string();

    let entry = Observation {
        fact_id: id.clone(),
        timestamp: Utc::now().to_rfc3339(),
        observation: obs.to_string(),
        source_url: url.to_string(),
        source_kind: kind.to_string(),
    };

    let line = match serde_json::to_string(&entry) {
        Ok(l) => l,
        Err(e) => return json!({ "error": e.to_string() }),
    };

    let path = observations_path();
    let mut file = match fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .await
    {
        Ok(f) => f,
        Err(e) => return json!({ "error": e.to_string() }),
    };

    use tokio::io::AsyncWriteExt;
    if let Err(e) = file.write_all(line.as_bytes()).await {
        return json!({ "error": e.to_string() });
    }
    if let Err(e) = file.write_all(b"\n").await {
        return json!({ "error": e.to_string() });
    }

    json!({ "fact_id": id, "status": "noted" })
}

async fn list_observations(limit: usize) -> Vec<Observation> {
    let path = observations_path();
    let content = match fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut obs = Vec::new();
    for line in content.lines() {
        if let Ok(o) = serde_json::from_str::<Observation>(line) {
            obs.push(o);
        }
    }
    obs.reverse();
    obs.truncate(limit);
    obs
}

// ══════════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// ══════════════════════════════════════════════════════════════════════════

async fn fetch_and_extract(url: &str) -> Result<(Option<String>, String)> {
    let client = build_client()?;
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(20))
        .send()
        .await
        .with_context(|| format!("fetch {}", url))?;

    let body = resp
        .text()
        .await
        .with_context(|| format!("read body {}", url))?;

    let (title, text, _) = parse_html(&body);
    Ok((title, text))
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(
        USER_AGENT,
        HeaderValue::from_static("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"),
    );
    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("build client")
}

fn parse_html(html: &str) -> (Option<String>, String, Vec<String>) {
    let doc = Html::parse_document(html);
    let title = doc
        .select(&Selector::parse("title").unwrap())
        .next()
        .map(|e| e.text().collect::<String>().trim().to_string());

    let text = extract_text(&doc);
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");

    let mut links = Vec::new();
    for el in doc.select(&Selector::parse("a[href]").unwrap()) {
        if let Some(h) = el.value().attr("href") {
            if h.starts_with("http") {
                links.push(h.to_string());
            }
        }
    }
    links.truncate(20);

    (title, text, links)
}

fn extract_text(doc: &Html) -> String {
    for sel in &["article", "main", ".content", "#content"] {
        if let Ok(s) = Selector::parse(sel) {
            if let Some(el) = doc.select(&s).next() {
                return el.text().collect::<Vec<_>>().join(" ");
            }
        }
    }
    if let Ok(s) = Selector::parse("body") {
        if let Some(el) = doc.select(&s).next() {
            return el.text().collect::<Vec<_>>().join(" ");
        }
    }
    doc.root_element().text().collect::<Vec<_>>().join(" ")
}

// ══════════════════════════════════════════════════════════════════════════
// SEARCH
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug)]
struct SearchItem {
    url: String,
    title: String,
}

async fn search_ddg(query: &str) -> Vec<SearchItem> {
    let encoded = urlencoding::encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

    let client = match build_client() {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let body = match client
        .get(&url)
        .timeout(std::time::Duration::from_secs(30))
        .send()
        .await
    {
        Ok(r) => match r.text().await {
            Ok(b) => b,
            Err(_) => return Vec::new(),
        },
        Err(_) => return Vec::new(),
    };

    let doc = Html::parse_document(&body);
    let mut items = Vec::new();

    let result_sel = Selector::parse(".result").unwrap_or_else(|_| Selector::parse("div").unwrap());
    let title_a = Selector::parse("a.result__a").unwrap_or_else(|_| Selector::parse("a").unwrap());

    for result in doc.select(&result_sel) {
        let title = result
            .select(
                &Selector::parse(".result__title")
                    .unwrap_or_else(|_| Selector::parse("h2").unwrap()),
            )
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let url = result
            .select(&title_a)
            .next()
            .and_then(|e| e.value().attr("href"))
            .map(|s| s.to_string())
            .unwrap_or_default();

        if !title.is_empty() && !url.is_empty() {
            items.push(SearchItem { url, title });
        }
    }

    items
}
