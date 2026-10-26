//! mcp-brave-search — Rust MCP server for the Brave Search API.
//!
//! A Rust port of `@modelcontextprotocol/server-brave-search` (v0.6.2).
//!
//! Tools:
//!   brave_web_search(query, count=10, offset=0)
//!   brave_local_search(query, count=5)
//!
//! The API key is read from `BRAVE_API_KEY`, falling back to
//! `~/.config/opencode/keys/brave.key` and `~/.config/openklawc/keys/brave.key`.

use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, ACCEPT_ENCODING};
use serde_json::{json, Value};
use tokio::sync::Mutex;

const WEB_ENDPOINT: &str = "https://api.search.brave.com/res/v1/web/search";
const LOCAL_POIS_ENDPOINT: &str = "https://api.search.brave.com/res/v1/local/pois";
const LOCAL_DESCRIPTIONS_ENDPOINT: &str = "https://api.search.brave.com/res/v1/local/descriptions";

/// Requests per second allowed by the Brave plan.
const RATE_LIMIT_PER_SECOND: u32 = 1;
/// Requests per month allowed by the Brave plan.
const RATE_LIMIT_PER_MONTH: u32 = 15_000;

/// Client-side rate limiter mirroring the upstream server.
#[derive(Debug)]
struct RateLimiter {
    second_count: u32,
    month_count: u32,
    second_started: Instant,
    month_started: Instant,
}

impl RateLimiter {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            second_count: 0,
            month_count: 0,
            second_started: now,
            month_started: now,
        }
    }

    fn check(&mut self) -> Result<()> {
        let now = Instant::now();
        if now.duration_since(self.second_started) >= Duration::from_secs(1) {
            self.second_count = 0;
            self.second_started = now;
        }
        if now.duration_since(self.month_started) >= Duration::from_secs(30 * 24 * 3600) {
            self.month_count = 0;
            self.month_started = now;
        }
        if self.second_count >= RATE_LIMIT_PER_SECOND {
            anyhow::bail!("Rate limit exceeded");
        }
        if self.month_count >= RATE_LIMIT_PER_MONTH {
            anyhow::bail!("Rate limit exceeded");
        }
        self.second_count += 1;
        self.month_count += 1;
        Ok(())
    }
}

/// Locate the Brave API key.
fn api_key() -> Option<String> {
    if let Ok(key) = std::env::var("BRAVE_API_KEY") {
        let key = key.trim().to_string();
        if !key.is_empty() {
            return Some(key);
        }
    }
    let mut candidates = Vec::new();
    if let Some(config) = dirs::config_dir() {
        candidates.push(config.join("opencode").join("keys").join("brave.key"));
        candidates.push(config.join("openklawc").join("keys").join("brave.key"));
    }
    for path in candidates {
        if let Ok(value) = std::fs::read_to_string(&path) {
            let value = value.trim().to_string();
            if !value.is_empty() {
                return Some(value);
            }
        }
    }
    None
}

/// Shared state for the tool handlers.
struct Brave {
    client: reqwest::Client,
    key: Option<String>,
    limiter: Arc<Mutex<RateLimiter>>,
}

impl Brave {
    fn headers(&self) -> Result<HeaderMap> {
        let key = self
            .key
            .as_ref()
            .ok_or_else(|| anyhow!("BRAVE_API_KEY is required for Brave Search"))?;
        let mut headers = HeaderMap::new();
        headers.insert(ACCEPT, HeaderValue::from_static("application/json"));
        headers.insert(ACCEPT_ENCODING, HeaderValue::from_static("gzip"));
        headers.insert(
            "X-Subscription-Token",
            HeaderValue::from_str(key).context("invalid BRAVE_API_KEY")?,
        );
        Ok(headers)
    }

    async fn get_json(&self, url: &str) -> Result<Value> {
        let response = self
            .client
            .get(url)
            .headers(self.headers()?)
            .send()
            .await
            .with_context(|| format!("requesting {}", url))?;

        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!("Brave API error: {} {}\n{}", status.as_u16(), status, body);
        }
        response.json().await.context("parsing Brave API response")
    }

    /// Run a general web search.
    async fn web_search(&self, query: &str, count: usize, offset: usize) -> Result<String> {
        self.limiter.lock().await.check()?;
        let count = count.clamp(1, 20);
        let url = format!(
            "{}?q={}&count={}&offset={}",
            WEB_ENDPOINT,
            urlencoding(query),
            count,
            offset
        );
        let data = self.get_json(&url).await?;

        let results = data
            .get("web")
            .and_then(|w| w.get("results"))
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default();

        let mut blocks = Vec::new();
        for result in results {
            blocks.push(format!(
                "Title: {}\nDescription: {}\nURL: {}",
                result
                    .get("title")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                result
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                result
                    .get("url")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
            ));
        }
        Ok(blocks.join("\n\n"))
    }

    /// Search for local businesses and places.
    async fn local_search(&self, query: &str, count: usize) -> Result<String> {
        self.limiter.lock().await.check()?;
        let count = count.clamp(1, 20);

        let url = format!(
            "{}?q={}&search_lang=en&result_filter=locations&count={}",
            WEB_ENDPOINT,
            urlencoding(query),
            count
        );
        let data = self.get_json(&url).await?;

        let location_ids: Vec<String> = data
            .get("locations")
            .and_then(|l| l.get("results"))
            .and_then(|r| r.as_array())
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.get("id"))
                    .filter_map(|id| {
                        id.as_str()
                            .map(|s| s.to_string())
                            .or_else(|| id.as_i64().map(|n| n.to_string()))
                    })
                    .collect()
            })
            .unwrap_or_default();

        if location_ids.is_empty() {
            return self.web_search(query, count, 0).await;
        }

        let ids = location_ids
            .iter()
            .map(|id| urlencoding(id))
            .collect::<Vec<_>>()
            .join("&ids=");

        let pois = self
            .get_json(&format!("{}?ids={}", LOCAL_POIS_ENDPOINT, ids))
            .await?;
        let descriptions = self
            .get_json(&format!("{}?ids={}", LOCAL_DESCRIPTIONS_ENDPOINT, ids))
            .await?;

        let description_map = descriptions
            .get("descriptions")
            .and_then(|d| d.as_object())
            .cloned()
            .unwrap_or_default();

        let mut blocks = Vec::new();
        for poi in pois
            .get("results")
            .and_then(|r| r.as_array())
            .cloned()
            .unwrap_or_default()
        {
            let id = poi
                .get("id")
                .map(|v| match v {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                })
                .unwrap_or_default();

            let address = poi
                .get("address")
                .map(|a| {
                    [
                        a.get("streetAddress"),
                        a.get("addressLocality"),
                        a.get("addressRegion"),
                        a.get("postalCode"),
                    ]
                    .iter()
                    .filter_map(|v| v.and_then(|v| v.as_str()))
                    .filter(|s| !s.is_empty())
                    .collect::<Vec<_>>()
                    .join(", ")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "N/A".to_string());

            let rating = poi
                .get("rating")
                .and_then(|r| r.get("ratingValue"))
                .map(|v| match v {
                    Value::Number(n) => n.to_string(),
                    Value::String(s) => s.clone(),
                    _ => "N/A".to_string(),
                })
                .unwrap_or_else(|| "N/A".to_string());
            let rating_count = poi
                .get("rating")
                .and_then(|r| r.get("ratingCount"))
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let hours = poi
                .get("openingHours")
                .and_then(|h| h.as_array())
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|v| v.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                })
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "N/A".to_string());

            let description = description_map
                .get(&id)
                .and_then(|v| v.as_str())
                .unwrap_or("No description available")
                .to_string();

            blocks.push(format!(
                "Name: {}\nAddress: {}\nPhone: {}\nRating: {} ({} reviews)\nPrice Range: {}\nHours: {}\nDescription: {}\n",
                poi.get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or_default(),
                address,
                poi.get("phone")
                    .and_then(|v| v.as_str())
                    .unwrap_or("N/A"),
                rating,
                rating_count,
                poi.get("priceRange")
                    .and_then(|v| v.as_str())
                    .unwrap_or("N/A"),
                hours,
                description,
            ));
        }

        if blocks.is_empty() {
            Ok("No local results found".to_string())
        } else {
            Ok(blocks.join("\n---\n"))
        }
    }
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

    let key = api_key();
    if key.is_none() {
        tracing::warn!("BRAVE_API_KEY not found — brave-search tools will fail until it is set");
    }

    let brave = Arc::new(Brave {
        client: reqwest::Client::builder()
            .timeout(Duration::from_secs(20))
            .build()?,
        key,
        limiter: Arc::new(Mutex::new(RateLimiter::new())),
    });

    let mut server = okc_mcp::McpServer::new("brave-search");

    let brave_web = brave.clone();
    server.add_tool(
        "brave_web_search",
        "Performs a web search using the Brave Search API, ideal for general queries, news, \
         articles, and online content. Use this for broad information gathering, recent events, \
         or when you need diverse web sources. Supports pagination and content filtering. \
         Maximum 20 results per request, with offset for pagination.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Search query (max 400 chars, 50 words)" },
                "count": { "type": "integer", "description": "Number of results (1-20, default 10)", "default": 10 },
                "offset": { "type": "integer", "description": "Pagination offset (max 9, default 0)", "default": 0 }
            },
            "required": ["query"]
        }),
        Box::new(move |args| {
            let brave = brave_web.clone();
            Box::pin(async move {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| anyhow!("query required"))?;
                let count = args["count"].as_u64().unwrap_or(10) as usize;
                let offset = args["offset"].as_u64().unwrap_or(0) as usize;
                Ok(json!(brave.web_search(query, count, offset).await?))
            })
        }),
    );

    let brave_local = brave.clone();
    server.add_tool(
        "brave_local_search",
        "Searches for local businesses and places using Brave's Local Search API. Best for \
         queries related to physical locations, businesses, restaurants, services. Returns \
         business names and addresses, ratings and review counts, phone numbers and opening \
         hours. Automatically falls back to web search if no local results are found.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string", "description": "Local search query (e.g. 'pizza near Central Park')" },
                "count": { "type": "integer", "description": "Number of results (1-20, default 5)", "default": 5 }
            },
            "required": ["query"]
        }),
        Box::new(move |args| {
            let brave = brave_local.clone();
            Box::pin(async move {
                let query = args["query"]
                    .as_str()
                    .ok_or_else(|| anyhow!("query required"))?;
                let count = args["count"].as_u64().unwrap_or(5) as usize;
                Ok(json!(brave.local_search(query, count).await?))
            })
        }),
    );

    server.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_is_encoded() {
        assert_eq!(urlencoding("a b&c"), "a+b%26c");
        assert_eq!(urlencoding("plain"), "plain");
    }

    #[test]
    fn rate_limiter_blocks_bursts() {
        let mut limiter = RateLimiter::new();
        assert!(limiter.check().is_ok());
        assert!(limiter.check().is_err());
    }
}
