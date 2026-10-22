//! mcp-web-research — Rust MCP server for web research.
//!
//! Tools:
//!   fetch_url(url, timeout=25)
//!   web_search(query, n=10)
//!   web_research(question, depth=2, per_round=3)
//!   providers()

use std::collections::HashSet;

use anyhow::{anyhow, Context, Result};
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use scraper::{Html, Selector};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut server = okc_mcp::McpServer::new("web_research");

    // ─── fetch_url ────────────────────────────────────────────────────────
    server.add_tool(
        "fetch_url",
        "Fetch one URL and return structured content.",
        json!({
            "type": "object",
            "properties": {
                "url": { "type": "string" },
                "timeout": { "type": "number", "default": 25.0 }
            },
            "required": ["url"]
        }),
        Box::new(|args| Box::pin(async move {
            let url = args["url"].as_str().ok_or_else(|| anyhow!("url required"))?;
            let timeout = args["timeout"].as_f64().unwrap_or(25.0);
            let result = fetch_url(url, timeout).await;
            Ok(json!(result))
        })),
    );

    // ─── web_search ───────────────────────────────────────────────────────
    server.add_tool(
        "web_search",
        "Search the web via DuckDuckGo HTML interface.",
        json!({
            "type": "object",
            "properties": {
                "query": { "type": "string" },
                "n": { "type": "integer", "default": 10 }
            },
            "required": ["query"]
        }),
        Box::new(|args| Box::pin(async move {
            let query = args["query"].as_str().ok_or_else(|| anyhow!("query required"))?;
            let n = args["n"].as_u64().unwrap_or(10) as usize;
            let result = web_search(query, n).await;
            Ok(json!(result))
        })),
    );

    // ─── web_research ─────────────────────────────────────────────────────
    server.add_tool(
        "web_research",
        "Multi-step research: search, fetch, synthesize, repeat.",
        json!({
            "type": "object",
            "properties": {
                "question": { "type": "string" },
                "depth": { "type": "integer", "default": 2 },
                "per_round": { "type": "integer", "default": 3 }
            },
            "required": ["question"]
        }),
        Box::new(|args| Box::pin(async move {
            let question = args["question"].as_str().ok_or_else(|| anyhow!("question required"))?;
            let depth = args["depth"].as_u64().unwrap_or(2) as usize;
            let per_round = args["per_round"].as_u64().unwrap_or(3) as usize;
            let result = web_research(question, depth, per_round).await;
            Ok(json!(result))
        })),
    );

    // ─── providers ────────────────────────────────────────────────────────
    server.add_tool(
        "providers",
        "List available fetch backends.",
        json!({ "type": "object", "properties": {} }),
        Box::new(|_| Box::pin(async move {
            Ok(json!({ "backends": ["reqwest"] }))
        })),
    );

    server.run().await
}

// ══════════════════════════════════════════════════════════════════════════
// FETCH
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, serde::Serialize)]
struct FetchResult {
    url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_url: Option<String>,
    status: u16,
    backend: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    markdown: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    links: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn fetch_url(url: &str, timeout_secs: f64) -> FetchResult {
    let client = match build_client() {
        Ok(c) => c,
        Err(e) => return FetchResult {
            url: url.to_string(),
            final_url: None,
            status: 0,
            backend: "reqwest".into(),
            title: None,
            text: None,
            markdown: None,
            links: None,
            error: Some(e.to_string()),
        },
    };

    let timeout = std::time::Duration::from_secs_f64(timeout_secs.max(1.0));
    let resp = match client.get(url).timeout(timeout).send().await {
        Ok(r) => r,
        Err(e) => return FetchResult {
            url: url.to_string(),
            final_url: None,
            status: 0,
            backend: "reqwest".into(),
            title: None,
            text: None,
            markdown: None,
            links: None,
            error: Some(e.to_string()),
        },
    };

    let status = resp.status().as_u16();
    let final_url = resp.url().to_string();

    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => return FetchResult {
            url: url.to_string(),
            final_url: Some(final_url),
            status,
            backend: "reqwest".into(),
            title: None,
            text: None,
            markdown: None,
            links: None,
            error: Some(e.to_string()),
        },
    };

    let (title, text, links) = parse_html(&body);

    FetchResult {
        url: url.to_string(),
        final_url: Some(final_url),
        status,
        backend: "reqwest".into(),
        title,
        text: Some(text),
        markdown: None,
        links: Some(links),
        error: None,
    }
}

fn build_client() -> Result<reqwest::Client> {
    let mut headers = HeaderMap::new();
    headers.insert(USER_AGENT, HeaderValue::from_static(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
    ));
    headers.insert("Accept", HeaderValue::from_static("text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"));
    headers.insert("Accept-Language", HeaderValue::from_static("en-US,en;q=0.9"));

    reqwest::Client::builder()
        .default_headers(headers)
        .build()
        .context("build http client")
}

fn parse_html(html: &str) -> (Option<String>, String, Vec<String>) {
    let document = Html::parse_document(html);

    let title = document
        .select(&Selector::parse("title").unwrap())
        .next()
        .map(|e| e.text().collect::<String>().trim().to_string());

    // Extract links
    let mut links = Vec::new();
    let a_sel = Selector::parse("a[href]").unwrap();
    for el in document.select(&a_sel) {
        if let Some(href) = el.value().attr("href") {
            if href.starts_with("http") && !href.starts_with("javascript:") {
                links.push(href.to_string());
            }
        }
    }
    links.dedup();
    links.truncate(50);

    // Extract text — prefer article/main content, fall back to body
    let text = extract_text(&document);
    let text = clean_text(&text);

    (title, text, links)
}

fn extract_text(doc: &Html) -> String {
    // Try article/main first
    for selector in &["article", "main", "[role='main']", ".content", "#content"] {
        if let Ok(sel) = Selector::parse(selector) {
            if let Some(el) = doc.select(&sel).next() {
                return el.text().collect::<Vec<_>>().join(" ");
            }
        }
    }
    // Fall back to body
    if let Ok(sel) = Selector::parse("body") {
        if let Some(el) = doc.select(&sel).next() {
            return el.text().collect::<Vec<_>>().join(" ");
        }
    }
    doc.root_element().text().collect::<Vec<_>>().join(" ")
}

fn clean_text(text: &str) -> String {
    let mut out = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if !out.is_empty() {
                out.push(' ');
            }
            out.push_str(trimmed);
        }
    }
    // Collapse multiple spaces
    out.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ══════════════════════════════════════════════════════════════════════════
// SEARCH
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, serde::Serialize)]
struct SearchResult {
    query: String,
    engine: String,
    items: Vec<SearchItem>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, serde::Serialize)]
struct SearchItem {
    url: String,
    title: String,
    snippet: String,
}

async fn web_search(query: &str, n: usize) -> SearchResult {
    let encoded = urlencoding::encode(query);
    let url = format!("https://html.duckduckgo.com/html/?q={}", encoded);

    let client = match build_client() {
        Ok(c) => c,
        Err(e) => return SearchResult {
            query: query.to_string(),
            engine: "duckduckgo".into(),
            items: Vec::new(),
            error: Some(e.to_string()),
        },
    };

    let resp = match client.get(&url).timeout(std::time::Duration::from_secs(30)).send().await {
        Ok(r) => r,
        Err(e) => return SearchResult {
            query: query.to_string(),
            engine: "duckduckgo".into(),
            items: Vec::new(),
            error: Some(e.to_string()),
        },
    };

    let body = match resp.text().await {
        Ok(b) => b,
        Err(e) => return SearchResult {
            query: query.to_string(),
            engine: "duckduckgo".into(),
            items: Vec::new(),
            error: Some(e.to_string()),
        },
    };

    let items = parse_ddg_results(&body);
    let items: Vec<_> = items.into_iter().take(n).collect();

    SearchResult {
        query: query.to_string(),
        engine: "duckduckgo".into(),
        items,
        error: None,
    }
}

fn parse_ddg_results(html: &str) -> Vec<SearchItem> {
    let doc = Html::parse_document(html);
    let mut items = Vec::new();

    // DuckDuckGo HTML results use .result class
    let result_sel = Selector::parse(".result").unwrap_or_else(|_| Selector::parse("div").unwrap());
    let title_sel = Selector::parse(".result__title a").unwrap_or_else(|_| Selector::parse("a").unwrap());
    let snippet_sel = Selector::parse(".result__snippet").unwrap_or_else(|_| Selector::parse("p").unwrap());
    let url_sel = Selector::parse(".result__url").unwrap_or_else(|_| Selector::parse(".result__url").unwrap());

    // More flexible selectors for different DDG HTML layouts
    let result_sel2 = Selector::parse(".web-result").unwrap_or_else(|_| Selector::parse("div").unwrap());

    for result in doc.select(&result_sel).chain(doc.select(&result_sel2)) {
        let title = result
            .select(&title_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let snippet = result
            .select(&snippet_sel)
            .next()
            .map(|e| e.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        let url = result
            .select(&Selector::parse("a.result__a").unwrap_or_else(|_| Selector::parse("a").unwrap()))
            .next()
            .and_then(|e| e.value().attr("href"))
            .map(|s| s.to_string())
            .or_else(|| {
                result
                    .select(&url_sel)
                    .next()
                    .map(|e| e.text().collect::<String>().trim().to_string())
            })
            .unwrap_or_default();

        if !title.is_empty() && !url.is_empty() {
            items.push(SearchItem { url, title, snippet });
        }
    }

    items
}

// ══════════════════════════════════════════════════════════════════════════
// MULTI-STEP RESEARCH
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, serde::Serialize)]
struct ResearchResult {
    question: String,
    depth: usize,
    rounds: Vec<ResearchRound>,
    synthesis: String,
}

#[derive(Debug, serde::Serialize)]
struct ResearchRound {
    round: usize,
    query: String,
    findings: Vec<RoundFinding>,
}

#[derive(Debug, serde::Serialize)]
struct RoundFinding {
    url: String,
    title: String,
    excerpt: String,
}

async fn web_research(question: &str, depth: usize, per_round: usize) -> ResearchResult {
    let mut rounds = Vec::new();
    let mut seen_urls: HashSet<String> = HashSet::new();

    // Initial search queries — start with the question, then derive follow-ups
    let mut queries = vec![question.to_string()];

    for round_num in 0..depth {
        let query = match queries.get(round_num) {
            Some(q) => q.clone(),
            None => break,
        };

        let search = web_search(&query, per_round).await;
        let mut findings = Vec::new();

        for item in &search.items {
            if seen_urls.contains(&item.url) {
                continue;
            }
            seen_urls.insert(item.url.clone());

            let fetched = fetch_url(&item.url, 15.0).await;
            let excerpt = fetched.text.as_ref()
                .map(|t| t.chars().take(800).collect())
                .unwrap_or_else(|| item.snippet.clone());

            findings.push(RoundFinding {
                url: item.url.clone(),
                title: fetched.title.unwrap_or_else(|| item.title.clone()),
                excerpt,
            });
        }

        rounds.push(ResearchRound {
            round: round_num + 1,
            query,
            findings,
        });

        // Derive a follow-up query from what we found
        if round_num + 1 < depth {
            let follow_up = format!("{} more details", question);
            queries.push(follow_up);
        }
    }

    let synthesis = synthesize_research(question, &rounds);

    ResearchResult {
        question: question.to_string(),
        depth,
        rounds,
        synthesis,
    }
}

fn synthesize_research(question: &str, rounds: &[ResearchRound]) -> String {
    let mut out = format!("Research on: {}\n", question);
    for round in rounds {
        out.push_str(&format!("\nRound {} (query: {}):\n", round.round, round.query));
        for finding in &round.findings {
            out.push_str(&format!("  - {}: {}\n", finding.title, finding.url));
            out.push_str(&format!("    {}\n", finding.excerpt.chars().take(200).collect::<String>()));
        }
    }
    out
}
