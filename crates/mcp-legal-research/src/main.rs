//! mcp-legal-research — Legal research MCP server.
//!
//! Tools:
//!   check_legal_status(question, jurisdiction="EU", n=3)
//!   fetch_eu_law(source, identifier, language="EN")
//!   note_legal_finding(title, summary, source_url, jurisdiction, citation)
//!   list_legal_findings(limit=20)

use std::path::PathBuf;

use anyhow::{anyhow, Context, Result};
use chrono::Utc;
use reqwest::header::{HeaderMap, HeaderValue, USER_AGENT};
use scraper::{Html, Selector};
use serde::{Deserialize, Serialize};
use serde_json::json;
use tokio::fs;

const DISCLAIMER: &str = "INFORMATIONAL ONLY. NOT LEGAL ADVICE. This tool retrieves and cites primary or mirror legal sources but does not render a verdict, opinion, or recommendation on the legality of any specific action in any jurisdiction. Consult a qualified lawyer for advice on your specific situation. Sources fetched may be unofficial mirrors.";

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();

    let mut server = okc_mcp::McpServer::new("legal_research");

    // ─── check_legal_status ───────────────────────────────────────────────
    server.add_tool(
        "check_legal_status",
        "Search authoritative legal sources and return observed legal text + citations.",
        json!({
            "type": "object",
            "properties": {
                "question": { "type": "string" },
                "jurisdiction": { "type": "string", "default": "EU" },
                "n": { "type": "integer", "default": 3 }
            },
            "required": ["question"]
        }),
        Box::new(|args| Box::pin(async move {
            let q = args["question"].as_str().ok_or_else(|| anyhow!("question required"))?;
            let jurisdiction = args["jurisdiction"].as_str().unwrap_or("EU");
            let n = args["n"].as_u64().unwrap_or(3) as usize;
            let result = check_legal_status(q, jurisdiction, n).await;
            Ok(json!(result))
        })),
    );

    // ─── fetch_eu_law ─────────────────────────────────────────────────────
    server.add_tool(
        "fetch_eu_law",
        "Fetch a specific legal text from a configured source.",
        json!({
            "type": "object",
            "properties": {
                "source": { "type": "string" },
                "identifier": { "type": "string" },
                "language": { "type": "string", "default": "EN" }
            },
            "required": ["source", "identifier"]
        }),
        Box::new(|args| Box::pin(async move {
            let source = args["source"].as_str().ok_or_else(|| anyhow!("source required"))?;
            let identifier = args["identifier"].as_str().ok_or_else(|| anyhow!("identifier required"))?;
            let language = args["language"].as_str().unwrap_or("EN");
            let result = fetch_eu_law(source, identifier, language).await;
            Ok(json!(result))
        })),
    );

    // ─── note_legal_finding ───────────────────────────────────────────────
    server.add_tool(
        "note_legal_finding",
        "Log a legal finding with provenance. Returns finding_id.",
        json!({
            "type": "object",
            "properties": {
                "title": { "type": "string" },
                "summary": { "type": "string" },
                "source_url": { "type": "string" },
                "jurisdiction": { "type": "string" },
                "citation": { "type": "string" }
            },
            "required": ["title", "summary", "source_url"]
        }),
        Box::new(|args| Box::pin(async move {
            let title = args["title"].as_str().ok_or_else(|| anyhow!("title required"))?;
            let summary = args["summary"].as_str().ok_or_else(|| anyhow!("summary required"))?;
            let url = args["source_url"].as_str().ok_or_else(|| anyhow!("source_url required"))?;
            let jurisdiction = args["jurisdiction"].as_str().unwrap_or("unspecified");
            let citation = args["citation"].as_str().unwrap_or("");
            let result = note_finding(title, summary, url, jurisdiction, citation).await;
            Ok(json!(result))
        })),
    );

    // ─── list_legal_findings ──────────────────────────────────────────────
    server.add_tool(
        "list_legal_findings",
        "List recent legal findings with provenance.",
        json!({
            "type": "object",
            "properties": {
                "limit": { "type": "integer", "default": 20 }
            }
        }),
        Box::new(|args| Box::pin(async move {
            let limit = args["limit"].as_u64().unwrap_or(20) as usize;
            let result = list_findings(limit).await;
            Ok(json!(result))
        })),
    );

    server.run().await
}

// ══════════════════════════════════════════════════════════════════════════
// DATA DIRECTORY
// ══════════════════════════════════════════════════════════════════════════

fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("legal_research")
        .join("data")
}

fn findings_path() -> PathBuf {
    data_dir().join("findings.jsonl")
}

async fn ensure_data_dir() -> Result<()> {
    fs::create_dir_all(data_dir()).await?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════════
// CHECK LEGAL STATUS
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct LegalStatusResult {
    question: String,
    jurisdiction: String,
    disclaimer: String,
    sources: Vec<LegalSource>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

#[derive(Debug, Serialize)]
struct LegalSource {
    url: String,
    title: Option<String>,
    excerpt: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

async fn check_legal_status(question: &str, jurisdiction: &str, n: usize) -> LegalStatusResult {
    let query = format!("{} {} law", question, jurisdiction);
    let search_items = search_ddg(&query).await;

    let mut sources = Vec::new();
    for item in search_items.into_iter().take(n) {
        let source = match fetch_and_extract(&item.url).await {
            Ok((title, text)) => LegalSource {
                url: item.url,
                title,
                excerpt: text.chars().take(2000).collect(),
                error: None,
            },
            Err(e) => LegalSource {
                url: item.url,
                title: None,
                excerpt: String::new(),
                error: Some(e.to_string()),
            },
        };
        sources.push(source);
    }

    LegalStatusResult {
        question: question.to_string(),
        jurisdiction: jurisdiction.to_string(),
        disclaimer: DISCLAIMER.to_string(),
        sources,
        error: None,
    }
}

// ══════════════════════════════════════════════════════════════════════════
// FETCH EU LAW
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize)]
struct EuLawResult {
    source: String,
    identifier: String,
    language: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    title: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    disclaimer: String,
}

async fn fetch_eu_law(source: &str, identifier: &str, language: &str) -> EuLawResult {
    let url = match source {
        "gdpr-info" | "gdpr" => "https://gdpr-info.eu".to_string(),
        "ai-act" | "eu-ai-act" => "https://artificialintelligenceact.eu".to_string(),
        "legislation-gov-uk" | "uk" => format!("https://www.legislation.gov.uk/{}", identifier),
        "eur-lex" => format!(
            "https://eur-lex.europa.eu/legal-content/{}/TXT/?uri=CELEX:{}",
            language, identifier
        ),
        _ => {
            return EuLawResult {
                source: source.to_string(),
                identifier: identifier.to_string(),
                language: language.to_string(),
                title: None,
                text: None,
                error: Some(format!("Unknown source: {}. Try: gdpr-info, ai-act, legislation-gov-uk, eur-lex", source)),
                disclaimer: DISCLAIMER.to_string(),
            };
        }
    };

    let (title, text) = match fetch_and_extract(&url).await {
        Ok((t, txt)) => (t, Some(txt)),
        Err(e) => {
            return EuLawResult {
                source: source.to_string(),
                identifier: identifier.to_string(),
                language: language.to_string(),
                title: None,
                text: None,
                error: Some(e.to_string()),
                disclaimer: DISCLAIMER.to_string(),
            };
        }
    };

    EuLawResult {
        source: source.to_string(),
        identifier: identifier.to_string(),
        language: language.to_string(),
        title,
        text,
        error: None,
        disclaimer: DISCLAIMER.to_string(),
    }
}

// ══════════════════════════════════════════════════════════════════════════
// FINDINGS
// ══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Serialize, Deserialize)]
struct LegalFinding {
    finding_id: String,
    timestamp: String,
    title: String,
    summary: String,
    source_url: String,
    jurisdiction: String,
    citation: String,
}

async fn note_finding(title: &str, summary: &str, url: &str, jurisdiction: &str, citation: &str) -> serde_json::Value {
    let _ = ensure_data_dir().await;
    let id = uuid::Uuid::new_v4().to_string();

    let entry = LegalFinding {
        finding_id: id.clone(),
        timestamp: Utc::now().to_rfc3339(),
        title: title.to_string(),
        summary: summary.to_string(),
        source_url: url.to_string(),
        jurisdiction: jurisdiction.to_string(),
        citation: citation.to_string(),
    };

    let line = match serde_json::to_string(&entry) {
        Ok(l) => l,
        Err(e) => return json!({ "error": e.to_string() }),
    };

    let path = findings_path();
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

    json!({ "finding_id": id, "status": "noted" })
}

async fn list_findings(limit: usize) -> Vec<LegalFinding> {
    let path = findings_path();
    let content = match fs::read_to_string(&path).await {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut findings = Vec::new();
    for line in content.lines() {
        if let Ok(f) = serde_json::from_str::<LegalFinding>(line) {
            findings.push(f);
        }
    }
    findings.reverse();
    findings.truncate(limit);
    findings
}

// ══════════════════════════════════════════════════════════════════════════
// HTTP HELPERS
// ══════════════════════════════════════════════════════════════════════════

async fn fetch_and_extract(url: &str) -> Result<(Option<String>, String)> {
    let client = build_client()?;
    let resp = client
        .get(url)
        .timeout(std::time::Duration::from_secs(25))
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
    headers.insert(USER_AGENT, HeaderValue::from_static(
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    ));
    headers.insert("Accept", HeaderValue::from_static("text/html,*/*"));
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
    for sel in &["article", "main", ".content", "#content", ".legislation"] {
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

    let body = match client.get(&url).timeout(std::time::Duration::from_secs(30)).send().await {
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
            .select(&Selector::parse(".result__title").unwrap_or_else(|_| Selector::parse("h2").unwrap()))
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
