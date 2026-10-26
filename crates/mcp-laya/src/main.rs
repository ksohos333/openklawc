//! mcp-laya — Rust MCP server exposing the Laya typed-decision tools.
//!
//! A Rust port of the `laya-mcp-server` tools that opencode uses. The server
//! answers over the TypeSafe `/v1/systemone` wire protocol from a running
//! Laya / Ollaya daemon, so no model weights are loaded in this process.
//!
//! Tools:
//!   laya_decide(state, schema, model?, min_confidence?)
//!   laya_route(state, questions, task?, lang?, lang_guess?)
//!   laya_route_batch(requests, hooks_timeout?)
//!   laya_status()
//!
//! Environment:
//!   LAYA_BASE_URL  default http://127.0.0.1:11435
//!   LAYA_API_KEY   optional bearer token
//!   LAYA_MODEL     default model name, default "laya"

use std::collections::BTreeMap;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use reqwest::Client;
use serde_json::{json, Map, Value};

/// Default endpoint of the Laya / Ollaya daemon.
const DEFAULT_BASE_URL: &str = "http://127.0.0.1:11435";
/// Default model asked for when the caller does not name one.
const DEFAULT_MODEL: &str = "laya";

fn base_url() -> String {
    std::env::var("LAYA_BASE_URL")
        .or_else(|_| std::env::var("LAYA_URL"))
        .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
        .trim_end_matches('/')
        .to_string()
}

fn default_model() -> String {
    std::env::var("LAYA_DEFAULT_MODEL")
        .or_else(|_| std::env::var("LAYA_MODEL"))
        .unwrap_or_else(|_| DEFAULT_MODEL.to_string())
}

/// Client for the TypeSafe HTTP API.
#[derive(Clone)]
struct Laya {
    http: Client,
    base_url: String,
    model: String,
    api_key: Option<String>,
}

impl Laya {
    fn new() -> Result<Self> {
        let http = Client::builder()
            .timeout(Duration::from_secs(300))
            .build()?;
        Ok(Self {
            http,
            base_url: base_url(),
            model: default_model(),
            api_key: std::env::var("LAYA_API_KEY").ok().filter(|k| !k.is_empty()),
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let url = format!("{}{}", self.base_url, path);
        let mut builder = self.http.request(method, &url);
        if let Some(key) = &self.api_key {
            builder = builder.bearer_auth(key);
        }
        builder
    }

    /// Answer a set of questions about a state.
    async fn systemone(&self, body: Value) -> Result<Value> {
        let response = self
            .request(reqwest::Method::POST, "/v1/systemone")
            .json(&body)
            .send()
            .await
            .with_context(|| format!("requesting {}/v1/systemone", self.base_url))?;

        let status = response.status();
        if !status.is_success() {
            let detail = response.text().await.unwrap_or_default();
            anyhow::bail!("laya-serve error {}: {}", status.as_u16(), detail);
        }
        response
            .json::<Value>()
            .await
            .context("parsing laya response")
    }

    /// Report the daemon's health and the models it knows.
    async fn status(&self) -> Result<Value> {
        let healthy = match self.request(reqwest::Method::GET, "/health").send().await {
            Ok(response) => response.status().is_success(),
            Err(_) => false,
        };

        let mut models: Vec<String> = Vec::new();
        if let Ok(response) = self
            .request(reqwest::Method::GET, "/v1/models")
            .send()
            .await
        {
            if response.status().is_success() {
                if let Ok(body) = response.json::<Value>().await {
                    if let Some(items) = body.get("models").and_then(|m| m.as_array()) {
                        for item in items {
                            let id = item
                                .get("id")
                                .or_else(|| item.get("name"))
                                .or_else(|| item.get("model"))
                                .and_then(|v| v.as_str());
                            if let Some(id) = id {
                                models.push(id.to_string());
                            }
                        }
                    }
                }
            }
        }

        Ok(json!({
            "healthy": healthy,
            "base_url": self.base_url,
            "default_model": self.model,
            "models": models,
        }))
    }
}

/// One question in TypeSafe's wire form.
fn question_json(kind: &str, instructions: &str, criteria: Value) -> Value {
    json!({
        "type": kind,
        "instructions": instructions,
        "criteria": criteria,
    })
}

/// Turn a JSON schema into TypeSafe questions.
///
/// `enum` properties become choice questions, bounded integers become score
/// questions, booleans become yes/no questions. Free-form string properties
/// are not decisions and are skipped.
fn schema_to_questions(schema: &Value) -> Result<(Map<String, Value>, Vec<String>)> {
    let properties = schema
        .get("properties")
        .and_then(|p| p.as_object())
        .ok_or_else(|| anyhow!("schema must have a 'properties' object"))?;

    let mut questions = Map::new();
    let mut decided = Vec::new();

    for (name, property) in properties {
        let instructions = property
            .get("description")
            .and_then(|v| v.as_str())
            .or_else(|| property.get("title").and_then(|v| v.as_str()))
            .unwrap_or(name.as_str())
            .to_string();

        if let Some(options) = property.get("enum").and_then(|e| e.as_array()) {
            let labels: Vec<String> = options
                .iter()
                .filter_map(|v| v.as_str().map(|s| s.to_string()))
                .collect();
            if labels.is_empty() {
                continue;
            }
            let mut criteria = Map::new();
            for label in &labels {
                criteria.insert(label.clone(), json!(label));
            }
            questions.insert(
                name.clone(),
                question_json("choice", &instructions, Value::Object(criteria)),
            );
            decided.push(name.clone());
            continue;
        }

        let property_type = property.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match property_type {
            "boolean" => {
                questions.insert(
                    name.clone(),
                    question_json("noul", &instructions, json!({"true": "yes", "false": "no"})),
                );
                decided.push(name.clone());
            }
            "integer" | "number" => {
                let min = property
                    .get("minimum")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);
                let max = property
                    .get("maximum")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(5.0);
                if max <= min {
                    anyhow::bail!("property '{}' needs a maximum above its minimum", name);
                }
                let steps = property
                    .get("max_levels")
                    .and_then(|v| v.as_u64())
                    .unwrap_or_else(|| ((max - min).min(10.0) as u64) + 1);
                let levels: Vec<Value> = (0..steps)
                    .map(|i| {
                        let value = min + (max - min) * (i as f64) / ((steps - 1).max(1) as f64);
                        json!(format!("{:.2}", value))
                    })
                    .collect();
                questions.insert(
                    name.clone(),
                    question_json("score", &instructions, Value::Array(levels)),
                );
                decided.push(name.clone());
            }
            _ => {}
        }
    }

    if questions.is_empty() {
        anyhow::bail!("schema has no decision-shaped properties (enum, boolean or bounded number)");
    }

    Ok((questions, decided))
}

/// Project the answers back onto the schema.
fn project_answers(
    answers: &Map<String, Value>,
    schema: &Value,
    decided: &[String],
    min_confidence: Option<f64>,
) -> (Value, Value, Value) {
    let mut values = Map::new();
    let mut confidence = Map::new();
    let mut probabilities = Map::new();

    for name in decided {
        let Some(answer) = answers.get(name) else {
            values.insert(name.clone(), Value::Null);
            continue;
        };

        let kind = answer.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let value = match kind {
            "choice" => answer
                .get("choice")
                .and_then(|v| v.as_str())
                .map(|s| json!(s))
                .unwrap_or(Value::Null),
            "score" => {
                let raw = answer.get("score").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let property_type = schema
                    .get("properties")
                    .and_then(|p| p.get(name))
                    .and_then(|p| p.get("type"))
                    .and_then(|v| v.as_str())
                    .unwrap_or("number");
                if property_type == "integer" {
                    json!(raw.round() as i64)
                } else {
                    json!(raw)
                }
            }
            "noul" => {
                let probability = answer.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.0);
                json!(probability >= 0.5)
            }
            _ => Value::Null,
        };

        let answer_confidence =
            answer
                .get("confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(match kind {
                    "noul" => answer.get("noul").and_then(|v| v.as_f64()).unwrap_or(0.5),
                    _ => 0.5,
                });

        confidence.insert(name.clone(), json!(answer_confidence));
        if let Some(probs) = answer.get("probabilities") {
            probabilities.insert(name.clone(), probs.clone());
        }

        let below_threshold = min_confidence
            .map(|min| answer_confidence < min)
            .unwrap_or(false);

        values.insert(
            name.clone(),
            if below_threshold { Value::Null } else { value },
        );
    }

    (
        Value::Object(values),
        Value::Object(confidence),
        Value::Object(probabilities),
    )
}

/// Guess the language of a state from its script.
fn detect_language(state: &str) -> (&'static str, bool) {
    let mut non_latin = false;
    let mut latin = false;
    for ch in state.chars() {
        if !ch.is_alphabetic() {
            continue;
        }
        if ch.is_ascii() || ('\u{0400}'..='\u{04FF}').contains(&ch) {
            latin = true;
        } else {
            non_latin = true;
        }
    }
    if non_latin {
        ("multilingual", true)
    } else {
        ("english", latin)
    }
}

/// Decide which checkpoint would answer a request.
fn route_state(
    state: &str,
    task: Option<&str>,
    lang: Option<&str>,
    lang_guess: Option<&str>,
    model: Option<&str>,
) -> Value {
    if let Some(model) = model.filter(|m| !m.is_empty() && *m != "auto") {
        return json!({
            "model": model,
            "route": "explicit",
            "reason": "model pinned by the caller",
        });
    }
    if let Some(task) = task.filter(|t| !t.is_empty()) {
        return json!({
            "model": task,
            "route": "typed_decisions",
            "reason": "task pinned by the caller",
        });
    }
    if let Some(lang) = lang.filter(|l| !l.is_empty()) {
        let english = lang.to_ascii_lowercase().starts_with("en");
        return json!({
            "model": if english { "laya:en" } else { "laya:multilingual" },
            "route": if english { "english" } else { "multilingual" },
            "reason": format!("lang={} pinned by the caller", lang),
        });
    }
    let (route, has_text) = detect_language(state);
    if let Some(guess) = lang_guess.filter(|g| !g.is_empty()) {
        return json!({
            "model": if guess.to_ascii_lowercase().starts_with("en") {
                "laya:en"
            } else {
                "laya:multilingual"
            },
            "route": guess,
            "reason": "lang_guess used for routing",
        });
    }
    json!({
        "model": if route == "english" { "laya:en" } else { "laya:multilingual" },
        "route": route,
        "reason": if has_text {
            "detected from the text of the state"
        } else {
            "no textual evidence — falling back to the default model"
        },
    })
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

    let laya = Laya::new()?;
    let mut server = okc_mcp::McpServer::new("laya");

    // ─── laya_decide ──────────────────────────────────────────────────────
    let decide_client = laya.clone();
    server.add_tool(
        "laya_decide",
        "Answer a JSON-schema-shaped decision and return the decided values. The schema's \
         enum, bounded-integer and boolean properties become Laya questions, answered in one \
         forward pass and projected back onto the schema. Use for structured decisions only — \
         not for open Q&A, summarization, code or multi-hop reasoning.",
        json!({
            "type": "object",
            "properties": {
                "state": { "type": "string", "description": "The state / context to decide on" },
                "schema": { "type": "object", "description": "JSON schema with enum, boolean or bounded-number properties" },
                "model": { "type": "string", "default": "auto", "description": "null or 'auto' lets the router pick; anything else names a checkpoint" },
                "min_confidence": { "type": "number", "description": "answers below this confidence come back as null" }
            },
            "required": ["state", "schema"]
        }),
        Box::new(move |args| {
            let laya = decide_client.clone();
            Box::pin(async move {
                let state = args["state"]
                    .as_str()
                    .ok_or_else(|| anyhow!("state required"))?;
                let schema = args["schema"]
                    .as_object()
                    .ok_or_else(|| anyhow!("schema required"))?;
                let model = args["model"].as_str().filter(|m| !m.is_empty());
                let min_confidence = args["min_confidence"].as_f64();

                let (questions, decided) = schema_to_questions(&Value::Object(schema.clone()))?;

                let mut body = json!({
                    "model": model.unwrap_or("auto"),
                    "state": state,
                    "questions": Value::Object(questions),
                });
                if let Some(min) = min_confidence {
                    body["min_confidence"] = json!(min);
                }

                let response = laya.systemone(body).await?;
                let answers = response
                    .get("answers")
                    .and_then(|a| a.as_object())
                    .cloned()
                    .unwrap_or_default();
                let (values, confidence, probabilities) =
                    project_answers(&answers, &Value::Object(schema.clone()), &decided, min_confidence);

                Ok(json!({
                    "values": values,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "model": response.get("model").cloned().unwrap_or(Value::Null),
                    "usage": response.get("usage").cloned().unwrap_or(Value::Null),
                }))
            })
        }),
    );

    // ─── laya_route ───────────────────────────────────────────────────────
    let route_client = laya.clone();
    server.add_tool(
        "laya_route",
        "Decide which Laya checkpoint would answer a request, without answering it. \
         'task' names a checkpoint by what the work is, 'lang' forces a language, and \
         'lang_guess' is a soft language that participates in routing.",
        json!({
            "type": "object",
            "properties": {
                "state": { "type": "string" },
                "questions": { "type": "object", "description": "The questions that would be asked" },
                "task": { "type": "string", "description": "e.g. 'typed_decisions'" },
                "lang": { "type": "string", "description": "language code, e.g. 'de' or 'en-US'" },
                "lang_guess": { "type": "string", "description": "a probable-but-uncertain language" }
            },
            "required": ["state"]
        }),
        Box::new(move |args| {
            let _ = &route_client;
            Box::pin(async move {
                let state = args["state"]
                    .as_str()
                    .ok_or_else(|| anyhow!("state required"))?;
                let task = args["task"].as_str();
                let lang = args["lang"].as_str();
                let lang_guess = args["lang_guess"].as_str();
                let model = args["model"].as_str();
                Ok(json!(route_state(state, task, lang, lang_guess, model)))
            })
        }),
    );

    // ─── laya_route_batch ─────────────────────────────────────────────────
    server.add_tool(
        "laya_route_batch",
        "Route several requests at once. Each request carries its own state, task, lang or \
         lang_guess. The answers come back in input order.",
        json!({
            "type": "object",
            "properties": {
                "requests": { "type": "array", "items": { "type": "object" } },
                "hooks_timeout": { "type": "number", "description": "accepted and ignored — hooks run on the server" }
            },
            "required": ["requests"]
        }),
        Box::new(move |args| {
            Box::pin(async move {
                let requests = args["requests"]
                    .as_array()
                    .ok_or_else(|| anyhow!("requests required"))?;
                let mut results: Vec<Value> = Vec::new();
                for request in requests {
                    let state = request
                        .get("state")
                        .and_then(|v| v.as_str())
                        .unwrap_or_default();
                    let route = route_state(
                        state,
                        request.get("task").and_then(|v| v.as_str()),
                        request.get("lang").and_then(|v| v.as_str()),
                        request.get("lang_guess").and_then(|v| v.as_str()),
                        request.get("model").and_then(|v| v.as_str()),
                    );
                    results.push(route);
                }
                Ok(json!(results))
            })
        }),
    );

    // ─── laya_status ──────────────────────────────────────────────────────
    let status_client = laya.clone();
    server.add_tool(
        "laya_status",
        "Report the Laya daemon's health, endpoint and known models. Does not decide anything.",
        json!({ "type": "object", "properties": {} }),
        Box::new(move |_args| {
            let laya = status_client.clone();
            Box::pin(async move { Ok(json!(laya.status().await?)) })
        }),
    );

    // Keep BTreeMap in the import list meaningful for future ordered output.
    let _ = BTreeMap::<String, String>::new();

    server.run().await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enum_becomes_a_choice_question() {
        let schema = json!({
            "type": "object",
            "properties": {
                "colour": { "type": "string", "enum": ["red", "green"] },
                "note": { "type": "string" }
            }
        });
        let (questions, decided) = schema_to_questions(&schema).unwrap();
        assert_eq!(questions.len(), 1);
        assert_eq!(decided, vec!["colour".to_string()]);
        let question = &questions["colour"];
        assert_eq!(question["type"], "choice");
        assert!(question["criteria"]["red"].is_string());
    }

    #[test]
    fn boolean_becomes_a_noul_question() {
        let schema = json!({
            "type": "object",
            "properties": { "ok": { "type": "boolean" } }
        });
        let (questions, _) = schema_to_questions(&schema).unwrap();
        assert_eq!(questions["ok"]["type"], "noul");
    }

    #[test]
    fn bounded_number_becomes_a_score_question() {
        let schema = json!({
            "type": "object",
            "properties": { "risk": { "type": "integer", "minimum": 0, "maximum": 4 } }
        });
        let (questions, _) = schema_to_questions(&schema).unwrap();
        assert_eq!(questions["risk"]["type"], "score");
        assert_eq!(questions["risk"]["criteria"].as_array().unwrap().len(), 5);
    }

    #[test]
    fn answers_project_onto_the_schema() {
        let schema = json!({
            "type": "object",
            "properties": {
                "colour": { "type": "string", "enum": ["red", "green"] },
                "ok": { "type": "boolean" }
            }
        });
        let answers = json!({
            "colour": {"type": "choice", "choice": "green", "confidence": 0.9},
            "ok": {"type": "noul", "noul": 0.8}
        })
        .as_object()
        .unwrap()
        .clone();
        let (values, confidence, _) = project_answers(
            &answers,
            &schema,
            &["colour".to_string(), "ok".to_string()],
            None,
        );
        assert_eq!(values["colour"], json!("green"));
        assert_eq!(values["ok"], json!(true));
        assert_eq!(confidence["colour"], json!(0.9));
    }

    #[test]
    fn low_confidence_answers_become_null() {
        let schema = json!({
            "type": "object",
            "properties": { "colour": { "type": "string", "enum": ["red", "green"] } }
        });
        let answers = json!({
            "colour": {"type": "choice", "choice": "green", "confidence": 0.2}
        })
        .as_object()
        .unwrap()
        .clone();
        let (values, _, _) = project_answers(&answers, &schema, &["colour".to_string()], Some(0.5));
        assert_eq!(values["colour"], Value::Null);
    }

    #[test]
    fn routing_follows_the_script() {
        let route = route_state("hello world", None, None, None, None);
        assert_eq!(route["route"], "english");
        let route = route_state("こんにちは世界", None, None, None, None);
        assert_eq!(route["route"], "multilingual");
    }

    #[test]
    fn routing_honours_an_explicit_model() {
        let route = route_state("hello", None, None, None, Some("winnow:e4b"));
        assert_eq!(route["model"], "winnow:e4b");
        assert_eq!(route["route"], "explicit");
    }
}
