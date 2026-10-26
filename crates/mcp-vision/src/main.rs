//! mcp-vision — Rust MCP server for local image understanding and audio
//! transcription.
//!
//! A Rust port of `vision-mcp-server.py`. All processing stays local or on the
//! LAN — no cloud, no telemetry.
//!
//! Tools:
//!   vision_analyze(image_path, question=…, backend="local"|"macstudio")
//!   audio_transcribe(audio_path, model_size=…, language=…)
//!
//! Environment:
//!   VISION_LOCAL_URL          default http://localhost:11434
//!   VISION_MACSTUDIO_URL      default http://100.92.174.127:11434
//!   VISION_LOCAL_MODEL        default qwen2.5vl:3b
//!   VISION_MACSTUDIO_MODEL    default huihui-qwen3-vl-235b-abliterated:latest
//!   VISION_MCP_WHISPER_MODEL  default small
//!   OPENCLAWC_STT_MODEL       path to a GGUF model file (overrides the download)
//!   OPENCLAWC_STT_BACKEND     handy (default) | server | cli
//!   WHISPER_SERVER_URL        default http://127.0.0.1:8080

mod stt;

use std::path::Path;
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use base64::Engine;
use reqwest::Client;
use serde_json::{json, Value};

/// The question asked when the caller does not supply one.
const DEFAULT_QUESTION: &str =
    "Describe this image in detail. Note any composition, framing, text legibility or visual-quality issues.";

/// A vision-language backend.
struct Backend {
    name: &'static str,
    url: String,
    model: String,
    timeout: Duration,
}

fn local_backend() -> Backend {
    Backend {
        name: "local",
        url: std::env::var("VISION_LOCAL_URL")
            .unwrap_or_else(|_| "http://localhost:11434".to_string())
            .trim_end_matches('/')
            .to_string(),
        model: std::env::var("VISION_LOCAL_MODEL").unwrap_or_else(|_| "qwen2.5vl:3b".to_string()),
        timeout: Duration::from_secs(600),
    }
}

fn macstudio_backend() -> Backend {
    Backend {
        name: "macstudio",
        url: std::env::var("VISION_MACSTUDIO_URL")
            .unwrap_or_else(|_| "http://100.92.174.127:11434".to_string())
            .trim_end_matches('/')
            .to_string(),
        model: std::env::var("VISION_MACSTUDIO_MODEL")
            .unwrap_or_else(|_| "huihui-qwen3-vl-235b-abliterated:latest".to_string()),
        timeout: Duration::from_secs(1800),
    }
}

/// Analyze an image with a vision-language model.
async fn vision_analyze(
    client: &Client,
    image_path: &str,
    question: &str,
    backend_name: &str,
) -> Result<String> {
    let path = expand_home(image_path);
    if !Path::new(&path).exists() {
        return Ok(format!("ERROR: image not found: {}", path));
    }

    let backend = match backend_name {
        "macstudio" => macstudio_backend(),
        _ => local_backend(),
    };

    let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path))?;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let payload = json!({
        "model": backend.model,
        "stream": false,
        "options": { "temperature": 0.2 },
        "messages": [{
            "role": "user",
            "content": question,
            "images": [encoded],
        }],
    });

    let url = format!("{}/api/chat", backend.url);
    let response = client
        .post(&url)
        .json(&payload)
        .timeout(backend.timeout)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?;

    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_default();
        anyhow::bail!("Ollama error: {}\n{}", status.as_u16(), body);
    }

    let data = response
        .json::<Value>()
        .await
        .context("parsing Ollama response")?;
    let message = data
        .get("message")
        .and_then(|m| m.get("content"))
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    let eval_count = data
        .get("eval_count")
        .map(|v| v.to_string())
        .unwrap_or_else(|| "?".to_string());

    Ok(format!(
        "[backend={} model={} tokens={}]\n{}",
        backend.name, backend.model, eval_count, message
    ))
}

/// One transcribed segment.
struct Segment {
    start: f64,
    end: f64,
    text: String,
}

/// Transcribe an audio or video file.
///
/// Three backends are supported. The default, `handy`, runs transcribe.cpp —
/// the same Rust engine Handy's desktop app uses — against a GGUF model from
/// the `handy-computer` HuggingFace org. `server` talks to a whisper.cpp HTTP
/// server and `cli` shells out to a local `whisper-cli` binary.
async fn audio_transcribe(
    client: &Client,
    audio_path: &str,
    model_size: &str,
    language: &str,
) -> Result<String> {
    let path = expand_home(audio_path);
    if !Path::new(&path).exists() {
        return Ok(format!("ERROR: file not found: {}", path));
    }

    let size = if model_size.is_empty() {
        std::env::var("VISION_MCP_WHISPER_MODEL").unwrap_or_else(|_| "small".to_string())
    } else {
        model_size.to_string()
    };

    let backend = std::env::var("OPENCLAWC_STT_BACKEND")
        .unwrap_or_else(|_| "handy".to_string())
        .trim()
        .to_ascii_lowercase();

    match backend.as_str() {
        "server" => audio_transcribe_server(client, &path, &size, language).await,
        "cli" => Ok(
            local_whisper_cli(&path, &size, language).unwrap_or_else(|| {
                "ERROR: no whisper-cli binary with WHISPER_MODEL found".to_string()
            }),
        ),
        _ => audio_transcribe_handy(client, &path, &size, language).await,
    }
}

/// Transcribe with transcribe.cpp — Handy's engine.
async fn audio_transcribe_handy(
    client: &Client,
    path: &str,
    size: &str,
    language: &str,
) -> Result<String> {
    let (explicit, spec) = stt::requested_model(size)?;
    let model_path = stt::ensure_model(client, &spec, explicit.as_deref()).await?;

    let path = path.to_string();
    let language = language.to_string();
    let model_path_for_task = model_path.clone();

    // The native inference is blocking — keep it off the async runtime.
    let transcript = tokio::task::spawn_blocking(move || {
        let pcm = stt::decode_to_16k_mono(std::path::Path::new(&path))?;
        stt::transcribe_blocking(&model_path_for_task, &pcm, Some(language.as_str()))
    })
    .await
    .context("the transcription task panicked")??;

    let name = spec.file.trim_end_matches(".gguf").to_string();
    Ok(stt::format_transcript(&name, &transcript))
}

/// Transcribe through a whisper.cpp HTTP server.
async fn audio_transcribe_server(
    client: &Client,
    path: &str,
    size: &str,
    language: &str,
) -> Result<String> {
    let server_url = std::env::var("WHISPER_SERVER_URL")
        .unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
        .trim_end_matches('/')
        .to_string();

    let audio = std::fs::read(path).with_context(|| format!("reading {}", path))?;

    // Preferred: a whisper.cpp HTTP server that accepts a multipart upload.
    let part = reqwest::multipart::Part::bytes(audio.clone())
        .file_name(
            Path::new(&path)
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| "audio".to_string()),
        )
        .mime_str("application/octet-stream")?;

    let mut form = reqwest::multipart::Form::new().part("file", part);
    if !language.is_empty() {
        form = form.text("language", language.to_string());
    }
    form = form.text("response_format", "verbose_json".to_string());

    let response = client
        .post(format!("{}/inference", server_url))
        .multipart(form)
        .send()
        .await;

    match response {
        Ok(response) if response.status().is_success() => {
            let data = response.json::<Value>().await.unwrap_or(Value::Null);
            let text = data
                .get("text")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            let segments: Vec<Segment> = data
                .get("segments")
                .and_then(|v| v.as_array())
                .map(|items| {
                    items
                        .iter()
                        .map(|item| Segment {
                            start: item
                                .get("offsets")
                                .and_then(|o| o.get("from"))
                                .and_then(|v| v.as_f64())
                                .unwrap_or_else(|| {
                                    item.get("start").and_then(|v| v.as_f64()).unwrap_or(0.0)
                                })
                                / 1000.0,
                            end: item
                                .get("offsets")
                                .and_then(|o| o.get("to"))
                                .and_then(|v| v.as_f64())
                                .unwrap_or_else(|| {
                                    item.get("end").and_then(|v| v.as_f64()).unwrap_or(0.0)
                                })
                                / 1000.0,
                            text: item
                                .get("text")
                                .and_then(|v| v.as_str())
                                .unwrap_or("")
                                .trim()
                                .to_string(),
                        })
                        .collect()
                })
                .unwrap_or_default();

            let language = data
                .get("language")
                .and_then(|v| v.as_str())
                .unwrap_or(language);
            let duration = data.get("duration").and_then(|v| v.as_f64()).unwrap_or(0.0);

            Ok(format_transcript(
                &format!("whisper-{}", size),
                language,
                1.0,
                duration,
                &segments,
                &text,
            ))
        }
        Ok(response) => {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            anyhow::bail!(
                "whisper server at {} returned {} {}\n{}",
                server_url,
                status.as_u16(),
                status,
                body
            );
        }
        Err(err) => {
            // Fall back to a local whisper.cpp CLI.
            match local_whisper_cli(path, size, language) {
                Some(text) => Ok(text),
                None => Err(anyhow!(
                    "no transcription backend available: the whisper.cpp server at {} is unreachable ({}) \
                     and no whisper-cli/whisper-cpp binary with WHISPER_MODEL was found",
                    server_url,
                    err
                )),
            }
        }
    }
}

/// Run a local whisper.cpp CLI if one is installed.
fn local_whisper_cli(path: &str, size: &str, language: &str) -> Option<String> {
    let model = std::env::var("WHISPER_MODEL").ok()?;
    if model.trim().is_empty() {
        return None;
    }
    for binary in ["whisper-cli", "whisper-cpp", "main"] {
        if which(binary).is_none() {
            continue;
        }
        let mut command = std::process::Command::new(binary);
        command.arg("-m").arg(&model).arg("-f").arg(path);
        if !language.is_empty() {
            command.arg("-l").arg(language);
        }
        command.arg("-oj");
        let output = command.output().ok()?;
        if !output.status.success() {
            continue;
        }
        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut segments = Vec::new();
        let mut in_transcript = false;
        for line in stdout.lines() {
            if line.contains("[00:00.000 -->") {
                in_transcript = true;
                continue;
            }
            if in_transcript {
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    in_transcript = false;
                    continue;
                }
                segments.push(Segment {
                    start: 0.0,
                    end: 0.0,
                    text: trimmed.to_string(),
                });
            }
        }
        let body = segments
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        return Some(format_transcript(
            &format!("whisper-{}", size),
            language,
            1.0,
            0.0,
            &segments,
            &body,
        ));
    }
    None
}

/// Format a transcript the way the upstream server does.
fn format_transcript(
    model: &str,
    language: &str,
    probability: f64,
    duration: f64,
    segments: &[Segment],
    plain_text: &str,
) -> String {
    let body = if segments.is_empty() {
        if plain_text.trim().is_empty() {
            "(no speech detected)".to_string()
        } else {
            plain_text.to_string()
        }
    } else {
        segments
            .iter()
            .map(|s| format!("[{:.2} -> {:.2}] {}", s.start, s.end, s.text))
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "[model={} language={} p={:.2} duration={:.1}s]\n{}",
        model, language, probability, duration, body
    )
}

/// Find a binary on `PATH`.
fn which(binary: &str) -> Option<std::path::PathBuf> {
    let paths = std::env::var_os("PATH")?;
    std::env::split_paths(&paths)
        .map(|dir| dir.join(binary))
        .find(|candidate| candidate.is_file())
}

/// Expand a leading `~`.
fn expand_home(input: &str) -> String {
    if input == "~" {
        return std::env::var("HOME").unwrap_or_else(|_| input.to_string());
    }
    if let Some(rest) = input.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return format!("{}/{}", home, rest);
        }
    }
    input.to_string()
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
        .connect_timeout(Duration::from_secs(10))
        .build()?;

    let mut server = okc_mcp::McpServer::new("vision");

    let analyze_client = client.clone();
    server.add_tool(
        "vision_analyze",
        "Analyze an image with a local vision-language model. backend=\"local\" is fast \
         (seconds); backend=\"macstudio\" uses a much stronger model over the LAN but is \
         slow — reserve it for final review.",
        json!({
            "type": "object",
            "properties": {
                "image_path": { "type": "string" },
                "question": { "type": "string", "default": DEFAULT_QUESTION },
                "backend": { "type": "string", "enum": ["local", "macstudio"], "default": "local" }
            },
            "required": ["image_path"]
        }),
        Box::new(move |args| {
            let client = analyze_client.clone();
            Box::pin(async move {
                let image_path = args["image_path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("image_path required"))?;
                let question = args["question"]
                    .as_str()
                    .filter(|q| !q.is_empty())
                    .unwrap_or(DEFAULT_QUESTION);
                let backend = args["backend"].as_str().unwrap_or("local");
                Ok(json!(
                    vision_analyze(&client, image_path, question, backend).await?
                ))
            })
        }),
    );

    let transcribe_client = client.clone();
    server.add_tool(
        "audio_transcribe",
        "Transcribe an audio or video file locally with whisper.cpp. model_size selects the \
         model (tiny/base/small/medium/large-v3).",
        json!({
            "type": "object",
            "properties": {
                "audio_path": { "type": "string" },
                "model_size": { "type": "string", "default": "" },
                "language": { "type": "string", "default": "" }
            },
            "required": ["audio_path"]
        }),
        Box::new(move |args| {
            let client = transcribe_client.clone();
            Box::pin(async move {
                let audio_path = args["audio_path"]
                    .as_str()
                    .ok_or_else(|| anyhow!("audio_path required"))?;
                let model_size = args["model_size"].as_str().unwrap_or("");
                let language = args["language"].as_str().unwrap_or("");
                Ok(json!(
                    audio_transcribe(&client, audio_path, model_size, language).await?
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
    fn transcript_format_matches_upstream() {
        let segments = vec![Segment {
            start: 0.0,
            end: 1.25,
            text: "hello".into(),
        }];
        let text = format_transcript("whisper-small", "en", 0.97, 3.5, &segments, "");
        assert!(text.starts_with("[model=whisper-small language=en p=0.97 duration=3.5s]"));
        assert!(text.contains("[0.00 -> 1.25] hello"));
    }

    #[test]
    fn empty_transcript_is_reported() {
        let text = format_transcript("whisper-small", "en", 1.0, 0.0, &[], "");
        assert!(text.ends_with("(no speech detected)"));
    }
}
