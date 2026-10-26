//! Speech-to-text through transcribe.cpp — the same Rust engine Handy uses.
//!
//! Handy (github.com/cjpais/Handy) runs every model through `transcribe-cpp`,
//! a set of Rust bindings around transcribe.cpp / ggml. This module uses that
//! exact stack: a GGUF model from the `handy-computer` HuggingFace org, audio
//! decoded with rodio and resampled to 16 kHz mono with rubato, and the
//! inference done by the native library (Vulkan on this machine).

use std::path::{Path, PathBuf};

use anyhow::{anyhow, Context, Result};
use rodio::Source;
use rubato::{
    Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction,
};
use transcribe_cpp::{Model, RunOptions, TimestampKind};

/// The sample rate every transcription model expects.
const TARGET_RATE: u32 = 16_000;

/// Frames per resampler chunk.
const CHUNK_SIZE: usize = 1024;

/// A GGUF model on the HuggingFace hub.
#[derive(Debug, Clone, Copy)]
pub struct ModelSpec {
    pub repo: &'static str,
    pub file: &'static str,
}

/// Map a friendly size name onto one of Handy's whisper models.
pub fn model_for_size(size: &str) -> ModelSpec {
    match size.trim().to_ascii_lowercase().as_str() {
        "base" => ModelSpec {
            repo: "handy-computer/whisper-base-gguf",
            file: "whisper-base-Q5_K_M.gguf",
        },
        "small" => ModelSpec {
            repo: "handy-computer/whisper-small-gguf",
            file: "whisper-small-Q5_K_M.gguf",
        },
        "medium" => ModelSpec {
            repo: "handy-computer/whisper-medium-gguf",
            file: "whisper-medium-Q5_K_M.gguf",
        },
        "large" | "large-v3" | "large-v3-turbo" => ModelSpec {
            repo: "handy-computer/whisper-large-v3-turbo-gguf",
            file: "whisper-large-v3-turbo-Q5_K_M.gguf",
        },
        _ => ModelSpec {
            repo: "handy-computer/whisper-tiny-gguf",
            file: "whisper-tiny-Q5_K_M.gguf",
        },
    }
}

/// The model the caller asked for: an explicit path, or a named size.
pub fn requested_model(size: &str) -> Result<(Option<PathBuf>, ModelSpec)> {
    if let Ok(path) = std::env::var("OPENCLAWC_STT_MODEL") {
        let path = PathBuf::from(path.trim());
        if path.as_os_str().is_empty() {
            return Err(anyhow!("OPENCLAWC_STT_MODEL is set but empty"));
        }
        return Ok((Some(path), model_for_size(size)));
    }
    Ok((None, model_for_size(size)))
}

/// Where downloaded models are cached.
fn cache_dir() -> Result<PathBuf> {
    let base = dirs::cache_dir()
        .or_else(dirs::home_dir)
        .ok_or_else(|| anyhow!("no cache directory available"))?;
    Ok(base.join("openklawc").join("stt"))
}

/// The on-disk path of a cached model.
fn cached_path(spec: &ModelSpec) -> Result<PathBuf> {
    let repo_dir = spec.repo.replace('/', "--");
    Ok(cache_dir()?.join(repo_dir).join(spec.file))
}

/// Make sure the model file is on disk, downloading it when it is not.
pub async fn ensure_model(
    http: &reqwest::Client,
    spec: &ModelSpec,
    explicit: Option<&Path>,
) -> Result<PathBuf> {
    if let Some(path) = explicit {
        if !path.is_file() {
            return Err(anyhow!("model file not found: {}", path.display()));
        }
        return Ok(path.to_path_buf());
    }

    let target = cached_path(spec)?;
    if target.is_file() {
        return Ok(target);
    }

    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = pick_gguf(http, spec).await?;
    let url = format!("https://huggingface.co/{}/resolve/main/{}", spec.repo, file);

    tracing::info!("downloading transcription model {} ({})", file, spec.repo);
    let response = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("requesting {}", url))?
        .error_for_status()
        .with_context(|| format!("downloading {}", url))?;

    let temp = target.with_extension("part");
    let mut file_out = tokio::fs::File::create(&temp)
        .await
        .with_context(|| format!("creating {}", temp.display()))?;
    let mut stream = response.bytes_stream();
    use futures_util::StreamExt;
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        tokio::io::AsyncWriteExt::write_all(&mut file_out, &chunk).await?;
    }
    tokio::io::AsyncWriteExt::flush(&mut file_out).await?;
    drop(file_out);
    tokio::fs::rename(&temp, &target).await?;
    Ok(target)
}

/// Pick the GGUF file to download: the requested one when the repo has it,
/// otherwise any GGUF the repo publishes.
async fn pick_gguf(http: &reqwest::Client, spec: &ModelSpec) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{}/tree/main", spec.repo);
    let listing: Vec<serde_json::Value> = http
        .get(&url)
        .send()
        .await
        .with_context(|| format!("listing {}", spec.repo))?
        .error_for_status()?
        .json()
        .await
        .context("parsing the model listing")?;

    let files: Vec<String> = listing
        .iter()
        .filter_map(|item| item.get("path").and_then(|p| p.as_str()))
        .filter(|path| path.ends_with(".gguf"))
        .map(|path| path.to_string())
        .collect();

    if files.is_empty() {
        return Err(anyhow!("{} publishes no GGUF files", spec.repo));
    }
    if files.iter().any(|path| path == spec.file) {
        return Ok(spec.file.to_string());
    }
    // Prefer a quantised build, then whatever is there.
    for needle in ["Q5_K_M", "Q8_0", "Q4_K_M", "F16"] {
        if let Some(found) = files.iter().find(|path| path.contains(needle)) {
            return Ok(found.clone());
        }
    }
    Ok(files[0].clone())
}

/// Decode an audio file into 16 kHz mono samples in [-1, 1].
pub fn decode_to_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let source = rodio::Decoder::new(std::io::BufReader::new(file))
        .with_context(|| format!("decoding {}", path.display()))?;

    let sample_rate = source.sample_rate().get();
    let channels = source.channels().get() as usize;
    let interleaved: Vec<f32> = source.collect();

    // Mix down to mono.
    let mono: Vec<f32> = interleaved
        .chunks(channels)
        .map(|frame| frame.iter().sum::<f32>() / frame.len() as f32)
        .collect();

    if sample_rate == TARGET_RATE {
        return Ok(mono);
    }

    resample(&mono, sample_rate, TARGET_RATE)
}

/// Band-limited resample to the model's sample rate.
fn resample(input: &[f32], from_rate: u32, to_rate: u32) -> Result<Vec<f32>> {
    let ratio = to_rate as f64 / from_rate as f64;
    let parameters = SincInterpolationParameters {
        sinc_len: 128,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Cubic,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(ratio, 1.0, parameters, CHUNK_SIZE, 1)
        .context("creating the resampler")?;

    let mut output: Vec<f32> =
        Vec::with_capacity((input.len() as f64 * ratio) as usize + CHUNK_SIZE);
    for chunk in input.chunks(CHUNK_SIZE) {
        // rubato wants fixed-size chunks; the tail is zero-padded.
        let mut padded = Vec::with_capacity(CHUNK_SIZE);
        padded.extend_from_slice(chunk);
        padded.resize(CHUNK_SIZE, 0.0);
        let produced = resampler
            .process(&[padded.as_slice()], None)
            .map_err(|e| anyhow!("resampling audio: {:?}", e))?;
        for channel in produced {
            output.extend_from_slice(&channel);
        }
    }
    Ok(output)
}

/// One transcribed segment.
pub struct Segment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

/// The result of a transcription.
pub struct Transcript {
    pub text: String,
    pub language: Option<String>,
    pub duration_secs: f64,
    pub segments: Vec<Segment>,
}

/// Run the model over a 16 kHz mono buffer. Blocking — call from a blocking task.
pub fn transcribe_blocking(
    model_path: &Path,
    pcm: &[f32],
    language: Option<&str>,
) -> Result<Transcript> {
    let model = Model::load(model_path)
        .map_err(|e| anyhow!("loading model {}: {}", model_path.display(), e))?;
    let mut session = model
        .session()
        .map_err(|e| anyhow!("opening a session: {}", e))?;

    let options = RunOptions {
        timestamps: TimestampKind::Segment,
        language: language.filter(|l| !l.is_empty()).map(|l| l.to_string()),
        ..RunOptions::default()
    };

    let transcript = session
        .run(pcm, &options)
        .map_err(|e| anyhow!("running the model: {}", e))?;

    let segments = transcript
        .segments
        .iter()
        .map(|segment| Segment {
            start_ms: segment.t0_ms,
            end_ms: segment.t1_ms,
            text: segment.text.trim().to_string(),
        })
        .filter(|segment| !segment.text.is_empty())
        .collect();

    Ok(Transcript {
        text: transcript.text.trim().to_string(),
        language: transcript.language.clone(),
        duration_secs: pcm.len() as f64 / TARGET_RATE as f64,
        segments,
    })
}

/// Format a transcript the way the upstream server does.
pub fn format_transcript(model_name: &str, transcript: &Transcript) -> String {
    let language = transcript.language.as_deref().unwrap_or("und");
    let body = if transcript.segments.is_empty() {
        if transcript.text.is_empty() {
            "(no speech detected)".to_string()
        } else {
            transcript.text.clone()
        }
    } else {
        transcript
            .segments
            .iter()
            .map(|segment| {
                format!(
                    "[{:7.2} -> {:7.2}] {}",
                    segment.start_ms as f64 / 1000.0,
                    segment.end_ms as f64 / 1000.0,
                    segment.text
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };

    format!(
        "[model={} language={} p=1.00 duration={:.1}s]\n{}",
        model_name, language, transcript.duration_secs, body
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sizes_map_to_handy_models() {
        assert_eq!(
            model_for_size("tiny").repo,
            "handy-computer/whisper-tiny-gguf"
        );
        assert_eq!(
            model_for_size("base").repo,
            "handy-computer/whisper-base-gguf"
        );
        assert_eq!(
            model_for_size("large-v3").repo,
            "handy-computer/whisper-large-v3-turbo-gguf"
        );
        // Anything unknown falls back to the tiny model.
        assert_eq!(
            model_for_size("nonsense").repo,
            "handy-computer/whisper-tiny-gguf"
        );
    }

    #[test]
    fn transcript_formatting_matches_upstream() {
        let transcript = Transcript {
            text: "hello there".into(),
            language: Some("en".into()),
            duration_secs: 3.5,
            segments: vec![Segment {
                start_ms: 0,
                end_ms: 1250,
                text: "hello there".into(),
            }],
        };
        let text = format_transcript("whisper-tiny", &transcript);
        assert!(text.starts_with("[model=whisper-tiny language=en p=1.00 duration=3.5s]"));
        assert!(text.contains("[   0.00 ->    1.25] hello there"));
    }

    #[test]
    fn empty_transcripts_are_reported() {
        let transcript = Transcript {
            text: String::new(),
            language: None,
            duration_secs: 0.0,
            segments: Vec::new(),
        };
        let text = format_transcript("whisper-tiny", &transcript);
        assert!(text.ends_with("(no speech detected)"));
        assert!(text.contains("language=und"));
    }
}
