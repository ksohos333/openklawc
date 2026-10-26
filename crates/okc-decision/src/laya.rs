//! Laya / Ollaya HTTP client.
//!
//! Communicates with the Ollaya daemon (default: http://127.0.0.1:11435)
//! using the TypeSafe-compatible `/v1/systemone` endpoint.

use anyhow::{Context, Result};
use reqwest::Client;
use tracing::{debug, error, info};

use crate::{DecisionRequest, DecisionResponse};

/// HTTP client for the Ollaya decision-model daemon.
#[derive(Debug, Clone)]
pub struct LayaClient {
    http: Client,
    base_url: String,
    /// Default model to use if not specified per-request.
    default_model: String,
    /// Timeout for decision requests (these are fast: ~10-100ms).
    timeout_ms: u64,
}

impl Default for LayaClient {
    fn default() -> Self {
        Self::new("http://127.0.0.1:11435", "laya")
    }
}

impl LayaClient {
    pub fn new(base_url: impl Into<String>, default_model: impl Into<String>) -> Self {
        Self {
            http: Client::builder()
                .timeout(std::time::Duration::from_secs(5))
                .build()
                .expect("reqwest client build"),
            base_url: base_url.into(),
            default_model: default_model.into(),
            timeout_ms: 5000,
        }
    }

    /// Set a custom reqwest client (e.g. with custom TLS config).
    pub fn with_http_client(mut self, client: Client) -> Self {
        self.http = client;
        self
    }

    /// Override the default model.
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.default_model = model.into();
        self
    }

    /// Override the request timeout.
    pub fn with_timeout_ms(mut self, ms: u64) -> Self {
        self.timeout_ms = ms;
        self
    }

    /// Check if the Ollaya daemon is reachable.
    pub async fn is_healthy(&self) -> bool {
        let url = format!("{}/health", self.base_url);
        match self.http.get(&url).send().await {
            Ok(resp) => resp.status().is_success(),
            Err(e) => {
                debug!("Ollaya health check failed: {}", e);
                false
            }
        }
    }

    /// List available decision models from the daemon.
    pub async fn list_models(&self) -> Result<Vec<String>> {
        let url = format!("{}/v1/models", self.base_url);
        let resp = self
            .http
            .get(&url)
            .send()
            .await
            .context("failed to list models from ollaya")?;

        if !resp.status().is_success() {
            anyhow::bail!("ollaya returned status {}", resp.status());
        }

        #[derive(serde::Deserialize)]
        struct ModelsResponse {
            models: Vec<ModelInfo>,
        }
        #[derive(serde::Deserialize)]
        struct ModelInfo {
            id: String,
        }

        let body: ModelsResponse = resp
            .json()
            .await
            .context("failed to parse models response")?;

        Ok(body.models.into_iter().map(|m| m.id).collect())
    }

    /// Send a decision request and get calibrated answers.
    pub async fn decide(&self, mut request: DecisionRequest) -> Result<DecisionResponse> {
        if request.model.is_none() {
            request.model = Some(self.default_model.clone());
        }

        let url = format!("{}/v1/systemone", self.base_url);
        debug!(
            "Sending decision request to {} (model={})",
            url,
            request.model.as_ref().unwrap_or(&self.default_model)
        );

        let start = std::time::Instant::now();
        let resp = self
            .http
            .post(&url)
            .json(&request)
            .send()
            .await
            .context("failed to send decision request to ollaya")?;

        let elapsed = start.elapsed();

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            error!("Ollaya returned error status {}: {}", status, body);
            anyhow::bail!("ollaya decision request failed: {} — {}", status, body);
        }

        let mut decision: DecisionResponse = resp
            .json()
            .await
            .context("failed to parse decision response")?;

        decision.elapsed_ms = elapsed.as_millis() as f32;

        info!(
            "Decision received in {:.1}ms (model={})",
            decision.elapsed_ms, decision.model
        );

        Ok(decision)
    }

    /// Convenience: validate a single yes/no question.
    pub async fn ask_yes_no(&self, state: &str, question: &str) -> Result<(bool, f32)> {
        let request = DecisionRequest {
            state: state.to_string(),
            questions: vec![crate::Question::Noul {
                key: "answer".to_string(),
                text: question.to_string(),
            }],
            model: None,
        };

        let response = self.decide(request).await?;
        let answer = response
            .answers
            .get("answer")
            .context("missing answer key")?;
        let yes = answer.value.as_str().unwrap_or("no") == "yes";
        Ok((yes, answer.confidence))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_client_default_url() {
        let client = LayaClient::default();
        assert_eq!(client.base_url, "http://127.0.0.1:11435");
        assert_eq!(client.default_model, "laya");
    }

    #[test]
    fn test_client_custom() {
        let client = LayaClient::new("http://localhost:9999", "winnow:e4b").with_timeout_ms(1000);
        assert_eq!(client.base_url, "http://localhost:9999");
        assert_eq!(client.default_model, "winnow:e4b");
        assert_eq!(client.timeout_ms, 1000);
    }
}
