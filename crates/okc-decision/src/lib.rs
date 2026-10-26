//! okc-decision — Decision model integration for openklawc.
//!
//! Provides a fast calibrated decision layer via Laya / Ollaya:
//!   - the LLM proposes an approach
//!   - Laya scores confidence / correctness in ~10ms
//!   - the orchestrator uses the score to proceed, backtrack, or ask the user
//!
//! Laya (via Ollaya daemon) speaks the TypeSafe `/v1/systemone` wire format.
//! Default endpoint: http://127.0.0.1:11435

use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};
use tracing::debug;

pub mod laya;

pub use laya::LayaClient;

/// A decision request sent to the model.
#[derive(Debug, Clone, Serialize)]
pub struct DecisionRequest {
    /// The current state / context (e.g. the user's question + proposed approach).
    pub state: String,
    /// The typed questions to answer.
    pub questions: Vec<Question>,
    /// Optional: which model to use (e.g. "laya", "laya:en", "winnow:e4b").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

/// One typed question for the decision model.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum Question {
    /// Binary / multi-choice question.
    #[serde(rename = "choice")]
    Choice {
        key: String,
        text: String,
        choices: Vec<String>,
    },
    /// Score on a scale (e.g. 0–5).
    #[serde(rename = "score")]
    Score {
        key: String,
        text: String,
        min: f32,
        max: f32,
    },
    /// Yes / no / uncertain.
    #[serde(rename = "noul")]
    Noul { key: String, text: String },
}

/// A decision response from the model.
#[derive(Debug, Clone, Deserialize)]
pub struct DecisionResponse {
    pub answers: HashMap<String, Answer>,
    #[serde(default)]
    pub model: String,
    #[serde(default)]
    pub elapsed_ms: f32,
}

/// One answer from the decision model.
#[derive(Debug, Clone, Deserialize)]
pub struct Answer {
    /// The chosen / highest-confidence option.
    pub value: serde_json::Value,
    /// Confidence score (0.0 – 1.0).
    pub confidence: f32,
    /// All option probabilities (for choice questions).
    #[serde(default)]
    pub distribution: Option<HashMap<String, f32>>,
}

/// Result of a decision validation check.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the proposed approach should proceed.
    pub proceed: bool,
    /// Numerical confidence in the decision (0.0 – 1.0).
    pub confidence: f32,
    /// Human-readable explanation.
    pub explanation: String,
    /// Suggested action if not proceeding.
    pub suggestion: Option<String>,
}

/// High-level decision orchestrator that wraps a LayaClient.
#[derive(Clone)]
pub struct DecisionOrchestrator {
    client: LayaClient,
    /// Confidence threshold below which we should reconsider.
    confidence_threshold: f32,
    /// Whether to enable the decision layer at all.
    enabled: bool,
}

impl DecisionOrchestrator {
    pub fn new(client: LayaClient) -> Self {
        Self {
            client,
            confidence_threshold: 0.7,
            enabled: true,
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    pub fn disabled(mut self) -> Self {
        self.enabled = false;
        self
    }

    /// Validate a proposed tool call plan before execution.
    pub async fn validate_plan(
        &self,
        user_request: &str,
        proposed_tools: &[String],
    ) -> Result<ValidationResult> {
        if !self.enabled {
            return Ok(ValidationResult {
                proceed: true,
                confidence: 1.0,
                explanation: "Decision layer disabled — proceeding.".to_string(),
                suggestion: None,
            });
        }

        debug!("Validating plan with decision layer");

        let state = format!(
            "User request: {}\nProposed tools: {}",
            user_request,
            proposed_tools.join(", ")
        );

        let request = DecisionRequest {
            state,
            questions: vec![
                Question::Choice {
                    key: "correctness".to_string(),
                    text: "Is this approach correct for the user's request?".to_string(),
                    choices: vec![
                        "correct".to_string(),
                        "partially_correct".to_string(),
                        "incorrect".to_string(),
                    ],
                },
                Question::Score {
                    key: "confidence".to_string(),
                    text: "How confident are you that this plan will succeed?".to_string(),
                    min: 0.0,
                    max: 1.0,
                },
                Question::Noul {
                    key: "risks".to_string(),
                    text: "Are there significant risks with this approach?".to_string(),
                },
            ],
            model: None,
        };

        let response = self.client.decide(request).await?;

        let correctness = response
            .answers
            .get("correctness")
            .and_then(|a| a.value.as_str())
            .unwrap_or("unknown");

        let confidence = response
            .answers
            .get("confidence")
            .and_then(|a| a.value.as_f64())
            .map(|v| v as f32)
            .unwrap_or(0.5);

        let has_risks = response
            .answers
            .get("risks")
            .and_then(|a| a.value.as_str())
            .map(|v| v == "yes")
            .unwrap_or(false);

        let proceed = correctness == "correct"
            || (correctness == "partially_correct"
                && confidence >= self.confidence_threshold
                && !has_risks);

        let explanation = format!(
            "Decision model says: {} (confidence: {:.2}, risks: {})",
            correctness,
            confidence,
            if has_risks { "yes" } else { "no" }
        );

        let suggestion = if !proceed {
            Some(
                if has_risks {
                    "Consider gathering more information with read-only tools before proceeding."
                } else {
                    "The plan may be incomplete. Reconsider the approach or ask the user for clarification."
                }
                .to_string(),
            )
        } else {
            None
        };

        Ok(ValidationResult {
            proceed,
            confidence,
            explanation,
            suggestion,
        })
    }

    /// Quick sentiment / intent classification on the user's message.
    pub async fn classify_intent(&self, message: &str) -> Result<IntentClassification> {
        let request = DecisionRequest {
            state: message.to_string(),
            questions: vec![
                Question::Choice {
                    key: "intent".to_string(),
                    text: "What is the user's primary intent?".to_string(),
                    choices: vec![
                        "code_change".to_string(),
                        "information_request".to_string(),
                        "debugging".to_string(),
                        "architecture_discussion".to_string(),
                        "other".to_string(),
                    ],
                },
                Question::Choice {
                    key: "urgency".to_string(),
                    text: "How urgent is this request?".to_string(),
                    choices: vec![
                        "immediate".to_string(),
                        "soon".to_string(),
                        "whenever".to_string(),
                    ],
                },
                Question::Noul {
                    key: "requires_context".to_string(),
                    text: "Does this require reading project files first?".to_string(),
                },
            ],
            model: None,
        };

        let response = self.client.decide(request).await?;

        Ok(IntentClassification {
            intent: response
                .answers
                .get("intent")
                .and_then(|a| a.value.as_str().map(String::from))
                .unwrap_or_else(|| "other".to_string()),
            urgency: response
                .answers
                .get("urgency")
                .and_then(|a| a.value.as_str().map(String::from))
                .unwrap_or_else(|| "whenever".to_string()),
            requires_context: response
                .answers
                .get("requires_context")
                .and_then(|a| a.value.as_str().map(|v| v == "yes"))
                .unwrap_or(true),
            confidence: response
                .answers
                .get("intent")
                .map(|a| a.confidence)
                .unwrap_or(0.5),
        })
    }
}

/// Classified intent of a user message.
#[derive(Debug, Clone)]
pub struct IntentClassification {
    pub intent: String,
    pub urgency: String,
    pub requires_context: bool,
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_request_serialize() {
        let req = DecisionRequest {
            state: "Test state".to_string(),
            questions: vec![Question::Choice {
                key: "test".to_string(),
                text: "Is this a test?".to_string(),
                choices: vec!["yes".to_string(), "no".to_string()],
            }],
            model: Some("laya".to_string()),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("choice"));
        assert!(json.contains("Is this a test?"));
    }

    #[test]
    fn test_validation_result_logic() {
        // When disabled, always proceed
        let orch = DecisionOrchestrator::new(LayaClient::default()).disabled();
        assert!(!orch.enabled);
    }
}
