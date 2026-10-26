//! Multi-agent delegation — parent agents spawn sub-agents for subtasks.
//!
//! Sub-agents run in isolated contexts with limited tools and message history.
//! Their results are aggregated back to the parent agent.

use okc_core::ChatMessage;

use crate::persona::Persona;

/// Result returned by a delegated sub-agent.
#[derive(Debug, Clone)]
pub struct DelegationResult {
    /// The sub-agent's final response text.
    pub text: String,
    /// Messages exchanged during the sub-agent's turn.
    pub messages: Vec<ChatMessage>,
    /// Reasoning log from the sub-agent.
    pub reasoning_log: Vec<okc_core::ReasoningStep>,
    /// Whether the sub-agent completed successfully.
    pub success: bool,
    /// Error message if the sub-agent failed.
    pub error: Option<String>,
}

/// Configuration for a delegated sub-agent.
#[derive(Debug, Clone)]
pub struct DelegationConfig {
    /// Prompt / task description for the sub-agent.
    pub prompt: String,
    /// Optional persona to apply.
    pub persona: Option<Persona>,
    /// Which tools the sub-agent may use (empty = inherit all).
    pub allowed_tools: Vec<String>,
    /// Max iterations for the sub-agent.
    pub max_iterations: usize,
    /// Whether to include parent messages as context.
    pub include_parent_context: bool,
    /// How many recent parent messages to include.
    pub parent_context_messages: usize,
}

impl Default for DelegationConfig {
    fn default() -> Self {
        Self {
            prompt: String::new(),
            persona: None,
            allowed_tools: Vec::new(),
            max_iterations: 5,
            include_parent_context: true,
            parent_context_messages: 3,
        }
    }
}

/// A sub-agent spawned by a parent agent to handle a subtask.
pub struct SubAgent {
    pub agent: crate::Agent,
    pub config: DelegationConfig,
}

impl SubAgent {
    /// Create a new sub-agent from a parent agent and delegation config.
    pub fn from_parent(parent: &crate::Agent, config: DelegationConfig) -> Self {
        let mut agent = crate::Agent::new(
            parent.llm_client.clone(),
            parent.provider.clone(),
            parent.model_id.clone(),
            parent.tools.clone(),
            parent.autonomy,
            parent.event_bus.clone(),
        );
        agent.max_iterations = config.max_iterations;
        agent.config = parent.config.clone();

        // Apply persona if specified, otherwise inherit parent's
        if let Some(ref persona) = config.persona {
            agent = agent.with_persona(persona.clone());
        } else if let Some(ref parent_persona) = parent.persona {
            if parent_persona.can_delegate {
                agent = agent.with_persona(parent_persona.clone());
            }
        }

        SubAgent { agent, config }
    }

    /// Run the sub-agent with optional parent message context.
    pub async fn run(&mut self, parent_messages: &[ChatMessage]) -> DelegationResult {
        let mut messages = Vec::new();

        if self.config.include_parent_context {
            let start = parent_messages
                .len()
                .saturating_sub(self.config.parent_context_messages);
            messages.extend_from_slice(&parent_messages[start..]);
        }

        let result = self.agent.run_turn(&self.config.prompt, messages).await;

        // Extract the final assistant message
        let final_text = result
            .iter()
            .rev()
            .find(|m| m.role == okc_core::Role::Assistant)
            .map(|m| m.content.clone())
            .unwrap_or_default();

        DelegationResult {
            text: final_text,
            messages: result,
            reasoning_log: self.agent.reasoning_log.clone(),
            success: true,
            error: None,
        }
    }
}

/// A registry of active sub-agents managed by a parent.
#[derive(Debug, Clone, Default)]
pub struct DelegationRegistry {
    pub results: Vec<DelegationResult>,
}

impl DelegationRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(&mut self, result: DelegationResult) {
        self.results.push(result);
    }

    /// Aggregate all sub-agent results into a single summary string.
    pub fn summarize(&self) -> String {
        let mut summary = String::from("# Delegated Sub-Agent Results\n\n");
        for (i, result) in self.results.iter().enumerate() {
            summary.push_str(&format!(
                "## Sub-Agent {} ({}):\n{}\n\n",
                i + 1,
                if result.success { "success" } else { "failed" },
                result.text
            ));
        }
        summary
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_delegation_config_default() {
        let cfg = DelegationConfig::default();
        assert_eq!(cfg.max_iterations, 5);
        assert!(cfg.include_parent_context);
        assert_eq!(cfg.parent_context_messages, 3);
    }

    #[test]
    fn test_registry_summarize() {
        let mut reg = DelegationRegistry::new();
        reg.record(DelegationResult {
            text: "Result A".into(),
            messages: vec![],
            reasoning_log: vec![],
            success: true,
            error: None,
        });
        reg.record(DelegationResult {
            text: "Result B".into(),
            messages: vec![],
            reasoning_log: vec![],
            success: true,
            error: None,
        });

        let summary = reg.summarize();
        assert!(summary.contains("Sub-Agent 1"));
        assert!(summary.contains("Result A"));
        assert!(summary.contains("Result B"));
    }
}
