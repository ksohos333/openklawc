//! Agent personas / profiles.
//!
//! Personas override the system prompt, restrict tool access, and can
//! change model parameters. They let the same agent runtime behave as
//! an architect, security auditor, tester, etc.

use std::collections::HashSet;

use okc_core::ToolSchema;

/// Runtime persona state.
#[derive(Debug, Clone, Default)]
pub struct Persona {
    /// Name of the active persona.
    pub name: Option<String>,
    /// System prompt fragment to inject.
    pub system_prompt: String,
    /// Tool whitelist (empty = all allowed).
    pub allowed_tools: HashSet<String>,
    /// Optional model override.
    pub model_override: Option<String>,
    /// Temperature override.
    pub temperature: Option<f32>,
    /// Max iterations override.
    pub max_iterations: Option<usize>,
    /// Whether this persona can delegate to sub-agents.
    pub can_delegate: bool,
}

impl Persona {
    /// Build a persona from config.
    pub fn from_config(name: &str, cfg: &okc_config::PersonaConfig) -> Self {
        Self {
            name: Some(name.to_string()),
            system_prompt: cfg.system_prompt.clone(),
            allowed_tools: cfg.allowed_tools.iter().cloned().collect(),
            model_override: cfg.model.clone(),
            temperature: Some(cfg.temperature),
            max_iterations: cfg.max_iterations,
            can_delegate: cfg.can_delegate,
        }
    }

    /// Check if a tool is allowed under this persona.
    pub fn allows_tool(&self, name: &str) -> bool {
        self.allowed_tools.is_empty() || self.allowed_tools.contains(name)
    }

    /// Filter a list of tool schemas to only those allowed.
    pub fn filter_tools<'a>(&self, tools: &'a [ToolSchema]) -> Vec<&'a ToolSchema> {
        if self.allowed_tools.is_empty() {
            tools.iter().collect()
        } else {
            tools
                .iter()
                .filter(|t| self.allowed_tools.contains(&t.name))
                .collect()
        }
    }
}

/// Pre-defined personas bundled with openklawc.
pub fn builtin_personas() -> HashSet<&'static str> {
    let mut set = HashSet::new();
    set.insert("architect");
    set.insert("reviewer");
    set.insert("tester");
    set.insert("security_auditor");
    set
}

/// Return the default system prompt for a built-in persona.
pub fn builtin_persona_prompt(name: &str) -> Option<String> {
    match name {
        "architect" => Some(
            "You are a software architect. Focus on high-level design, trade-offs, and maintainability. \
             Prefer documenting structure over making edits. When proposing changes, consider backwards compatibility, \
             testing strategy, and deployment impact."
                .to_string(),
        ),
        "reviewer" => Some(
            "You are a code reviewer. Be thorough and critical. Look for bugs, performance issues, \
             security vulnerabilities, and style violations. Do NOT write code — only suggest improvements \
             and explain why they matter. Rate each issue as [critical], [warning], or [nit]."
                .to_string(),
        ),
        "tester" => Some(
            "You are a QA engineer. Your goal is comprehensive test coverage. Propose unit tests, \
             integration tests, and edge cases. Prefer property-based testing where applicable. \
             Always run the test suite after making changes."
                .to_string(),
        ),
        "security_auditor" => Some(
            "You are a security auditor. Scan for injection risks, unsafe memory usage, secret leakage, \
             insecure dependencies, and missing auth checks. Flag anything that touches file paths, \
             user input, network requests, or cryptography. Be paranoid — false positives are better \
             than missed vulnerabilities."
                .to_string(),
        ),
        _ => None,
    }
}
