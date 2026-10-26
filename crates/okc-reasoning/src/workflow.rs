//! Workflow engine — scripted agent pipelines.
//!
//! Workflows are declarative multi-step pipelines defined in YAML or TOML.
//! They support sequential steps, parallel branches, and conditional logic.
//!
//! Example YAML workflow:
//! ```yaml
//! name: code_review
//! description: Review a pull request
//! steps:
//!   - name: read_files
//!     type: tool
//!     tool: read_file
//!     arguments:
//!       path: "{pr_files}"
//!   - name: analyze
//!     type: llm
//!     prompt: "Review these files for bugs and style issues: {read_files.result}"
//!     persona: reviewer
//!   - name: comment
//!     type: tool
//!     tool: shell
//!     arguments:
//!       command: "gh pr comment {pr_number} --body '{analyze.result}'"
//! ```

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_yaml::Value as YamlValue;

/// A workflow is a named collection of steps.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Workflow {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub inputs: Vec<WorkflowInput>,
    pub steps: Vec<WorkflowStep>,
}

/// An input parameter the workflow expects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowInput {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub default: Option<String>,
    #[serde(default)]
    pub required: bool,
}

/// A single step in a workflow.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WorkflowStep {
    /// Run a tool.
    Tool {
        name: String,
        tool: String,
        #[serde(default)]
        arguments: HashMap<String, YamlValue>,
        #[serde(default)]
        condition: Option<String>,
    },
    /// Run an LLM prompt.
    Llm {
        name: String,
        prompt: String,
        #[serde(default)]
        persona: Option<String>,
        #[serde(default)]
        model: Option<String>,
        #[serde(default)]
        condition: Option<String>,
    },
    /// Run multiple steps in parallel.
    Parallel {
        name: String,
        steps: Vec<WorkflowStep>,
        #[serde(default)]
        condition: Option<String>,
    },
    /// Pause and wait for human input.
    Human {
        name: String,
        prompt: String,
        #[serde(default)]
        condition: Option<String>,
    },
    /// Delegate to a sub-agent.
    Agent {
        name: String,
        prompt: String,
        #[serde(default)]
        persona: Option<String>,
        #[serde(default)]
        tools: Vec<String>,
        #[serde(default)]
        max_iterations: Option<usize>,
        #[serde(default)]
        condition: Option<String>,
    },
}

impl Workflow {
    /// Load a workflow from a YAML file.
    pub fn from_yaml_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading workflow from {}", path.display()))?;
        Self::from_yaml(&text)
    }

    /// Parse a workflow from a YAML string.
    pub fn from_yaml(text: &str) -> Result<Self> {
        let workflow: Workflow = serde_yaml::from_str(text).context("parsing workflow YAML")?;
        // Validate step names are unique
        let mut names = std::collections::HashSet::new();
        for step in &workflow.steps {
            let name = step_name(step);
            if !names.insert(name) {
                anyhow::bail!("duplicate step name: {}", name);
            }
        }
        Ok(workflow)
    }

    /// Load a workflow from a TOML file.
    pub fn from_toml_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading workflow from {}", path.display()))?;
        Self::from_toml(&text)
    }

    /// Parse a workflow from a TOML string.
    pub fn from_toml(text: &str) -> Result<Self> {
        let workflow: Workflow = toml::from_str(text).context("parsing workflow TOML")?;
        let mut names = std::collections::HashSet::new();
        for step in &workflow.steps {
            let name = step_name(step);
            if !names.insert(name) {
                anyhow::bail!("duplicate step name: {}", name);
            }
        }
        Ok(workflow)
    }
}

/// Get the name of a workflow step.
pub fn step_name(step: &WorkflowStep) -> &str {
    match step {
        WorkflowStep::Tool { name, .. } => name,
        WorkflowStep::Llm { name, .. } => name,
        WorkflowStep::Parallel { name, .. } => name,
        WorkflowStep::Human { name, .. } => name,
        WorkflowStep::Agent { name, .. } => name,
    }
}

/// Execution context for a running workflow.
#[derive(Debug, Clone, Default)]
pub struct WorkflowContext {
    /// User-provided inputs.
    pub inputs: HashMap<String, String>,
    /// Results from executed steps.
    pub results: HashMap<String, String>,
    /// Current step index.
    pub current_step: usize,
    /// Whether the workflow is paused waiting for human input.
    pub paused: bool,
    pub pause_prompt: Option<String>,
    /// Error if the workflow failed.
    pub error: Option<String>,
}

impl WorkflowContext {
    pub fn new(inputs: HashMap<String, String>) -> Self {
        Self {
            inputs,
            results: HashMap::new(),
            current_step: 0,
            paused: false,
            pause_prompt: None,
            error: None,
        }
    }

    /// Replace `{step_name.result}` and `{input_name}` placeholders.
    pub fn interpolate(&self, template: &str) -> String {
        let mut result = template.to_string();
        for (key, val) in &self.inputs {
            result = result.replace(&format!("{{{}}}", key), val);
        }
        for (key, val) in &self.results {
            result = result.replace(&format!("{{{}.result}}", key), val);
        }
        result
    }

    /// Evaluate a condition string (simple truthiness check).
    pub fn evaluate_condition(&self, condition: &str) -> bool {
        let interpolated = self.interpolate(condition);
        !interpolated.is_empty() && interpolated != "false" && interpolated != "0"
    }
}

/// Workflow executor — drives a workflow to completion.
pub struct WorkflowExecutor;

impl WorkflowExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Execute a workflow step (stub — full implementation ties into Agent).
    pub async fn execute_step(
        &self,
        step: &WorkflowStep,
        ctx: &mut WorkflowContext,
    ) -> Result<String> {
        // Evaluate condition if present
        if let Some(cond) = step_condition(step) {
            if !ctx.evaluate_condition(cond) {
                return Ok(String::from("(skipped)"));
            }
        }

        match step {
            WorkflowStep::Tool {
                tool, arguments, ..
            } => {
                let interpolated: HashMap<String, YamlValue> = arguments
                    .iter()
                    .map(|(k, v)| {
                        let s = match v {
                            YamlValue::String(s) => ctx.interpolate(s),
                            _ => serde_yaml::to_string(v)
                                .unwrap_or_else(|_| v.as_str().unwrap_or("").to_string()),
                        };
                        (k.clone(), YamlValue::String(s))
                    })
                    .collect();
                let args_json = serde_json::to_string(&interpolated)?;
                Ok(format!("tool:{}:{}", tool, args_json))
            }
            WorkflowStep::Llm { prompt, .. } => {
                let interpolated = ctx.interpolate(prompt);
                Ok(format!("llm:{}", interpolated))
            }
            WorkflowStep::Parallel { steps, .. } => {
                // In a real implementation, run all sub-steps concurrently
                let mut results = Vec::new();
                for s in steps {
                    results.push(Box::pin(self.execute_step(s, ctx)).await?);
                }
                Ok(results.join("\n---\n"))
            }
            WorkflowStep::Human { prompt, .. } => {
                ctx.paused = true;
                ctx.pause_prompt = Some(ctx.interpolate(prompt));
                Ok(String::from("(paused for human input)"))
            }
            WorkflowStep::Agent { prompt, .. } => {
                let interpolated = ctx.interpolate(prompt);
                Ok(format!("agent:{}", interpolated))
            }
        }
    }
}

impl Default for WorkflowExecutor {
    fn default() -> Self {
        Self::new()
    }
}

/// Get the condition field from any step variant.
fn step_condition(step: &WorkflowStep) -> Option<&String> {
    match step {
        WorkflowStep::Tool { condition, .. } => condition.as_ref(),
        WorkflowStep::Llm { condition, .. } => condition.as_ref(),
        WorkflowStep::Parallel { condition, .. } => condition.as_ref(),
        WorkflowStep::Human { condition, .. } => condition.as_ref(),
        WorkflowStep::Agent { condition, .. } => condition.as_ref(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate() {
        let mut ctx = WorkflowContext::default();
        ctx.inputs.insert("name".into(), "world".into());
        ctx.results.insert("step1".into(), "hello".into());

        assert_eq!(
            ctx.interpolate("{name} says {step1.result}"),
            "world says hello"
        );
    }

    #[test]
    fn test_yaml_workflow() {
        let yaml = r#"
name: test_workflow
description: A test
steps:
  - name: greet
    type: llm
    prompt: "Hello {name}"
  - name: check
    type: tool
    tool: shell
    arguments:
      command: "echo hello"
"#;
        let wf = Workflow::from_yaml(yaml).unwrap();
        assert_eq!(wf.name, "test_workflow");
        assert_eq!(wf.steps.len(), 2);
    }
}
