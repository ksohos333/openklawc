//! System prompts for the autonomy engine.

use okc_core::ToolSchema;

/// Build the system prompt given available tools and autonomy level.
pub fn build_system_prompt(tools: &[ToolSchema], cwd: &str) -> String {
    let mut prompt = String::new();
    prompt.push_str("You are openklawc, an autonomous coding agent running in a Rust TUI.\n");
    prompt.push_str("You have access to tools. When you need to use a tool, output JSON like:\n");
    prompt.push_str(r#"{"tool": "tool_name", "arguments": {...}}"#);
    prompt.push('\n');
    prompt.push_str("After each tool result, reflect and decide the next step.\n\n");
    prompt.push_str(&format!("Current working directory: {}\n", cwd));
    prompt.push_str("Available tools:\n");

    for tool in tools {
        prompt.push_str(&format!("- {}: {}\n", tool.name, tool.description));
    }

    prompt.push_str("\nRespond concisely. Think step by step.");
    prompt
}
