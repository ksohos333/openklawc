//! System prompts for the agent loop.

use std::fmt::Write;

/// Build a system prompt that instructs the LLM on how to use tools.
pub fn build_agent_prompt(tool_descriptions: &str, cwd: &str, autonomy_level: &str) -> String {
    let mut prompt = String::new();

    prompt.push_str("You are openklawc, an autonomous coding agent running in a Rust TUI.\n\n");

    prompt.push_str("TOOL USAGE:\n");
    prompt.push_str("When you need to use a tool, output ONE standalone JSON block per line like:\n");
    prompt.push_str(r#"{"tool": "tool_name", "arguments": {"param": "value"}}"#);
    prompt.push('\n');
    prompt.push_str("Do NOT wrap tool calls in markdown code fences. Place the JSON on its own line.\n");
    prompt.push_str("After each tool result is returned, you will get a chance to continue.\n");
    prompt.push_str("When you have enough information to answer, stop calling tools and respond normally.\n\n");

    prompt.push_str(&format!("AUTONOMY LEVEL: {}\n", autonomy_level));
    prompt.push_str("At this autonomy level, the agent may execute tools without asking.\n\n");

    if !tool_descriptions.is_empty() {
        let _ = write!(prompt, "AVAILABLE TOOLS:\n{}\n", tool_descriptions);
    }

    prompt.push_str(&format!("\nCURRENT WORKING DIRECTORY: {}\n", cwd));
    prompt.push_str("\nRespond concisely. Think step by step. Call tools only when necessary.");

    prompt
}

/// Format a list of tool schemas into a human-readable description block.
pub fn format_tool_descriptions(tools: &[okc_core::ToolSchema]) -> String {
    let mut out = String::new();
    for tool in tools {
        let _ = writeln!(out, "\n### {}", tool.name);
        let _ = writeln!(out, "{}", tool.description);
        let schema_str = serde_json::to_string_pretty(&tool.parameters).unwrap_or_default();
        let _ = writeln!(out, "  Parameters: {}", schema_str);
    }
    out
}
