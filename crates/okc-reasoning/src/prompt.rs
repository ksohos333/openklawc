//! System prompts for the agent loop — REASON analytical framework.
//!
//! All prompts embed the REASON methodology:
//!   R — Restate & Reframe
//!   E — Explore Options
//!   A — Analyze Trade-offs
//!   S — Select & Justify
//!   N — Next Steps

use std::fmt::Write;

/// Build a system prompt that instructs the LLM on how to use tools AND reason.
/// Optionally includes project context for workspace-aware behavior.
pub fn build_agent_prompt(
    tool_descriptions: &str,
    cwd: &str,
    autonomy_level: &str,
    project_context: Option<&str>,
) -> String {
    let mut prompt = String::new();

    prompt.push_str("You are openklawc, an autonomous coding agent running in a Rust TUI.\n\n");

    // ─── REASON Framework ─────────────────────────────────────────────────
    prompt.push_str("# Analytical Reasoning Framework\n\n");
    prompt.push_str("Before taking any non-trivial action, apply the REASON method:\n\n");
    prompt.push_str("## R — Restate & Reframe\n");
    prompt.push_str("Restate the user's request in your own words. Identify the core goal, constraints, and success criteria.\n\n");
    prompt.push_str("## E — Explore Options\n");
    prompt.push_str("Generate 2-3 genuinely distinct approaches. These must differ in methodology, not just be superficial variations.\n\n");
    prompt.push_str("## A — Analyze Trade-offs\n");
    prompt.push_str("For each option, evaluate: Correctness, Maintainability, Performance, Risk, Scope (how much code change).\n\n");
    prompt.push_str("## S — Select & Justify\n");
    prompt.push_str("State which approach you choose and WHY, referencing specific trade-offs. Do not assert superiority without evidence.\n\n");
    prompt.push_str("## N — Next Steps\n");
    prompt.push_str("Briefly outline the execution plan.\n\n");

    prompt.push_str("## Output Format\n");
    prompt.push_str("Embed lightweight decision metadata in your reasoning using this format:\n");
    prompt.push_str("  <analysis>\n");
    prompt.push_str("  Situation: [Restated problem]\n");
    prompt.push_str("  Options:\n");
    prompt.push_str("    1. [Approach A] — [one-line description]\n");
    prompt.push_str("    2. [Approach B] — [one-line description]\n");
    prompt.push_str("  Evaluation:\n");
    prompt.push_str("    - Option 1: [pros/cons]\n");
    prompt.push_str("    - Option 2: [pros/cons]\n");
    prompt.push_str("  Decision: [Chosen option] because [specific justification]\n");
    prompt.push_str("  Confidence: [high/medium/low] — [brief rationale]\n");
    prompt.push_str("  KeyTradeoff: [primary concern that decided between top options]\n");
    prompt.push_str("  </analysis>\n\n");

    prompt.push_str("## Constraints\n");
    prompt.push_str("- Do NOT include the <analysis> block in your final user-facing text. Use reasoning parts for analytical content.\n");
    prompt.push_str("- Keep each analysis concise. Over-analysis wastes tokens and user time.\n");
    prompt.push_str("- If a task is trivial (single file edit, obvious fix), skip the full framework and act directly.\n");
    prompt.push_str("- Never use REASON as an excuse for delay. Analysis must be brief and immediately followed by execution.\n\n");

    // ─── Confidence & Backtracking ────────────────────────────────────────
    prompt.push_str("# Confidence & Backtracking\n\n");
    prompt.push_str(
        "- Assign confidence honestly. 'high' means you are certain this is the right approach.\n",
    );
    prompt.push_str(
        "- 'low' confidence should trigger caution: prefer read-only inspection before mutation.\n",
    );
    prompt.push_str(
        "- If a tool call fails or returns unexpected results, DO NOT continue blindly.\n",
    );
    prompt.push_str("  Stop, reflect, and either backtrack to an alternative or ask the user for clarification.\n");
    prompt.push_str("- When backtracking, explicitly state: 'Backtracking: [reason]. Trying [alternative].'\n\n");

    // ─── Communication Discipline ─────────────────────────────────────────
    prompt.push_str("# Communication Discipline\n\n");
    prompt.push_str("- NEVER use filler phrases such as 'let me', 'I will', 'I am going to', or 'now I will'.\n");
    prompt.push_str(
        "- State actions directly: 'Running tests...', 'Fixing error in src/lib.rs...'\n",
    );
    prompt.push_str("- Do not narrate your internal planning process. Only communicate outcomes, progress, or questions.\n\n");

    // ─── Project Context ────────────────────────────────────────────────
    if let Some(ctx) = project_context {
        if !ctx.is_empty() {
            prompt.push_str("# Project Context\n\n");
            prompt.push_str(ctx);
            prompt.push_str("\n\n");
        }
    }

    // ─── Tool Usage ───────────────────────────────────────────────────────
    prompt.push_str("# Tool Usage\n\n");
    prompt
        .push_str("When you need to use a tool, output ONE standalone JSON block per line like:\n");
    prompt.push_str(r#"{"tool": "tool_name", "arguments": {"param": "value"}}"#);
    prompt.push('\n');
    prompt.push_str(
        "Do NOT wrap tool calls in markdown code fences. Place the JSON on its own line.\n",
    );
    prompt.push_str("After each tool result is returned, you will get a chance to continue.\n");
    prompt.push_str(
        "When you have enough information to answer, stop calling tools and respond normally.\n\n",
    );

    prompt.push_str(&format!("AUTONOMY LEVEL: {}\n", autonomy_level));
    prompt.push_str("At this autonomy level, the agent may execute tools without asking.\n\n");

    if !tool_descriptions.is_empty() {
        let _ = write!(prompt, "AVAILABLE TOOLS:\n{}\n", tool_descriptions);
    }

    prompt.push_str(&format!("\nCURRENT WORKING DIRECTORY: {}\n", cwd));
    prompt.push_str("\nRespond concisely. Think step by step. Call tools only when necessary.");

    prompt
}

/// Extract decision metadata from reasoning text.
/// Parses voluntary labels embedded by the LLM during REASON analysis.
pub fn extract_decision_metadata(text: &str) -> okc_core::DecisionMetadata {
    use okc_core::DecisionMetadata;
    let mut meta = DecisionMetadata::default();

    // Confidence: high | medium | low
    if let Some(cap) = regex::Regex::new(r"[Cc]onfidence\s*[:=]\s*(high|medium|low)")
        .ok()
        .and_then(|re| re.captures(text))
    {
        meta.confidence = cap.get(1).map(|m| m.as_str().to_lowercase().to_string());
    }

    // Decision summary
    if let Some(cap) = regex::Regex::new(r"[Dd]ecision\s*[:=]\s*(.+?)(?:\n|$)")
        .ok()
        .and_then(|re| re.captures(text))
    {
        meta.chosen_approach = cap.get(1).map(|m| m.as_str().trim().to_string());
    }

    // Chosen approach (alternative label)
    if meta.chosen_approach.is_none() {
        if let Some(cap) = regex::Regex::new(r"[Cc]hosen[_\s]*[Aa]pproach\s*[:=]\s*(.+?)(?:\n|$)")
            .ok()
            .and_then(|re| re.captures(text))
        {
            meta.chosen_approach = cap.get(1).map(|m| m.as_str().trim().to_string());
        }
    }

    // Alternatives considered
    if let Some(cap) =
        regex::Regex::new(r"[Aa]lternatives[_\s]*[Cc]onsidered\s*[:=]\s*(.+?)(?:\n|$)")
            .ok()
            .and_then(|re| re.captures(text))
    {
        if let Some(m) = cap.get(1) {
            meta.alternatives = Some(
                m.as_str()
                    .split([',', ';'].as_ref())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect(),
            );
        }
    }

    // Key tradeoff
    if let Some(cap) = regex::Regex::new(r"[Kk]ey[_\s]*[Tt]radeoff\s*[:=]\s*(.+?)(?:\n|$)")
        .ok()
        .and_then(|re| re.captures(text))
    {
        meta.key_tradeoff = cap.get(1).map(|m| m.as_str().trim().to_string());
    }

    // Backtrack detection
    if regex::Regex::new(r"[Bb]acktracking[:\s]|backtrack|abandoning.*approach")
        .ok()
        .map(|re| re.is_match(text))
        .unwrap_or(false)
    {
        meta.is_backtrack = true;
    }

    meta
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_metadata_basic() {
        let text = r#"
        <analysis>
        Situation: User wants to refactor the auth module
        Options:
          1. Extract into a standalone crate
          2. Keep inline but reorganize modules
        Evaluation:
          - Option 1: Better separation but more build complexity
          - Option 2: Simpler, less disruption
        Decision: Option 2 because it minimizes risk
        Confidence: high — the module is well-tested
        KeyTradeoff: separation vs disruption
        </analysis>
        "#;

        let meta = extract_decision_metadata(text);
        assert_eq!(meta.confidence, Some("high".to_string()));
        assert_eq!(
            meta.chosen_approach,
            Some("Option 2 because it minimizes risk".to_string())
        );
        assert_eq!(
            meta.key_tradeoff,
            Some("separation vs disruption".to_string())
        );
    }

    #[test]
    fn test_extract_backtrack() {
        let text = "Backtracking: the test suite failed after the refactoring. Trying alternative approach.";
        let meta = extract_decision_metadata(text);
        assert!(meta.is_backtrack);
    }
}
