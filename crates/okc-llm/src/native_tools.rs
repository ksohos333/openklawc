//! Native tool wire-format converters for OpenAI and Anthropic.
//!
//! `okc-core` owns the conceptual `NativeToolDef/Call/Result` types.
//! This module converts them to each provider's wire format and back.

use okc_core::{NativeToolCall, NativeToolDef, NativeToolResult, TurnOutput};
use serde_json::Value;

// ═══════════════════════════════════════════════════════════════════════════════
// OPENAI WIRE FORMAT
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert our tool definitions to OpenAI's `tools` request field.
pub fn to_openai_tools(tools: &[NativeToolDef]) -> Value {
    let defs: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
            })
        })
        .collect();
    serde_json::json!(defs)
}

/// Parse OpenAI `choices[0].message` into TurnOutput.
pub fn parse_openai_message(message: &Value) -> TurnOutput {
    let mut out = TurnOutput {
        text: message
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        tool_calls: Vec::new(),
        usage: None,
    };

    let Some(tool_calls) = message.get("tool_calls").and_then(|v| v.as_array()) else {
        return out;
    };

    for tc in tool_calls {
        let tc_type = tc.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if tc_type != "function" {
            continue;
        }
        let id = tc
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let func = tc.get("function").unwrap_or(&Value::Null);
        let name = func
            .get("name")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        let args_str = func
            .get("arguments")
            .and_then(|v| v.as_str())
            .unwrap_or("{}");
        let arguments = serde_json::from_str(args_str).unwrap_or(Value::Null);
        out.tool_calls.push(NativeToolCall {
            id,
            name,
            arguments,
        });
    }

    out
}

/// Build an OpenAI `role: "tool"` message for a tool result.
pub fn to_openai_tool_message(result: &NativeToolResult) -> Value {
    serde_json::json!({
        "role": "tool",
        "tool_call_id": result.call_id,
        "content": if result.is_error {
            format!("Error: {}", result.content)
        } else {
            result.content.clone()
        },
    })
}

// ═══════════════════════════════════════════════════════════════════════════════
// ANTHROPIC WIRE FORMAT
// ═══════════════════════════════════════════════════════════════════════════════

/// Convert our tool definitions to Anthropic's `tools` request field.
pub fn to_anthropic_tools(tools: &[NativeToolDef]) -> Value {
    let defs: Vec<Value> = tools
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
        })
        .collect();
    serde_json::json!(defs)
}

/// Parse Anthropic `content` blocks into TurnOutput.
pub fn parse_anthropic_content(content: &[Value]) -> TurnOutput {
    let mut out = TurnOutput::default();
    for block in content {
        let ty = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
        match ty {
            "text" => {
                if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                    out.text.push_str(text);
                }
            }
            "tool_use" => {
                let id = block
                    .get("id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let name = block
                    .get("name")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let arguments = block.get("input").cloned().unwrap_or(Value::Null);
                out.tool_calls.push(NativeToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }
    out
}

/// Build multiple Anthropic `tool_result` content blocks.
pub fn to_anthropic_tool_results(results: &[NativeToolResult]) -> Vec<Value> {
    results.iter().map(to_anthropic_tool_result).collect()
}

/// Build a single Anthropic `tool_result` content block.
pub fn to_anthropic_tool_result(result: &NativeToolResult) -> Value {
    let text = if result.is_error {
        format!("Error: {}", result.content)
    } else {
        result.content.clone()
    };
    serde_json::json!({
        "type": "tool_result",
        "tool_use_id": result.call_id,
        "content": [{ "type": "text", "text": text }],
        "is_error": result.is_error,
    })
}
