//! Harmony (GPT-OSS) tool call grammar.
//!
//! Unlike other formats, Harmony tool calls are detected at the token level
//! by the `HarmonyContext` reasoning parser — not by text prefixes.  This
//! module only provides the grammar for constraining the JSON argument body
//! once the reasoning parser signals that a tool call has started.

use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::json;

use crate::Tool;

/// Build a grammar for Harmony tool call arguments, optionally using a
/// strict tool's parameters schema when `tool_name` matches a tool with
/// `strict: true`.
pub fn tool_call_grammar_for_tool(
    tool_name: Option<&str>,
    tools: Option<&[Tool]>,
) -> TopLevelGrammar {
    let args_schema = tool_name
        .and_then(|name| {
            let tools = tools?;
            // Harmony recipients use "functions.tool_name" format;
            // strip the prefix when matching against tool definitions.
            let bare_name = name.strip_prefix("functions.").unwrap_or(name);
            tools.iter().find(|t| t.function.name == bare_name)
        })
        .and_then(|t| t.function.strict_parameters_schema())
        .unwrap_or_else(|| json!({"type": "object"}));

    let json_body = GrammarWithLexer {
        json_schema: Some(args_schema),
        ..Default::default()
    };
    TopLevelGrammar {
        grammars: vec![json_body],
        max_tokens: None,
    }
}
