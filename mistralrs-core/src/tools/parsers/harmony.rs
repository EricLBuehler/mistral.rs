//! Harmony (GPT-OSS) tool call grammar.
//!
//! Unlike other formats, Harmony tool calls are detected at the token level
//! by the `HarmonyContext` reasoning parser — not by text prefixes.  This
//! module only provides the grammar for constraining the JSON argument body
//! once the reasoning parser signals that a tool call has started.

use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::json;

/// Build a grammar that constrains generation to a valid JSON object.
/// The tool name is already known from the Harmony recipient field, so
/// only the arguments body needs constraining.
pub fn tool_call_grammar() -> TopLevelGrammar {
    let json_body = GrammarWithLexer {
        json_schema: Some(json!({"type": "object"})),
        ..Default::default()
    };
    TopLevelGrammar {
        grammars: vec![json_body],
        max_tokens: None,
    }
}
