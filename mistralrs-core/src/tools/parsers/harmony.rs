//! Harmony (GPT-OSS) tool call grammar.
//!
//! Unlike other formats, Harmony tool calls are detected at the token level
//! by the `HarmonyContext` reasoning parser — not by text prefixes.  This
//! module only provides the grammar for constraining the JSON argument body
//! once the reasoning parser signals that a tool call has started.

use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::json;

use crate::Tool;

pub fn required_tool_call_grammar(tools: &[Tool], needs_message_boundary: bool) -> TopLevelGrammar {
    let mut branches = Vec::new();
    let mut rules = Vec::new();
    let mut grammars = Vec::new();

    for (idx, tool) in tools.iter().enumerate() {
        let branch = format!("harmony_tool_{idx}");
        let json_body = format!("json_body_{idx}");
        let recipient = harmony_recipient_for_tool(&tool.function.name);
        let header = format!("commentary to={recipient} ");
        rules.push(format!(
            r#"{branch}: <|channel|> {header:?} <|constrain|> "json" <|message|> @{json_body} <|call|>"#
        ));
        branches.push(branch);
        grammars.push(GrammarWithLexer {
            name: Some(json_body),
            json_schema: Some(
                tool.function
                    .strict_parameters_schema()
                    .unwrap_or_else(|| json!({"type": "object"})),
            ),
            ..Default::default()
        });
    }

    let start = if needs_message_boundary {
        format!(
            r#"start: <|end|> <|start|> "assistant" harmony_tool
harmony_tool: {}"#,
            branches.join(" | ")
        )
    } else {
        format!("start: {}", branches.join(" | "))
    };
    let top = GrammarWithLexer::from_lark(format!("{start}\n{}", rules.join("\n")));
    let mut all_grammars = vec![top];
    all_grammars.extend(grammars);
    TopLevelGrammar {
        grammars: all_grammars,
        max_tokens: None,
    }
}

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

fn harmony_recipient_for_tool(name: &str) -> String {
    if name.starts_with("browser.") || name == "python" {
        name.to_string()
    } else {
        format!("functions.{name}")
    }
}
