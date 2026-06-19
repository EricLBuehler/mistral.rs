//! DeepSeek tool call parser.
//!
//! Format:
//! ~~~text
//! <｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME
//! ```json
//! {"key": "value"}
//! ```
//! <｜tool▁call▁end｜>
//! ~~~

use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use regex::Regex;
use serde_json::{json, Value};
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct DeepSeekParser;

impl ToolFormatParser for DeepSeekParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<｜tool▁call▁begin｜>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::DeepSeek
    }

    /// Grammar activates after the ` ```json\n` fence.  Covers the JSON
    /// arguments object, closing ` ``` ` fence, and `<｜tool▁call▁end｜>`
    /// end delimiter (matched as a special token via bare angle-bracket
    /// syntax).
    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar {
        let args_schema = extract_deepseek_tool_name(text)
            .and_then(|name| tools.iter().find(|t| t.function.name == name))
            .and_then(|t| t.function.strict_parameters_schema())
            .unwrap_or_else(|| json!({"type": "object"}));

        let lark = r#"start: @json_body "\n```\n" <｜tool▁call▁end｜>"#.to_string();
        let top = GrammarWithLexer::from_lark(lark);
        let json_body = GrammarWithLexer {
            name: Some("json_body".to_string()),
            json_schema: Some(args_schema),
            ..Default::default()
        };
        TopLevelGrammar {
            grammars: vec![top, json_body],
            max_tokens: None,
        }
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        let mut branches = Vec::new();
        let mut rules = Vec::new();
        let mut grammars = Vec::new();

        for (idx, tool) in tools.iter().enumerate() {
            let branch = format!("deepseek_tool_{idx}");
            let json_body = format!("json_body_{idx}");
            let tool_name = format!("{:?}", tool.function.name);
            rules.push(format!(
                r#"{branch}: <｜tool▁call▁begin｜> "function" <｜tool▁sep｜> {tool_name} "\n```json\n" @{json_body} "\n```\n" <｜tool▁call▁end｜>"#
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

        if branches.is_empty() {
            branches.push("deepseek_tool".to_string());
            rules.push(
                r#"deepseek_tool: <｜tool▁call▁begin｜> "function" <｜tool▁sep｜> /[a-zA-Z_][a-zA-Z0-9_]*/ "\n```json\n" @json_body "\n```\n" <｜tool▁call▁end｜>"#
                    .to_string(),
            );
            grammars.push(GrammarWithLexer {
                name: Some("json_body".to_string()),
                json_schema: Some(json!({"type": "object"})),
                ..Default::default()
            });
        }

        let lark = format!("start: {}\n{}", branches.join(" | "), rules.join("\n"));
        let top = GrammarWithLexer::from_lark(lark);
        let mut all_grammars = vec![top];
        all_grammars.extend(grammars);
        TopLevelGrammar {
            grammars: all_grammars,
            max_tokens: None,
        }
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let re = DEEPSEEK_REGEX.get_or_init(|| {
            Regex::new(
                r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
            )
            .unwrap()
        });

        if !re.is_match(message) {
            return Ok(None);
        }

        #[derive(serde::Serialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }

        let mut calls = Vec::new();
        for caps in re.captures_iter(message) {
            let name = caps
                .name("name")
                .ok_or("Could not capture function name")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim()
                .to_string();
            let json_str = caps
                .name("json")
                .ok_or("Could not capture JSON arguments")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim();
            let arguments: Value =
                serde_json::from_str(json_str).map_err(candle_core::Error::msg)?;
            calls.push(ToolCall { name, arguments });
        }

        Ok(Some(
            serde_json::to_string(&calls).map_err(candle_core::Error::msg)?,
        ))
    }
}

/// Extract the tool name from a DeepSeek prefix.
/// Pattern: `<｜tool▁call▁begin｜>function<｜tool▁sep｜>NAME\n```json\n`
fn extract_deepseek_tool_name(text: &str) -> Option<&str> {
    let sep = "<｜tool▁sep｜>";
    let sep_pos = text.rfind(sep)?;
    let after_sep = &text[sep_pos + sep.len()..];
    let name_end = after_sep.find('\n')?;
    Some(after_sep[..name_end].trim())
}
