//! Llama tool call parser.
//!
//! Format: `<|python_tag|>{"name":"...", "parameters":{...}}`

use llguidance::api::TopLevelGrammar;

use super::ToolFormatParser;
use crate::Tool;

pub struct LlamaParser;

impl ToolFormatParser for LlamaParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<|python_tag|>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Llama
    }

    fn tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body"#.to_string(),
            tools,
            "parameters",
            false,
        )
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        if let Some(rest) = message.strip_prefix("<|python_tag|>") {
            Ok(Some(rest.to_string()))
        } else {
            Ok(None)
        }
    }
}
