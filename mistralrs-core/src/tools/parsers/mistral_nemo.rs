//! Mistral Nemo tool call parser.
//!
//! Format: `[TOOL_CALLS][{"name":"...", "arguments":{...}}]`

use llguidance::api::TopLevelGrammar;

use super::ToolFormatParser;
use crate::Tool;

pub struct MistralNemoParser;

impl ToolFormatParser for MistralNemoParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("[TOOL_CALLS]")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::MistralNemo
    }

    fn tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body"#.to_string(),
            tools,
            "arguments",
            true,
        )
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        if let Some(inner) = message
            .strip_prefix("[TOOL_CALLS][")
            .and_then(|s| s.strip_suffix("]"))
        {
            Ok(Some(inner.to_string()))
        } else {
            Ok(None)
        }
    }
}
