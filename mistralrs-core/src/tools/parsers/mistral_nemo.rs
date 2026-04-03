//! Mistral Nemo tool call parser.
//!
//! Format: `[TOOL_CALLS][{"name":"...", "arguments":{...}}]`

use super::ToolFormatParser;

pub struct MistralNemoParser;

impl ToolFormatParser for MistralNemoParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("[TOOL_CALLS]")
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
