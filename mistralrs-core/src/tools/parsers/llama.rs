//! Llama tool call parser.
//!
//! Format: `<|python_tag|>{"name":"...", "parameters":{...}}`

use super::ToolFormatParser;

pub struct LlamaParser;

impl ToolFormatParser for LlamaParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<|python_tag|>")
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        if let Some(rest) = message.strip_prefix("<|python_tag|>") {
            Ok(Some(rest.to_string()))
        } else {
            Ok(None)
        }
    }
}
