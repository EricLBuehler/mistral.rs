//! Qwen tool call parser.
//!
//! Format: `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`

use regex::Regex;
use std::sync::OnceLock;

use super::ToolFormatParser;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let re = QWEN_REGEX
            .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

        if let Some(caps) = re.captures(message) {
            let inner = caps.name("inner").unwrap().as_str();
            Ok(Some(inner.trim().to_string()))
        } else {
            Ok(None)
        }
    }
}
