//! Qwen tool call parser.
//!
//! Format: `<tool_call>{"name":"...", "arguments":{...}}</tool_call>`

use llguidance::api::TopLevelGrammar;
use regex::Regex;
use std::sync::OnceLock;

use super::ToolFormatParser;
use crate::Tool;

static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

pub struct QwenParser;

impl ToolFormatParser for QwenParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Qwen
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        // `</tool_call>` is matched as a special token via bare
        // angle-bracket syntax (not a string literal).
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body </tool_call>"#.to_string(),
            tools,
            "arguments",
            false,
        )
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
