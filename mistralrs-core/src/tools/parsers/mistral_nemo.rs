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

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: @json_body"#.to_string(),
            tools,
            "arguments",
            true,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: "[TOOL_CALLS]" @json_body"#.to_string(),
            tools,
            "arguments",
            true,
        )
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let prefix = "[TOOL_CALLS]";
        if let Some(pos) = message.find(prefix) {
            let rest = &message[pos + prefix.len()..];
            if let Some(inner) = rest.strip_prefix('[').and_then(|s| s.strip_suffix(']')) {
                Ok(Some(inner.to_string()))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::MistralNemoParser;
    use crate::tools::parsers::ToolFormatParser;

    #[test]
    fn parses_tool_call_after_text() {
        let parsed = MistralNemoParser
            .parse(r#"I'll check.[TOOL_CALLS][{"name":"search","arguments":{"query":"rust"}}]"#)
            .unwrap()
            .unwrap();
        assert_eq!(parsed, r#"{"name":"search","arguments":{"query":"rust"}}"#);
    }
}
