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
            let rest = message[pos + prefix.len()..].trim();
            if rest.starts_with('[') && rest.ends_with(']') {
                Ok(Some(rest.to_string()))
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
        assert_eq!(
            parsed,
            r#"[{"name":"search","arguments":{"query":"rust"}}]"#
        );
    }

    #[test]
    fn keeps_parallel_calls_as_one_array() {
        let parsed = MistralNemoParser
            .parse(r#"[TOOL_CALLS][{"name":"a","arguments":{}}, {"name":"b","arguments":{}}]"#)
            .unwrap()
            .unwrap();
        assert_eq!(
            parsed,
            r#"[{"name":"a","arguments":{}}, {"name":"b","arguments":{}}]"#
        );
    }

    #[test]
    fn ignores_trailing_whitespace_after_the_array() {
        let parsed = MistralNemoParser
            .parse("[TOOL_CALLS][{\"name\":\"a\",\"arguments\":{}}]\n")
            .unwrap()
            .unwrap();
        assert_eq!(parsed, r#"[{"name":"a","arguments":{}}]"#);
    }

    #[test]
    fn waits_for_the_closing_bracket() {
        assert!(MistralNemoParser
            .parse(r#"[TOOL_CALLS][{"name":"a","arguments":{"#)
            .unwrap()
            .is_none());
    }
}
