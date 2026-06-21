//! Hunyuan tool call parser.
//!
//! Format: `<tool_calls>[{"name":"...", "arguments":{...}}]</tool_calls>`

use llguidance::api::TopLevelGrammar;

use super::ToolFormatParser;
use crate::{tools::CalledFunctionParameters, Tool};

const PREFIX: &str = "<tool_calls";
const START: &str = "<tool_calls>";
const END: &str = "</tool_calls>";

pub struct HunyuanParser;

impl ToolFormatParser for HunyuanParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains(START) || text.ends_with(PREFIX)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Hunyuan
    }

    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar {
        let start = if text.ends_with(PREFIX) {
            r#"start: ">" @json_body "</tool_calls>""#
        } else {
            r#"start: @json_body "</tool_calls>""#
        };
        crate::tools::grammar::build_json_format_grammar(
            start.to_string(),
            tools,
            "arguments",
            true,
        )
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        crate::tools::grammar::build_json_format_grammar(
            r#"start: "<tool_calls>" @json_body "</tool_calls>""#.to_string(),
            tools,
            "arguments",
            true,
        )
    }

    fn parse(&self, message: &str) -> candle_core::Result<Option<String>> {
        let Some(start) = message.find(START) else {
            return Ok(None);
        };
        let rest = &message[start + START.len()..];
        let Some(end) = rest.find(END) else {
            return Ok(None);
        };
        let body = rest[..end].trim();
        let Ok(calls) = serde_json::from_str::<Vec<CalledFunctionParameters>>(body) else {
            return Ok(None);
        };
        if calls.is_empty() {
            return Ok(None);
        }
        Ok(Some(body.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::HunyuanParser;
    use crate::tools::parsers::{extract_model_specific_message, ToolFormatParser};

    #[test]
    fn parses_parallel_tool_calls() {
        let message = r#"<tool_calls>[{"name":"search","arguments":{"query":"rust"}},{"name":"weather","arguments":{"city":"Paris"}}]</tool_calls>"#;
        let parsed = HunyuanParser.parse(message).unwrap().unwrap();

        assert_eq!(
            parsed,
            r#"[{"name":"search","arguments":{"query":"rust"}},{"name":"weather","arguments":{"city":"Paris"}}]"#
        );
    }

    #[test]
    fn extracts_calls_without_discarding_text() {
        let message = r#"before<tool_calls>[{"name":"search","arguments":{}}]</tool_calls>after"#;
        let (calls, content) = extract_model_specific_message(message).unwrap().unwrap();

        assert_eq!(calls, r#"[{"name":"search","arguments":{}}]"#);
        assert_eq!(content, "beforeafter");
    }

    #[test]
    fn leaves_incomplete_call_unparsed() {
        let message = r#"<tool_calls>[{"name":"search","arguments":{}}]"#;

        assert!(HunyuanParser.parse(message).unwrap().is_none());
    }
}
