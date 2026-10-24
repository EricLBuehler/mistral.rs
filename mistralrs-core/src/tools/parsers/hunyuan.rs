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
        Ok(extract_call(message).map(|(_, body)| body.to_string()))
    }
}

fn extract_call(message: &str) -> Option<(std::ops::Range<usize>, &str)> {
    let start = message.find(START)?;
    let rest = &message[start + START.len()..];
    // Decode the JSON first: the wrapper end may also occur inside a string value.
    let mut stream =
        serde_json::Deserializer::from_str(rest).into_iter::<Vec<CalledFunctionParameters>>();
    let calls = stream.next()?.ok()?;
    if calls.is_empty() {
        return None;
    }
    let offset = stream.byte_offset();
    let tail = rest[offset..].trim_start();
    tail.strip_prefix(END)?;
    let end = message.len() - tail.len() + END.len();
    Some((start..end, rest[..offset].trim()))
}

pub(super) fn strip_tool_calls(message: &str) -> String {
    let mut rest = message;
    let mut output = String::new();
    while let Some((range, _)) = extract_call(rest) {
        output.push_str(&rest[..range.start]);
        rest = &rest[range.end..];
    }
    output.push_str(rest);
    output
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

    #[test]
    fn wrapper_tags_inside_json_strings_are_data() {
        let message = r#"before<tool_calls>[{"name":"echo","arguments":{"value":"</tool_calls> and <tool_calls> kept"}}]</tool_calls>after"#;

        let parsed = HunyuanParser
            .parse(message)
            .unwrap()
            .expect("wrapper tags inside a JSON string must not end the call");
        let calls: serde_json::Value = serde_json::from_str(&parsed).unwrap();
        assert_eq!(
            calls[0]["arguments"]["value"],
            "</tool_calls> and <tool_calls> kept"
        );

        let (calls_json, content) = extract_model_specific_message(message).unwrap().unwrap();
        assert_eq!(calls_json, parsed);
        assert_eq!(content, "beforeafter");
    }
}
