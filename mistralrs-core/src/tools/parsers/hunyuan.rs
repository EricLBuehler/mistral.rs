//! Hunyuan tool call parser.
//!
//! Format: `<tool_calls>[{"name":"...", "arguments":{...}}]</tool_calls>`

use llguidance::api::TopLevelGrammar;

use super::ToolFormatParser;
use crate::{tools::CalledFunctionParameters, Tool};

const PREFIX: &str = "<tool_calls";
pub(super) const START: &str = "<tool_calls>";
pub(super) const END: &str = "</tool_calls>";

/// Parses one JSON value from `rest` and requires it be immediately followed, modulo
/// whitespace, by `</tool_calls>` - so a closing tag inside a string argument can't end the
/// call early the way a plain `str::find` would. Returns the raw JSON body text and the
/// offset in `rest` right after the closing tag.
pub(super) fn locate_call(rest: &str) -> Option<(&str, usize)> {
    let mut stream =
        serde_json::Deserializer::from_str(rest).into_iter::<Vec<CalledFunctionParameters>>();
    let calls = stream.next()?.ok()?;
    if calls.is_empty() {
        return None;
    }
    let consumed = stream.byte_offset();
    let after = &rest[consumed..];
    let trimmed_after = after.trim_start();
    if !trimmed_after.starts_with(END) {
        return None;
    }
    let end_of_tag = consumed + (after.len() - trimmed_after.len()) + END.len();
    Some((rest[..consumed].trim(), end_of_tag))
}

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
        let Some((body, _)) = locate_call(rest) else {
            return Ok(None);
        };
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
    fn keeps_closing_tag_found_inside_a_string_argument() {
        let message = r#"before<tool_calls>[{"name":"echo","arguments":{"value":"</tool_calls> and <tool_calls> kept"}}]</tool_calls>after"#;
        let (calls, content) = extract_model_specific_message(message).unwrap().unwrap();

        let calls: serde_json::Value = serde_json::from_str(&calls).unwrap();
        assert_eq!(
            calls[0]["arguments"]["value"],
            "</tool_calls> and <tool_calls> kept"
        );
        assert_eq!(content, "beforeafter");
    }

    #[test]
    fn leaves_incomplete_call_unparsed() {
        let message = r#"<tool_calls>[{"name":"search","arguments":{}}]"#;

        assert!(HunyuanParser.parse(message).unwrap().is_none());
    }
}
