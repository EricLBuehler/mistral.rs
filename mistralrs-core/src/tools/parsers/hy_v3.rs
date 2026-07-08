//! HYV3 tool call parser.
//!
//! Format:
//! `<tool_calls[:suffix]>`
//! `<tool_call[:suffix]>name<tool_sep[:suffix]>`
//! `<arg_key[:suffix]>key</arg_key[:suffix]>`
//! `<arg_value[:suffix]>value</arg_value[:suffix]>`
//! `</tool_call[:suffix]>`
//! `</tool_calls[:suffix]>`

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::{Map, Value};

use super::ToolFormatParser;
use crate::Tool;

const CALLS_PREFIX: &str = "<tool_calls";
const CALL_PREFIX: &str = "<tool_call";

#[derive(serde::Serialize)]
struct HyV3ToolCall {
    name: String,
    arguments: Value,
}

pub struct HyV3Parser;

impl ToolFormatParser for HyV3Parser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains(CALLS_PREFIX)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::HyV3
    }

    fn tool_call_grammar(&self, tools: &[Tool], text: &str) -> TopLevelGrammar {
        hy_v3_tool_call_grammar(tools, false, text)
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        hy_v3_tool_call_grammar(tools, true, "")
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        if !message.contains(CALLS_PREFIX) && !message.contains(CALL_PREFIX) {
            return Ok(None);
        }
        parse_hy_v3_tool_calls(message)
    }
}

fn hy_v3_tool_call_grammar(tools: &[Tool], include_wrapper: bool, text: &str) -> TopLevelGrammar {
    let tool_alts = crate::tools::grammar::lark_tool_name_alternatives(tools);
    let start = if include_wrapper {
        r#"start: tool_calls_plain | tool_calls_open
tool_calls_plain: "<tool_calls>" body_plain
tool_calls_open: "<tool_calls:opensource>" body_open"#
    } else if text.ends_with("<tool_calls") {
        r#"start: ">" body_plain | ":opensource>" body_open"#
    } else if text.ends_with("<tool_calls:opensource") {
        r#"start: ">" body_open"#
    } else if text.ends_with("<tool_call") {
        r#"start: ">" standalone_plain | ":opensource>" standalone_open"#
    } else if text.ends_with("<tool_call:opensource") {
        r#"start: ">" standalone_open"#
    } else if text.ends_with("<tool_call>") {
        r#"start: standalone_plain"#
    } else if text.ends_with("<tool_call:opensource>") {
        r#"start: standalone_open"#
    } else if standalone_name_without_sep(text, "") {
        r#"start: "<tool_sep>" arg_plain+ "</tool_call>""#
    } else if standalone_name_without_sep(text, ":opensource") {
        r#"start: "<tool_sep:opensource>" arg_open+ "</tool_call:opensource>""#
    } else if text.contains("<tool_calls:opensource>") {
        r#"start: body_open"#
    } else {
        r#"start: body_plain"#
    };

    let lark = format!(
        r#"{start}
body_plain: tool_call_plain+ "</tool_calls>"
body_open: tool_call_open+ "</tool_calls:opensource>"
standalone_plain: TOOL_NAME "<tool_sep>" arg_plain+ "</tool_call>"
standalone_open: TOOL_NAME "<tool_sep:opensource>" arg_open+ "</tool_call:opensource>"
tool_call_plain: "<tool_call>" standalone_plain
tool_call_open: "<tool_call:opensource>" standalone_open
arg_plain: "<arg_key>" KEY "</arg_key>" "<arg_value>" value "</arg_value>"
arg_open: "<arg_key:opensource>" KEY "</arg_key:opensource>" "<arg_value:opensource>" value "</arg_value:opensource>"
TOOL_NAME: {tool_alts}
KEY: /[a-zA-Z_][a-zA-Z0-9_]*/
value: VALUE?
VALUE: /[^<]+/
%ignore /[ \t\r\n]+/
"#
    );

    TopLevelGrammar {
        grammars: vec![GrammarWithLexer::from_lark(lark)],
        max_tokens: None,
    }
}

pub(crate) fn parse_hy_v3_tool_calls(message: &str) -> Result<Option<String>> {
    let calls = if let Some((body, suffix)) = extract_tool_calls_body(message) {
        parse_calls(body, suffix)?
    } else if let Some((body, suffix)) = extract_standalone_call_body(message) {
        parse_standalone_call(body, suffix)?
    } else {
        return Ok(None);
    };
    if calls.is_empty() {
        return Ok(None);
    }
    serde_json::to_string(&calls)
        .map(Some)
        .map_err(candle_core::Error::msg)
}

pub(crate) fn strip_hy_v3_tool_calls(message: &str) -> String {
    let mut rest = message;
    let mut out = String::new();
    while let Some(start) = find_next_tool_segment(rest) {
        out.push_str(&rest[..start]);
        let is_calls = rest[start..].starts_with(CALLS_PREFIX);
        let prefix = if is_calls { CALLS_PREFIX } else { CALL_PREFIX };
        let after_start_prefix = &rest[start + prefix.len()..];
        let Some(close) = after_start_prefix.find('>') else {
            return out;
        };
        let suffix = &after_start_prefix[..close];
        let end_tag = if is_calls {
            format!("</tool_calls{suffix}>")
        } else {
            format!("</tool_call{suffix}>")
        };
        let after_start = &after_start_prefix[close + 1..];
        let Some(end) = after_start.find(&end_tag) else {
            return out;
        };
        rest = &after_start[end + end_tag.len()..];
    }
    out.push_str(rest);
    out
}

fn standalone_name_without_sep(text: &str, suffix: &str) -> bool {
    let start_tag = format!("<tool_call{suffix}>");
    let Some((_, after_start)) = text.rsplit_once(&start_tag) else {
        return false;
    };
    !after_start.is_empty() && !after_start.contains(&format!("<tool_sep{suffix}>"))
}

fn find_next_tool_segment(text: &str) -> Option<usize> {
    match (text.find(CALLS_PREFIX), text.find(CALL_PREFIX)) {
        (Some(calls), Some(call)) => Some(calls.min(call)),
        (Some(calls), None) => Some(calls),
        (None, Some(call)) => Some(call),
        (None, None) => None,
    }
}

fn extract_standalone_call_body(message: &str) -> Option<(&str, &str)> {
    let start = message.find(CALL_PREFIX)?;
    if message[start..].starts_with(CALLS_PREFIX) {
        return None;
    }
    let after_start_prefix = &message[start + CALL_PREFIX.len()..];
    let close = after_start_prefix.find('>')?;
    let suffix = &after_start_prefix[..close];
    let end_tag = format!("</tool_call{suffix}>");
    let body_start = start + CALL_PREFIX.len() + close + 1;
    let rest = &message[body_start..];
    let body_end = rest.find(&end_tag)?;
    Some((&rest[..body_end], suffix))
}

fn parse_standalone_call(body: &str, suffix: &str) -> Result<Vec<HyV3ToolCall>> {
    let tool_sep = format!("<tool_sep{suffix}>");
    let Some(sep) = body.find(&tool_sep) else {
        return Ok(Vec::new());
    };
    let name = body[..sep].trim().to_string();
    let arguments = parse_args(&body[sep + tool_sep.len()..], suffix)?;
    Ok(vec![HyV3ToolCall {
        name,
        arguments: Value::Object(arguments),
    }])
}

fn extract_tool_calls_body(message: &str) -> Option<(&str, &str)> {
    let start = message.find(CALLS_PREFIX)?;
    let after_start_prefix = &message[start + CALLS_PREFIX.len()..];
    let close = after_start_prefix.find('>')?;
    let suffix = &after_start_prefix[..close];
    let end_tag = format!("</tool_calls{suffix}>");
    let body_start = start + CALLS_PREFIX.len() + close + 1;
    let rest = &message[body_start..];
    let body_end = rest.find(&end_tag)?;
    Some((&rest[..body_end], suffix))
}

fn parse_calls(body: &str, suffix: &str) -> Result<Vec<HyV3ToolCall>> {
    let call_start = format!("<tool_call{suffix}>");
    let call_end = format!("</tool_call{suffix}>");
    let tool_sep = format!("<tool_sep{suffix}>");
    let mut calls = Vec::new();
    let mut search = 0;

    while let Some(rel_start) = body[search..].find(&call_start) {
        let name_start = search + rel_start + call_start.len();
        let Some(rel_sep) = body[name_start..].find(&tool_sep) else {
            return Ok(Vec::new());
        };
        let sep = name_start + rel_sep;
        let name = body[name_start..sep].trim().to_string();
        let args_start = sep + tool_sep.len();
        let Some(rel_end) = body[args_start..].find(&call_end) else {
            return Ok(Vec::new());
        };
        let args_end = args_start + rel_end;
        let arguments = parse_args(&body[args_start..args_end], suffix)?;
        calls.push(HyV3ToolCall {
            name,
            arguments: Value::Object(arguments),
        });
        search = args_end + call_end.len();
    }

    Ok(calls)
}

fn parse_args(raw: &str, suffix: &str) -> Result<Map<String, Value>> {
    let key_start = format!("<arg_key{suffix}>");
    let key_end = format!("</arg_key{suffix}>");
    let value_start = format!("<arg_value{suffix}>");
    let value_end = format!("</arg_value{suffix}>");
    let mut args = Map::new();
    let mut search = 0;

    while let Some(rel_key_start) = raw[search..].find(&key_start) {
        let key_start_abs = search + rel_key_start + key_start.len();
        let Some(rel_key_end) = raw[key_start_abs..].find(&key_end) else {
            return Ok(Map::new());
        };
        let key_end_abs = key_start_abs + rel_key_end;
        let key = raw[key_start_abs..key_end_abs].trim();
        let after_key = key_end_abs + key_end.len();
        let Some(rel_value_start) = raw[after_key..].find(&value_start) else {
            return Ok(Map::new());
        };
        let value_start_abs = after_key + rel_value_start + value_start.len();
        let Some(rel_value_end) = raw[value_start_abs..].find(&value_end) else {
            return Ok(Map::new());
        };
        let value_end_abs = value_start_abs + rel_value_end;
        let value_raw = raw[value_start_abs..value_end_abs].trim();
        args.insert(key.to_string(), parse_value(value_raw));
        search = value_end_abs + value_end.len();
    }

    Ok(args)
}

fn parse_value(raw: &str) -> Value {
    if raw.is_empty() {
        return Value::String(String::new());
    }
    serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.to_string()))
}

#[cfg(test)]
mod tests {
    use super::HyV3Parser;
    use crate::tools::parsers::{extract_model_specific_message, ToolFormatParser};

    #[test]
    fn parses_opensource_suffix_tool_call() {
        let message = "<tool_calls:opensource>\n<tool_call:opensource>get_weather<tool_sep:opensource>\n<arg_key:opensource>city</arg_key:opensource>\n<arg_value:opensource>Paris</arg_value:opensource>\n<arg_key:opensource>days</arg_key:opensource>\n<arg_value:opensource>3</arg_value:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>";
        let parsed = HyV3Parser.parse(message).unwrap().unwrap();

        assert_eq!(
            parsed,
            r#"[{"name":"get_weather","arguments":{"city":"Paris","days":3}}]"#
        );
    }

    #[test]
    fn parses_plain_suffix_tool_call() {
        let message = "<tool_calls>\n<tool_call>search<tool_sep>\n<arg_key>query</arg_key>\n<arg_value>{\"term\":\"rust\"}</arg_value>\n</tool_call>\n</tool_calls>";
        let parsed = HyV3Parser.parse(message).unwrap().unwrap();

        assert_eq!(
            parsed,
            r#"[{"name":"search","arguments":{"query":{"term":"rust"}}}]"#
        );
    }

    #[test]
    fn parses_standalone_tool_call() {
        let message = "<tool_call>get_weather<tool_sep><arg_key>city</arg_key><arg_value>Tokyo</arg_value></tool_call>";
        let parsed = HyV3Parser.parse(message).unwrap().unwrap();

        assert_eq!(
            parsed,
            r#"[{"name":"get_weather","arguments":{"city":"Tokyo"}}]"#
        );
    }

    #[test]
    fn extracts_calls_without_discarding_text() {
        let message = "before<tool_calls:opensource>\n<tool_call:opensource>search<tool_sep:opensource>\n</tool_call:opensource>\n</tool_calls:opensource>after";
        let (calls, content) = extract_model_specific_message(message).unwrap().unwrap();

        assert_eq!(calls, r#"[{"name":"search","arguments":{}}]"#);
        assert_eq!(content, "beforeafter");
    }

    #[test]
    fn leaves_incomplete_call_unparsed() {
        let message =
            "<tool_calls:opensource>\n<tool_call:opensource>search<tool_sep:opensource>\n";

        assert!(HyV3Parser.parse(message).unwrap().is_none());
    }
}
