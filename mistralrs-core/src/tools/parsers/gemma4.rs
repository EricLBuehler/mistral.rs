//! Gemma 4 tool call parser.
//!
//! Format: `<|tool_call>call:NAME{key:<|"|>value<|"|>,key2:42}<tool_call|>`

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::Value;

use super::ToolFormatParser;
use crate::Tool;

/// Gemma 4 string delimiter token.
const GEMMA4_STR_DELIM: &str = "<|\"|>";

pub struct Gemma4Parser;

impl ToolFormatParser for Gemma4Parser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains("<|tool_call>")
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Gemma4
    }

    /// Pure Lark grammar for Gemma 4's non-JSON tool call format.
    /// Special tokens (`<|"|>`, `<tool_call|>`) use bare angle-bracket
    /// syntax so llguidance matches them via the trie.
    fn tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        let tool_alts = crate::tools::grammar::lark_tool_name_alternatives(tools);
        // Use r##"..."## because the grammar contains `<|"|>` which has
        // a `"#` sequence that would close an r#"..."# literal.
        let lark = format!(
            r##"start: "call:" TOOL_NAME "{{" args "}}" <tool_call|>
TOOL_NAME: {tool_alts}
args: pair ("," pair)* |
pair: KEY ":" value
KEY: /[a-zA-Z_][a-zA-Z0-9_]*/
value: gemma_string | number | "true" | "false" | "null" | array | object
gemma_string: <|"|> /[^<]*/ <|"|>
number: /-?(0|[1-9][0-9]*)(\.[0-9]+)?/
array: "[" (value ("," value)*)? "]"
object: "{{" (pair ("," pair)*)? "}}"
"##
        );
        let top = GrammarWithLexer::from_lark(lark);
        TopLevelGrammar {
            grammars: vec![top],
            max_tokens: None,
        }
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        if !message.contains("<|tool_call>") {
            return Ok(None);
        }
        parse_gemma4_tool_calls(message)
    }
}

// ── Parsing ────────────────────────────────────────────────────────────────

/// Parse Gemma 4 tool calls from model output text.
/// Returns a JSON string of tool calls, or `None` if no complete calls found.
pub(crate) fn parse_gemma4_tool_calls(message: &str) -> Result<Option<String>> {
    let message = message
        .trim_end()
        .strip_suffix("<|tool_response>")
        .unwrap_or(message);

    let prefix = "<|tool_call>call:";
    let suffix = "<tool_call|>";

    if !message.contains(prefix) {
        return Ok(None);
    }

    #[derive(serde::Serialize)]
    struct ToolCall {
        name: String,
        arguments: Value,
    }

    let mut calls = Vec::new();
    let mut search_start = 0;

    while let Some(rel_pos) = message[search_start..].find(prefix) {
        let abs_start = search_start + rel_pos + prefix.len();

        let Some(brace_rel) = message[abs_start..].find('{') else {
            return Ok(None);
        };
        let name = message[abs_start..abs_start + brace_rel].trim().to_string();
        let brace_abs = abs_start + brace_rel;

        let Some((inner, after_brace)) = extract_matched_braces(message, brace_abs) else {
            return Ok(None);
        };

        let arguments = gemma4_args_to_json(inner)?;
        calls.push(ToolCall { name, arguments });

        let remaining = &message[after_brace..];
        if let Some(suf_pos) = remaining.find(suffix) {
            search_start = after_brace + suf_pos + suffix.len();
        } else {
            search_start = after_brace;
        }
    }

    if calls.is_empty() {
        return Ok(None);
    }

    let json = serde_json::to_string(&calls).map_err(candle_core::Error::msg)?;
    Ok(Some(json))
}

// ── Gemma 4 argument format conversion ─────────────────────────────────────

/// Convert Gemma 4 custom argument format to a JSON value.
pub(crate) fn gemma4_args_to_json(raw: &str) -> std::result::Result<Value, candle_core::Error> {
    let with_braces = format!("{{{raw}}}");
    let with_braces = escape_inner_quotes(&with_braces);
    let with_quotes = with_braces.replace(GEMMA4_STR_DELIM, "\"");
    let json_str = quote_unquoted_keys(&with_quotes);

    serde_json::from_str(&json_str).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to parse Gemma 4 tool call arguments: {e}\nConverted JSON: {json_str}"
        ))
    })
}

/// Extract content between matched braces, respecting `<|"|>` delimiters.
pub(crate) fn extract_matched_braces(s: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut i = start;
    while i < s.len() {
        if in_string {
            if s[i..].starts_with(GEMMA4_STR_DELIM) {
                in_string = false;
                i += GEMMA4_STR_DELIM.len();
                continue;
            }
            i += 1;
            continue;
        }
        if s[i..].starts_with(GEMMA4_STR_DELIM) {
            in_string = true;
            i += GEMMA4_STR_DELIM.len();
            continue;
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some((&s[start + 1..i], i + 1));
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Escape literal `"` inside `<|"|>…<|"|>` delimited strings.
pub(crate) fn escape_inner_quotes(input: &str) -> String {
    let mut result = String::with_capacity(input.len());
    let mut rest = input;
    loop {
        let Some(open) = rest.find(GEMMA4_STR_DELIM) else {
            result.push_str(rest);
            break;
        };
        result.push_str(&rest[..open]);
        result.push_str(GEMMA4_STR_DELIM);
        rest = &rest[open + GEMMA4_STR_DELIM.len()..];

        let close = rest.find(GEMMA4_STR_DELIM).unwrap_or(rest.len());
        let inner = &rest[..close];
        for ch in inner.chars() {
            if ch == '"' {
                result.push('\\');
            }
            result.push(ch);
        }
        if close < rest.len() {
            result.push_str(GEMMA4_STR_DELIM);
            rest = &rest[close + GEMMA4_STR_DELIM.len()..];
        } else {
            rest = &rest[close..];
        }
    }
    result
}

/// Quote bare (unquoted) keys in a JSON-like string.
pub(crate) fn quote_unquoted_keys(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 32);
    let chars: Vec<char> = input.chars().collect();
    let len = chars.len();
    let mut i = 0;
    let mut in_string = false;

    while i < len {
        if in_string {
            result.push(chars[i]);
            if chars[i] == '"' {
                in_string = false;
            } else if chars[i] == '\\' && i + 1 < len {
                i += 1;
                result.push(chars[i]);
            }
            i += 1;
            continue;
        }

        if chars[i] == '"' {
            in_string = true;
            result.push(chars[i]);
            i += 1;
            continue;
        }

        if chars[i].is_alphabetic() || chars[i] == '_' {
            let key_start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let key: String = chars[key_start..i].iter().collect();
            if i < len && chars[i] == ':' {
                result.push('"');
                result.push_str(&key);
                result.push('"');
            } else {
                result.push_str(&key);
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::CalledFunctionParameters;

    /// Helper: parse through the full registry (same as the real code path).
    fn parse(msg: &str) -> String {
        crate::tools::parsers::process_model_specific_message(msg).unwrap()
    }

    #[test]
    fn single_tool_call() {
        let msg = "<|tool_call>call:get_weather{city:<|\"|\x3eLondon<|\"|\x3e,units:<|\"|\x3ecelsius<|\"|\x3e}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(parsed[0].parameters["city"], "London");
        assert_eq!(parsed[0].parameters["units"], "celsius");
    }

    #[test]
    fn tool_call_with_number() {
        let msg = "<|tool_call>call:set_temp{value:42}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].name, "set_temp");
        assert_eq!(parsed[0].parameters["value"], 42);
    }

    #[test]
    fn tool_call_with_boolean() {
        let msg = "<|tool_call>call:toggle{enabled:true}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].parameters["enabled"], true);
    }

    #[test]
    fn tool_call_nested_object() {
        let msg = "<|tool_call>call:api{config:{url:<|\"|\x3ehttps://example.com<|\"|\x3e,retries:3}}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].name, "api");
        assert_eq!(parsed[0].parameters["config"]["url"], "https://example.com");
        assert_eq!(parsed[0].parameters["config"]["retries"], 3);
    }

    #[test]
    fn strips_tool_response_eos() {
        let msg =
            "<|tool_call>call:search{query:<|\"|\x3eweather<|\"|\x3e}<tool_call|><|tool_response>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].parameters["query"], "weather");
    }

    #[test]
    fn multiple_tool_calls() {
        let msg = "<|tool_call>call:func_a{x:1}<tool_call|><|tool_call>call:func_b{y:<|\"|\x3ehello<|\"|\x3e}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "func_a");
        assert_eq!(parsed[1].name, "func_b");
    }

    #[test]
    fn with_thinking_prefix() {
        let msg = "<|channel>thought\nLet me search.\n<channel|><|tool_call>call:search{q:<|\"|\x3etest<|\"|\x3e}<tool_call|><|tool_response>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "search");
    }

    #[test]
    fn incomplete_no_brace() {
        let msg = "<|tool_call>call:";
        assert_eq!(parse(msg), msg);
    }

    #[test]
    fn incomplete_unmatched_braces() {
        let msg = "<|tool_call>call:func{key:<|\"|\x3evalue";
        assert_eq!(parse(msg), msg);
    }

    #[test]
    fn incomplete_just_prefix_token() {
        let msg = "<|tool_call>";
        assert_eq!(parse(msg), msg);
    }

    #[test]
    fn incomplete_name_no_closing() {
        let msg = "<|tool_call>call:search_the_web{query:<|\"|\x3etest<|\"|\x3e";
        assert_eq!(parse(msg), msg);
    }

    #[test]
    fn non_tool_message_unchanged() {
        let msg = "Hello, how can I help you?";
        assert_eq!(parse(msg), msg);
    }

    #[test]
    fn with_array() {
        let msg = "<|tool_call>call:multi{items:[1,2,3]}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].parameters["items"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn empty_args() {
        let msg = "<|tool_call>call:no_args{}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].parameters, serde_json::json!({}));
    }

    #[test]
    fn string_with_special_chars() {
        let msg = "<|tool_call>call:test{query:<|\"|\x3ekey:val, {nested}<|\"|\x3e}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].parameters["query"], "key:val, {nested}");
    }

    #[test]
    #[allow(clippy::approx_constant)]
    fn negative_number() {
        let msg = "<|tool_call>call:offset{x:-5,y:3.14}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert_eq!(parsed[0].parameters["x"], -5);
        assert_eq!(parsed[0].parameters["y"], 3.14);
    }

    #[test]
    fn null_value() {
        let msg = "<|tool_call>call:test{val:null}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        assert!(parsed[0].parameters["val"].is_null());
    }

    #[test]
    fn inner_quotes_escaped() {
        let msg = "<|tool_call>call:google_search{queries:[<|\"|\x3e\"Review\" stuff<|\"|\x3e,<|\"|\x3eplain<|\"|\x3e]}<tool_call|>";
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parse(msg)).unwrap();
        let queries = parsed[0].parameters["queries"].as_array().unwrap();
        assert_eq!(queries[0], "\"Review\" stuff");
        assert_eq!(queries[1], "plain");
    }

    // ── Unit tests for helper functions ─────────────────────────────────

    #[test]
    fn quote_keys_basic() {
        assert_eq!(
            quote_unquoted_keys(r#"{key:"value",num:42}"#),
            r#"{"key":"value","num":42}"#
        );
    }

    #[test]
    fn quote_keys_nested() {
        assert_eq!(
            quote_unquoted_keys(r#"{outer:{inner:"val"}}"#),
            r#"{"outer":{"inner":"val"}}"#
        );
    }

    #[test]
    fn quote_keys_preserves_strings() {
        assert_eq!(
            quote_unquoted_keys(r#"{key:"has:colon,and{brace}"}"#),
            r#"{"key":"has:colon,and{brace}"}"#
        );
    }

    #[test]
    fn quote_keys_array_of_strings() {
        assert_eq!(
            quote_unquoted_keys(r#"{items:["a","b"]}"#),
            r#"{"items":["a","b"]}"#
        );
    }

    #[test]
    fn quote_keys_multibyte() {
        assert_eq!(
            quote_unquoted_keys(r#"{first:"café",second:42}"#),
            r#"{"first":"café","second":42}"#
        );
    }

    #[test]
    fn quote_keys_booleans_not_quoted() {
        assert_eq!(
            quote_unquoted_keys(r#"{a:true,b:false,c:null}"#),
            r#"{"a":true,"b":false,"c":null}"#
        );
    }

    #[test]
    fn matched_braces_basic() {
        let s = "{a:1,b:{c:2}}rest";
        let (inner, pos) = extract_matched_braces(s, 0).unwrap();
        assert_eq!(inner, "a:1,b:{c:2}");
        assert_eq!(pos, 13);
    }

    #[test]
    fn matched_braces_with_strings() {
        let s = "{key:<|\"|\x3e{not a brace}<|\"|\x3e}after";
        let (inner, _) = extract_matched_braces(s, 0).unwrap();
        assert_eq!(inner, "key:<|\"|\x3e{not a brace}<|\"|\x3e");
    }

    #[test]
    fn args_to_json_mixed_types() {
        let raw = "name:<|\"|\x3eAlice<|\"|\x3e,age:30,active:true,data:{score:9.5}";
        let val = gemma4_args_to_json(raw).unwrap();
        assert_eq!(val["name"], "Alice");
        assert_eq!(val["age"], 30);
        assert_eq!(val["active"], true);
        assert_eq!(val["data"]["score"], 9.5);
    }

    #[test]
    fn escape_inner_quotes_basic() {
        let input = r#"<|"|>hello "world"<|"|>"#;
        assert_eq!(escape_inner_quotes(input), r#"<|"|>hello \"world\"<|"|>"#);
    }

    #[test]
    fn escape_inner_quotes_no_quotes() {
        let input = r#"<|"|>hello world<|"|>"#;
        assert_eq!(escape_inner_quotes(input), input);
    }

    #[test]
    fn escape_inner_quotes_multiple() {
        let input = r#"key:<|"|>"a" and "b"<|"|>,other:<|"|>no quotes<|"|>"#;
        assert_eq!(
            escape_inner_quotes(input),
            r#"key:<|"|>\"a\" and \"b\"<|"|>,other:<|"|>no quotes<|"|>"#
        );
    }

    #[test]
    fn prefix_detection() {
        assert!(Gemma4Parser.could_be_tool_call("<|tool_call>call:test{}<tool_call|>"));
        assert!(!Gemma4Parser.could_be_tool_call("some random text"));
    }
}
