//! Gemma 4 tool call parser.
//!
//! Format: `<|tool_call>call:NAME{key:<|"|>value<|"|>,key2:42}<tool_call|>`

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::Value;

use super::gemma4_strict::GemmaLarkBuilder;
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

    /// Lark grammar for Gemma 4's non-JSON tool call format.
    ///
    /// When any tool has `strict: true`, generates a branching grammar with
    /// per-tool schema-constrained argument rules.  Otherwise falls back to
    /// the generic grammar that accepts any key-value pairs.
    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        let any_strict = tools.iter().any(|t| t.function.strict == Some(true));

        let lark = if any_strict {
            let mut builder = GemmaLarkBuilder::new();
            let branches: Vec<String> = tools.iter().map(|t| builder.emit_tool_branch(t)).collect();
            builder.emit_shared_rules();
            builder.build(&branches)
        } else {
            let tool_alts = crate::tools::grammar::lark_tool_name_alternatives(tools);
            // r##"..."## because the grammar contains `<|"|>` which has
            // a `"#` sequence that would close an r#"..."# literal.
            format!(
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
            )
        };

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

/// Parse Gemma 4's `<|"|>`-delimited arg format into a `Value`. Not JSON, so we build the tree directly to avoid escaping pain.
/// Example input: `code:<|"|>print("hello\nworld")<|"|>,count:42`
pub(crate) fn gemma4_args_to_json(raw: &str) -> std::result::Result<Value, candle_core::Error> {
    parse_gemma4_value(&format!("{{{raw}}}")).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to parse Gemma 4 tool call arguments: {e}\nRaw: {raw}"
        ))
    })
}

/// Parse a Gemma 4 value starting at position `pos` in `s`.
/// Returns the parsed value and the position after it.
fn parse_gemma4_value(s: &str) -> std::result::Result<Value, String> {
    let s = s.trim();
    if s.is_empty() {
        return Ok(Value::Null);
    }

    // String: <|"|>...<|"|>
    if let Some(rest) = s.strip_prefix(GEMMA4_STR_DELIM) {
        let end = rest
            .find(GEMMA4_STR_DELIM)
            .ok_or_else(|| "Unterminated string (missing closing <|\"|>)".to_string())?;
        let content = &rest[..end];
        return Ok(Value::String(content.to_string()));
    }

    // Object: { key:value, key2:value2 }
    if s.starts_with('{') {
        return parse_gemma4_object(s);
    }

    // Array: [ value, value, ... ]
    if s.starts_with('[') {
        return parse_gemma4_array(s);
    }

    // Boolean
    if s == "true" {
        return Ok(Value::Bool(true));
    }
    if s == "false" {
        return Ok(Value::Bool(false));
    }
    if s == "null" {
        return Ok(Value::Null);
    }

    // Number
    if let Ok(n) = s.parse::<i64>() {
        return Ok(Value::Number(n.into()));
    }
    if let Ok(n) = s.parse::<f64>() {
        if let Some(n) = serde_json::Number::from_f64(n) {
            return Ok(Value::Number(n));
        }
    }

    // Fallback: treat as unquoted string
    Ok(Value::String(s.to_string()))
}

/// Parse a Gemma 4 object: `{ key:value, key2:value2 }`
fn parse_gemma4_object(s: &str) -> std::result::Result<Value, String> {
    let s = s.trim();
    // Strip outer braces
    let inner = s
        .strip_prefix('{')
        .and_then(|s| s.strip_suffix('}'))
        .ok_or_else(|| format!("Expected object braces: {}", &s[..s.len().min(50)]))?
        .trim();

    if inner.is_empty() {
        return Ok(Value::Object(serde_json::Map::new()));
    }

    let mut map = serde_json::Map::new();
    let tokens = split_gemma4_top_level(inner, ',');

    for token in tokens {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        // Find the first `:` that's not inside <|"|> delimiters
        let colon = find_colon_outside_strings(token).ok_or_else(|| {
            format!(
                "Missing ':' in key-value pair: {}",
                &token[..token.len().min(80)]
            )
        })?;

        let key = token[..colon].trim();
        // Strip <|"|> from key if present
        let key = key
            .strip_prefix(GEMMA4_STR_DELIM)
            .and_then(|k| k.strip_suffix(GEMMA4_STR_DELIM))
            .unwrap_or(key);

        let val_str = token[colon + 1..].trim();
        let value = parse_gemma4_value(val_str)?;
        map.insert(key.to_string(), value);
    }

    Ok(Value::Object(map))
}

/// Parse a Gemma 4 array: `[ value, value, ... ]`
fn parse_gemma4_array(s: &str) -> std::result::Result<Value, String> {
    let s = s.trim();
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| format!("Expected array brackets: {}", &s[..s.len().min(50)]))?
        .trim();

    if inner.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    let tokens = split_gemma4_top_level(inner, ',');
    let mut arr = Vec::new();
    for token in tokens {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }
        arr.push(parse_gemma4_value(token)?);
    }
    Ok(Value::Array(arr))
}

/// Split a string by `sep` at the top level, respecting `<|"|>` strings,
/// `{}`/`[]` nesting.
fn split_gemma4_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize; // {} and [] nesting
    let mut in_string = false;
    let bytes = s.as_bytes();
    let delim_bytes = GEMMA4_STR_DELIM.as_bytes();
    let mut start = 0;
    let mut i = 0;

    while i < bytes.len() {
        if in_string {
            if bytes[i..].starts_with(delim_bytes) {
                in_string = false;
                i += delim_bytes.len();
                continue;
            }
            i += utf8_char_len(bytes[i]);
            continue;
        }
        if bytes[i..].starts_with(delim_bytes) {
            in_string = true;
            i += delim_bytes.len();
            continue;
        }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            c if c == sep as u8 && depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
        i += 1;
    }

    if start < s.len() {
        parts.push(&s[start..]);
    }
    parts
}

/// Find the first `:` not inside `<|"|>` strings or nested `{}`/`[]`.
fn find_colon_outside_strings(s: &str) -> Option<usize> {
    let mut in_string = false;
    let mut depth = 0usize;
    let bytes = s.as_bytes();
    let delim_bytes = GEMMA4_STR_DELIM.as_bytes();
    let mut i = 0;

    while i < bytes.len() {
        if in_string {
            if bytes[i..].starts_with(delim_bytes) {
                in_string = false;
                i += delim_bytes.len();
                continue;
            }
            i += utf8_char_len(bytes[i]);
            continue;
        }
        if bytes[i..].starts_with(delim_bytes) {
            in_string = true;
            i += delim_bytes.len();
            continue;
        }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth = depth.saturating_sub(1),
            b':' if depth == 0 => return Some(i),
            _ => {}
        }
        i += 1;
    }
    None
}

/// Extract content between matched braces, respecting `<|"|>` delimiters.
pub(crate) fn extract_matched_braces(s: &str, start: usize) -> Option<(&str, usize)> {
    let bytes = s.as_bytes();
    let delim_bytes = GEMMA4_STR_DELIM.as_bytes();
    if bytes.get(start) != Some(&b'{') {
        return None;
    }
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut i = start;
    while i < bytes.len() {
        if in_string {
            if bytes[i..].starts_with(delim_bytes) {
                in_string = false;
                i += delim_bytes.len();
                continue;
            }
            // Advance by the UTF-8 byte length of the current character.
            i += utf8_char_len(bytes[i]);
            continue;
        }
        if bytes[i..].starts_with(delim_bytes) {
            in_string = true;
            i += delim_bytes.len();
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

/// Return the byte length of a UTF-8 character from its leading byte.
fn utf8_char_len(b: u8) -> usize {
    match b {
        0..=0x7F => 1,
        0xC0..=0xDF => 2,
        0xE0..=0xEF => 3,
        0xF0..=0xF7 => 4,
        _ => 1, // continuation byte; shouldn't happen at a start position
    }
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
    fn prefix_detection() {
        assert!(Gemma4Parser.could_be_tool_call("<|tool_call>call:test{}<tool_call|>"));
        assert!(!Gemma4Parser.could_be_tool_call("some random text"));
    }

    #[test]
    fn strict_grammar_no_strict_unchanged() {
        // When no tools are strict, the grammar uses the non-branching
        // TOOL_NAME path.  This test guards the structural invariant
        // (single Lark grammar, no json_schema).
        use mistralrs_mcp::{Function, ToolType};
        let tools = vec![crate::Tool {
            tp: ToolType::Function,
            function: Function {
                name: "search".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        }];
        let grm = Gemma4Parser.tool_call_grammar(&tools, "");
        assert_eq!(grm.grammars.len(), 1);
        assert!(grm.grammars[0].json_schema.is_none());
    }
}
