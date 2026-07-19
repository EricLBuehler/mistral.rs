//! Liquid LFM2.5 tool call parser.
//!
//! Format: `<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>`

use candle_core::Result;
use llguidance::api::{GrammarWithLexer, TopLevelGrammar};
use serde_json::{Map, Number, Value};

use super::ToolFormatParser;
use crate::Tool;

const START: &str = "<|tool_call_start|>";
const END: &str = "<|tool_call_end|>";

#[derive(serde::Serialize)]
struct LiquidToolCall {
    name: String,
    arguments: Value,
}

pub struct LiquidParser;

impl ToolFormatParser for LiquidParser {
    fn could_be_tool_call(&self, text: &str) -> bool {
        text.contains(START)
    }

    fn format(&self) -> super::ToolCallFormat {
        super::ToolCallFormat::Liquid
    }

    fn tool_call_grammar(&self, tools: &[Tool], _text: &str) -> TopLevelGrammar {
        liquid_tool_call_grammar(tools, false)
    }

    fn required_tool_call_grammar(&self, tools: &[Tool]) -> TopLevelGrammar {
        liquid_tool_call_grammar(tools, true)
    }

    fn parse(&self, message: &str) -> Result<Option<String>> {
        if !message.contains(START) {
            return Ok(None);
        }
        parse_liquid_tool_calls(message)
    }
}

fn liquid_tool_call_grammar(tools: &[Tool], include_wrapper: bool) -> TopLevelGrammar {
    let tool_alts = crate::tools::grammar::lark_tool_name_alternatives(tools);
    let start = if include_wrapper {
        r#"start: <|tool_call_start|> "[" tool_call ("," tool_call)* "]" <|tool_call_end|>"#
    } else {
        r#"start: "[" tool_call ("," tool_call)* "]" <|tool_call_end|>"#
    };
    let lark = format!(
        r#"{start}
tool_call: TOOL_NAME "(" args? ")"
TOOL_NAME: {tool_alts}
args: arg ("," arg)*
arg: KEY "=" value
KEY: /[a-zA-Z_][a-zA-Z0-9_]*/
value: string | number | "true" | "false" | "True" | "False" | "null" | "None" | array | object
string: DOUBLE_STRING | SINGLE_STRING
DOUBLE_STRING: /"([^"\\]|\\.)*"/
SINGLE_STRING: /'([^'\\]|\\.)*'/
number: /-?(0|[1-9][0-9]*)(\.[0-9]+)?/
array: "[" (value ("," value)*)? "]"
object: "{{" (object_pair ("," object_pair)*)? "}}"
object_pair: (string | KEY) ":" value
"#
    );

    TopLevelGrammar {
        grammars: vec![GrammarWithLexer::from_lark(lark)],
        max_tokens: None,
    }
}

fn parse_liquid_tool_calls(message: &str) -> Result<Option<String>> {
    let mut calls = Vec::new();
    let mut search_start = 0;

    while let Some(rel_start) = message[search_start..].find(START) {
        let body_start = search_start + rel_start + START.len();
        let Some(rel_end) = message[body_start..].find(END) else {
            return Ok(None);
        };
        let body = &message[body_start..body_start + rel_end];
        let parsed = parse_liquid_body(body).map_err(candle_core::Error::msg)?;
        calls.extend(parsed);
        search_start = body_start + rel_end + END.len();
    }

    if calls.is_empty() {
        return Ok(None);
    }

    serde_json::to_string(&calls)
        .map(Some)
        .map_err(candle_core::Error::msg)
}

fn parse_liquid_body(body: &str) -> std::result::Result<Vec<LiquidToolCall>, String> {
    let body = body.trim();
    let inner = body
        .strip_prefix('[')
        .and_then(|body| body.strip_suffix(']'))
        .ok_or_else(|| "Liquid tool call body must be a bracketed list".to_string())?
        .trim();

    if inner.is_empty() {
        return Ok(Vec::new());
    }

    split_top_level(inner, ',')
        .into_iter()
        .map(parse_liquid_call)
        .collect()
}

fn parse_liquid_call(call: &str) -> std::result::Result<LiquidToolCall, String> {
    let call = call.trim();
    let open = call
        .find('(')
        .ok_or_else(|| format!("Liquid tool call is missing `(`: {call}"))?;
    let close = find_matching_delimiter(call, open, b'(', b')')
        .ok_or_else(|| format!("Liquid tool call is missing `)`: {call}"))?;
    if !call[close + 1..].trim().is_empty() {
        return Err(format!("Unexpected text after Liquid tool call: {call}"));
    }

    let name = call[..open].trim();
    if name.is_empty() {
        return Err("Liquid tool call name is empty".to_string());
    }

    let args = parse_liquid_args(&call[open + 1..close])?;
    Ok(LiquidToolCall {
        name: name.to_string(),
        arguments: Value::Object(args),
    })
}

fn parse_liquid_args(args: &str) -> std::result::Result<Map<String, Value>, String> {
    let mut parsed = Map::new();
    let args = args.trim();
    if args.is_empty() {
        return Ok(parsed);
    }

    for arg in split_top_level(args, ',') {
        let eq = find_top_level_char(arg, '=')
            .ok_or_else(|| format!("Liquid tool argument is missing `=`: {arg}"))?;
        let key = arg[..eq].trim();
        if key.is_empty() {
            return Err(format!("Liquid tool argument key is empty: {arg}"));
        }
        parsed.insert(key.to_string(), parse_liquid_value(&arg[eq + 1..])?);
    }

    Ok(parsed)
}

fn parse_liquid_value(raw: &str) -> std::result::Result<Value, String> {
    let raw = raw.trim();
    if raw.is_empty() {
        return Ok(Value::String(String::new()));
    }

    if matches!(raw.as_bytes().first(), Some(b'"' | b'\'')) {
        return parse_quoted_string(raw).map(Value::String);
    }

    match raw {
        "true" | "True" => return Ok(Value::Bool(true)),
        "false" | "False" => return Ok(Value::Bool(false)),
        "null" | "None" => return Ok(Value::Null),
        _ => {}
    }

    if raw.starts_with('[') && raw.ends_with(']') {
        return parse_liquid_array(raw);
    }
    if raw.starts_with('{') && raw.ends_with('}') {
        return parse_liquid_object(raw);
    }

    if let Ok(value) = raw.parse::<i64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = raw.parse::<u64>() {
        return Ok(Value::Number(value.into()));
    }
    if let Ok(value) = raw.parse::<f64>() {
        if let Some(value) = Number::from_f64(value) {
            return Ok(Value::Number(value));
        }
    }

    Ok(Value::String(raw.to_string()))
}

fn parse_liquid_array(raw: &str) -> std::result::Result<Value, String> {
    let inner = raw
        .trim()
        .strip_prefix('[')
        .and_then(|raw| raw.strip_suffix(']'))
        .ok_or_else(|| format!("Expected array brackets: {raw}"))?
        .trim();
    if inner.is_empty() {
        return Ok(Value::Array(Vec::new()));
    }

    split_top_level(inner, ',')
        .into_iter()
        .map(parse_liquid_value)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map(Value::Array)
}

fn parse_liquid_object(raw: &str) -> std::result::Result<Value, String> {
    let inner = raw
        .trim()
        .strip_prefix('{')
        .and_then(|raw| raw.strip_suffix('}'))
        .ok_or_else(|| format!("Expected object braces: {raw}"))?
        .trim();
    if inner.is_empty() {
        return Ok(Value::Object(Map::new()));
    }

    let mut parsed = Map::new();
    for pair in split_top_level(inner, ',') {
        let colon = find_top_level_char(pair, ':')
            .ok_or_else(|| format!("Liquid object pair is missing `:`: {pair}"))?;
        let key = parse_liquid_key(&pair[..colon])?;
        parsed.insert(key, parse_liquid_value(&pair[colon + 1..])?);
    }
    Ok(Value::Object(parsed))
}

fn parse_liquid_key(raw: &str) -> std::result::Result<String, String> {
    let raw = raw.trim();
    if matches!(raw.as_bytes().first(), Some(b'"' | b'\'')) {
        parse_quoted_string(raw)
    } else if raw.is_empty() {
        Err("Liquid object key is empty".to_string())
    } else {
        Ok(raw.to_string())
    }
}

fn parse_quoted_string(raw: &str) -> std::result::Result<String, String> {
    let mut chars = raw.char_indices();
    let Some((_, quote @ ('"' | '\''))) = chars.next() else {
        return Err(format!("Expected quoted string: {raw}"));
    };

    let mut out = String::new();
    let mut escaped = false;
    for (idx, ch) in chars {
        if escaped {
            match ch {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'b' => out.push('\u{0008}'),
                'f' => out.push('\u{000c}'),
                '\\' => out.push('\\'),
                '"' => out.push('"'),
                '\'' => out.push('\''),
                other => out.push(other),
            }
            escaped = false;
            continue;
        }
        if ch == '\\' {
            escaped = true;
            continue;
        }
        if ch == quote {
            if raw[idx + ch.len_utf8()..].trim().is_empty() {
                return Ok(out);
            }
            return Err(format!("Unexpected text after quoted string: {raw}"));
        }
        out.push(ch);
    }

    Err(format!("Unterminated quoted string: {raw}"))
}

fn split_top_level(s: &str, sep: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let mut start = 0;

    for (idx, ch) in s.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ch if ch == sep && depth == 0 => {
                parts.push(s[start..idx].trim());
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    if start <= s.len() {
        let tail = s[start..].trim();
        if !tail.is_empty() {
            parts.push(tail);
        }
    }
    parts
}

fn find_top_level_char(s: &str, target: char) -> Option<usize> {
    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;

    for (idx, ch) in s.char_indices() {
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            continue;
        }

        match ch {
            '"' | '\'' => quote = Some(ch),
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth = depth.saturating_sub(1),
            ch if ch == target && depth == 0 => return Some(idx),
            _ => {}
        }
    }

    None
}

fn find_matching_delimiter(s: &str, start: usize, open: u8, close: u8) -> Option<usize> {
    let bytes = s.as_bytes();
    if bytes.get(start) != Some(&open) {
        return None;
    }

    let mut depth = 0usize;
    let mut quote = None;
    let mut escaped = false;
    let mut idx = start;

    while idx < bytes.len() {
        let ch = s[idx..].chars().next()?;
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if ch == '\\' {
                escaped = true;
            } else if ch == active_quote {
                quote = None;
            }
            idx += ch.len_utf8();
            continue;
        }

        match ch {
            '"' | '\'' => quote = Some(ch),
            ch if ch as u8 == open => depth += 1,
            ch if ch as u8 == close => {
                depth -= 1;
                if depth == 0 {
                    return Some(idx);
                }
            }
            _ => {}
        }
        idx += ch.len_utf8();
    }

    None
}

#[cfg(test)]
mod tests {
    use super::LiquidParser;
    use crate::tools::parsers::{extract_model_specific_message, ToolFormatParser};
    use crate::tools::{CalledFunctionParameters, ToolCallingMatcher, ToolChoice};
    use crate::{Function, Tool, ToolType};

    fn tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                name: name.to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn parses_single_call() {
        let parsed = LiquidParser
            .parse(r#"<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>"#)
            .unwrap()
            .unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parsed).unwrap();

        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(parsed[0].parameters["location"], "Paris");
    }

    #[test]
    fn parses_multiple_calls_and_value_types() {
        let parsed = LiquidParser
            .parse(
                r#"<|tool_call_start|>[search(query='rust, candle'), set_flags(enabled=True, limit=3, ratio=0.5, meta={"source":"web"}, tags=['a', 'b'], value=None)]<|tool_call_end|>"#,
            )
            .unwrap()
            .unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&parsed).unwrap();

        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "search");
        assert_eq!(parsed[0].parameters["query"], "rust, candle");
        assert_eq!(parsed[1].name, "set_flags");
        assert_eq!(parsed[1].parameters["enabled"], true);
        assert_eq!(parsed[1].parameters["limit"], 3);
        assert_eq!(parsed[1].parameters["ratio"], 0.5);
        assert_eq!(parsed[1].parameters["meta"]["source"], "web");
        assert_eq!(parsed[1].parameters["tags"], serde_json::json!(["a", "b"]));
        assert!(parsed[1].parameters["value"].is_null());
    }

    #[test]
    fn extracts_calls_without_losing_surrounding_text() {
        let message = r#"before <|tool_call_start|>[search(query="rust")]<|tool_call_end|> after"#;
        let (calls, content) = extract_model_specific_message(message).unwrap().unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&calls).unwrap();

        assert_eq!(parsed[0].name, "search");
        assert_eq!(content, "before  after");
    }

    #[test]
    fn matcher_returns_structured_tool_calls_and_content() {
        let tools = vec![tool("get_weather")];
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&tools)).unwrap();
        let (content, calls) = matcher
            .get_call_with_content(
                r#"<|tool_call_start|>[get_weather(location="Paris")]<|tool_call_end|>Checking."#,
            )
            .unwrap();

        assert_eq!(content.as_deref(), Some("Checking."));
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].function.name, "get_weather");
        assert_eq!(calls[0].function.arguments, r#"{"location":"Paris"}"#);
    }

    #[test]
    fn leaves_incomplete_call_unparsed() {
        let message = r#"<|tool_call_start|>[search(query="rust")]"#;

        assert!(LiquidParser.parse(message).unwrap().is_none());
    }
}
