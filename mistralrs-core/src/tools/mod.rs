mod request;
mod response;

use candle_core::Result;
use regex::Regex;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, OnceLock};
use uuid::Uuid;

use crate::Pipeline;
use mistralrs_mcp::CalledFunction;

// Re-export the types so they're accessible as tools::Type
pub use mistralrs_mcp::{ToolCallback, ToolCallbackWithTool};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

fn contains_tool_call_prefix(prefix: &str) -> bool {
    prefix.contains("<tool_call>")
        || prefix.contains("<|tool_call>") // Gemma 4
        || prefix.contains("<｜tool▁call▁begin｜>")
        || prefix.contains("<|python_tag|>")
        || prefix.contains("[TOOL_CALLS]")
}

/// Gemma 4 string delimiter token used in tool call arguments.
const GEMMA4_STR_DELIM: &str = "<|\"|\x3e";

/// Extract content between matched braces starting at `start`, respecting
/// Gemma 4 `<|"|>` string delimiters (braces inside strings are ignored).
/// Returns `(inner_content, position_after_closing_brace)`.
fn extract_matched_braces(s: &str, start: usize) -> Option<(&str, usize)> {
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

/// Convert Gemma 4 custom argument format to a JSON string.
///
/// Input:  `key1:<|"|>value<|"|>,key2:42,nested:{k:<|"|>v<|"|>}`
/// Output: `{"key1":"value","key2":42,"nested":{"k":"v"}}`
fn gemma4_args_to_json(raw: &str) -> std::result::Result<Value, candle_core::Error> {
    // Step 1: wrap in braces so it's a full object
    let with_braces = format!("{{{raw}}}");

    // Step 2: replace <|"|> with "
    let with_quotes = with_braces.replace(GEMMA4_STR_DELIM, "\"");

    // Step 3: quote unquoted keys using a state machine
    let json_str = quote_unquoted_keys(&with_quotes);

    serde_json::from_str(&json_str).map_err(|e| {
        candle_core::Error::Msg(format!(
            "Failed to parse Gemma 4 tool call arguments: {e}\nConverted JSON: {json_str}"
        ))
    })
}

/// Quote bare (unquoted) keys in a JSON-like string.
/// Tracks whether we're inside a `"..."` string to avoid quoting content.
fn quote_unquoted_keys(input: &str) -> String {
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

        // Check for unquoted key: alphanumeric/underscore sequence followed by ':'
        if chars[i].is_alphabetic() || chars[i] == '_' {
            let key_start = i;
            while i < len && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            // Collect from the chars slice (not byte-index the string) to
            // stay safe with multi-byte UTF-8 content elsewhere in the input.
            let key: String = chars[key_start..i].iter().collect();
            if i < len && chars[i] == ':' {
                result.push('"');
                result.push_str(&key);
                result.push('"');
            } else {
                // Not a key (e.g., `true`, `false`, `null`, or bare value)
                result.push_str(&key);
            }
            continue;
        }

        result.push(chars[i]);
        i += 1;
    }
    result
}

/// Parse Gemma 4 tool calls from model output text.
/// Returns a JSON string of `Vec<CalledFunctionParameters>`, or `None` if no
/// complete Gemma 4 tool calls are found (incomplete calls also return `None`).
fn parse_gemma4_tool_calls(message: &str) -> Result<Option<String>> {
    // Strip trailing <|tool_response> (EOS marker)
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

        // Find the opening brace for arguments.
        // If not found, the tool call is still being generated — return None.
        let Some(brace_rel) = message[abs_start..].find('{') else {
            return Ok(None);
        };
        let name = message[abs_start..abs_start + brace_rel].trim().to_string();
        let brace_abs = abs_start + brace_rel;

        // Extract matched braces.
        // If braces don't match, the tool call is still being generated.
        let Some((inner, after_brace)) = extract_matched_braces(message, brace_abs) else {
            return Ok(None);
        };

        let arguments = gemma4_args_to_json(inner)?;
        calls.push(ToolCall { name, arguments });

        // Skip past <tool_call|> suffix
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

fn process_model_specific_message(message: &str) -> Result<String> {
    static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();
    static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();

    // These are reasoning models so we need a regex.
    let deepseek_regex = DEEPSEEK_REGEX.get_or_init(|| Regex::new(
        r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
    ).unwrap());
    let qwen_regex = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    // Gemma 4 uses <|tool_call>call:NAME{ARGS}<tool_call|> — check first since
    // <|tool_call> contains <tool_call> as a substring (avoids false Qwen match).
    if message.contains("<|tool_call>") {
        if let Some(json) = parse_gemma4_tool_calls(message)? {
            return Ok(json);
        }
    }

    if let Some(message) = message.strip_prefix("<|python_tag|>") {
        // Llama case
        Ok(message.to_string())
    } else if qwen_regex.is_match(message) {
        if let Some(caps) = qwen_regex.captures(message) {
            let inner = caps.name("inner").unwrap().as_str();
            return Ok(inner.trim().to_string());
        }
        Ok(message.to_string())
    } else if let Some(message) = message
        .strip_prefix("[TOOL_CALLS][")
        .and_then(|s| s.strip_suffix("]"))
    {
        // Mistral Nemo case
        Ok(message.to_string())
    } else if deepseek_regex.find(message).is_some() {
        #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
        struct ToolCall {
            name: String,
            arguments: Value,
        }
        let mut calls = Vec::new();
        for caps in deepseek_regex.captures_iter(message) {
            let name = caps
                .name("name")
                .ok_or("Could not capture function name")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim()
                .to_string();
            let json_str = caps
                .name("json")
                .ok_or("Could not capture JSON arguments")
                .map_err(candle_core::Error::msg)?
                .as_str()
                .trim();
            let arguments: Value =
                serde_json::from_str(json_str).map_err(candle_core::Error::msg)?;
            calls.push(ToolCall { name, arguments });
        }
        Ok(serde_json::to_string(&calls).map_err(candle_core::Error::msg)?)
    } else {
        Ok(message.to_string())
    }
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

// Same as CalledFunction, but has different cases for variations on the names
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    #[serde(alias = "function")]
    pub name: String,
    #[serde(alias = "arguments", deserialize_with = "flexible_args")]
    pub parameters: Value,
}

// Accept either `{...}` **or** a `"stringified { ... }"`
fn flexible_args<'de, D>(d: D) -> std::result::Result<Value, D::Error>
where
    D: Deserializer<'de>,
{
    struct ArgVisitor;

    impl<'de> Visitor<'de> for ArgVisitor {
        type Value = Value;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("an object or a JSON-encoded string containing an object")
        }

        // Case 1 – the good case: already a JSON object
        fn visit_map<M>(self, mut m: M) -> std::result::Result<Self::Value, M::Error>
        where
            M: MapAccess<'de>,
        {
            let mut map = Map::new();
            while let Some((k, v)) = m.next_entry()? {
                map.insert(k, v);
            }
            Ok(Value::Object(map))
        }

        // Case 2 – got a *string*; try parsing it as JSON
        fn visit_str<E>(self, s: &str) -> std::result::Result<Self::Value, E>
        where
            E: de::Error,
        {
            serde_json::from_str(s).map_err(|e| E::custom(format!("inner JSON error: {e}")))
        }
    }

    d.deserialize_any(ArgVisitor)
}

/// Fixup potentially broken JSON
/// 1) allow/handle arguments as maps in quotations
fn fix_broken_json(raw: &str) -> anyhow::Result<String> {
    // Only apply the fix if the first pattern matches - otherwise we might corrupt valid JSON
    // where arguments is a properly escaped string containing `}`
    if raw.contains(r#""arguments":"{"#) {
        // 1) Delete the opening quote that shouldn't be there
        let tmp = raw.replacen(r#""arguments":"{"#, r#""arguments":{"#, 1);
        // 2) Delete the closing quote that matches it
        let fixed = tmp.replacen(r#"}"}"#, r#"}}"#, 1);
        Ok(fixed)
    } else {
        Ok(raw.to_string())
    }
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    // Checks if the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(
        &self,
        _pipeline: &dyn Pipeline,
        message_prefix: &str,
    ) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let message_prefix = process_model_specific_message(message_prefix)?;
        let message_prefix = fix_broken_json(&message_prefix).unwrap();

        // Check if the prefix could be a JSON serialization of any of the following types.
        Ok([
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<Vec<CalledFunctionParameters>>,
        ]
        .iter()
        .find_map(|check| {
            let (could_be_tool, is_complete_tool) = check(&message_prefix);
            if could_be_tool || is_complete_tool {
                Some((could_be_tool, is_complete_tool))
            } else {
                None
            }
        })
        .unwrap_or((contains_tool_call_prefix(&message_prefix), false)))
    }

    pub fn get_call(
        &self,
        _pipeline: &dyn Pipeline,
        message: &str,
    ) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }
        let message = process_model_specific_message(message)?;
        let message = fix_broken_json(&message).unwrap();

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(&message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                index: 0,
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            Ok(deser
                .into_iter()
                .enumerate()
                .map(|(idx, deser)| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        index: idx,
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else {
            if matches!(self.tool_choice, ToolChoice::Tool(_)) {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            Ok(Vec::new())
        }
    }
}

/// Checks if the given prefix could be the start of, or the entire JSON serialization of a given type, `T`.
///
/// Returns a tuple of `(could_be_tool, is_entire_tool)`.
fn could_be_json<T>(text_prefix: &str) -> (bool, bool)
where
    T: serde::de::DeserializeOwned,
{
    if text_prefix.trim().is_empty() {
        return (false, false);
    }
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => (false, true),
        // EOF show that JSON parsing was successful up to the end of the entire string.
        Err(e) if e.is_eof() => (true, false),
        _ => (false, false),
    }
}

/// Takes raw UTf8 text and parses any possible tool calls from it.
pub fn parse_text_tools<'a>(
    pipeline: &dyn Pipeline,
    raw_text: &'a str,
    matcher: Option<Arc<ToolCallingMatcher>>,
) -> anyhow::Result<(Option<&'a str>, Vec<ToolCallResponse>)> {
    let mut tool_calls = Vec::new();
    let mut text_new = Some(raw_text);

    if let Some(ref matcher) = matcher {
        let calls = matcher
            .get_call(pipeline, raw_text)
            .map_err(candle_core::Error::msg)?;
        if !calls.is_empty() {
            text_new = None;
            tool_calls = calls;
        }
    };
    Ok((text_new, tool_calls))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemma4_single_tool_call() {
        let msg = "<|tool_call>call:get_weather{city:<|\"|\x3eLondon<|\"|\x3e,units:<|\"|\x3ecelsius<|\"|\x3e}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "get_weather");
        assert_eq!(parsed[0].parameters["city"], "London");
        assert_eq!(parsed[0].parameters["units"], "celsius");
    }

    #[test]
    fn gemma4_tool_call_with_number() {
        let msg = "<|tool_call>call:set_temp{value:42}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].name, "set_temp");
        assert_eq!(parsed[0].parameters["value"], 42);
    }

    #[test]
    fn gemma4_tool_call_with_boolean() {
        let msg = "<|tool_call>call:toggle{enabled:true}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].parameters["enabled"], true);
    }

    #[test]
    fn gemma4_tool_call_nested_object() {
        let msg = "<|tool_call>call:api{config:{url:<|\"|\x3ehttps://example.com<|\"|\x3e,retries:3}}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].name, "api");
        assert_eq!(parsed[0].parameters["config"]["url"], "https://example.com");
        assert_eq!(parsed[0].parameters["config"]["retries"], 3);
    }

    #[test]
    fn gemma4_tool_call_strips_tool_response() {
        let msg =
            "<|tool_call>call:search{query:<|\"|\x3eweather<|\"|\x3e}<tool_call|><|tool_response>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "search");
        assert_eq!(parsed[0].parameters["query"], "weather");
    }

    #[test]
    fn gemma4_multiple_tool_calls() {
        let msg = "<|tool_call>call:func_a{x:1}<tool_call|><|tool_call>call:func_b{y:<|\"|\x3ehello<|\"|\x3e}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].name, "func_a");
        assert_eq!(parsed[0].parameters["x"], 1);
        assert_eq!(parsed[1].name, "func_b");
        assert_eq!(parsed[1].parameters["y"], "hello");
    }

    #[test]
    fn gemma4_tool_call_with_thinking_prefix() {
        let msg = "<|channel>thought\nLet me search.\n<channel|><|tool_call>call:search{q:<|\"|\x3etest<|\"|\x3e}<tool_call|><|tool_response>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "search");
    }

    #[test]
    fn gemma4_incomplete_tool_call_no_brace() {
        // Model has only generated the prefix — should return original, not error
        let msg = "<|tool_call>call:";
        let result = process_model_specific_message(msg).unwrap();
        assert_eq!(result, msg);
    }

    #[test]
    fn gemma4_incomplete_tool_call_unmatched_braces() {
        let msg = "<|tool_call>call:func{key:<|\"|\x3evalue";
        let result = process_model_specific_message(msg).unwrap();
        assert_eq!(result, msg);
    }

    #[test]
    fn gemma4_contains_prefix_detects_pipe_variant() {
        assert!(contains_tool_call_prefix(
            "<|tool_call>call:test{}<tool_call|>"
        ));
        assert!(!contains_tool_call_prefix("some random text"));
    }

    #[test]
    fn gemma4_quote_unquoted_keys() {
        assert_eq!(
            quote_unquoted_keys(r#"{key:"value",num:42}"#),
            r#"{"key":"value","num":42}"#
        );
    }

    #[test]
    fn gemma4_quote_unquoted_keys_nested() {
        assert_eq!(
            quote_unquoted_keys(r#"{outer:{inner:"val"}}"#),
            r#"{"outer":{"inner":"val"}}"#
        );
    }

    #[test]
    fn gemma4_quote_keys_preserves_strings() {
        assert_eq!(
            quote_unquoted_keys(r#"{key:"has:colon,and{brace}"}"#),
            r#"{"key":"has:colon,and{brace}"}"#
        );
    }

    #[test]
    fn gemma4_extract_matched_braces() {
        let s = "{a:1,b:{c:2}}rest";
        let (inner, pos) = extract_matched_braces(s, 0).unwrap();
        assert_eq!(inner, "a:1,b:{c:2}");
        assert_eq!(pos, 13);
    }

    #[test]
    fn gemma4_extract_braces_with_strings() {
        let s = "{key:<|\"|\x3e{not a brace}<|\"|\x3e}after";
        let (inner, _) = extract_matched_braces(s, 0).unwrap();
        assert_eq!(inner, "key:<|\"|\x3e{not a brace}<|\"|\x3e");
    }

    #[test]
    fn gemma4_tool_call_with_array() {
        let msg = "<|tool_call>call:multi{items:[1,2,3]}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].name, "multi");
        assert_eq!(parsed[0].parameters["items"], serde_json::json!([1, 2, 3]));
    }

    #[test]
    fn gemma4_tool_call_empty_args() {
        let msg = "<|tool_call>call:no_args{}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].name, "no_args");
        assert_eq!(parsed[0].parameters, serde_json::json!({}));
    }

    #[test]
    fn gemma4_tool_call_string_with_special_chars() {
        // String containing commas, colons, and braces — must be preserved
        let msg = "<|tool_call>call:test{query:<|\"|\x3ekey:val, {nested}<|\"|\x3e}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].parameters["query"], "key:val, {nested}");
    }

    #[test]
    fn gemma4_tool_call_negative_number() {
        let msg = "<|tool_call>call:offset{x:-5,y:3.14}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert_eq!(parsed[0].parameters["x"], -5);
        assert_eq!(parsed[0].parameters["y"], 3.14);
    }

    #[test]
    fn gemma4_tool_call_null_value() {
        let msg = "<|tool_call>call:test{val:null}<tool_call|>";
        let result = process_model_specific_message(msg).unwrap();
        let parsed: Vec<CalledFunctionParameters> = serde_json::from_str(&result).unwrap();
        assert!(parsed[0].parameters["val"].is_null());
    }

    #[test]
    fn gemma4_incomplete_just_prefix_token() {
        // Just the <|tool_call> token, no "call:" yet
        let msg = "<|tool_call>";
        let result = process_model_specific_message(msg).unwrap();
        assert_eq!(result, msg); // No "call:" prefix found → falls through
    }

    #[test]
    fn gemma4_incomplete_name_no_closing() {
        // Function name started but no closing brace yet
        let msg = "<|tool_call>call:search_the_web{query:<|\"|\x3etest<|\"|\x3e";
        let result = process_model_specific_message(msg).unwrap();
        assert_eq!(result, msg);
    }

    #[test]
    fn gemma4_non_tool_message_unchanged() {
        let msg = "Hello, how can I help you?";
        let result = process_model_specific_message(msg).unwrap();
        assert_eq!(result, msg);
    }

    #[test]
    fn gemma4_args_to_json_mixed_types() {
        let raw = "name:<|\"|\x3eAlice<|\"|\x3e,age:30,active:true,data:{score:9.5}";
        let val = gemma4_args_to_json(raw).unwrap();
        assert_eq!(val["name"], "Alice");
        assert_eq!(val["age"], 30);
        assert_eq!(val["active"], true);
        assert_eq!(val["data"]["score"], 9.5);
    }

    #[test]
    fn gemma4_quote_keys_with_array_of_strings() {
        assert_eq!(
            quote_unquoted_keys(r#"{items:["a","b"]}"#),
            r#"{"items":["a","b"]}"#
        );
    }

    #[test]
    fn gemma4_quote_keys_with_multibyte_string_value() {
        // Multi-byte chars in a string value must not corrupt byte offsets for
        // keys that appear later in the input.
        assert_eq!(
            quote_unquoted_keys(r#"{first:"café",second:42}"#),
            r#"{"first":"café","second":42}"#
        );
    }

    #[test]
    fn gemma4_quote_keys_boolean_values_not_quoted() {
        // true/false/null should NOT get quotes since they're not followed by ':'
        assert_eq!(
            quote_unquoted_keys(r#"{a:true,b:false,c:null}"#),
            r#"{"a":true,"b":false,"c":null}"#
        );
    }
}
