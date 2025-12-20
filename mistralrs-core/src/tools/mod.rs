pub mod mistral_token_parser;
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
        || prefix.contains("<｜tool▁call▁begin｜>")
        || prefix.contains("<|python_tag|>")
        || prefix.contains("[TOOL_CALLS]")
}

/// Normalize model-emitted tool call text into a vector of `CalledFunctionParameters`.
/// Returns `None` when no tool-call sentinel is present or parsing fails.
pub fn normalize_tool_calls(raw: &str) -> Option<Vec<CalledFunctionParameters>> {
    if !contains_tool_call_prefix(raw) {
        return None;
    }
    let normalized = process_model_specific_message(raw).ok()?;
    if let Ok(single) = serde_json::from_str::<CalledFunctionParameters>(&normalized) {
        return Some(vec![single]);
    }
    serde_json::from_str::<Vec<CalledFunctionParameters>>(&normalized).ok()
}

fn process_model_specific_message(message: &str) -> Result<String> {
    static DEEPSEEK_REGEX: OnceLock<Regex> = OnceLock::new();
    static QWEN_REGEX: OnceLock<Regex> = OnceLock::new();
    static CODEX_SHELL_UNESCAPED_QUOTES_REGEX: OnceLock<Regex> = OnceLock::new();

    // These are reasoning models so we need a regex.
    let deepseek_regex = DEEPSEEK_REGEX.get_or_init(|| Regex::new(
        r"(?s)<｜tool▁call▁begin｜>function<｜tool▁sep｜>(?P<name>[^\n]+)\n```json\n(?P<json>.+?)\n```<｜tool▁call▁end｜>",
    ).unwrap());
    let qwen_regex = QWEN_REGEX
        .get_or_init(|| Regex::new(r"(?s)<tool_call>(?P<inner>.*?)</tool_call>").unwrap());

    #[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
    struct ToolCall {
        name: String,
        arguments: Value,
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
    } else if let Some(tool_pos) = message.find("[TOOL_CALLS]") {
        // Some models emit tool calls in a bracketed tag format:
        //   ...[TOOL_CALLS]tool_name[ARGS]{ ...json... }
        //   ...[TOOL_CALLS]tool_name { ...json... }           (Codex-style)
        // Optionally with no whitespace before the JSON:
        //   ...[TOOL_CALLS]tool_name{ ...json... }
        // Optionally repeated multiple times.
        //
        // Normalize to a JSON list of `{ name, arguments }` objects so it can be parsed by
        // `ToolCallingMatcher::get_call`.
        let mut rest = &message[tool_pos..];
        let mut calls = Vec::new();

        // Find the start of a JSON value (`{...}` or `[...]`) without accidentally matching
        // the control tokens themselves (e.g. the `[` in `[ARGS]`).
        let find_json_start = |s: &str| -> Option<usize> {
            for (idx, byte) in s.as_bytes().iter().enumerate() {
                match *byte {
                    b'{' => return Some(idx),
                    b'[' => {
                        let tail = &s[idx..];
                        if tail.starts_with("[TOOL_CALLS]")
                            || tail.starts_with("[ARGS]")
                            || tail.starts_with("[CALL_ID]")
                        {
                            continue;
                        }
                        return Some(idx);
                    }
                    _ => {}
                }
            }
            None
        };

        // If there is no JSON-looking content at all, bail early instead of emitting
        // noisy parse warnings on partial markers like "[TOOL_CALLS]shell[ARGS]".
        if find_json_start(rest).is_none() {
            return Ok(message.to_string());
        }

        // Some models (and especially Codex-style prompting) occasionally produce invalid JSON for
        // tool call arguments, typically by inserting unescaped `"` inside a JSON string (e.g.
        // `find . -name "*.md"`). Try a tiny repair pass for common shell patterns before giving up.
        //
        // NOTE: This intentionally focuses on repairing *likely* shell snippets, not arbitrary JSON,
        // to reduce the risk of corrupting otherwise-valid tool argument payloads.
        let repair_codex_shell_unescaped_quotes = |raw: &str| -> Option<String> {
            let re = CODEX_SHELL_UNESCAPED_QUOTES_REGEX.get_or_init(|| {
                Regex::new(r#"(?P<flag>-(?:name|path|glob|g))\s+"(?P<pat>[^"]+)""#).unwrap()
            });
            let mut current = raw.to_string();
            // Apply repeatedly; nested/duplicated patterns are common in `bash -lc` strings.
            loop {
                let next = re.replace_all(&current, "${flag} '${pat}'").to_string();
                if next == current {
                    break;
                }
                current = next;
            }
            (current != raw).then_some(current)
        };

        let sanitize_shell_args = |args: &mut Value| {
            // Codex CLI generally runs tools in the repo root on the client side; `workdir` is
            // optional and, when malformed, can cause confusing "sandbox" failures.
            let Value::Object(map) = args else { return };

            // Normalize `command` into the expected array form if a model emitted it as a string.
            // Codex expects `command: string[]`.
            if let Some(Value::String(cmd)) = map.get("command").cloned() {
                let cmd = cmd.trim().to_string();
                if !cmd.is_empty() {
                    map.insert(
                        "command".to_string(),
                        Value::Array(vec![
                            Value::String("bash".to_string()),
                            Value::String("-lc".to_string()),
                            Value::String(cmd),
                        ]),
                    );
                }
            }

            // Ensure `workdir` exists.
            if !map.contains_key("workdir") {
                map.insert("workdir".to_string(), Value::String(".".to_string()));
            }
            let Some(Value::String(workdir)) = map.get("workdir").cloned() else {
                return;
            };
            // If the model accidentally concatenated prose into the workdir field, drop it.
            let mut wd = workdir;
            // Hard cut at common sentinels.
            for sentinel in ["\n", "\r", "\t", "[TOOL_CALLS]", "{", "}", "  "] {
                if let Some(pos) = wd.find(sentinel) {
                    wd.truncate(pos);
                }
            }
            wd = wd.trim().to_string();
            // Paths with spaces are extremely uncommon in our usage; treat as malformed.
            if wd.contains(' ') {
                map.insert("workdir".to_string(), Value::String(".".to_string()));
                return;
            }
            if wd.is_empty() {
                map.insert("workdir".to_string(), Value::String(".".to_string()));
                return;
            }
            map.insert("workdir".to_string(), Value::String(wd));
        };

        loop {
            let Some(tool_prefix_pos) = rest.find("[TOOL_CALLS]") else {
                break;
            };
            let after_prefix = &rest[tool_prefix_pos + "[TOOL_CALLS]".len()..];
            let after_prefix = after_prefix.trim_start();
            let Some(json_start) = find_json_start(after_prefix) else {
                break;
            };

            // Everything before the first JSON delimiter is the raw name region; Codex-style
            // outputs sometimes repeat `[ARGS]` tokens (e.g. `shell[ARGS]update_plan[ARGS]{...}`).
            // Use the last segment before JSON as the function name.
            let pre_json = &after_prefix[..json_start];
            let mut name = pre_json;
            if pre_json.contains("[ARGS]") {
                if let Some(seg) = pre_json.rsplit("[ARGS]").find(|s| !s.trim().is_empty()) {
                    name = seg;
                }
            }
            let name = name.trim();
            let name = name.strip_prefix("[TOOL_CALLS]").unwrap_or(name).trim();
            let name = if let Some(pos) = name.find("[CALL_ID]") {
                name[..pos].trim()
            } else {
                name
            };

            let json_src = after_prefix[json_start..].trim_start();
            if name.is_empty() {
                break;
            }

            let try_parse_first_value =
                |src: &str| -> std::result::Result<(Value, usize), serde_json::Error> {
                    let mut stream = serde_json::Deserializer::from_str(src).into_iter::<Value>();
                    let first = stream.next().ok_or_else(|| {
                        serde_json::Error::io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "empty",
                        ))
                    })?;
                    match first {
                        Ok(v) => Ok((v, stream.byte_offset())),
                        Err(e) => Err(e),
                    }
                };

            let mut parsed = match try_parse_first_value(json_src) {
                Ok(ok) => Ok(ok),
                Err(e) => {
                    if let Some(repaired) = repair_codex_shell_unescaped_quotes(json_src) {
                        try_parse_first_value(&repaired)
                    } else {
                        Err(e)
                    }
                }
            };

            let (arguments, offset) = match &mut parsed {
                Ok((arguments, offset)) => (arguments.clone(), *offset),
                Err(e) => {
                    // While streaming we frequently observe incomplete JSON that fails parsing
                    // with various error categories (not always `Error::is_eof()`), e.g.
                    // `EOF while parsing a string` / `EOF while parsing an object`.
                    //
                    // Heuristic: since our tool args are expected to start with `{` or `[`,
                    // treat the JSON as *incomplete* unless the text currently ends with the
                    // matching closing delimiter (`}` for `{...}`, `]` for `[...]`).
                    if e.is_eof() {
                        // Partial JSON while streaming: defer parsing until more text arrives.
                        return Ok(message.to_string());
                    }

                    // Malformed JSON: break the sentinel so `prefix_could_be_tool` returns false
                    // and streaming can continue instead of hanging forever.
                    return Ok(message.replacen("[TOOL_CALLS]", "TOOL_CALLS", 1));
                }
            };
            let mut arguments = arguments;
            if name == "shell" {
                sanitize_shell_args(&mut arguments);
            }
            calls.push(ToolCall {
                name: name.to_string(),
                arguments,
            });

            rest = &json_src[offset..];
            rest = rest.trim_start();
        }
        if !calls.is_empty() {
            return serde_json::to_string(&calls).map_err(candle_core::Error::msg);
        }
        Ok(message.to_string())
    } else if deepseek_regex.find(message).is_some() {
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

        // If a Codex-style marker is present but there is no JSON delimiter yet, don't block
        // streaming; treat it as ordinary text until a `{`/`[` arrives.
        if message_prefix.contains("[TOOL_CALLS]")
            && !message_prefix.contains('{')
            && !message_prefix.contains('[')
        {
            return Ok((false, false));
        }

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
        // Fallback: don't block streaming on malformed `[TOOL_CALLS]` text.
        .unwrap_or((false, false)))
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
    fn parses_tool_calls_bracket_args_format_from_anywhere_in_message() {
        let msg = r#"Let me search.[TOOL_CALLS]shell[ARGS]{"command":["find","/tmp","-name","*.md"],"workdir":"/tmp"}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].parameters["workdir"], "/tmp");
        assert_eq!(calls[0].parameters["command"][0], "find");
    }

    #[test]
    fn parses_tool_calls_bracket_args_format_no_space() {
        let msg = r#"[TOOL_CALLS]shell[ARGS]{"command":["ls","-la"],"workdir":"/tmp"}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].parameters["command"][0], "ls");
    }

    #[test]
    fn parses_multiple_tool_calls_bracket_args_format() {
        let msg = r#"[TOOL_CALLS]a[ARGS]{"x":1}   [TOOL_CALLS]b[ARGS]{"y":2}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].parameters["x"], 1);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].parameters["y"], 2);
    }

    #[test]
    fn parses_tool_calls_bracket_space_json_format() {
        let msg = r#"[TOOL_CALLS]shell {"command":["cat","plan.md"],"workdir":"/tmp"}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].parameters["command"][0], "cat");
        assert_eq!(calls[0].parameters["workdir"], "/tmp");
    }

    #[test]
    fn parses_tool_calls_bracket_no_space_json_format() {
        let msg = r#"[TOOL_CALLS]shell{"command":["ls","-la"],"workdir":"/tmp"}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].parameters["command"][0], "ls");
    }

    #[test]
    fn parses_multiple_tool_calls_mixed_formats() {
        let msg = r#"One.[TOOL_CALLS]a {"x":1} Two.[TOOL_CALLS]b[ARGS]{"y":2}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "a");
        assert_eq!(calls[0].parameters["x"], 1);
        assert_eq!(calls[1].name, "b");
        assert_eq!(calls[1].parameters["y"], 2);
    }

    #[test]
    fn does_not_break_tool_call_sentinel_on_incomplete_json() {
        // This is a common streaming prefix: JSON started but is incomplete.
        let msg = r#"[TOOL_CALLS]shell[ARGS]{"command":["bash","-lc","ls -la"],"workdir":"#;
        let normalized = process_model_specific_message(msg).unwrap();
        // Must not replace the sentinel (otherwise streaming will treat it as normal text).
        assert!(normalized.contains("[TOOL_CALLS]"));
        // And it should not have been normalized into a JSON calls list yet.
        assert!(!normalized.trim_start().starts_with("[{"));
    }

    #[test]
    fn does_not_break_tool_call_sentinel_on_args_marker_without_json() {
        // This is another common streaming prefix: tool name + [ARGS], but the JSON hasn't
        // started yet. We must not mis-detect the `[` in `[ARGS]` as a JSON array delimiter.
        let msg = r#"[TOOL_CALLS]shell[ARGS]"#;
        let normalized = process_model_specific_message(msg).unwrap();
        assert_eq!(normalized, msg);
    }

    #[test]
    fn repairs_common_shell_unescaped_quotes_in_copied_commands() {
        // This is invalid JSON (the `"*.md"` is unescaped inside a JSON string), but the repair
        // pass should rewrite it to use single quotes so it becomes parseable.
        let msg = r##"[TOOL_CALLS]shell[ARGS]{"command":["bash","-lc","find . -name "*.md" -type f"],"workdir":"/tmp"}"##;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].parameters["command"][2],
            "find . -name '*.md' -type f"
        );
    }

    #[test]
    fn repairs_shell_unescaped_quotes_and_sanitizes_malformed_workdir() {
        // This is invalid JSON (the `".md"` is unescaped inside a JSON string), and the workdir
        // field contains appended prose. The repair should make it parseable, and the sanitizer
        // should clamp the workdir to ".".
        let msg = r##"[TOOL_CALLS]shell[ARGS]{"command":["bash","-lc","find . -name ".md" -type f"],"workdir":"/weka/users/gu to see what files are here"}"##;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(
            calls[0].parameters["command"][2],
            "find . -name '.md' -type f"
        );
        assert_eq!(calls[0].parameters["workdir"], ".");
    }

    #[test]
    fn normalize_tool_calls_handles_complete_payload() {
        let s =
            "[TOOL_CALLS]shell[ARGS]{\"command\":[\"bash\",\"-lc\",\"ls\"],\"workdir\":\"/tmp\"}";
        let parsed = normalize_tool_calls(s).expect("should parse");
        assert_eq!(parsed.len(), 1);
        let call = &parsed[0];
        assert_eq!(call.name, "shell");
        assert_eq!(call.parameters["command"][0], "bash");
        assert_eq!(call.parameters["workdir"], "/tmp");
    }

    #[test]
    fn normalize_tool_calls_succeeds_after_incremental_buffering() {
        let chunks = [
            "[TOOL_CALLS]shell[ARGS]{\"comm",
            "and\":[\"bash\",\"-lc\",\"find\"],\"w",
            "orkdir\":\"/home/user\"}",
        ];
        let mut buf = String::new();
        let mut parsed = None;
        for c in chunks {
            buf.push_str(c);
            parsed = normalize_tool_calls(&buf);
        }
        let parsed = parsed.expect("should parse after final chunk");
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].name, "shell");
        assert_eq!(parsed[0].parameters["workdir"], "/home/user");
    }

    #[test]
    fn normalize_tool_calls_returns_none_for_plain_text() {
        assert!(normalize_tool_calls("hello world").is_none());
        assert!(normalize_tool_calls("[TOOL_CALLS]garbled").is_none());
    }

    #[test]
    fn normalizes_shell_command_string_to_array() {
        // Some agents emit `command` as a single string, but our `shell` tool expects an array.
        let msg = r#"[TOOL_CALLS]shell[ARGS]{"command":"ls -la","workdir":"/tmp"}"#;
        let normalized = process_model_specific_message(msg).unwrap();
        let calls: Vec<CalledFunctionParameters> = serde_json::from_str(&normalized).unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "shell");
        assert_eq!(calls[0].parameters["command"][0], "bash");
        assert_eq!(calls[0].parameters["command"][1], "-lc");
        assert_eq!(calls[0].parameters["command"][2], "ls -la");
        assert_eq!(calls[0].parameters["workdir"], "/tmp");
    }
}
