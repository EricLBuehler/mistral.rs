pub(crate) mod grammar;
pub(crate) mod parsers;
mod request;
mod response;
pub(crate) mod state;
pub(crate) mod strategy;

use candle_core::Result;
pub(crate) use parsers::specialize_required_tool_call_grammar;
pub(crate) use parsers::ToolCallFormat;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
pub(crate) use state::ToolCallState;
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

use mistralrs_mcp::CalledFunction;

pub use mistralrs_mcp::{ToolCallback, ToolCallbackWithTool};

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

fn contains_tool_call_prefix(prefix: &str) -> bool {
    parsers::contains_tool_call_prefix(prefix)
}

fn process_model_specific_message(message: &str) -> Result<String> {
    parsers::process_model_specific_message(message)
}

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
    known_tool_names: Option<std::collections::HashSet<String>>,
    tools: Option<Arc<Vec<crate::Tool>>>,
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
    #[cfg(test)]
    pub fn new(tool_choice: ToolChoice, tools: Option<&[crate::Tool]>) -> anyhow::Result<Self> {
        Self::new_with_format(tool_choice, tools, None)
    }

    pub fn new_with_format(
        tool_choice: ToolChoice,
        tools: Option<&[crate::Tool]>,
        _preferred_tool_call_format: Option<ToolCallFormat>,
    ) -> anyhow::Result<Self> {
        let selected_tools = match &tool_choice {
            ToolChoice::Builtin(choice) => {
                anyhow::bail!(
                    "tool_choice forcing hosted tool `{}` is not supported.",
                    choice.tp.kind()
                );
            }
            ToolChoice::AllowedTools(choice) => {
                let tools = tools.unwrap_or_default();
                let mut seen = std::collections::HashSet::new();
                let mut matching_tools = Vec::new();
                for allowed_tool in &choice.tools {
                    let AllowedToolChoice::Function { name } = allowed_tool else {
                        anyhow::bail!(
                            "tool_choice.allowed_tools contains hosted tool `{}`; hosted tool forcing is not supported.",
                            allowed_tool.kind()
                        );
                    };
                    if !seen.insert(name.as_str()) {
                        continue;
                    }
                    let Some(tool) = tools.iter().find(|tool| tool.function.name == *name) else {
                        anyhow::bail!("tool_choice references unknown tool `{name}`.");
                    };
                    matching_tools.push(tool.clone());
                }
                if matching_tools.is_empty() {
                    anyhow::bail!("tool_choice.allowed_tools requires at least one function tool.");
                }
                Some(matching_tools)
            }
            _ => {
                if let Some(name) = tool_choice.forced_function_name() {
                    let tools = tools.unwrap_or_default();
                    let matching_tools = tools
                        .iter()
                        .filter(|tool| tool.function.name == name)
                        .cloned()
                        .collect::<Vec<_>>();
                    if matching_tools.is_empty() {
                        anyhow::bail!("tool_choice references unknown tool `{name}`.");
                    }
                    Some(matching_tools)
                } else {
                    tools.map(|tools| tools.to_vec())
                }
            }
        };
        let known_tool_names = selected_tools.as_ref().map(|t| {
            t.iter()
                .map(|tool| tool.function.name.clone())
                .collect::<std::collections::HashSet<_>>()
        });
        let tools_arc = selected_tools.map(Arc::new);
        Ok(Self {
            tool_choice,
            known_tool_names,
            tools: tools_arc,
        })
    }

    pub fn requires_tool_call(&self) -> bool {
        self.tool_choice.requires_tool_call()
    }

    pub(crate) fn allows_tool_call(&self) -> bool {
        !matches!(self.tool_choice, ToolChoice::None)
    }

    pub(crate) fn tools(&self) -> Option<&[crate::Tool]> {
        self.tools.as_ref().map(|tools| tools.as_slice())
    }

    // Checks if the `message_prefix` could be a tool call. If false, either
    // [`ToolChoice::None`] was selected, or the prefix could not match.
    //
    // If the start of a message could be a tool call, then it looks like an incomplete JSON of a given structure, e.g. `{"name": "foo", "param`.
    //
    // Returns a tuple of `(could_be_tool, is_complete_tool)`.
    pub fn prefix_could_be_tool(&self, message_prefix: &str) -> Result<(bool, bool)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((false, false));
        }
        let raw_prefix = message_prefix;
        let message_prefix = process_model_specific_message(message_prefix)?;
        let message_prefix = fix_broken_json(&message_prefix).map_err(candle_core::Error::msg)?;

        // Check if the prefix could be a JSON serialization of any of the following types.
        let (could_be_tool, is_complete_tool) = [
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
        .unwrap_or((contains_tool_call_prefix(&message_prefix), false));
        // Qwen emits parallel calls as consecutive blocks; keep buffering until the model closes the turn
        if is_complete_tool && qwen_more_calls_possible(raw_prefix) {
            return Ok((true, false));
        }
        Ok((could_be_tool, is_complete_tool))
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        self.get_call_with_content(message).map(|(_, calls)| calls)
    }

    pub fn get_call_with_content(
        &self,
        message: &str,
    ) -> anyhow::Result<(Option<String>, Vec<ToolCallResponse>)> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok((Some(message.to_string()), Vec::new()));
        }
        let (message, content) =
            if let Some((message, content)) = parsers::extract_model_specific_message(message)? {
                let content = content.trim_start().to_string();
                let content = if content.is_empty() {
                    None
                } else {
                    Some(content)
                };
                (message, content)
            } else {
                (process_model_specific_message(message)?, None)
            };
        let message = fix_broken_json(&message)?;

        let mut calls = if let Ok(deser) =
            serde_json::from_str::<CalledFunctionParameters>(&message)
        {
            let id = format!("call-{}", Uuid::new_v4());
            vec![ToolCallResponse {
                index: 0,
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }]
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(&message) {
            deser
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
                .collect::<anyhow::Result<Vec<_>>>()?
        } else {
            if self.tool_choice.requires_tool_call() {
                anyhow::bail!("Tool choice was required but no tools were called.")
            }
            return Ok((Some(message), Vec::new()));
        };

        if let Some(tools) = self.tools.as_deref() {
            coerce_arguments_by_schema(tools, &mut calls)?;
        }

        // Filter out hallucinated tool names.
        if let Some(ref known) = self.known_tool_names {
            let before = calls.len();
            calls.retain(|tc| {
                let valid = known.contains(&tc.function.name);
                if !valid {
                    tracing::warn!(
                        "Dropping hallucinated tool call `{}` (not in defined tools: {:?})",
                        tc.function.name,
                        known
                    );
                }
                valid
            });
            if calls.is_empty() && before > 0 && self.tool_choice.requires_tool_call() {
                anyhow::bail!("Tool choice was required but model called unknown tools.");
            }
        }

        Ok((content, calls))
    }
}

const QWEN_TOOL_CALL_OPEN: &str = "<tool_call>";
const QWEN_TOOL_CALL_CLOSE: &str = "</tool_call>";

// True while the text after the last complete <tool_call> block is empty or an unfinished opener of the next block
fn qwen_more_calls_possible(text: &str) -> bool {
    let Some(last_close) = text.rfind(QWEN_TOOL_CALL_CLOSE) else {
        return false;
    };
    let tail = text[last_close + QWEN_TOOL_CALL_CLOSE.len()..].trim_start();
    tail.is_empty()
        || QWEN_TOOL_CALL_OPEN.starts_with(tail)
        || tail.starts_with(QWEN_TOOL_CALL_OPEN)
}

/// XML-style tool formats (Qwen3.5) carry every argument as text; the tool schema decides its JSON type (vLLM parity).
fn coerce_arguments_by_schema(
    tools: &[crate::Tool],
    calls: &mut [ToolCallResponse],
) -> anyhow::Result<()> {
    for call in calls.iter_mut() {
        let Some(tool) = tools
            .iter()
            .find(|tool| tool.function.name == call.function.name)
        else {
            continue;
        };
        let Some(properties) = tool
            .function
            .parameters
            .as_ref()
            .and_then(|parameters| parameters.get("properties"))
            .and_then(Value::as_object)
        else {
            continue;
        };
        let Ok(Value::Object(mut arguments)) =
            serde_json::from_str::<Value>(&call.function.arguments)
        else {
            continue;
        };
        for (key, value) in arguments.iter_mut() {
            let Value::String(raw) = value else {
                continue;
            };
            let param_type = properties
                .get(key)
                .and_then(|property| property.get("type"))
                .and_then(Value::as_str)
                .unwrap_or("string");
            *value = coerce_param_value(raw, param_type);
        }
        call.function.arguments = serde_json::to_string(&arguments)?;
    }
    Ok(())
}

fn coerce_param_value(raw: &str, param_type: &str) -> Value {
    if raw.eq_ignore_ascii_case("null") {
        return Value::Null;
    }
    let param_type = param_type.trim().to_ascii_lowercase();
    let integer_like = ["int", "uint", "long", "short", "unsigned"]
        .iter()
        .any(|prefix| param_type.starts_with(prefix));
    let number_like = param_type.starts_with("num") || param_type.starts_with("float");
    let structured = matches!(param_type.as_str(), "object" | "array" | "arr" | "sequence")
        || param_type.starts_with("dict")
        || param_type.starts_with("list");
    if integer_like {
        raw.trim()
            .parse::<i64>()
            .map(Value::from)
            .unwrap_or_else(|_| Value::String(raw.to_string()))
    } else if number_like {
        let trimmed = raw.trim();
        trimmed
            .parse::<i64>()
            .map(Value::from)
            .or_else(|_| trimmed.parse::<f64>().map(Value::from))
            .unwrap_or_else(|_| Value::String(raw.to_string()))
    } else if matches!(param_type.as_str(), "boolean" | "bool" | "binary") {
        Value::Bool(raw.trim().eq_ignore_ascii_case("true"))
    } else if structured {
        serde_json::from_str(raw.trim()).unwrap_or_else(|_| Value::String(raw.to_string()))
    } else {
        Value::String(raw.to_string())
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Function, Tool, ToolType};
    use serde_json::json;

    fn test_tool(name: &str) -> Tool {
        Tool {
            tp: ToolType::Function,
            function: Function {
                description: None,
                name: name.to_string(),
                parameters: None,
                strict: None,
            },
        }
    }

    #[test]
    fn deserializes_responses_named_function_tool_choice() {
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_weather" })).unwrap();

        let ToolChoice::NamedFunction(choice) = choice else {
            panic!("expected named function tool choice");
        };
        assert_eq!(choice.name, "get_weather");
    }

    #[test]
    fn deserializes_chat_function_tool_choice() {
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "function",
            "function": { "name": "get_weather" }
        }))
        .unwrap();

        let ToolChoice::Tool(tool) = choice else {
            panic!("expected chat function tool choice");
        };
        assert_eq!(tool.function.name, "get_weather");
    }

    #[test]
    fn tool_call_allowed_tools_deserializes_required_function_subset() {
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{ "type": "function", "name": "get_weather" }]
        }))
        .unwrap();

        let ToolChoice::AllowedTools(choice) = choice else {
            panic!("expected allowed_tools tool choice");
        };
        assert_eq!(choice.mode, AllowedToolsMode::Required);
        assert_eq!(choice.tools.len(), 1);
    }

    #[test]
    fn specific_tool_choice_rejects_unknown_tool() {
        let tools = vec![test_tool("get_weather")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_customer" })).unwrap();

        assert!(ToolCallingMatcher::new(choice, Some(&tools)).is_err());
    }

    #[test]
    fn specific_tool_choice_constrains_called_tool() {
        let tools = vec![test_tool("get_weather"), test_tool("get_customer")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "function", "name": "get_weather" })).unwrap();
        let matcher = ToolCallingMatcher::new(choice, Some(&tools)).unwrap();

        assert!(matcher
            .get_call(r#"{"name":"get_customer","parameters":{}}"#)
            .is_err());
        let calls = matcher
            .get_call(r#"{"name":"get_weather","parameters":{}}"#)
            .unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_allowed_tools_required_constrains_called_tool() {
        let tools = vec![test_tool("get_weather"), test_tool("get_customer")];
        let choice: ToolChoice = serde_json::from_value(json!({
            "type": "allowed_tools",
            "mode": "required",
            "tools": [{ "type": "function", "name": "get_weather" }]
        }))
        .unwrap();
        let matcher = ToolCallingMatcher::new(choice, Some(&tools)).unwrap();

        assert!(matcher.requires_tool_call());
        assert!(matcher
            .get_call(r#"{"name":"get_customer","parameters":{}}"#)
            .is_err());
        let calls = matcher
            .get_call(r#"{"name":"get_weather","parameters":{}}"#)
            .unwrap();
        assert_eq!(calls[0].function.name, "get_weather");
    }

    #[test]
    fn tool_call_rejects_forced_hosted_tool_choice() {
        let tools = vec![test_tool("get_weather")];
        let choice: ToolChoice =
            serde_json::from_value(json!({ "type": "web_search_preview" })).unwrap();

        assert!(matches!(choice, ToolChoice::Builtin(_)));
        assert!(ToolCallingMatcher::new(choice, Some(&tools)).is_err());
    }

    #[test]
    fn qwen_xml_arguments_follow_the_tool_schema() {
        let mut tool = test_tool("get_weather");
        tool.function.parameters = Some(
            serde_json::from_value(json!({
                "type": "object",
                "properties": {
                    "city": { "type": "string" },
                    "days": { "type": "integer" },
                    "units": { "type": "string" },
                    "detailed": { "type": "boolean" },
                    "coords": { "type": "array" }
                }
            }))
            .unwrap(),
        );
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&[tool])).unwrap();
        let calls = matcher
            .get_call(
                "<tool_call>\n<function=get_weather>\n<parameter=city>\n12345\n</parameter>\n<parameter=days>\n3\n</parameter>\n<parameter=units>\n\"C\"\n</parameter>\n<parameter=detailed>\ntrue\n</parameter>\n<parameter=coords>\n[1, 2]\n</parameter>\n</function>\n</tool_call>",
            )
            .unwrap();
        let args: Value = serde_json::from_str(&calls[0].function.arguments).unwrap();
        assert_eq!(args["city"], "12345");
        assert_eq!(args["days"], 3);
        assert_eq!(args["units"], "\"C\"");
        assert_eq!(args["detailed"], true);
        assert_eq!(args["coords"], json!([1, 2]));
    }

    #[test]
    fn qwen_parallel_calls_keep_buffering_until_the_turn_ends() {
        let tools = vec![test_tool("get_weather"), test_tool("get_time")];
        let matcher = ToolCallingMatcher::new(ToolChoice::Auto, Some(&tools)).unwrap();
        let first = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>";
        assert_eq!(matcher.prefix_could_be_tool(first).unwrap(), (true, false));
        assert_eq!(
            matcher
                .prefix_could_be_tool(&format!("{first}\n<tool_call>\n<function=get_time>"))
                .unwrap(),
            (true, false)
        );
        assert_eq!(
            matcher
                .prefix_could_be_tool(&format!("{first}\nDone."))
                .unwrap(),
            (false, true)
        );
        let both = format!(
            "{first}\n<tool_call>\n<function=get_time>\n<parameter=zone>\nUTC\n</parameter>\n</function>\n</tool_call>"
        );
        assert_eq!(matcher.get_call(&both).unwrap().len(), 2);
    }
}
