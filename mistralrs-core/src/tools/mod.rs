pub(crate) mod grammar;
pub(crate) mod parsers;
mod request;
mod response;
pub(crate) mod state;
pub(crate) mod strategy;

use candle_core::Result;
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
        let message_prefix = process_model_specific_message(message_prefix)?;
        let message_prefix = fix_broken_json(&message_prefix).map_err(candle_core::Error::msg)?;

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
}
