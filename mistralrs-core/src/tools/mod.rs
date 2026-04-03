pub(crate) mod parsers;
mod request;
mod response;

use candle_core::Result;
pub use request::*;
pub use response::*;
use serde::de::{self, Deserializer, MapAccess, Visitor};
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use uuid::Uuid;

use crate::Pipeline;
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

