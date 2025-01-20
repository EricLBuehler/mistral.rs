mod request;
mod response;

pub use request::*;
pub use response::*;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

pub struct ToolCallingMatcher {
    tool_choice: ToolChoice,
}

// Same as CalledFunction, but uses `parameters`
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionParameters {
    pub name: String,
    pub parameters: HashMap<String, Value>,
}

// Same as CalledFunction, but uses `arguments``
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CalledFunctionArguments {
    pub name: String,
    pub arguments: HashMap<String, Value>,
}

impl ToolCallingMatcher {
    pub fn new(tool_choice: ToolChoice) -> anyhow::Result<Self> {
        Ok(Self { tool_choice })
    }

    // Checks if the the `message_prefix` could be a tool call.
    // If false, either [`ToolChoice::None`] was selected, or the prefix could not match.
    pub fn prefix_could_be_tool(&self, message_prefix: &str) -> bool {
        if matches!(self.tool_choice, ToolChoice::None) {
            return false;
        }

        // Check if the prefix could be a JSON serialization of any of the following types.
        [
            could_be_json::<CalledFunctionParameters>,
            could_be_json::<CalledFunctionArguments>,
            could_be_json::<Vec<CalledFunctionParameters>>,
            could_be_json::<Vec<CalledFunctionArguments>>,
        ]
        .iter()
        .any(|check| check(message_prefix))
    }

    pub fn get_call(&self, message: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
        if matches!(self.tool_choice, ToolChoice::None) {
            return Ok(Vec::new());
        }

        if let Ok(deser) = serde_json::from_str::<CalledFunctionParameters>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.parameters)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionParameters>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.parameters)?,
                        },
                    })
                })
                .collect::<anyhow::Result<Vec<_>>>()?)
        } else if let Ok(deser) = serde_json::from_str::<CalledFunctionArguments>(message) {
            let id = format!("call-{}", Uuid::new_v4());
            Ok(vec![ToolCallResponse {
                id,
                tp: ToolCallType::Function,
                function: CalledFunction {
                    name: deser.name,
                    arguments: serde_json::to_string(&deser.arguments)?,
                },
            }])
        } else if let Ok(deser) = serde_json::from_str::<Vec<CalledFunctionArguments>>(message) {
            Ok(deser
                .into_iter()
                .map(|deser| {
                    let id = format!("call-{}", Uuid::new_v4());
                    Ok(ToolCallResponse {
                        id,
                        tp: ToolCallType::Function,
                        function: CalledFunction {
                            name: deser.name,
                            arguments: serde_json::to_string(&deser.arguments)?,
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

/// Checks if the given prefix could be the start of the JSON serialization of a given type, `T`.
fn could_be_json<T>(text_prefix: &str) -> bool
where
    T: serde::de::DeserializeOwned,
{
    match serde_json::from_str::<T>(text_prefix) {
        Ok(_) => true,
        Err(e) if e.is_eof() => true,
        _ => false,
    }
}
