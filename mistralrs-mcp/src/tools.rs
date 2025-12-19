use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use std::sync::Arc;

/// Callback used for custom tool functions. Receives the called function
/// (name and JSON arguments) and returns the tool output as a string.
pub type ToolCallback = dyn Fn(&CalledFunction) -> anyhow::Result<String> + Send + Sync;

/// A tool callback with its associated Tool definition.
#[derive(Clone)]
pub struct ToolCallbackWithTool {
    pub callback: Arc<ToolCallback>,
    pub tool: Tool,
}

/// Collection of callbacks keyed by tool name.
pub type ToolCallbacks = HashMap<String, Arc<ToolCallback>>;

/// Collection of callbacks with their tool definitions keyed by tool name.
pub type ToolCallbacksWithTools = HashMap<String, ToolCallbackWithTool>;

/// Type of tool
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

/// Function definition for a tool
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: Option<HashMap<String, Value>>,
}

/// Tool definition
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum ToolDeserializeRepr {
    /// OpenAI Chat Completions-style tool definition:
    /// `{ "type": "function", "function": { "name": "...", ... } }`
    Nested {
        #[serde(rename = "type")]
        tp: ToolType,
        function: Function,
    },
    /// OpenAI Responses API-style tool definition:
    /// `{ "type": "function", "name": "...", "description": "...", "parameters": { ... } }`
    Flat {
        #[serde(rename = "type")]
        tp: ToolType,
        name: String,
        description: Option<String>,
        #[serde(alias = "arguments")]
        parameters: Option<HashMap<String, Value>>,
    },
}

impl<'de> Deserialize<'de> for Tool {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match ToolDeserializeRepr::deserialize(deserializer)? {
            ToolDeserializeRepr::Nested { tp, function } => Ok(Self { tp, function }),
            ToolDeserializeRepr::Flat {
                tp,
                name,
                description,
                parameters,
            } => Ok(Self {
                tp,
                function: Function {
                    name,
                    description,
                    parameters,
                },
            }),
        }
    }
}

/// Called function with name and arguments
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}
