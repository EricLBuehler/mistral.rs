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

/// Type of tool.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

/// Function definition for a tool.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: Option<HashMap<String, Value>>,
}

/// Tool definition.
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}

/// Called function with name and arguments.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}
