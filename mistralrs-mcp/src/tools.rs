use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
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
    /// When `true`, the tool's `parameters` JSON schema is enforced on the
    /// generated arguments via constrained decoding (llguidance).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub strict: Option<bool>,
}

impl Function {
    /// Returns the parameters as a JSON Schema [`Value`] when strict mode is
    /// enabled.  Returns `None` when strict is absent or `false`.
    pub fn strict_parameters_schema(&self) -> Option<Value> {
        if self.strict != Some(true) {
            return None;
        }
        match &self.parameters {
            Some(p) => match serde_json::to_value(p) {
                Ok(v) => Some(v),
                Err(e) => {
                    tracing::warn!(
                        "Failed to serialize parameters for strict tool `{}`: {e}. \
                         Falling back to generic object schema.",
                        self.name,
                    );
                    Some(json!({"type": "object"}))
                }
            },
            None => {
                tracing::warn!(
                    "Tool `{}` has strict: true but no parameters schema defined. \
                     Cannot enforce strict mode — falling back to generic object schema.",
                    self.name,
                );
                Some(json!({"type": "object"}))
            }
        }
    }
}

/// Tool definition
#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}

/// Called function with name and arguments
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}
