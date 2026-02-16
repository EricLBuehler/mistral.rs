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
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}

#[cfg(all(test, feature = "mcp"))]
mod tests {
    use super as local;

    #[test]
    fn tool_types_match_mcp_shapes() {
        let local_tool = local::Tool {
            tp: local::ToolType::Function,
            function: local::Function {
                description: Some("d".to_string()),
                name: "n".to_string(),
                parameters: None,
            },
        };
        let mcp_tool = mistralrs_mcp::Tool {
            tp: mistralrs_mcp::ToolType::Function,
            function: mistralrs_mcp::Function {
                description: Some("d".to_string()),
                name: "n".to_string(),
                parameters: None,
            },
        };
        assert_eq!(
            serde_json::to_value(local_tool).unwrap(),
            serde_json::to_value(mcp_tool).unwrap()
        );

        let local_called = local::CalledFunction {
            name: "n".to_string(),
            arguments: "{}".to_string(),
        };
        let mcp_called = mistralrs_mcp::CalledFunction {
            name: "n".to_string(),
            arguments: "{}".to_string(),
        };
        assert_eq!(
            serde_json::to_value(local_called).unwrap(),
            serde_json::to_value(mcp_called).unwrap()
        );
    }
}
