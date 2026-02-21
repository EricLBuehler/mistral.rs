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

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_type_serializes_as_function() {
        let v = serde_json::to_value(ToolType::Function).unwrap();
        assert_eq!(v, json!("function"));
    }

    #[test]
    fn function_accepts_parameters_alias_arguments() {
        let v = json!({
            "name": "weather",
            "description": "Get weather",
            "arguments": { "city": "Paris" }
        });
        let f: Function = serde_json::from_value(v).unwrap();
        assert_eq!(f.name, "weather");
        assert_eq!(
            f.parameters.unwrap().get("city").unwrap(),
            &json!("Paris")
        );
    }

    #[test]
    fn tool_roundtrip_stable_shape() {
        let tool = Tool {
            tp: ToolType::Function,
            function: Function {
                description: Some("desc".to_string()),
                name: "search".to_string(),
                parameters: Some(HashMap::from([("q".to_string(), json!("term"))])),
            },
        };
        let v = serde_json::to_value(&tool).unwrap();
        assert_eq!(v.get("type").unwrap(), "function");
        assert_eq!(v.get("function").unwrap().get("name").unwrap(), "search");
        let de: Tool = serde_json::from_value(v).unwrap();
        assert_eq!(de.function.name, "search");
    }

    #[test]
    fn called_function_roundtrip() {
        let called = CalledFunction {
            name: "tool_name".to_string(),
            arguments: "{\"a\":1}".to_string(),
        };
        let v = serde_json::to_value(&called).unwrap();
        let de: CalledFunction = serde_json::from_value(v).unwrap();
        assert_eq!(de.name, "tool_name");
        assert_eq!(de.arguments, "{\"a\":1}");
    }
}
