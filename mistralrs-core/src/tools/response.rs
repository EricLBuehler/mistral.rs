/// The type of a tool call (currently only function calls).
#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    Function,
}

impl std::fmt::Display for ToolCallType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolCallType::Function => write!(f, "function"),
        }
    }
}

use mistralrs_mcp::CalledFunction;

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCallResponse {
    pub index: usize,
    pub id: String,
    #[serde(rename = "type")]
    pub tp: ToolCallType,
    pub function: CalledFunction,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializes_index_field() {
        let resp = ToolCallResponse {
            index: 0,
            id: "call-1".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "foo".to_string(),
                arguments: "{}".to_string(),
            },
        };

        let json = serde_json::to_value(&resp).unwrap();
        assert_eq!(json.get("index").and_then(|v| v.as_u64()), Some(0));
    }
}
