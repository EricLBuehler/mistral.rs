#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    Function,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize)]
pub struct ToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: ToolCallType,
    pub function: CalledFunction,
}



#[cfg(test)]
mod test {
    use super::*;
    use serde_json::json;

    // test the serde of the ToolCallResponse enum 
    #[test]
    fn test_tool_call_response_serde() {
        let tool_call_response = ToolCallResponse {
            id: "1".to_string(),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: "test".to_string(),
                arguments: "test".to_string(),
            },
        };

        let json = json!({
            "id": "1",
            "type": "function",
            "function": {
                "name": "test",
                "arguments": "test"
            }
        });


        let string_json = serde_json::to_string(&tool_call_response).unwrap();

        assert_eq!(string_json, json.to_string());

    }
}

