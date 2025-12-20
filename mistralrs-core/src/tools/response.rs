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

#[cfg(feature = "pyo3_macros")]
mod pyo3_impls {
    use super::ToolCallType;
    use pyo3::prelude::*;
    use std::convert::Infallible;

    impl<'py> FromPyObject<'py> for ToolCallType {
        fn extract_bound(obj: &Bound<'py, PyAny>) -> PyResult<Self> {
            let s: &str = obj.extract()?;
            match s {
                "function" => Ok(ToolCallType::Function),
                other => Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "invalid ToolCallType: {other}"
                ))),
            }
        }
    }

    impl<'py> IntoPyObject<'py> for ToolCallType {
        type Target = pyo3::types::PyString;
        type Output = Bound<'py, Self::Target>;
        type Error = Infallible;

        fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
            Ok(pyo3::types::PyString::new(py, &self.to_string()))
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
