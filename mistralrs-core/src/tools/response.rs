#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass(eq, eq_int))]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ToolCallType {
    Function,
}

#[cfg_attr(feature = "pyo3_macros", pyo3::pyclass)]
#[cfg_attr(feature = "pyo3_macros", pyo3(get_all))]
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
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
