use std::collections::HashMap;

use serde_json::Value;

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    /// Disallow selection of tools.
    None,
    #[serde(rename = "auto")]
    /// Allow automatic selection of any given tool, or none.
    Auto,
    #[serde(untagged)]
    /// Force selection of a given tool.
    Tool(Tool),
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    #[serde(alias = "arguments")]
    pub parameters: Option<HashMap<String, Value>>,
}

#[cfg_attr(feature = "utoipa", derive(utoipa::ToSchema))]
#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}
