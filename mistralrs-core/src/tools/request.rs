use std::collections::HashMap;

use serde_json::Value;

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(untagged)]
    Tool(Tool),
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    pub parameters: Option<HashMap<String, Value>>,
}

#[derive(Clone, Debug, serde::Deserialize, serde::Serialize)]
pub struct Tool {
    #[serde(rename = "type")]
    pub tp: ToolType,
    pub function: Function,
}
