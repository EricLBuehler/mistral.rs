use std::collections::HashMap;

use serde_json::Value;

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolChoice {
    #[serde(rename = "none")]
    None,
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "required")]
    Required,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Function {
    pub description: Option<String>,
    pub name: String,
    pub parameters: Option<HashMap<String, Value>>,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct Tool {
    pub tp: ToolType,
    pub function: Function,
}
