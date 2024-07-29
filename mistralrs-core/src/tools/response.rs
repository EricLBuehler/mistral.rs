#[derive(Clone, Debug, serde::Deserialize)]
#[serde(untagged)]
pub enum ToolCallType {
    #[serde(rename = "function")]
    Function,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct CalledFunction {
    pub name: String,
    pub arguments: String,
}

#[derive(Clone, Debug, serde::Deserialize)]
pub struct ToolCallResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub tp: ToolCallType,
    pub function: CalledFunction,
}
