use serde::{Deserialize, Serialize};

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ExecutorRequest {
    #[serde(rename = "execute")]
    Execute { code: String },
    #[serde(rename = "reset")]
    Reset,
}

#[derive(Debug, Deserialize)]
pub struct ExecuteResponse {
    pub stdout: String,
    pub stderr: String,
    pub exception: Option<String>,
    pub last_expr_repr: Option<String>,
    pub last_expr_type: Option<String>,
    #[serde(default)]
    pub images: Vec<ImageOutput>,
    #[serde(default)]
    pub execution_time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct ImageOutput {
    #[allow(dead_code)]
    pub format: String,
    pub data_base64: String,
}

#[derive(Debug, Deserialize)]
pub struct ResetResponse {
    #[allow(dead_code)]
    pub success: bool,
}
