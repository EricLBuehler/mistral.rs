use serde::{Deserialize, Serialize};

/// One requested output file. The executor reads it from the working dir after user code runs.
#[derive(Debug, Clone, Serialize, Default)]
pub struct ExecuteOutputSpec {
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
}

#[derive(Serialize)]
#[serde(tag = "type")]
pub enum ExecutorRequest<'a> {
    #[serde(rename = "execute")]
    Execute {
        code: String,
        #[serde(skip_serializing_if = "<[_]>::is_empty")]
        outputs: &'a [ExecuteOutputSpec],
    },
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
    pub video_frames: Vec<ImageOutput>,
    /// One entry per name in the request's `outputs` list. Missing files have `error` set.
    #[serde(default)]
    pub files: Vec<ExecuteFile>,
    #[serde(default)]
    pub execution_time_ms: u64,
}

#[derive(Debug, Deserialize)]
pub struct ImageOutput {
    #[allow(dead_code)]
    pub format: String,
    pub data_base64: String,
}

/// A file read from the working directory. Exactly one of `text`, `data_base64`, or `error` is set.
#[derive(Debug, Clone, Deserialize)]
pub struct ExecuteFile {
    pub name: String,
    /// Lowercased extension or explicit override.
    pub format: String,
    #[serde(default)]
    pub mime_type: Option<String>,
    #[serde(default)]
    pub size_bytes: u64,
    #[serde(default)]
    pub text: Option<String>,
    #[serde(default)]
    pub data_base64: Option<String>,
    #[serde(default)]
    pub error: Option<String>,
}

impl ExecuteFile {
    pub fn is_error(&self) -> bool {
        self.error.is_some()
    }

    pub fn is_text(&self) -> bool {
        self.text.is_some()
    }
}

#[derive(Debug, Deserialize)]
pub struct ResetResponse {
    #[allow(dead_code)]
    pub success: bool,
}
