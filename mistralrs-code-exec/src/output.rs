use base64::Engine;
use image::DynamicImage;

use crate::protocol::{ExecuteFile, ExecuteResponse};

/// Processed result from a code execution, ready for tool output.
pub struct CodeExecResult {
    /// JSON tool-response text (stdout/stderr/exception/etc., NOT file
    /// bodies). The engine composes the model-facing text with file
    /// summary on top of this.
    pub text: String,
    pub images: Vec<DynamicImage>,
    pub video_frames: Vec<DynamicImage>,
    /// Files declared in the request's `outputs` parameter, read out of
    /// the working directory by the executor.
    pub files: Vec<ExecuteFile>,
}

impl CodeExecResult {
    /// Create a timeout result with no output.
    pub fn timeout(timeout_secs: u64, interrupted: bool) -> Self {
        let text = serde_json::json!({
            "status": "timeout",
            "timeout_info": {
                "timeout_secs": timeout_secs,
                "interrupted": interrupted,
                "killed": !interrupted,
            }
        })
        .to_string();
        Self {
            text,
            images: vec![],
            video_frames: vec![],
            files: vec![],
        }
    }

    /// Create an error result.
    pub fn error(msg: &str) -> Self {
        let text = serde_json::json!({
            "status": "error",
            "exception": msg,
        })
        .to_string();
        Self {
            text,
            images: vec![],
            video_frames: vec![],
            files: vec![],
        }
    }

    /// Build from a successful Python execution response.
    pub fn from_response(response: ExecuteResponse, work_dir: &str) -> Self {
        let images = decode_images(&response.images);
        let video_frames = decode_images(&response.video_frames);

        let status = if response.exception.is_some() {
            "error"
        } else {
            "success"
        };

        let mut result = serde_json::json!({
            "status": status,
            "execution_time_ms": response.execution_time_ms,
            "working_directory": work_dir,
        });

        if !response.stdout.is_empty() {
            result["stdout"] = serde_json::Value::String(response.stdout);
        }
        if !response.stderr.is_empty() {
            result["stderr"] = serde_json::Value::String(response.stderr);
        }
        if let Some(ref exc) = response.exception {
            result["exception"] = serde_json::Value::String(exc.clone());
        }
        if let Some(ref repr) = response.last_expr_repr {
            result["result"] = serde_json::Value::String(repr.clone());
        }
        if let Some(ref tp) = response.last_expr_type {
            result["result_type"] = serde_json::Value::String(tp.clone());
        }
        if !images.is_empty() {
            result["images_generated"] = serde_json::Value::Number(images.len().into());
        }
        if !video_frames.is_empty() {
            result["video_frames_generated"] = serde_json::Value::Number(video_frames.len().into());
        }

        Self {
            text: result.to_string(),
            images,
            video_frames,
            files: response.files,
        }
    }
}

fn decode_images(outputs: &[crate::protocol::ImageOutput]) -> Vec<DynamicImage> {
    let b64 = base64::engine::general_purpose::STANDARD;
    outputs
        .iter()
        .filter_map(|img| {
            let bytes = b64.decode(&img.data_base64).ok()?;
            image::load_from_memory(&bytes).ok()
        })
        .collect()
}
