//! OpenAI-compatible Files endpoints. Read/list/delete only; files arrive via agentic tool calls, not uploads.

use axum::{
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{File as CoreFile, FileContent};
use serde::Serialize;

use crate::types::{ExtractedMistralRsState, SharedMistralRsState};

const PURPOSE: &str = "agent_output";

/// OpenAI file metadata + mistral.rs extensions (`format`, `mime_type`, `source`, `truncated`).
#[derive(Serialize)]
pub struct FileMetadata {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub mime_type: String,
    pub source: SourceMeta,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

#[derive(Serialize)]
pub struct SourceMeta {
    pub tool: String,
    pub round: usize,
    pub turn: usize,
}

pub async fn get_file(State(state): ExtractedMistralRsState, Path(id): Path<String>) -> Response {
    match state.find_file(&id) {
        Some(f) => Json(metadata(&f)).into_response(),
        None => not_found(&id),
    }
}

pub async fn get_file_content(
    State(state): ExtractedMistralRsState,
    Path(id): Path<String>,
) -> Response {
    serve_bytes(state, &id).unwrap_or_else(|(code, msg)| {
        (code, [(header::CONTENT_TYPE, "application/json")], msg).into_response()
    })
}

pub async fn list_files(State(state): ExtractedMistralRsState) -> Response {
    let data: Vec<FileMetadata> = state.list_files().iter().map(|f| metadata(f)).collect();
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

pub async fn delete_file(
    State(state): ExtractedMistralRsState,
    Path(id): Path<String>,
) -> Response {
    if !state.remove_file(&id) {
        return not_found(&id);
    }
    Json(serde_json::json!({
        "id": id,
        "object": "file",
        "deleted": true,
    }))
    .into_response()
}

fn metadata(f: &CoreFile) -> FileMetadata {
    FileMetadata {
        id: f.id.clone(),
        object: "file",
        bytes: f.bytes,
        created_at: f.created_at,
        filename: f.name.clone(),
        purpose: PURPOSE,
        format: f.format.clone(),
        mime_type: f
            .mime_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".to_string()),
        source: SourceMeta {
            tool: f.source.tool.clone(),
            round: f.source.round,
            turn: f.source.turn,
        },
        truncated: f.is_truncated(),
    }
}

fn serve_bytes(state: SharedMistralRsState, id: &str) -> Result<Response, (StatusCode, String)> {
    let file = state.find_file(id).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            json_error("file not found or expired"),
        )
    })?;

    let mime = file
        .mime_type
        .clone()
        .unwrap_or_else(|| "application/octet-stream".to_string());

    let bytes: Vec<u8> = match &file.content {
        FileContent::Text { text: Some(t), .. } => t.as_bytes().to_vec(),
        FileContent::Text { text: None, .. } => {
            return Err((
                StatusCode::GONE,
                json_error("text body was elided; not available via fetch"),
            ));
        }
        FileContent::Binary {
            data_base64: Some(b),
        } => STANDARD.decode(b).map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                json_error(&format!("base64 decode failed: {e}")),
            )
        })?,
        FileContent::Binary { data_base64: None } => {
            return Err((
                StatusCode::GONE,
                json_error("binary body was elided; not available via fetch"),
            ));
        }
        FileContent::Error { message, .. } => {
            return Err((StatusCode::UNPROCESSABLE_ENTITY, json_error(message)));
        }
    };

    let len = bytes.len();
    let disposition = format!(
        "inline; filename=\"{}\"; filename*=UTF-8''{}",
        ascii_safe_filename(&file.name),
        percent_encode_filename(&file.name),
    );
    Ok((
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, mime),
            (header::CONTENT_LENGTH, len.to_string()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        bytes,
    )
        .into_response())
}

fn not_found(id: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": format!("File '{id}' not found or expired"),
            "type": "invalid_request_error",
            "code": "file_not_found",
        }
    });
    (StatusCode::NOT_FOUND, Json(body)).into_response()
}

fn json_error(msg: &str) -> String {
    serde_json::json!({ "error": msg }).to_string()
}

fn ascii_safe_filename(name: &str) -> String {
    let cleaned: String = name
        .chars()
        .map(|c| {
            if c.is_ascii_graphic() && c != '"' && c != '\\' {
                c
            } else if c == ' ' {
                ' '
            } else {
                '_'
            }
        })
        .collect();
    if cleaned.is_empty() {
        "file".to_string()
    } else {
        cleaned
    }
}

/// RFC 5987 attr-char set.
fn percent_encode_filename(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for &b in name.as_bytes() {
        let safe = b.is_ascii_alphanumeric()
            || matches!(
                b,
                b'!' | b'#' | b'$' | b'&' | b'+' | b'-' | b'.' | b'^' | b'_' | b'`' | b'|' | b'~'
            );
        if safe {
            out.push(b as char);
        } else {
            use std::fmt::Write;
            let _ = write!(out, "%{:02X}", b);
        }
    }
    out
}
