//! OpenAI-compatible Files endpoints for uploaded request files and agent-produced files.

use axum::{
    extract::{Multipart, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{File as CoreFile, FileContent, FileSource, FILE_PURPOSE_USER_DATA};
use serde::Serialize;
use utoipa::ToSchema;

use crate::types::{ExtractedMistralRsState, SharedMistralRsState};

const MAX_FILE_UPLOAD_BYTES: usize = 64 * 1024 * 1024;

struct FileUpload {
    filename: String,
    mime_type: Option<String>,
    purpose: String,
    bytes: Vec<u8>,
}

/// OpenAI file metadata + mistral.rs extensions (`format`, `mime_type`, `source`, `truncated`).
#[derive(Serialize, ToSchema)]
pub struct FileMetadata {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub purpose: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub mime_type: String,
    pub source: SourceMeta,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    pub truncated: bool,
}

/// Which agentic tool produced the file, and when in the session.
#[derive(Serialize, ToSchema)]
pub struct SourceMeta {
    pub tool: String,
    pub round: usize,
    pub turn: usize,
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/files",
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Uploaded file metadata", body = FileMetadata),
        (status = 400, description = "Invalid upload"),
    )
)]
pub async fn upload_file(State(state): ExtractedMistralRsState, multipart: Multipart) -> Response {
    match parse_upload(multipart).await {
        Ok(upload) => {
            let file = CoreFile::from_bytes(
                CoreFile::make_upload_id(),
                upload.filename,
                upload.mime_type,
                upload.purpose,
                FileSource {
                    tool: "user_upload".to_string(),
                    round: 0,
                    turn: 0,
                },
                upload.bytes,
            );
            if let Err(e) = state.insert_file(None, file.clone(), None) {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "error": e.to_string() })),
                )
                    .into_response();
            }
            Json(metadata(&file)).into_response()
        }
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
    }
}

async fn parse_upload(mut multipart: Multipart) -> anyhow::Result<FileUpload> {
    let mut purpose = None;
    let mut file = None;

    while let Some(field) = multipart.next_field().await? {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            "purpose" => {
                let value = field.text().await?;
                if !value.trim().is_empty() {
                    purpose = Some(value);
                }
            }
            "file" => {
                let filename = field
                    .file_name()
                    .ok_or_else(|| anyhow::anyhow!("Uploaded file is missing a filename."))?
                    .to_string();
                let mime_type = field.content_type().map(ToString::to_string);
                let bytes = field.bytes().await?.to_vec();
                if bytes.len() > MAX_FILE_UPLOAD_BYTES {
                    anyhow::bail!("File upload exceeds the {MAX_FILE_UPLOAD_BYTES} byte limit.");
                }
                file = Some((filename, mime_type, bytes));
            }
            _ => {}
        }
    }

    let purpose = purpose.ok_or_else(|| {
        anyhow::anyhow!(
            "File upload requires multipart field `purpose` such as `{}`.",
            FILE_PURPOSE_USER_DATA
        )
    })?;
    let (filename, mime_type, bytes) =
        file.ok_or_else(|| anyhow::anyhow!("File upload requires multipart field `file`."))?;

    Ok(FileUpload {
        filename,
        mime_type,
        purpose,
        bytes,
    })
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/files/{id}",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "File metadata", body = FileMetadata),
        (status = 404, description = "File not found or expired"),
    )
)]
pub async fn get_file(State(state): ExtractedMistralRsState, Path(id): Path<String>) -> Response {
    match state.find_file(&id) {
        Some(f) => Json(metadata(&f)).into_response(),
        None => not_found(&id),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/files/{id}/content",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "Raw file bytes with the file's MIME type"),
        (status = 404, description = "File not found or expired"),
        (status = 410, description = "File body was elided and is no longer fetchable"),
    )
)]
pub async fn get_file_content(
    State(state): ExtractedMistralRsState,
    Path(id): Path<String>,
) -> Response {
    serve_bytes(state, &id).unwrap_or_else(|(code, msg)| {
        (code, [(header::CONTENT_TYPE, "application/json")], msg).into_response()
    })
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/files",
    responses((status = 200, description = "List of file metadata", body = [FileMetadata]))
)]
pub async fn list_files(State(state): ExtractedMistralRsState) -> Response {
    let data: Vec<FileMetadata> = state.list_files().iter().map(|f| metadata(f)).collect();
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

#[utoipa::path(
    delete,
    tag = "Mistral.rs",
    path = "/v1/files/{id}",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "File deleted"),
        (status = 404, description = "File not found or expired"),
    )
)]
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
        purpose: f.purpose.clone(),
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
