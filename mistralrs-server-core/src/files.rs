//! OpenAI-compatible Files endpoints for uploaded request files and agent-produced files.

use axum::{
    extract::{multipart::MultipartRejection, Multipart, Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{
    File as CoreFile, FileContent, FileSource, MistralRs, MistralRsError, FILE_PURPOSE_USER_DATA,
};
use serde::Serialize;
use utoipa::ToSchema;

use crate::{
    handler_core::{openai_error_response, ApiError, ApiErrorKind},
    types::{ExtractedMistralRsState, SharedMistralRsState},
};

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

/// OpenAI-compatible container file metadata backed by the same in-process file store.
#[derive(Serialize, ToSchema)]
pub struct ContainerFileMetadata {
    pub id: String,
    pub object: &'static str,
    pub bytes: u64,
    pub created_at: u64,
    pub filename: String,
    pub container_id: String,
    pub source: SourceMeta,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    pub mime_type: String,
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
pub async fn upload_file(
    State(state): ExtractedMistralRsState,
    payload: Result<Multipart, MultipartRejection>,
) -> Response {
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => {
            return openai_error_response(ApiError::from_status(error.status(), error.body_text()));
        }
    };
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
                MistralRs::maybe_log_error(state, &e);
                return openai_error_response(ApiError::internal());
            }
            Json(metadata(&file)).into_response()
        }
        Err(error) => openai_error_response(error),
    }
}

async fn parse_upload(mut multipart: Multipart) -> Result<FileUpload, ApiError> {
    let mut purpose = None;
    let mut file = None;

    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let field_name = field.name().unwrap_or_default().to_string();
        match field_name.as_str() {
            "purpose" => {
                let value = field.text().await.map_err(multipart_error)?;
                if !value.trim().is_empty() {
                    purpose = Some(value);
                }
            }
            "file" => {
                let filename = field
                    .file_name()
                    .ok_or_else(|| {
                        ApiError::new(
                            ApiErrorKind::InvalidRequest,
                            "Uploaded file is missing a filename.",
                            Some("invalid_file"),
                            Some("file"),
                        )
                    })?
                    .to_string();
                let mime_type = field.content_type().map(ToString::to_string);
                let bytes = field.bytes().await.map_err(multipart_error)?.to_vec();
                if bytes.len() > MAX_FILE_UPLOAD_BYTES {
                    return Err(ApiError::new(
                        ApiErrorKind::PayloadTooLarge,
                        format!("File upload exceeds the {MAX_FILE_UPLOAD_BYTES} byte limit."),
                        Some("file_too_large"),
                        Some("file"),
                    ));
                }
                file = Some((filename, mime_type, bytes));
            }
            _ => {}
        }
    }

    let purpose = purpose.ok_or_else(|| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            format!(
                "File upload requires multipart field `purpose` such as `{}`.",
                FILE_PURPOSE_USER_DATA
            ),
            Some("missing_required_parameter"),
            Some("purpose"),
        )
    })?;
    let (filename, mime_type, bytes) = file.ok_or_else(|| {
        ApiError::new(
            ApiErrorKind::InvalidRequest,
            "File upload requires multipart field `file`.",
            Some("missing_required_parameter"),
            Some("file"),
        )
    })?;

    Ok(FileUpload {
        filename,
        mime_type,
        purpose,
        bytes,
    })
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::from_status(error.status(), error.body_text())
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/files/{id}",
    params(("id" = String, Path, description = "File ID")),
    responses(
        (status = 200, description = "File metadata", body = FileMetadata),
        (status = 404, description = "File not found or expired"),
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn get_file(State(state): ExtractedMistralRsState, Path(id): Path<String>) -> Response {
    match state.try_find_file(&id) {
        Ok(Some(f)) => Json(metadata(&f)).into_response(),
        Ok(None) => not_found(&id),
        Err(error) => file_store_error(state, &error),
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
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn get_file_content(
    State(state): ExtractedMistralRsState,
    Path(id): Path<String>,
) -> Response {
    serve_bytes(state, &id)
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/files",
    responses(
        (status = 200, description = "List of file metadata", body = [FileMetadata]),
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn list_files(State(state): ExtractedMistralRsState) -> Response {
    let files = match state.try_list_files() {
        Ok(files) => files,
        Err(error) => return file_store_error(state, &error),
    };
    let data: Vec<FileMetadata> = files.iter().map(|f| metadata(f)).collect();
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
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn delete_file(
    State(state): ExtractedMistralRsState,
    Path(id): Path<String>,
) -> Response {
    match state.try_remove_file(&id) {
        Ok(true) => {}
        Ok(false) => return not_found(&id),
        Err(error) => return file_store_error(state, &error),
    }
    Json(serde_json::json!({
        "id": id,
        "object": "file",
        "deleted": true,
    }))
    .into_response()
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/containers/{container_id}/files",
    params(("container_id" = String, Path, description = "Container ID")),
    responses(
        (status = 200, description = "List of container file metadata", body = [ContainerFileMetadata]),
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn list_container_files(
    State(state): ExtractedMistralRsState,
    Path(container_id): Path<String>,
) -> Response {
    let files = match state.try_list_files() {
        Ok(files) => files,
        Err(error) => return file_store_error(state, &error),
    };
    let data: Vec<ContainerFileMetadata> = files
        .iter()
        .map(|f| container_metadata(&container_id, f))
        .collect();
    Json(serde_json::json!({ "object": "list", "data": data })).into_response()
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/containers/{container_id}/files/{file_id}",
    params(
        ("container_id" = String, Path, description = "Container ID"),
        ("file_id" = String, Path, description = "File ID")
    ),
    responses(
        (status = 200, description = "Container file metadata", body = ContainerFileMetadata),
        (status = 404, description = "File not found or expired"),
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn get_container_file(
    State(state): ExtractedMistralRsState,
    Path((container_id, file_id)): Path<(String, String)>,
) -> Response {
    match state.try_find_file(&file_id) {
        Ok(Some(f)) => Json(container_metadata(&container_id, &f)).into_response(),
        Ok(None) => not_found(&file_id),
        Err(error) => file_store_error(state, &error),
    }
}

#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/containers/{container_id}/files/{file_id}/content",
    params(
        ("container_id" = String, Path, description = "Container ID"),
        ("file_id" = String, Path, description = "File ID")
    ),
    responses(
        (status = 200, description = "Raw file bytes with the file's MIME type"),
        (status = 404, description = "File not found or expired"),
        (status = 410, description = "File body was elided and is no longer fetchable"),
        (status = 500, description = "Internal server error"),
    )
)]
pub async fn get_container_file_content(
    State(state): ExtractedMistralRsState,
    Path((_container_id, file_id)): Path<(String, String)>,
) -> Response {
    serve_bytes(state, &file_id)
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

fn container_metadata(container_id: &str, f: &CoreFile) -> ContainerFileMetadata {
    ContainerFileMetadata {
        id: f.id.clone(),
        object: "container.file",
        bytes: f.bytes,
        created_at: f.created_at,
        filename: f.name.clone(),
        container_id: container_id.to_string(),
        source: SourceMeta {
            tool: f.source.tool.clone(),
            round: f.source.round,
            turn: f.source.turn,
        },
        format: f.format.clone(),
        mime_type: f
            .mime_type
            .clone()
            .unwrap_or_else(|| "application/octet-stream".to_string()),
    }
}

fn serve_bytes(state: SharedMistralRsState, id: &str) -> Response {
    let file = match state.try_find_file(id) {
        Ok(Some(file)) => file,
        Ok(None) => return not_found(id),
        Err(error) => return file_store_error(state, &error),
    };

    let mime = file
        .mime_type
        .clone()
        .unwrap_or_else(|| "application/octet-stream".to_string());

    let bytes: Vec<u8> = match &file.content {
        FileContent::Text { text: Some(t), .. } => t.as_bytes().to_vec(),
        FileContent::Text { text: None, .. } => {
            return content_gone("Text body was elided and is no longer available.");
        }
        FileContent::Binary {
            data_base64: Some(b),
        } => match STANDARD.decode(b) {
            Ok(bytes) => bytes,
            Err(error) => {
                tracing::error!(%error, file_id = id, "failed to decode stored file content");
                return openai_error_response(ApiError::internal());
            }
        },
        FileContent::Binary { data_base64: None } => {
            return content_gone("Binary body was elided and is no longer available.");
        }
        FileContent::Error { message, .. } => {
            return openai_error_response(ApiError::new(
                ApiErrorKind::InvalidRequest,
                message,
                Some("file_content_error"),
                Some("file_id"),
            ));
        }
    };

    let len = bytes.len();
    let disposition = format!(
        "inline; filename=\"{}\"; filename*=UTF-8''{}",
        ascii_safe_filename(&file.name),
        percent_encode_filename(&file.name),
    );
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, mime),
            (header::CONTENT_LENGTH, len.to_string()),
            (header::CONTENT_DISPOSITION, disposition),
        ],
        bytes,
    )
        .into_response()
}

fn file_store_error(state: SharedMistralRsState, error: &MistralRsError) -> Response {
    MistralRs::maybe_log_error(state, error);
    openai_error_response(ApiError::from_error(error, ApiErrorKind::Internal))
}

fn not_found(id: &str) -> Response {
    openai_error_response(ApiError::new(
        ApiErrorKind::NotFound,
        format!("File '{id}' not found or expired."),
        Some("file_not_found"),
        Some("file_id"),
    ))
}

fn content_gone(message: &str) -> Response {
    let mut response = openai_error_response(ApiError::new(
        ApiErrorKind::NotFound,
        message,
        Some("file_content_unavailable"),
        Some("file_id"),
    ));
    *response.status_mut() = StatusCode::GONE;
    response
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
