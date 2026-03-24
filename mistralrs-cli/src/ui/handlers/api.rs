use axum::{
    extract::{Extension, Multipart},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use serde::Deserialize;
use serde_json::json;
use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::fs;
use tracing::error;
use uuid::Uuid;

use mistralrs::speech_utils;

use crate::ui::chat::append_chat_message;
use crate::ui::types::{
    AppState, ChatFile, DeleteChatRequest, LoadChatRequest, NewChatRequest, RenameChatRequest,
    SelectRequest,
};
use crate::ui::utils::get_cache_dir;

fn validate_image_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if let Some(mime) = content_type {
        if !mime.starts_with("image/") {
            return Err("File must be an image");
        }
    }

    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "bmp" | "svg" => Ok(ext),
        "" => Err("No file extension"),
        _ => Err("Unsupported image format"),
    }
}

fn validate_audio_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if let Some(mime) = content_type {
        if !mime.starts_with("audio/") {
            return Err("File must be an audio file");
        }
    }

    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "wav" | "mp3" | "ogg" | "flac" | "m4a" | "aac" | "opus" | "webm" => Ok(ext),
        "" => Err("No file extension"),
        _ => Err("Unsupported audio format"),
    }
}

fn validate_text_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    if let Some(mime) = content_type {
        if !mime.starts_with("text/")
            && mime != "application/json"
            && mime != "application/javascript"
            && !matches!(
                mime,
                "application/octet-stream"
                    | "application/x-python"
                    | "application/x-rust"
                    | "application/x-sh"
            )
        {
            return Err("File must be a text file");
        }
    }

    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        "txt" | "md" | "markdown" | "log" | "csv" | "tsv" | "json" | "xml" | "yaml" | "yml"
        | "toml" | "ini" | "cfg" | "conf" => Ok(ext),
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "html" | "htm" | "css" | "scss" | "sass"
        | "less" => Ok(ext),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" | "java" | "kt" | "swift" | "go"
        | "rb" | "php" => Ok(ext),
        "cu" | "cuh" | "cl" | "ptx" | "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc"
        | "tese" | "hlsl" | "metal" | "wgsl" => Ok(ext),
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" | "sql" | "dockerfile"
        | "makefile" => Ok(ext),
        "r" | "R" | "scala" | "clj" | "cljs" | "hs" | "elm" | "ex" | "exs" | "erl" | "fs"
        | "fsx" | "ml" | "mli" => Ok(ext),
        "vue" | "svelte" | "astro" | "lua" | "nim" | "zig" | "d" | "dart" | "jl" | "pl" | "pm"
        | "tcl" => Ok(ext),
        "gitignore" | "dockerignore" | "editorconfig" | "env" | "htaccess" => Ok(ext),
        "" => {
            if let Some(name) = filename {
                let name_lower = name.to_lowercase();
                if matches!(
                    name_lower.as_str(),
                    "readme"
                        | "license"
                        | "changelog"
                        | "makefile"
                        | "dockerfile"
                        | "vagrantfile"
                        | "gemfile"
                        | "rakefile"
                ) {
                    return Ok("txt".to_string());
                }
            }
            Err("No file extension")
        }
        _ => Err("Unsupported text file format"),
    }
}

pub async fn upload_audio(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Ok(Some(field)) = multipart.next_field().await {
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let ext = match validate_audio_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(ext) => ext,
            Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "audio too large (limit 50 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let uploads_dir = get_cache_dir().join("uploads");
        if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
            error!("create uploads dir error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to create uploads directory",
            )
                .into_response();
        }

        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let filepath = uploads_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&filepath, &data).await {
            error!("write upload error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save audio").into_response();
        }

        let path = filepath.to_string_lossy().to_string();
        let url = format!("uploads/{filename}");
        (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "missing audio part").into_response()
    }
}

pub async fn upload_image(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Ok(Some(field)) = multipart.next_field().await {
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let ext = match validate_image_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(extension) => extension,
            Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "image too large (limit 50 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let uploads_dir = get_cache_dir().join("uploads");
        if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
            error!("create uploads dir error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to create uploads directory",
            )
                .into_response();
        }

        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let filepath = uploads_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&filepath, &data).await {
            error!("write upload error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "failed to save image").into_response();
        }

        let path = filepath.to_string_lossy().to_string();
        let url = format!("uploads/{filename}");
        (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "missing image part").into_response()
    }
}

pub async fn upload_text(
    Extension(_app): Extension<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    if let Ok(Some(field)) = multipart.next_field().await {
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let ext = match validate_text_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(ext) => ext,
            Err(msg) => return (StatusCode::BAD_REQUEST, msg).into_response(),
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "file too large (limit 50 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let uploads_dir = get_cache_dir().join("uploads");
        if let Err(e) = tokio::fs::create_dir_all(&uploads_dir).await {
            error!("create uploads dir error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to create uploads directory",
            )
                .into_response();
        }

        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let filepath = uploads_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&filepath, &data).await {
            error!("write upload error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "failed to save text file",
            )
                .into_response();
        }

        let path = filepath.to_string_lossy().to_string();
        let url = format!("uploads/{filename}");
        (StatusCode::OK, Json(json!({ "path": path, "url": url }))).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "missing text part").into_response()
    }
}

pub async fn list_models(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    let models: Vec<_> = app.models.values().cloned().collect();
    Json(json!({ "models": models }))
}

pub async fn select_model(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> impl IntoResponse {
    if !app.models.contains_key(&req.name) {
        return (
            StatusCode::BAD_REQUEST,
            format!("Model '{}' not found", req.name),
        )
            .into_response();
    }
    let mut cur = app.current.write().await;
    *cur = Some(req.name);
    (StatusCode::OK, "Selected").into_response()
}

pub async fn list_chats(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    let dir = &app.chats_dir;
    let mut chats = Vec::new();

    if let Ok(mut entries) = fs::read_dir(dir).await {
        while let Ok(Some(entry)) = entries.next_entry().await {
            if let Ok(bytes) = fs::read(entry.path()).await {
                if let Ok(chat) = serde_json::from_slice::<ChatFile>(&bytes) {
                    chats.push(chat);
                }
            }
        }
    }
    chats.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    Json(json!({ "chats": chats }))
}

pub async fn new_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<NewChatRequest>,
) -> impl IntoResponse {
    if !app.models.contains_key(&req.model) {
        return (StatusCode::BAD_REQUEST, "Unknown model").into_response();
    }

    let mut next_id = app.next_chat_id.write().await;
    let chat_id = format!("chat_{}", *next_id);
    *next_id += 1;

    let now = Utc::now().to_rfc3339();
    let kind = app
        .models
        .get(&req.model)
        .map(|m| m.kind.clone())
        .unwrap_or_else(|| "text".to_string());
    let chat = ChatFile {
        title: None,
        model: req.model,
        kind,
        created_at: now,
        messages: Vec::new(),
    };

    let path = format!("{}/{}.json", app.chats_dir, chat_id);
    if let Err(e) = fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap()).await {
        error!("write chat error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "write failed").into_response();
    }

    let mut cur = app.current_chat.write().await;
    *cur = Some(chat_id.clone());
    Json(json!({ "id": chat_id })).into_response()
}

pub async fn delete_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<DeleteChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    match fs::remove_file(&path).await {
        Ok(_) => (StatusCode::OK, "Deleted").into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "Chat not found").into_response(),
    }
}

pub async fn load_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<LoadChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Ok(bytes) = fs::read(&path).await {
        if let Ok(chat) = serde_json::from_slice::<ChatFile>(&bytes) {
            let mut cur = app.current_chat.write().await;
            *cur = Some(req.id.clone());
            return Json(chat).into_response();
        }
    }
    (StatusCode::NOT_FOUND, "Chat not found").into_response()
}

pub async fn rename_chat(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<RenameChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Ok(bytes) = fs::read(&path).await {
        if let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&bytes) {
            chat.title = Some(req.title);
            if fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap())
                .await
                .is_ok()
            {
                return (StatusCode::OK, "Renamed").into_response();
            }
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "rename failed").into_response()
}

#[derive(Deserialize)]
pub struct AppendMessageRequest {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

pub async fn append_message(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<AppendMessageRequest>,
) -> impl IntoResponse {
    if let Err(e) = append_chat_message(&app, &req.id, &req.role, &req.content, req.images).await {
        error!("append message error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "append failed").into_response();
    }
    (StatusCode::OK, "Appended").into_response()
}

#[derive(Deserialize)]
pub struct GenerateSpeechRequest {
    pub text: String,
}

pub async fn get_settings(Extension(app): Extension<Arc<AppState>>) -> impl IntoResponse {
    Json(json!({
        "defaults": {
            "temperature": app.default_params.temperature,
            "top_p": app.default_params.top_p,
            "top_k": app.default_params.top_k,
            "max_tokens": app.default_params.max_tokens,
            "repetition_penalty": app.default_params.repetition_penalty,
            "system_prompt": app.default_params.system_prompt,
        },
        "search_enabled": app.search_enabled,
        "search_embedding_model": app.search_embedding_model.map(|m| m.to_string()),
    }))
}

pub async fn generate_speech(
    Extension(app): Extension<Arc<AppState>>,
    Json(req): Json<GenerateSpeechRequest>,
) -> impl IntoResponse {
    let model_name = {
        let cur = app.current.read().await;
        if let Some(name) = &*cur {
            name.clone()
        } else {
            return (StatusCode::BAD_REQUEST, "No model selected").into_response();
        }
    };

    let kind = app
        .models
        .get(&model_name)
        .map(|m| m.kind.as_str())
        .unwrap_or("text");
    if kind != "speech" {
        return (
            StatusCode::BAD_REQUEST,
            "Selected model is not a speech model",
        )
            .into_response();
    }

    let (pcm, rate, channels) = match app
        .model
        .generate_speech_with_model(req.text, Some(&model_name))
        .await
    {
        Ok(res) => res,
        Err(e) => {
            error!("speech generation error: {}", e);
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "speech generation failed",
            )
                .into_response();
        }
    };

    let filename = format!("{}.wav", Uuid::new_v4());
    let filepath = PathBuf::from(&app.speech_dir).join(&filename);
    if let Err(e) = File::create(&filepath).and_then(|mut f| {
        speech_utils::write_pcm_as_wav(&mut f, &pcm, rate as u32, channels as u16)
    }) {
        error!("failed to write wav file: {}", e);
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "failed to write wav file",
        )
            .into_response();
    }

    let url = format!("speech/{filename}");
    (StatusCode::OK, Json(json!({ "url": url }))).into_response()
}
