use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use chrono::Utc;
use serde_json::json;
use std::sync::Arc;
use tokio::fs;
use tracing::error;
use uuid::Uuid;

use mistralrs::speech_utils;
use std::fs::File;
use std::path::PathBuf;

use crate::models::LoadedModel;
use crate::types::{
    AppState,
    ChatFile,
    DeleteChatRequest,
    LoadChatRequest,
    NewChatRequest,
    RenameChatRequest,
    SelectRequest,
    // Append partial assistant messages
    // (defined below)
};
use crate::utils::get_cache_dir;
use serde::Deserialize;

fn validate_image_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    // Check MIME type first
    if let Some(mime) = content_type {
        if !mime.starts_with("image/") {
            return Err("File must be an image");
        }
    }

    // Validate file extension
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
    // Check MIME type first
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

/// Accepts multipart audio upload, stores under `cache/uploads/`, returns its URL.
pub async fn upload_audio(
    State(_app): State<Arc<AppState>>,
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

        // Read bytes (limit 50MB like images)
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

        // Ensure upload directory exists
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

        let url = filepath.to_string_lossy();
        (StatusCode::OK, Json(json!({ "url": url }))).into_response()
    } else {
        (StatusCode::BAD_REQUEST, "missing audio part").into_response()
    }
}

fn validate_text_upload(
    filename: Option<&str>,
    content_type: Option<&str>,
) -> Result<String, &'static str> {
    // Check MIME type first (allow text/* and application/json)
    if let Some(mime) = content_type {
        if !mime.starts_with("text/")
            && mime != "application/json"
            && mime != "application/javascript"
        {
            // Also allow common binary MIME types that are actually text
            if !matches!(
                mime,
                "application/octet-stream"
                    | "application/x-python"
                    | "application/x-rust"
                    | "application/x-sh"
            ) {
                return Err("File must be a text file");
            }
        }
    }

    // Validate file extension
    let ext = if let Some(name) = filename {
        name.rsplit('.').next().unwrap_or("").to_lowercase()
    } else {
        return Err("No filename provided");
    };

    match ext.as_str() {
        // Text files
        "txt" | "md" | "markdown" | "log" | "csv" | "tsv" | "json" | "xml" | "yaml" | "yml"
        | "toml" | "ini" | "cfg" | "conf" => Ok(ext),
        // Code files
        "rs" | "py" | "js" | "ts" | "jsx" | "tsx" | "html" | "htm" | "css" | "scss" | "sass"
        | "less" => Ok(ext),
        "c" | "cpp" | "cc" | "cxx" | "h" | "hpp" | "hxx" | "java" | "kt" | "swift" | "go"
        | "rb" | "php" => Ok(ext),
        // GPU and shader languages
        "cu" | "cuh" | "cl" | "ptx" | "glsl" | "vert" | "frag" | "geom" | "comp" | "tesc"
        | "tese" | "hlsl" | "metal" | "wgsl" => Ok(ext),
        // Shell and other scripts
        "sh" | "bash" | "zsh" | "fish" | "ps1" | "bat" | "cmd" | "sql" | "dockerfile"
        | "makefile" => Ok(ext),
        "r" | "R" | "scala" | "clj" | "cljs" | "hs" | "elm" | "ex" | "exs" | "erl" | "fs"
        | "fsx" | "ml" | "mli" => Ok(ext),
        "vue" | "svelte" | "astro" | "lua" | "nim" | "zig" | "d" | "dart" | "jl" | "pl" | "pm"
        | "tcl" => Ok(ext),
        // Config and other text-like files
        "gitignore" | "dockerignore" | "editorconfig" | "env" | "htaccess" => Ok(ext),
        "" => {
            // Files without extension - check the filename
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

/// Accepts multipart image upload, stores it under `cache/uploads/`, and returns its URL.
pub async fn upload_image(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Expect single "image" part
    if let Ok(Some(field)) = multipart.next_field().await {
        // Clone filename and content type to avoid borrowing `field`
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let ext = match validate_image_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(extension) => extension,
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
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

        const MAX_SIZE: usize = 50 * 1024 * 1024; // 50MB
        if data.len() > MAX_SIZE {
            return (StatusCode::BAD_REQUEST, "image too large (limit 50 MB)").into_response();
        }

        if image::load_from_memory(&data).is_err() {
            return (StatusCode::BAD_REQUEST, "invalid image file").into_response();
        }

        // Determine upload directory under cache
        let base_cache = get_cache_dir();
        let upload_dir = base_cache.join("uploads");
        if let Err(e) = tokio::fs::create_dir_all(&upload_dir).await {
            error!("mkdir error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }
        // Build unique filename and write
        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let path = upload_dir.join(&filename);
        if let Err(e) = tokio::fs::write(&path, &data).await {
            error!("write error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }
        return (
            StatusCode::OK,
            Json(json!({ "url": path.to_string_lossy() })),
        )
            .into_response();
    }

    (StatusCode::BAD_REQUEST, "no image field").into_response()
}

/// Accepts multipart text file upload and returns the file content
pub async fn upload_text(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Expect single "file" part
    if let Ok(Some(field)) = multipart.next_field().await {
        // Clone filename and content type to avoid borrowing `field`
        let orig_filename = field.file_name().map(|s| s.to_string());
        let content_type_opt = field.content_type().map(|s| s.to_string());

        let _ext = match validate_text_upload(orig_filename.as_deref(), content_type_opt.as_deref())
        {
            Ok(extension) => extension,
            Err(msg) => {
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        let data = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
                error!("multipart bytes error: {}", e);
                let msg = if e.to_string().contains("exceeded") {
                    "file too large (limit 10 MB)"
                } else {
                    "failed to read upload"
                };
                return (StatusCode::BAD_REQUEST, msg).into_response();
            }
        };

        const MAX_SIZE: usize = 10 * 1024 * 1024; // 10MB for text files
        if data.len() > MAX_SIZE {
            return (StatusCode::BAD_REQUEST, "file too large (limit 10 MB)").into_response();
        }

        // Try to decode as UTF-8
        let content = match String::from_utf8(data.to_vec()) {
            Ok(text) => text,
            Err(_) => {
                return (StatusCode::BAD_REQUEST, "file is not valid UTF-8 text").into_response();
            }
        };

        // Limit content length for safety
        const MAX_CHARS: usize = 1_000_000; // 1 million characters
        if content.len() > MAX_CHARS {
            return (
                StatusCode::BAD_REQUEST,
                "file content too large (limit 1M characters)",
            )
                .into_response();
        }

        return (
            StatusCode::OK,
            Json(json!({
                "content": content,
                "filename": orig_filename.unwrap_or_else(|| "untitled".to_string()),
                "size": data.len()
            })),
        )
            .into_response();
    }

    (StatusCode::BAD_REQUEST, "no file field").into_response()
}

pub async fn list_models(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let items: Vec<_> = app
        .models
        .iter()
        .map(|(n, m)| {
            let kind = match m {
                LoadedModel::Text(_) => "text",
                LoadedModel::Vision(_) => "vision",
                LoadedModel::Speech(_) => "speech",
            };
            json!({ "name": n, "kind": kind })
        })
        .collect();
    Json(json!({ "models": items }))
}

pub async fn select_model(
    State(app): State<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> impl IntoResponse {
    if let Some(model_loaded) = app.models.get(&req.name) {
        {
            let mut cur = app.current.write().await;
            *cur = Some(req.name.clone());
        }
        // --- sync the active chat file so future loads use the correct model ---
        if let Some(chat_id) = app.current_chat.read().await.clone() {
            let path = format!("{}/{}.json", app.chats_dir, chat_id);
            if let Ok(data) = fs::read(&path).await {
                if let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&data) {
                    chat.model = req.name.clone();
                    chat.kind = match model_loaded {
                        LoadedModel::Text(_) => "text".into(),
                        LoadedModel::Vision(_) => "vision".into(),
                        LoadedModel::Speech(_) => "speech".into(),
                    };
                    // ignore write errors; not fatal for select_model
                    if let Ok(bytes) = serde_json::to_vec_pretty(&chat) {
                        let _ = tokio::fs::write(&path, bytes).await;
                    }
                }
            }
        }
        (StatusCode::OK, "Model selected").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Model not found").into_response()
    }
}

pub async fn list_chats(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let mut chats = Vec::new();
    if let Ok(mut dir) = fs::read_dir(&app.chats_dir).await {
        while let Ok(Some(entry)) = dir.next_entry().await {
            if let Some(name) = entry.file_name().to_str() {
                if name.ends_with(".json") {
                    let id = name.trim_end_matches(".json");
                    let data = fs::read(format!("{}/{}", app.chats_dir, name)).await.ok();
                    let (title, created) = data
                        .and_then(|bytes| serde_json::from_slice::<ChatFile>(&bytes).ok())
                        .map(|c| (c.title, c.created_at))
                        .map(|(title, created)| (title.unwrap_or_default(), created))
                        .unwrap_or_else(|| (String::new(), String::new()));
                    chats.push(json!({ "id": id, "title": title, "created_at": created }));
                }
            }
        }
    }
    Json(json!({ "chats": chats }))
}

pub async fn new_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<NewChatRequest>,
) -> impl IntoResponse {
    let mut id_guard = app.next_chat_id.write().await;
    let id = *id_guard;
    *id_guard += 1;
    drop(id_guard);

    let chat_id = format!("chat_{id}");
    let path = format!("{}/{}.json", app.chats_dir, chat_id);

    let kind = if let Some(m) = app.models.get(&req.model) {
        match m {
            LoadedModel::Text(_) => "text",
            LoadedModel::Vision(_) => "vision",
            LoadedModel::Speech(_) => "speech",
        }
    } else {
        "text"
    }
    .to_string();

    let chat = ChatFile {
        title: None,
        model: req.model.clone(),
        kind,
        created_at: Utc::now().to_rfc3339(),
        messages: Vec::new(),
    };
    let _ = fs::write(&path, serde_json::to_vec_pretty(&chat).unwrap()).await;

    {
        let mut cur_chat = app.current_chat.write().await;
        *cur_chat = Some(chat_id.clone());
        let mut cur_model = app.current.write().await;
        *cur_model = Some(req.model.clone());
    }

    Json(json!({ "id": chat_id }))
}

pub async fn delete_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<DeleteChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Err(e) = tokio::fs::remove_file(&path).await {
        error!("delete chat error: {}", e);
        return (StatusCode::NOT_FOUND, "chat not found").into_response();
    }
    {
        let mut cur_chat = app.current_chat.write().await;
        if cur_chat.as_ref() == Some(&req.id) {
            *cur_chat = None;
            let mut cur_model = app.current.write().await;
            *cur_model = None;
        }
    }
    (StatusCode::OK, "Deleted").into_response()
}

pub async fn load_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<LoadChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    match fs::read(&path).await {
        Ok(data) => match serde_json::from_slice::<ChatFile>(&data) {
            Ok(chat) => {
                {
                    let mut cur_chat = app.current_chat.write().await;
                    *cur_chat = Some(req.id.clone());
                    if app.models.contains_key(&chat.model) {
                        let mut cur_model = app.current.write().await;
                        *cur_model = Some(chat.model.clone());
                    }
                }
                Json(json!({
                    "id": req.id,
                    "title": chat.title.clone().unwrap_or_default(),
                    "model": chat.model,
                    "kind": chat.kind,
                    "created_at": chat.created_at.clone(),
                    "messages": chat.messages
                }))
                .into_response()
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "corrupt chat").into_response(),
        },
        Err(_) => (StatusCode::NOT_FOUND, "chat not found").into_response(),
    }
}

pub async fn rename_chat(
    State(app): State<Arc<AppState>>,
    Json(req): Json<RenameChatRequest>,
) -> impl IntoResponse {
    let path = format!("{}/{}.json", app.chats_dir, req.id);
    if let Ok(data) = fs::read(&path).await {
        if let Ok(mut chat) = serde_json::from_slice::<ChatFile>(&data) {
            chat.title = Some(req.title.clone());
            if let Ok(bytes) = serde_json::to_vec_pretty(&chat) {
                let _ = tokio::fs::write(&path, bytes).await;
                return (StatusCode::OK, "Renamed").into_response();
            }
        }
    }
    (StatusCode::INTERNAL_SERVER_ERROR, "rename failed").into_response()
}

/// Request to append a (partial) assistant message to a chat
#[derive(Deserialize)]
pub struct AppendMessageRequest {
    pub id: String,
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Option<Vec<String>>,
}

/// Appends a partial assistant response (or any role) to the chat file.
pub async fn append_message(
    State(app): State<Arc<AppState>>,
    Json(req): Json<AppendMessageRequest>,
) -> impl IntoResponse {
    if let Err(e) =
        crate::chat::append_chat_message(&app, &req.id, &req.role, &req.content, req.images).await
    {
        error!("append message error: {}", e);
        return (StatusCode::INTERNAL_SERVER_ERROR, "append failed").into_response();
    }
    (StatusCode::OK, "Appended").into_response()
}
/// Request to generate speech from text
#[derive(Deserialize)]
pub struct GenerateSpeechRequest {
    pub text: String,
}

/// Get current settings (default generation params and search status)
pub async fn get_settings(State(app): State<Arc<AppState>>) -> impl IntoResponse {
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
    }))
}

/// Endpoint to generate speech (.wav) for a given prompt using a speech model
pub async fn generate_speech(
    State(app): State<Arc<AppState>>,
    Json(req): Json<GenerateSpeechRequest>,
) -> impl IntoResponse {
    // Determine selected model
    let model_name = {
        let cur = app.current.read().await;
        if let Some(name) = &*cur {
            name.clone()
        } else {
            return (StatusCode::BAD_REQUEST, "No model selected").into_response();
        }
    };
    // Ensure model exists and is a speech model
    if let Some(LoadedModel::Speech(m)) = app.models.get(&model_name) {
        // Generate speech
        let (pcm, rate, channels) = match m.generate_speech(req.text).await {
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
        // Write WAV file
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
        // Return URL for client download
        let url = format!("/speech/{filename}");
        (StatusCode::OK, Json(json!({ "url": url }))).into_response()
    } else {
        (
            StatusCode::BAD_REQUEST,
            "Selected model is not a speech model",
        )
            .into_response()
    }
}
