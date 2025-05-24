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

use crate::models::LoadedModel;
use crate::types::{
    AppState, ChatFile, DeleteChatRequest, LoadChatRequest, NewChatRequest, RenameChatRequest,
    SelectRequest,
};

/// Accepts multipart image upload, stores it under `cache/uploads/`, and returns its URL.
pub async fn upload_image(
    State(_app): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> impl IntoResponse {
    // Expect single "image" part
    if let Ok(Some(field)) = multipart.next_field().await {
        let ext = field
            .file_name()
            .and_then(|n| n.rsplit('.').next())
            .unwrap_or("bin")
            .to_string();

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

        // Ensure dir exists
        let upload_dir = format!("{}/cache/uploads", env!("CARGO_MANIFEST_DIR"));
        if let Err(e) = tokio::fs::create_dir_all(&upload_dir).await {
            error!("mkdir error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }

        // Build unique filename
        let filename = format!("{}.{}", Uuid::new_v4(), ext);
        let path = format!("{}/{}", upload_dir, filename);

        if let Err(e) = tokio::fs::write(&path, &data).await {
            error!("write error: {}", e);
            return (StatusCode::INTERNAL_SERVER_ERROR, "server error").into_response();
        }

        return (StatusCode::OK, Json(json!({ "url": path }))).into_response();
    }

    (StatusCode::BAD_REQUEST, "no image field").into_response()
}

pub async fn list_models(State(app): State<Arc<AppState>>) -> impl IntoResponse {
    let items: Vec<_> = app
        .models
        .iter()
        .map(|(n, m)| {
            let kind = match m {
                LoadedModel::Text(_) => "text",
                LoadedModel::Vision(_) => "vision",
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

    let chat_id = format!("chat_{}", id);
    let path = format!("{}/{}.json", app.chats_dir, chat_id);

    let kind = if let Some(m) = app.models.get(&req.model) {
        match m {
            LoadedModel::Text(_) => "text",
            LoadedModel::Vision(_) => "vision",
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
