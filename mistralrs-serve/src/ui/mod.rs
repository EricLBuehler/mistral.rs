use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::http::{Response, StatusCode};
use axum::routing::{get, get_service, post};
use axum::Router;
use include_dir::{include_dir, Dir};
use indexmap::IndexMap;
use tracing::info;
use mistralrs::{Model, SearchEmbeddingModel};
use mistralrs_core::{MistralRs, ModelCategory};
use tokio::fs;
use tower_http::services::ServeDir;

use crate::ui::handlers::{api::*, websocket::ws_handler};
use crate::ui::types::{AppState, GenerationParams, UiModelInfo};
use crate::ui::utils::get_cache_dir;

mod chat;
mod handlers;
mod types;
mod utils;

static STATIC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/static");

#[derive(Clone)]
struct UiState {
    override_dir: Option<Arc<PathBuf>>,
}

async fn static_handler(
    axum::extract::State(state): axum::extract::State<UiState>,
    uri: axum::http::Uri,
) -> Response<Body> {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };

    // Try disk override first
    if let Some(ref dir) = state.override_dir {
        let file_path = dir.join(path);
        if let Ok(contents) = tokio::fs::read(&file_path).await {
            let mime = mime_guess::from_path(path).first_or_octet_stream();
            return Response::builder()
                .status(StatusCode::OK)
                .header(axum::http::header::CONTENT_TYPE, mime.as_ref())
                .body(Body::from(contents))
                .unwrap();
        }
    }

    // Fall back to embedded files
    if let Some(file) = STATIC_DIR.get_file(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.contents()))
            .unwrap()
    } else {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap()
    }
}

fn build_model_list(mistralrs: &Arc<MistralRs>) -> IndexMap<String, UiModelInfo> {
    let mut models = IndexMap::new();
    if let Ok(list) = mistralrs.list_models() {
        for model_id in list {
            if let Ok(category) = mistralrs.get_model_category(Some(&model_id)) {
                let kind = match category {
                    ModelCategory::Text => "text",
                    ModelCategory::Multimodal { .. } => "multimodal",
                    ModelCategory::Speech => "speech",
                    ModelCategory::Audio => "audio",
                    ModelCategory::Embedding => "embedding",
                    ModelCategory::Diffusion => "diffusion",
                };
                if matches!(kind, "text" | "multimodal" | "speech") {
                    let generation_defaults = mistralrs
                        .config(Some(&model_id))
                        .ok()
                        .and_then(|cfg| cfg.generation_defaults);
                    models.insert(
                        model_id.clone(),
                        UiModelInfo {
                            name: model_id,
                            kind: kind.to_string(),
                            generation_defaults: GenerationParams::from_model_defaults(
                                generation_defaults.as_ref(),
                            ),
                        },
                    );
                }
            }
        }
    }
    models
}

pub async fn build_ui_router(
    mistralrs: Arc<MistralRs>,
    enable_search: bool,
    search_embedding_model: Option<SearchEmbeddingModel>,
) -> Result<Router> {
    let models = build_model_list(&mistralrs);
    let model_wrapper = Model::new(mistralrs.clone());

    // Detect UI override directory next to the executable
    let ui_override_dir = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.parent().map(|p| p.join("ui")))
        .filter(|dir| dir.is_dir());
    if let Some(ref dir) = ui_override_dir {
        info!("UI override directory detected: {}", dir.display());
    }
    let ui_override_dir = ui_override_dir.map(Arc::new);

    let base_cache = get_cache_dir();
    let chats_dir = base_cache.join("chats");
    fs::create_dir_all(&chats_dir).await?;
    let speech_dir = base_cache.join("speech");
    fs::create_dir_all(&speech_dir).await?;
    let uploads_dir = base_cache.join("uploads");
    fs::create_dir_all(&uploads_dir).await?;

    let mut next_id = 1u32;
    if let Ok(mut dir) = fs::read_dir(&chats_dir).await {
        while let Ok(Some(entry)) = dir.next_entry().await {
            if let Some(name) = entry.file_name().to_str() {
                if let Some(num) = name
                    .strip_prefix("chat_")
                    .and_then(|s| s.strip_suffix(".json"))
                {
                    if let Ok(n) = num.parse::<u32>() {
                        next_id = next_id.max(n + 1);
                    }
                }
            }
        }
    }

    let default_model = mistralrs
        .get_default_model_id()
        .ok()
        .flatten()
        .or_else(|| models.keys().next().cloned());

    let app_state = Arc::new(AppState {
        model: model_wrapper,
        models,
        current: tokio::sync::RwLock::new(default_model),
        chats_dir: chats_dir.to_string_lossy().to_string(),
        speech_dir: speech_dir.to_string_lossy().to_string(),
        current_chat: tokio::sync::RwLock::new(None),
        next_chat_id: tokio::sync::RwLock::new(next_id),
        default_params: GenerationParams::default(),
        search_enabled: enable_search,
        search_embedding_model,
    });

    let router = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/upload_image", post(upload_image))
        .route("/api/upload_text", post(upload_text))
        .route("/api/upload_audio", post(upload_audio))
        .route("/api/list_models", get(list_models))
        .route("/api/select_model", post(select_model))
        .route("/api/list_chats", get(list_chats))
        .route("/api/new_chat", post(new_chat))
        .route("/api/delete_chat", post(delete_chat))
        .route("/api/load_chat", post(load_chat))
        .route("/api/rename_chat", post(rename_chat))
        .route("/api/append_message", post(append_message))
        .route("/api/settings", get(get_settings))
        .route("/api/generate_speech", post(generate_speech))
        .route("/api/stop", post(stop_generation))
        .nest_service("/speech", get_service(ServeDir::new(speech_dir.clone())))
        .nest_service("/uploads", get_service(ServeDir::new(uploads_dir.clone())))
        .route("/", get(static_handler))
        .route("/{*path}", get(static_handler))
        .with_state(UiState {
            override_dir: ui_override_dir,
        })
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(axum::extract::Extension(app_state));

    Ok(router)
}
