use std::sync::Arc;

use anyhow::Result;
use axum::body::Body;
use axum::extract::DefaultBodyLimit;
use axum::http::{Response, StatusCode};
use axum::routing::{get, get_service, post};
use axum::Router;
use include_dir::{include_dir, Dir};
use indexmap::IndexMap;
use mistralrs::{Model, SearchEmbeddingModel};
use mistralrs_core::{MistralRs, ModelCategory, SupportedModality};
use mistralrs_server_core::route_registry::{RouteInfo, RouteKind};
use tokio::fs;
use tower_http::services::ServeDir;

use crate::ui::handlers::api::*;
use crate::ui::types::{AppState, GenerationParams, UiModelInfo};
use crate::ui::utils::get_cache_dir;

mod chat;
mod handlers;
mod types;
mod utils;

static STATIC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/static");

pub(crate) const UI_UPLOAD_IMAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_image", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_VIDEO_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_video", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_TEXT_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_text", "POST", RouteKind::Ui);
pub(crate) const UI_UPLOAD_AUDIO_ROUTE: RouteInfo =
    RouteInfo::new("/api/upload_audio", "POST", RouteKind::Ui);
pub(crate) const UI_LIST_MODELS_ROUTE: RouteInfo =
    RouteInfo::new("/api/list_models", "GET", RouteKind::Ui);
pub(crate) const UI_SELECT_MODEL_ROUTE: RouteInfo =
    RouteInfo::new("/api/select_model", "POST", RouteKind::Ui);
pub(crate) const UI_LIST_CHATS_ROUTE: RouteInfo =
    RouteInfo::new("/api/list_chats", "GET", RouteKind::Ui);
pub(crate) const UI_NEW_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/new_chat", "POST", RouteKind::Ui);
pub(crate) const UI_DELETE_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/delete_chat", "POST", RouteKind::Ui);
pub(crate) const UI_LOAD_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/load_chat", "POST", RouteKind::Ui);
pub(crate) const UI_RENAME_CHAT_ROUTE: RouteInfo =
    RouteInfo::new("/api/rename_chat", "POST", RouteKind::Ui);
pub(crate) const UI_APPEND_MESSAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/append_message", "POST", RouteKind::Ui);
pub(crate) const UI_EDIT_MESSAGE_ROUTE: RouteInfo =
    RouteInfo::new("/api/edit_message", "POST", RouteKind::Ui);
pub(crate) const UI_SET_TAIL_ROUTE: RouteInfo =
    RouteInfo::new("/api/set_tail", "POST", RouteKind::Ui);
pub(crate) const UI_FORK_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/fork_session", "POST", RouteKind::Ui);
pub(crate) const UI_SAVE_CHAT_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/save_chat_session", "POST", RouteKind::Ui);
pub(crate) const UI_RESTORE_CHAT_SESSION_ROUTE: RouteInfo =
    RouteInfo::new("/api/restore_chat_session", "POST", RouteKind::Ui);
pub(crate) const UI_SETTINGS_ROUTE: RouteInfo =
    RouteInfo::new("/api/settings", "GET", RouteKind::Ui);
pub(crate) const UI_CAPABILITIES_ROUTE: RouteInfo =
    RouteInfo::new("/api/capabilities", "GET", RouteKind::Ui);
pub(crate) const UI_MCP_TOOLS_ROUTE: RouteInfo =
    RouteInfo::new("/api/mcp_tools", "GET", RouteKind::Ui);
pub(crate) const UI_GENERATE_SPEECH_ROUTE: RouteInfo =
    RouteInfo::new("/api/generate_speech", "POST", RouteKind::Ui);
pub(crate) const UI_SPEECH_ROUTE: RouteInfo = RouteInfo::new("/speech", "GET", RouteKind::Ui);
pub(crate) const UI_UPLOADS_ROUTE: RouteInfo = RouteInfo::new("/uploads", "GET", RouteKind::Ui);
pub(crate) const UI_ROOT_ROUTE: RouteInfo = RouteInfo::new("/", "GET", RouteKind::Ui);
pub(crate) const UI_STATIC_ROUTE: RouteInfo = RouteInfo::new("/{*path}", "GET", RouteKind::Ui);

async fn static_handler(uri: axum::http::Uri) -> Response<Body> {
    let path = uri.path().trim_start_matches('/');
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(file) = STATIC_DIR.get_file(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(axum::http::header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.contents()))
            .unwrap()
    } else {
        // SPA fallback: serve index.html for unrecognized paths
        if let Some(file) = STATIC_DIR.get_file("index.html") {
            let mime = mime_guess::from_path("index.html").first_or_octet_stream();
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
}

fn modality_label(m: &SupportedModality) -> String {
    match m {
        SupportedModality::Text => "text",
        SupportedModality::Audio => "audio",
        SupportedModality::Vision => "vision",
        SupportedModality::Video => "video",
        SupportedModality::Embedding => "embedding",
    }
    .to_string()
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
                    let cfg = mistralrs.config(Some(&model_id)).ok();
                    let generation_defaults =
                        cfg.as_ref().and_then(|c| c.generation_defaults.clone());
                    let (input_modalities, output_modalities) = cfg
                        .as_ref()
                        .map(|c| {
                            (
                                c.modalities.input.iter().map(modality_label).collect(),
                                c.modalities.output.iter().map(modality_label).collect(),
                            )
                        })
                        .unwrap_or_default();
                    models.insert(
                        model_id.clone(),
                        UiModelInfo {
                            name: model_id,
                            kind: kind.to_string(),
                            input_modalities,
                            output_modalities,
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
    enable_code_execution: bool,
    tool_dispatch_url: Option<String>,
) -> Result<Router> {
    let models = build_model_list(&mistralrs);
    let model_wrapper = Model::new(mistralrs.clone());

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
        code_execution_enabled: enable_code_execution,
        tool_dispatch_url,
    });

    let router = Router::new()
        .route(UI_UPLOAD_IMAGE_ROUTE.path, post(upload_image))
        .route(UI_UPLOAD_VIDEO_ROUTE.path, post(upload_video))
        .route(UI_UPLOAD_TEXT_ROUTE.path, post(upload_text))
        .route(UI_UPLOAD_AUDIO_ROUTE.path, post(upload_audio))
        .route(UI_LIST_MODELS_ROUTE.path, get(list_models))
        .route(UI_SELECT_MODEL_ROUTE.path, post(select_model))
        .route(UI_LIST_CHATS_ROUTE.path, get(list_chats))
        .route(UI_NEW_CHAT_ROUTE.path, post(new_chat))
        .route(UI_DELETE_CHAT_ROUTE.path, post(delete_chat))
        .route(UI_LOAD_CHAT_ROUTE.path, post(load_chat))
        .route(UI_RENAME_CHAT_ROUTE.path, post(rename_chat))
        .route(UI_APPEND_MESSAGE_ROUTE.path, post(append_message))
        .route(UI_EDIT_MESSAGE_ROUTE.path, post(edit_message))
        .route(UI_SET_TAIL_ROUTE.path, post(set_tail))
        .route(UI_FORK_SESSION_ROUTE.path, post(fork_session))
        .route(UI_SAVE_CHAT_SESSION_ROUTE.path, post(save_chat_session))
        .route(
            UI_RESTORE_CHAT_SESSION_ROUTE.path,
            post(restore_chat_session),
        )
        .route(UI_SETTINGS_ROUTE.path, get(get_settings))
        .route(UI_CAPABILITIES_ROUTE.path, get(get_capabilities))
        .route(UI_MCP_TOOLS_ROUTE.path, get(list_mcp_tools))
        .route(UI_GENERATE_SPEECH_ROUTE.path, post(generate_speech))
        .nest_service(
            UI_SPEECH_ROUTE.path,
            get_service(ServeDir::new(speech_dir.clone())),
        )
        .nest_service(
            UI_UPLOADS_ROUTE.path,
            get_service(ServeDir::new(uploads_dir.clone())),
        )
        .route(UI_ROOT_ROUTE.path, get(static_handler))
        .route(UI_STATIC_ROUTE.path, get(static_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(axum::extract::Extension(app_state));

    Ok(router)
}
