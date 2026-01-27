use anyhow::Result;
use axum::body::Body;
use axum::{
    extract::DefaultBodyLimit,
    response::IntoResponse,
    routing::{get, get_service, post},
    Router,
};
use clap::Parser;
use http::{Response, StatusCode};
use hyper::Uri;
use include_dir::{include_dir, Dir};
use indexmap::IndexMap;
use mistralrs::{
    best_device, parse_isq_value, IsqType, SearchEmbeddingModel, SpeechLoaderType,
    SpeechModelBuilder, TextModelBuilder, VisionModelBuilder,
};
use std::sync::Arc;
use tokio::{fs, net::TcpListener};
use tower_http::services::ServeDir;

mod chat;
mod handlers;
mod models;
mod types;
mod utils;

use handlers::{api::*, websocket::ws_handler};
use models::LoadedModel;
use types::{AppState, Cli, GenerationParams};

static STATIC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/static");

async fn static_handler(uri: Uri) -> impl IntoResponse {
    let path = uri.path().trim_start_matches('/');
    // Serve index.html when requesting the root path
    let path = if path.is_empty() { "index.html" } else { path };
    if let Some(file) = STATIC_DIR.get_file(path) {
        let mime = mime_guess::from_path(path).first_or_octet_stream();
        Response::builder()
            .status(StatusCode::OK)
            .header(http::header::CONTENT_TYPE, mime.as_ref())
            .body(Body::from(file.contents()))
            .unwrap()
    } else {
        Response::builder()
            .status(StatusCode::NOT_FOUND)
            .body(Body::from("Not Found"))
            .unwrap()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    eprintln!("⚠️  mistralrs-web-chat is deprecated. Please use `mistralrs serve --ui` from mistralrs-cli instead.");

    let cli = Cli::parse();
    if cli.text_models.is_empty() && cli.vision_models.is_empty() && cli.speech_models.is_empty() {
        eprintln!("At least one --text-model, --vision-model, or --speech-model is required");
        std::process::exit(1);
    }

    let default_isq = if cfg!(feature = "metal") {
        IsqType::AFQ6
    } else {
        IsqType::Q6K
    };

    let device = best_device(false)?;

    let isq = cli
        .isq
        .as_ref()
        .and_then(|isq| parse_isq_value(isq, Some(&device)).ok());

    // Determine embedding model for web search if enabled
    let search_embedding_model: Option<SearchEmbeddingModel> = if cli.enable_search {
        Some(cli.search_embedding_model.unwrap_or_default())
    } else {
        None
    };
    let mut models: IndexMap<String, LoadedModel> = IndexMap::new();

    // Insert text models first
    for path in cli.text_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("text-model")
            .to_string();
        println!("📝 Loading text model: {name}");
        let mut builder = TextModelBuilder::new(path)
            .with_isq(isq.unwrap_or(default_isq))
            .with_logging()
            .with_throughput_logging();
        if let Some(ref search_embedding_model) = search_embedding_model {
            builder = builder.with_search(*search_embedding_model);
        }
        let m = builder.build().await?;
        models.insert(name, LoadedModel::Text(Arc::new(m)));
    }

    // Then insert vision models (preserving order)
    for path in cli.vision_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("vision-model")
            .to_string();
        println!("🖼️  Loading vision model: {name}");
        let mut builder = VisionModelBuilder::new(path)
            .with_isq(isq.unwrap_or(default_isq))
            .with_logging()
            .with_throughput_logging();
        if let Some(ref search_embedding_model) = search_embedding_model {
            builder = builder.with_search(*search_embedding_model);
        }
        let m = builder.build().await?;
        models.insert(name, LoadedModel::Vision(Arc::new(m)));
    }

    // Then insert speech models (text-to-speech)
    for path in cli.speech_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("speech-model")
            .to_string();
        println!("🔊 Loading speech model: {name}");
        let m = SpeechModelBuilder::new(path.clone(), SpeechLoaderType::Dia)
            .with_logging()
            .build()
            .await?;
        models.insert(name, LoadedModel::Speech(Arc::new(m)));
    }

    // Initialize cache directory and chats subdirectory
    let base_cache = utils::get_cache_dir();
    println!("🔧 Using cache directory: {}", base_cache.display());
    let chats_dir = base_cache.join("chats").to_string_lossy().to_string();
    tokio::fs::create_dir_all(&chats_dir).await?;
    // Initialize speech output directory for generated wav files
    let speech_dir = base_cache.join("speech").to_string_lossy().to_string();
    tokio::fs::create_dir_all(&speech_dir).await?;
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

    // Build default generation parameters from CLI
    let default_params = GenerationParams {
        temperature: cli.temperature,
        top_p: cli.top_p,
        top_k: cli.top_k,
        max_tokens: cli.max_tokens,
        repetition_penalty: cli.repetition_penalty,
        system_prompt: cli.system_prompt.clone(),
    };

    let app_state = Arc::new(AppState {
        models,
        current: tokio::sync::RwLock::new(None),
        chats_dir,
        speech_dir: speech_dir.clone(),
        current_chat: tokio::sync::RwLock::new(None),
        next_chat_id: tokio::sync::RwLock::new(next_id),
        default_params,
        search_enabled: cli.enable_search,
    });

    let app = Router::new()
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
        // Generation settings endpoints
        .route("/api/settings", get(get_settings))
        // Text-to-speech generation endpoint
        .route("/api/generate_speech", post(generate_speech))
        // Serve generated speech files
        .nest_service("/speech", get_service(ServeDir::new(speech_dir.clone())))
        // Serve embedded static assets for the root path
        .route("/", get(static_handler))
        .route("/{*path}", get(static_handler))
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(app_state.clone());

    let host = cli.host.as_deref().unwrap_or("0.0.0.0");
    let port = cli.port.unwrap_or(1234);
    let bind_addr = format!("{}:{}", host, port);
    let listener = TcpListener::bind(&bind_addr).await?;
    println!("🔌 Listening on http://{}", bind_addr);
    axum::serve(listener, app).await?;
    Ok(())
}
