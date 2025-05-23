use anyhow::Result;
const CLEAR_CMD: &str = "__CLEAR__";
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::DefaultBodyLimit,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, get_service, post},
    Json, Router,
};
use clap::Parser;
use futures_util::stream::StreamExt;
use mistralrs::{
    IsqType, Model, TextMessageRole, TextMessages, TextModelBuilder, VisionMessages,
    VisionModelBuilder,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::mem;
use std::{net::SocketAddr, sync::Arc};
use tokio::{net::TcpListener, sync::RwLock};
use tower_http::services::ServeDir;
use tracing::error;
use uuid::Uuid;

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// Repeated flag for text‑only models
    #[arg(long = "text-model")]
    text_models: Vec<String>,

    /// Repeated flag for vision models
    #[arg(long = "vision-model")]
    vision_models: Vec<String>,
}

/// Distinguish at runtime which kind of model we have loaded.
#[derive(Clone)]
enum LoadedModel {
    Text(Arc<Model>),
    Vision(Arc<Model>),
}

struct AppState {
    models: HashMap<String, LoadedModel>,
    current: RwLock<Option<String>>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.text_models.is_empty() && cli.vision_models.is_empty() {
        eprintln!("At least one --text-model or --vision-model is required");
        std::process::exit(1);
    }

    let isq = if cfg!(feature = "metal") {
        IsqType::AFQ6
    } else {
        IsqType::Q6K
    };
    let mut models: HashMap<String, LoadedModel> = HashMap::new();

    for path in cli.text_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("text-model")
            .to_string();
        println!("📝 Loading text model: {name}");
        let m = TextModelBuilder::new(path)
            .with_isq(isq)
            .with_logging()
            .with_throughput_logging()
            .build()
            .await?;
        models.insert(name, LoadedModel::Text(Arc::new(m)));
    }

    for path in cli.vision_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("vision-model")
            .to_string();
        println!("🖼️  Loading vision model: {name}");
        let m = VisionModelBuilder::new(path)
            .with_isq(isq)
            .with_logging()
            .with_throughput_logging()
            .build()
            .await?;
        models.insert(name, LoadedModel::Vision(Arc::new(m)));
    }

    let app_state = Arc::new(AppState {
        models,
        current: RwLock::new(None),
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/upload_image", post(upload_image))
        .route("/api/list_models", get(list_models))
        .route("/api/select_model", post(select_model))
        .nest_service(
            "/",
            get_service(ServeDir::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/static"
            ))),
        )
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(app_state.clone());

    let addr: SocketAddr = ([0, 0, 0, 0], 3000).into();
    let listener = TcpListener::bind(addr).await?;
    println!("🔌 listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Accepts multipart image upload, stores it under `static/uploads/`, and returns its URL.
async fn upload_image(
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
        let upload_dir = format!("{}/static/uploads", env!("CARGO_MANIFEST_DIR"));
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

#[derive(Deserialize)]
struct SelectRequest {
    name: String,
}

async fn list_models(State(app): State<Arc<AppState>>) -> impl IntoResponse {
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

async fn select_model(
    State(app): State<Arc<AppState>>,
    Json(req): Json<SelectRequest>,
) -> impl IntoResponse {
    if app.models.contains_key(&req.name) {
        let mut cur = app.current.write().await;
        *cur = Some(req.name.clone());
        (StatusCode::OK, "Model selected").into_response()
    } else {
        (StatusCode::NOT_FOUND, "Model not found").into_response()
    }
}

/// Upgrades an HTTP request to a WebSocket connection.
async fn ws_handler(ws: WebSocketUpgrade, State(app): State<Arc<AppState>>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

/// Generic helper to stream tokens and forward them to the websocket.
async fn stream_and_forward<Msgs, F, E>(
    model: &Arc<Model>,
    msgs: Msgs,
    socket: &mut WebSocket,
    mut on_token: F,
    mut on_end: E,
) -> Result<(), anyhow::Error>
where
    Msgs: mistralrs::RequestLike + Send + 'static,
    F: FnMut(&str),
    E: FnMut(),
{
    match model.stream_chat_request(msgs).await {
        Ok(mut stream) => {
            let mut assistant_reply = String::new();
            while let Some(chunk) = stream.next().await {
                if let mistralrs::Response::Chunk(resp) = chunk {
                    if let Some(choice) = resp.choices.first() {
                        if let Some(token) = &choice.delta.content {
                            if socket.send(Message::Text(token.clone())).await.is_err() {
                                break;
                            }
                            assistant_reply.push_str(token);
                        }
                    }
                }
            }
            on_token(&assistant_reply);
            on_end();
            Ok(())
        }
        Err(e) => {
            let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
            Err(e)
        }
    }
}

/// Per-connection task.
async fn handle_socket(mut socket: WebSocket, app: Arc<AppState>) {
    let mut text_msgs = TextMessages::new();
    let mut vision_msgs = VisionMessages::new();
    let mut image_buffer: Vec<image::DynamicImage> = Vec::new();
    // `true` while we are streaming a reply back to the client.
    let mut streaming = false;

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        // Allow client to request a context reset without closing the socket
        if user_msg == CLEAR_CMD {
            if streaming {
                let _ = socket
                    .send(Message::Text(
                        "Cannot clear while assistant is replying.".into(),
                    ))
                    .await;
            } else {
                text_msgs = TextMessages::new();
                vision_msgs = VisionMessages::new();
                image_buffer.clear();
                let _ = socket.send(Message::Text("[Context cleared]".into())).await;
            }
            continue;
        }
        let model_name_opt = { app.current.read().await.clone() };
        let Some(model_name) = model_name_opt else {
            let _ = socket
                .send(Message::Text(
                    "No model selected. Choose one in the sidebar.".into(),
                ))
                .await;
            continue;
        };
        let Some(model_loaded) = app.models.get(&model_name).cloned() else {
            let _ = socket
                .send(Message::Text("Selected model not found.".into()))
                .await;
            continue;
        };

        match model_loaded {
            LoadedModel::Text(model) => {
                text_msgs = text_msgs.add_message(TextMessageRole::User, &user_msg);
                let msgs_snapshot = text_msgs.clone();

                streaming = true;
                let stream_res = stream_and_forward(
                    &model,
                    msgs_snapshot,
                    &mut socket,
                    |tok| {
                        let cur = mem::take(&mut text_msgs);
                        text_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                    },
                    || streaming = false,
                )
                .await;
                if let Err(e) = stream_res {
                    error!("stream error: {}", e);
                }
            }
            LoadedModel::Vision(model) => {
                // --- Vision input routing ---
                if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
                    // Case 1: pure image payload => buffer it and wait for a prompt
                    if let Some(url) = val.get("image").and_then(|v| v.as_str()) {
                        // load & decode
                        match tokio::fs::read(&url).await {
                            Ok(bytes) => match image::load_from_memory(&bytes) {
                                Ok(img) => {
                                    image_buffer.push(img);
                                }
                                Err(e) => {
                                    error!("image decode error: {}", e);
                                    let _ =
                                        socket.send(Message::Text(format!("Error: {}", e))).await;
                                }
                            },
                            Err(e) => {
                                error!("image read error: {}", e);
                                let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
                            }
                        }
                        // Skip sending to model until we get a prompt
                        continue;
                    } else {
                        // Fallback: treat whole JSON as text
                        vision_msgs = vision_msgs.add_message(TextMessageRole::User, &user_msg);
                    }
                } else {
                    // Plain-text prompt arrives here
                    if image_buffer.is_empty() {
                        vision_msgs = vision_msgs.add_message(TextMessageRole::User, &user_msg);
                    } else {
                        match vision_msgs.clone().add_image_message(
                            TextMessageRole::User,
                            &user_msg,
                            image_buffer.clone(),
                            &model,
                        ) {
                            Ok(updated) => {
                                vision_msgs = updated;
                                image_buffer.clear();
                            }
                            Err(e) => {
                                error!("image prompt error: {}", e);
                                let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
                            }
                        }
                    }
                }

                let msgs_snapshot = vision_msgs.clone();
                streaming = true;
                let stream_res = stream_and_forward(
                    &model,
                    msgs_snapshot,
                    &mut socket,
                    |tok| {
                        let cur = mem::take(&mut vision_msgs);
                        vision_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                    },
                    || streaming = false,
                )
                .await;
                if let Err(e) = stream_res {
                    error!("stream error: {}", e);
                }
            }
        }
    }
}
