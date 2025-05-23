use anyhow::Result;
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::DefaultBodyLimit,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, get_service, post},
    Json, Router,
};
use futures_util::stream::StreamExt;
use mistralrs::{
    IsqType, Model, TextMessageRole, TextMessages, TextModelBuilder, VisionMessages,
    VisionModelBuilder,
};
use serde::Deserialize;
use serde_json::{json, Value};
use std::mem;
use std::{net::SocketAddr, sync::Arc};
use tokio::{net::TcpListener, sync::RwLock};
use tower_http::services::ServeDir;
use tracing::error;
use uuid::Uuid;

/// Distinguish at runtime which kind of model we have loaded.
#[derive(Clone)]
enum LoadedModel {
    Text(Arc<Model>),
    Vision(Arc<Model>),
}

#[derive(Clone)]
struct AppState {
    model: Arc<RwLock<Option<LoadedModel>>>,
}

#[derive(Deserialize)]
struct LoadRequest {
    path: String,
    kind: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initially no model is loaded
    let app_state = AppState {
        model: Arc::new(RwLock::new(None)),
    };

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/load_model", post(load_model))
        .route("/api/upload_image", post(upload_image))
        .nest_service(
            "/",
            get_service(ServeDir::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/static"
            ))),
        )
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024)) // allow uploads up to 50 MB
        .with_state(app_state);

    let addr: SocketAddr = ([0, 0, 0, 0], 3000).into();
    let listener = TcpListener::bind(addr).await?;
    println!("🔌 listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Accepts multipart image upload, stores it under `static/uploads/`, and returns its URL.
async fn upload_image(State(_app): State<AppState>, mut multipart: Multipart) -> impl IntoResponse {
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

/// Loads a model from a path and kind.
async fn load_model(
    State(app): State<AppState>,
    Json(req): Json<LoadRequest>,
) -> impl IntoResponse {
    let isq = if cfg!(feature = "metal") {
        IsqType::AFQ6
    } else {
        IsqType::Q6K
    };

    // use mistralrs::PagedAttentionMetaBuilder;
    let result = (|| async {
        match req.kind.as_str() {
            "text" => TextModelBuilder::new(req.path)
                .with_isq(isq)
                .with_logging()
                // .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                .with_throughput_logging()
                .build()
                .await
                .map(|m| LoadedModel::Text(Arc::new(m))),
            "vision" => VisionModelBuilder::new(req.path)
                .with_isq(isq)
                .with_logging()
                // .with_paged_attn(|| PagedAttentionMetaBuilder::default().build())?
                .with_throughput_logging()
                .build()
                .await
                .map(|m| LoadedModel::Vision(Arc::new(m))),
            _ => Err(anyhow::anyhow!("unknown model kind")),
        }
    })()
    .await;

    match result {
        Ok(model) => {
            let mut guard = app.model.write().await;
            *guard = Some(model);
            (StatusCode::OK, "Model loaded successfully").into_response()
        }
        Err(e) => {
            error!("load error: {}", e);
            (StatusCode::INTERNAL_SERVER_ERROR, format!("Error: {}", e)).into_response()
        }
    }
}

/// Upgrades an HTTP request to a WebSocket connection.
async fn ws_handler(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

/// Generic helper to stream tokens and forward them to the websocket.
async fn stream_and_forward<Msgs, F>(
    model: &Arc<Model>,
    msgs: Msgs,
    socket: &mut WebSocket,
    mut on_assistant_complete: F,
) -> Result<(), anyhow::Error>
where
    Msgs: mistralrs::RequestLike + Send + 'static,
    F: FnMut(&str),
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
            on_assistant_complete(&assistant_reply);
            Ok(())
        }
        Err(e) => {
            let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
            Err(e)
        }
    }
}

/// Per-connection task.
async fn handle_socket(mut socket: WebSocket, app: AppState) {
    let mut text_msgs = TextMessages::new();
    let mut vision_msgs = VisionMessages::new();
    let mut image_buffer: Vec<image::DynamicImage> = Vec::new();

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        let model_opt = { app.model.read().await.clone() };
        let Some(model_loaded) = model_opt else {
            let _ = socket
                .send(Message::Text(
                    "No model loaded. Use the UI to load one.".into(),
                ))
                .await;
            continue;
        };

        match model_loaded {
            LoadedModel::Text(model) => {
                text_msgs = text_msgs.add_message(TextMessageRole::User, &user_msg);
                let msgs_snapshot = text_msgs.clone();

                if let Err(e) = stream_and_forward(&model, msgs_snapshot, &mut socket, |tok| {
                    // move the current value out, update, then place it back
                    let cur = mem::take(&mut text_msgs);
                    text_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                })
                .await
                {
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
                if let Err(e) = stream_and_forward(&model, msgs_snapshot, &mut socket, |tok| {
                    // move the current value out, update, then place it back
                    let cur = mem::take(&mut vision_msgs);
                    vision_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                })
                .await
                {
                    error!("stream error: {}", e);
                }
            }
        }
    }
}
