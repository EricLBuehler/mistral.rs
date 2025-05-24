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
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use chrono::Utc;
use clap::Parser;
use futures_util::stream::StreamExt;
use indexmap::IndexMap;
use mistralrs::{
    best_device, parse_isq_value, IsqType, Model, RequestLike, TextMessageRole, TextMessages,
    TextModelBuilder, VisionMessages, VisionModelBuilder,
};
use serde::Deserialize;
use serde::Serialize;
use serde_json::{json, Value};
use std::io::Cursor;
use std::mem;
use std::{net::SocketAddr, sync::Arc};
use tokio::fs;
use tokio::{net::TcpListener, sync::RwLock};
use tower_http::services::ServeDir;
use tracing::error;
use uuid::Uuid;

const CLEAR_CMD: &str = "__CLEAR__";

#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Cli {
    /// In-situ quantization to apply, defaults to 6-bit.
    #[arg(long = "isq")]
    isq: Option<String>,

    /// Repeated flag for text‑only models
    #[arg(long = "text-model")]
    text_models: Vec<String>,

    /// Repeated flag for vision models
    #[arg(long = "vision-model")]
    vision_models: Vec<String>,

    /// Port to listen on (default: 8080)
    #[arg(long = "port")]
    port: Option<u16>,
}

/// Distinguish at runtime which kind of model we have loaded.
#[derive(Clone)]
enum LoadedModel {
    Text(Arc<Model>),
    Vision(Arc<Model>),
}

#[derive(Serialize, Deserialize)]
struct ChatMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    images: Option<Vec<String>>,
}

#[derive(Serialize, Deserialize)]
struct ChatFile {
    #[serde(default)]
    title: Option<String>,
    model: String,
    kind: String,
    created_at: String,
    messages: Vec<ChatMessage>,
}

struct AppState {
    models: IndexMap<String, LoadedModel>,
    current: RwLock<Option<String>>,
    chats_dir: String,
    current_chat: RwLock<Option<String>>,
    next_chat_id: RwLock<u32>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    if cli.text_models.is_empty() && cli.vision_models.is_empty() {
        eprintln!("At least one --text-model or --vision-model is required");
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

    let mut models: IndexMap<String, LoadedModel> = IndexMap::new();

    // Insert text models first
    for path in cli.text_models {
        let name = std::path::Path::new(&path)
            .file_name()
            .and_then(|p| p.to_str())
            .unwrap_or("text-model")
            .to_string();
        println!("📝 Loading text model: {name}");
        let m = TextModelBuilder::new(path)
            .with_isq(isq.unwrap_or(default_isq))
            .with_logging()
            .with_throughput_logging()
            .build()
            .await?;
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
        let m = VisionModelBuilder::new(path)
            .with_isq(isq.unwrap_or(default_isq))
            .with_logging()
            .with_throughput_logging()
            .build()
            .await?;
        models.insert(name, LoadedModel::Vision(Arc::new(m)));
    }

    let chats_dir = format!("{}/cache/chats", env!("CARGO_MANIFEST_DIR"));
    tokio::fs::create_dir_all(&chats_dir).await?;
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

    let app_state = Arc::new(AppState {
        models,
        current: RwLock::new(None),
        chats_dir,
        current_chat: RwLock::new(None),
        next_chat_id: RwLock::new(next_id),
    });

    let app = Router::new()
        .route("/ws", get(ws_handler))
        .route("/api/upload_image", post(upload_image))
        .route("/api/list_models", get(list_models))
        .route("/api/select_model", post(select_model))
        .route("/api/list_chats", get(list_chats))
        .route("/api/new_chat", post(new_chat))
        .route("/api/delete_chat", post(delete_chat))
        .route("/api/load_chat", post(load_chat))
        .route("/api/rename_chat", post(rename_chat))
        .nest_service(
            "/",
            get_service(ServeDir::new(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/static"
            ))),
        )
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .with_state(app_state.clone());

    let addr: SocketAddr = ([0, 0, 0, 0], cli.port.unwrap_or(8080)).into();
    let listener = TcpListener::bind(addr).await?;
    println!("🔌 listening on http://{}", addr);
    axum::serve(listener, app).await?;
    Ok(())
}

/// Accepts multipart image upload, stores it under `cache/uploads/`, and returns its URL.
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

/// ---------- Chat‑history helpers & endpoints ----------
async fn list_chats(State(app): State<Arc<AppState>>) -> impl IntoResponse {
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

#[derive(Deserialize)]
struct NewChatRequest {
    model: String,
}

async fn new_chat(
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
#[derive(Deserialize)]
struct DeleteChatRequest {
    id: String,
}

async fn delete_chat(
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

#[derive(Deserialize)]
struct LoadChatRequest {
    id: String,
}

async fn load_chat(
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

#[derive(Deserialize)]
struct RenameChatRequest {
    id: String,
    title: String,
}

async fn rename_chat(
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

async fn append_chat_message(
    app: &Arc<AppState>,
    role: &str,
    content: &str,
    images: Option<Vec<String>>,
) -> Result<()> {
    // Ignore replay helpers sent from the front‑end
    if content.trim_start().starts_with("{\"restore\":") {
        return Ok(());
    }
    let chat_opt = app.current_chat.read().await.clone();
    let Some(chat_id) = chat_opt else {
        return Ok(());
    };
    let path = format!("{}/{}.json", app.chats_dir, chat_id);

    let mut chat: ChatFile = if let Ok(data) = fs::read(&path).await {
        serde_json::from_slice(&data).unwrap_or(ChatFile {
            title: None,
            model: app.current.read().await.clone().unwrap_or_default(),
            kind: String::new(),
            created_at: Utc::now().to_rfc3339(),
            messages: Vec::new(),
        })
    } else {
        ChatFile {
            title: None,
            model: app.current.read().await.clone().unwrap_or_default(),
            kind: String::new(),
            created_at: Utc::now().to_rfc3339(),
            messages: Vec::new(),
        }
    };

    chat.messages.push(ChatMessage {
        role: role.into(),
        content: content.into(),
        images,
    });
    fs::write(&path, serde_json::to_vec_pretty(&chat)?).await?;
    Ok(())
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
    // Track which chat file this websocket has prepared context for.
    // When the user switches chats via /api/load_chat, app.current_chat changes;
    // we must reset local state so conversations don't leak across chats.
    let mut active_chat_id: Option<String> = {
        // Whatever chat (if any) was active when the socket was opened.
        app.current_chat.read().await.clone()
    };

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        // ----- Detect chat switch -----
        let cur_chat_id = app.current_chat.read().await.clone();
        if cur_chat_id != active_chat_id {
            // A new chat has been selected: wipe the message buffers so
            // previous conversation state does not bleed into this chat.
            text_msgs = TextMessages::new();
            vision_msgs = VisionMessages::new();
            image_buffer.clear();
            active_chat_id = cur_chat_id;
        }
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
        // Handle front‑end replay helper messages without triggering inference
        if user_msg.trim_start().starts_with("{\"restore\":") {
            if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
                if let Some(obj) = val.get("restore") {
                    // Handle restoring saved messages (with optional images)
                    if let (Some(role), Some(content)) = (
                        obj.get("role").and_then(|v| v.as_str()),
                        obj.get("content").and_then(|v| v.as_str()),
                    ) {
                        let has_images = obj
                            .get("images")
                            .and_then(|v| v.as_array())
                            .is_some_and(|arr| !arr.is_empty());
                        match app.current.read().await.as_deref() {
                            Some(model_name) if app.models.get(model_name).is_some() => {
                                match app.models.get(model_name).unwrap() {
                                    LoadedModel::Text(_) => {
                                        // Text-only context
                                        text_msgs = text_msgs.add_message(
                                            if role == "assistant" {
                                                TextMessageRole::Assistant
                                            } else {
                                                TextMessageRole::User
                                            },
                                            content,
                                        );
                                    }
                                    LoadedModel::Vision(model) => {
                                        let role_enum = if role == "assistant" {
                                            TextMessageRole::Assistant
                                        } else {
                                            TextMessageRole::User
                                        };
                                        if has_images {
                                            // Collect restored images
                                            let mut imgs_b64 = Vec::new();
                                            if let Some(arr) =
                                                obj.get("images").and_then(|v| v.as_array())
                                            {
                                                for img_val in arr {
                                                    if let Some(src) = img_val.as_str() {
                                                        if let Some(idx) = src.find(',') {
                                                            let b64_data = &src[idx + 1..];
                                                            imgs_b64.push(format!(
                                                                "data:image/png;base64,{}",
                                                                b64_data
                                                            ));
                                                            if let Ok(img_bytes) =
                                                                BASE64.decode(b64_data.as_bytes())
                                                            {
                                                                if let Ok(img) =
                                                                    image::load_from_memory(
                                                                        &img_bytes,
                                                                    )
                                                                {
                                                                    image_buffer.push(img);
                                                                }
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                            // Restore as an image message
                                            if let Ok(updated) =
                                                vision_msgs.clone().add_image_message(
                                                    role_enum,
                                                    content,
                                                    image_buffer.clone(),
                                                    model,
                                                )
                                            {
                                                vision_msgs = updated;
                                            }
                                            // Clear buffer after use
                                            image_buffer.clear();
                                        } else {
                                            vision_msgs =
                                                vision_msgs.add_message(role_enum, content);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            // Skip normal handling (no saving, no inference)
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
                if let Err(e) = append_chat_message(&app, "user", &user_msg, None).await {
                    error!("chat save error: {}", e);
                }
                let mut assistant_content = String::new();
                let msgs_snapshot = text_msgs.clone();

                streaming = true;
                let stream_res = stream_and_forward(
                    &model,
                    msgs_snapshot,
                    &mut socket,
                    |tok| {
                        assistant_content = tok.to_string();
                        let cur = mem::take(&mut text_msgs);
                        text_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                    },
                    || streaming = false,
                )
                .await;
                if !assistant_content.is_empty() {
                    let _ = append_chat_message(&app, "assistant", &assistant_content, None).await;
                }
                if let Err(e) = stream_res {
                    error!("stream error: {}", e);
                }
            }
            LoadedModel::Vision(model) => {
                // Track the exact set of messages that will be sent *this* turn.
                let mut msgs_for_stream: Option<VisionMessages> = None;
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
                        msgs_for_stream = Some(vision_msgs.clone());
                        if let Err(e) = append_chat_message(&app, "user", &user_msg, None).await {
                            error!("chat save error: {}", e);
                        }
                    }
                } else {
                    // Plain-text prompt arrives here
                    if image_buffer.is_empty() {
                        vision_msgs = vision_msgs.add_message(TextMessageRole::User, &user_msg);
                        // Send the text‑only context to the model
                        msgs_for_stream = Some(vision_msgs.clone());
                        if let Err(e) = append_chat_message(&app, "user", &user_msg, None).await {
                            error!("chat save error: {}", e);
                        }
                    } else {
                        match vision_msgs.clone().add_image_message(
                            TextMessageRole::User,
                            &user_msg,
                            image_buffer.clone(),
                            &model,
                        ) {
                            Ok(updated) => {
                                // Keep the *text‑only* conversation in our long‑term state,
                                // but build a one‑off request that includes images.
                                let temp_msgs = updated;
                                vision_msgs =
                                    vision_msgs.add_message(TextMessageRole::User, &user_msg);
                                msgs_for_stream = Some(temp_msgs.clone());
                                // ---- persist user message with images ----
                                let mut imgs_b64 = Vec::new();
                                for img in &image_buffer {
                                    let mut buf = Vec::new();
                                    if img
                                        .write_to(
                                            &mut Cursor::new(&mut buf),
                                            image::ImageFormat::Png,
                                        )
                                        .is_ok()
                                    {
                                        imgs_b64.push(format!(
                                            "data:image/png;base64,{}",
                                            BASE64.encode(&buf)
                                        ));
                                    }
                                }
                                if let Err(e) =
                                    append_chat_message(&app, "user", &user_msg, Some(imgs_b64))
                                        .await
                                {
                                    error!("chat save error: {}", e);
                                }
                                image_buffer.clear();
                            }
                            Err(e) => {
                                error!("image prompt error: {}", e);
                                let _ = socket.send(Message::Text(format!("Error: {}", e))).await;
                            }
                        }
                    }
                }

                streaming = true;
                let mut assistant_content = String::new();
                dbg!(&msgs_for_stream
                    .as_ref()
                    .expect("msgs_for_stream must be set")
                    .messages_ref());
                let stream_res = stream_and_forward(
                    &model,
                    msgs_for_stream.expect("msgs_for_stream must be set"),
                    &mut socket,
                    |tok| {
                        assistant_content = tok.to_string();
                        let cur = mem::take(&mut vision_msgs);
                        vision_msgs = cur.add_message(TextMessageRole::Assistant, tok);
                    },
                    || streaming = false,
                )
                .await;
                if !assistant_content.is_empty() {
                    let _ = append_chat_message(&app, "assistant", &assistant_content, None).await;
                }
                if let Err(e) = stream_res {
                    error!("stream error: {}", e);
                }
            }
        }
    }
}
