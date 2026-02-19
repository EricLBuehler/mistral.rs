use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine as _};
use futures_util::stream::StreamExt;
use mistralrs::{
    Model, RequestBuilder, TextMessageRole, TextMessages, VisionMessages, WebSearchOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::io::Cursor;
use std::mem;
use std::path::Path;
use std::sync::Arc;
use tracing::error;

use mistralrs::AudioInput;

use crate::chat::append_chat_message;
use crate::models::LoadedModel;
use crate::types::{AppState, GenerationParams};
use crate::utils::get_cache_dir;

/// Generation parameters that can be sent per-message from the frontend
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MessageGenerationParams {
    #[serde(default)]
    pub temperature: Option<f64>,
    #[serde(default)]
    pub top_p: Option<f64>,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default)]
    pub max_tokens: Option<usize>,
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
}

const CLEAR_CMD: &str = "__CLEAR__";
/// Context for managing vision messages and image buffer.
pub struct VisionContext<'a> {
    pub msgs: &'a mut VisionMessages,
    pub image_buffer: &'a mut Vec<image::DynamicImage>,
    pub audio_buffer: &'a mut Vec<AudioInput>,
}

pub struct HandlerParams<'a> {
    pub socket: &'a mut WebSocket,
    pub app: &'a Arc<AppState>,
    pub streaming: &'a mut bool,
    pub active_chat_id: &'a Option<String>,
    pub gen_params: Option<MessageGenerationParams>,
    #[allow(dead_code)] // Reserved for future use (e.g., per-request system prompt override)
    pub system_prompt: &'a Option<String>,
}

/// Apply generation parameters to a RequestBuilder
fn apply_gen_params(
    mut builder: RequestBuilder,
    params: &Option<MessageGenerationParams>,
    defaults: &GenerationParams,
) -> RequestBuilder {
    // Use provided params or fall back to defaults
    let temp = params
        .as_ref()
        .and_then(|p| p.temperature)
        .unwrap_or(defaults.temperature);
    let top_p = params
        .as_ref()
        .and_then(|p| p.top_p)
        .unwrap_or(defaults.top_p);
    let top_k = params
        .as_ref()
        .and_then(|p| p.top_k)
        .unwrap_or(defaults.top_k);
    let max_tokens = params
        .as_ref()
        .and_then(|p| p.max_tokens)
        .unwrap_or(defaults.max_tokens);
    let rep_penalty = params
        .as_ref()
        .and_then(|p| p.repetition_penalty)
        .unwrap_or(defaults.repetition_penalty);

    builder = builder
        .set_sampler_temperature(temp)
        .set_sampler_topp(top_p)
        .set_sampler_topk(top_k)
        .set_sampler_max_len(max_tokens)
        .set_sampler_frequency_penalty(rep_penalty);

    builder
}

/// Upgrades an HTTP request to a WebSocket connection.
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(app): State<Arc<AppState>>,
) -> impl IntoResponse {
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
                            if socket
                                .send(Message::Text(token.clone().into()))
                                .await
                                .is_err()
                            {
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
            let _ = socket
                .send(Message::Text(format!("Error: {e}").into()))
                .await;
            Err(e.into())
        }
    }
}

/// Validate that a file path is safe and within the uploads directory
fn validate_image_path(path: &str) -> Result<String, &'static str> {
    // Determine upload directory under user cache
    let uploads_dir = get_cache_dir().join("uploads");
    let uploads_path = uploads_dir.as_path();

    // Resolve the full path
    let file_path = Path::new(path);

    // Check if it's an absolute path and starts with our uploads directory
    if !file_path.is_absolute() {
        return Err("Path must be absolute");
    }

    // Normalize and check if the path is within uploads directory
    match file_path.canonicalize() {
        Ok(canonical_path) => match uploads_path.canonicalize() {
            Ok(canonical_uploads) => {
                if canonical_path.starts_with(canonical_uploads) {
                    Ok(path.to_string())
                } else {
                    Err("Path is outside uploads directory")
                }
            }
            Err(_) => Err("Uploads directory not found"),
        },
        Err(_) => Err("Invalid file path"),
    }
}

fn validate_audio_path(path: &str) -> Result<String, &'static str> {
    // Same safety check as images
    let uploads_dir = get_cache_dir().join("uploads");
    let uploads_path = uploads_dir.as_path();

    let file_path = Path::new(path);
    if !file_path.is_absolute() {
        return Err("Path must be absolute");
    }

    match file_path.canonicalize() {
        Ok(canonical_path) => match uploads_path.canonicalize() {
            Ok(canonical_uploads) => {
                if canonical_path.starts_with(canonical_uploads) {
                    Ok(path.to_string())
                } else {
                    Err("Path is outside uploads directory")
                }
            }
            Err(_) => Err("Uploads directory not found"),
        },
        Err(_) => Err("Invalid file path"),
    }
}

/// Per-connection task.
pub async fn handle_socket(mut socket: WebSocket, app: Arc<AppState>) {
    let mut text_msgs = TextMessages::new();
    let mut vision_msgs = VisionMessages::new();
    let mut image_buffer: Vec<image::DynamicImage> = Vec::new();
    let mut audio_buffer: Vec<AudioInput> = Vec::new();
    // `true` while we are streaming a reply back to the client.
    let mut streaming = false;
    // Track per-connection chat ID; set by client via WebSocket control
    let mut active_chat_id: Option<String> = None;
    // Per-connection system prompt (can be set via WebSocket message)
    let mut system_prompt: Option<String> = app.default_params.system_prompt.clone();
    // Track whether we've added the system prompt to the context
    let mut system_prompt_added = false;

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        // Handle per-connection chat ID setting
        if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
            if let Some(id) = val.get("chat_id").and_then(|v| v.as_str()) {
                if active_chat_id.as_deref() != Some(id) {
                    active_chat_id = Some(id.to_string());
                    text_msgs = TextMessages::new();
                    vision_msgs = VisionMessages::new();
                    image_buffer.clear();
                    system_prompt_added = false;
                }
                continue;
            }
            // Handle system prompt updates
            if let Some(prompt) = val.get("set_system_prompt") {
                system_prompt = if prompt.is_null() {
                    None
                } else {
                    prompt.as_str().map(|s| s.to_string())
                };
                system_prompt_added = false;
                let _ = socket
                    .send(Message::Text("[System prompt updated]".into()))
                    .await;
                continue;
            }
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
                system_prompt_added = false;
                let _ = socket.send(Message::Text("[Context cleared]".into())).await;
            }
            continue;
        }
        // Handle front-end replay helper messages without triggering inference
        if user_msg.trim_start().starts_with("{\"restore\":") {
            handle_restore_message(
                &user_msg,
                &app,
                &mut text_msgs,
                &mut vision_msgs,
                &mut image_buffer,
            )
            .await;
            continue;
        }
        // Handle chat messages with optional web search options and generation params provided as JSON
        if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
            if let Some(content) = val.get("content").and_then(|v| v.as_str()) {
                // Extract web search options if provided
                let web_search_opts = if let Some(opts_val) = val.get("web_search_options") {
                    match serde_json::from_value::<WebSearchOptions>(opts_val.clone()) {
                        Ok(opts) => Some(opts),
                        Err(e) => {
                            let _ = socket
                                .send(Message::Text(
                                    format!("Error parsing web_search_options: {e}").into(),
                                ))
                                .await;
                            None
                        }
                    }
                } else {
                    None
                };
                // Extract generation parameters if provided
                let gen_params = val.get("generation_params").and_then(|v| {
                    serde_json::from_value::<MessageGenerationParams>(v.clone()).ok()
                });

                // Determine selected model
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

                // Add system prompt if not already added
                if !system_prompt_added {
                    if let Some(ref prompt) = system_prompt {
                        text_msgs = text_msgs.add_message(TextMessageRole::System, prompt);
                        vision_msgs = vision_msgs.add_message(TextMessageRole::System, prompt);
                    }
                    system_prompt_added = true;
                }

                match model_loaded {
                    LoadedModel::Text(model) => {
                        let mut params = HandlerParams {
                            socket: &mut socket,
                            app: &app,
                            streaming: &mut streaming,
                            active_chat_id: &active_chat_id,
                            gen_params: gen_params.clone(),
                            system_prompt: &system_prompt,
                        };
                        handle_text_model(
                            &model,
                            content,
                            web_search_opts.clone(),
                            &mut text_msgs,
                            &mut params,
                        )
                        .await;
                    }
                    LoadedModel::Vision(model) => {
                        let mut vision_ctx = VisionContext {
                            msgs: &mut vision_msgs,
                            image_buffer: &mut image_buffer,
                            audio_buffer: &mut audio_buffer,
                        };
                        let mut params = HandlerParams {
                            socket: &mut socket,
                            app: &app,
                            streaming: &mut streaming,
                            active_chat_id: &active_chat_id,
                            gen_params: gen_params.clone(),
                            system_prompt: &system_prompt,
                        };
                        handle_vision_model(
                            &model,
                            content,
                            web_search_opts.clone(),
                            &mut vision_ctx,
                            &mut params,
                        )
                        .await;
                    }
                    LoadedModel::Speech(_) => {
                        let _ = socket
                            .send(Message::Text(
                                "Speech models are not supported over WebSocket".into(),
                            ))
                            .await;
                    }
                }
                continue;
            }
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

        // Add system prompt if not already added
        if !system_prompt_added {
            if let Some(ref prompt) = system_prompt {
                text_msgs = text_msgs.add_message(TextMessageRole::System, prompt);
                vision_msgs = vision_msgs.add_message(TextMessageRole::System, prompt);
            }
            system_prompt_added = true;
        }

        match model_loaded {
            LoadedModel::Text(model) => {
                let mut params = HandlerParams {
                    socket: &mut socket,
                    app: &app,
                    streaming: &mut streaming,
                    active_chat_id: &active_chat_id,
                    gen_params: None,
                    system_prompt: &system_prompt,
                };
                handle_text_model(&model, &user_msg, None, &mut text_msgs, &mut params).await;
            }
            LoadedModel::Vision(model) => {
                let mut vision_ctx = VisionContext {
                    msgs: &mut vision_msgs,
                    image_buffer: &mut image_buffer,
                    audio_buffer: &mut audio_buffer,
                };
                let mut params = HandlerParams {
                    socket: &mut socket,
                    app: &app,
                    streaming: &mut streaming,
                    active_chat_id: &active_chat_id,
                    gen_params: None,
                    system_prompt: &system_prompt,
                };
                handle_vision_model(&model, &user_msg, None, &mut vision_ctx, &mut params).await;
            }
            // Speech models should use HTTP endpoint; not handled here
            LoadedModel::Speech(_) => {
                let _ = socket
                    .send(Message::Text(
                        "Speech models are not supported over WebSocket".into(),
                    ))
                    .await;
            }
        }
    }
}

async fn handle_restore_message(
    user_msg: &str,
    app: &Arc<AppState>,
    text_msgs: &mut TextMessages,
    vision_msgs: &mut VisionMessages,
    image_buffer: &mut Vec<image::DynamicImage>,
) {
    if let Ok(val) = serde_json::from_str::<Value>(user_msg) {
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
                                *text_msgs = text_msgs.clone().add_message(
                                    if role == "assistant" {
                                        TextMessageRole::Assistant
                                    } else {
                                        TextMessageRole::User
                                    },
                                    content,
                                );
                            }
                            LoadedModel::Vision(_model) => {
                                let role_enum = if role == "assistant" {
                                    TextMessageRole::Assistant
                                } else {
                                    TextMessageRole::User
                                };
                                if has_images {
                                    // Collect restored images
                                    let mut imgs_b64 = Vec::new();
                                    if let Some(arr) = obj.get("images").and_then(|v| v.as_array())
                                    {
                                        for img_val in arr {
                                            if let Some(src) = img_val.as_str() {
                                                if let Some(idx) = src.find(',') {
                                                    let b64_data = &src[idx + 1..];
                                                    imgs_b64.push(format!(
                                                        "data:image/png;base64,{b64_data}"
                                                    ));
                                                    if let Ok(img_bytes) =
                                                        BASE64.decode(b64_data.as_bytes())
                                                    {
                                                        if let Ok(img) =
                                                            image::load_from_memory(&img_bytes)
                                                        {
                                                            image_buffer.push(img);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    // Restore as an image message
                                    *vision_msgs = vision_msgs.clone().add_image_message(
                                        role_enum,
                                        content,
                                        image_buffer.clone(),
                                    );
                                    // Clear buffer after use
                                    image_buffer.clear();
                                } else {
                                    *vision_msgs =
                                        vision_msgs.clone().add_message(role_enum, content);
                                }
                            }
                            // Speech models do not support context restore over WebSocket
                            LoadedModel::Speech(_) => {}
                        }
                    }
                    _ => {}
                }
            }
        }
    }
}

async fn handle_text_model(
    model: &Arc<Model>,
    user_msg: &str,
    web_search_opts: Option<WebSearchOptions>,
    text_msgs: &mut TextMessages,
    params: &mut HandlerParams<'_>,
) {
    // Local aliases keep the original body unchanged.
    let socket = &mut *params.socket;
    let app = params.app;
    let streaming = &mut *params.streaming;
    let active_chat_id = params.active_chat_id;
    *text_msgs = text_msgs
        .clone()
        .add_message(TextMessageRole::User, user_msg);
    if let Some(chat_id) = active_chat_id {
        if let Err(e) = append_chat_message(app, chat_id, "user", user_msg, None).await {
            error!("chat save error: {}", e);
        }
    }
    let mut assistant_content = String::new();
    let msgs_snapshot = text_msgs.clone();
    let mut request_builder = RequestBuilder::from(msgs_snapshot);

    // Apply generation parameters
    request_builder = apply_gen_params(request_builder, &params.gen_params, &app.default_params);

    if let Some(opts) = web_search_opts {
        request_builder = request_builder.with_web_search_options(opts);
    }

    *streaming = true;
    let stream_res = stream_and_forward(
        model,
        request_builder,
        socket,
        |tok| {
            assistant_content = tok.to_string();
            let cur = mem::take(text_msgs);
            *text_msgs = cur.add_message(TextMessageRole::Assistant, tok);
        },
        || *streaming = false,
    )
    .await;
    if !assistant_content.is_empty() {
        if let Some(chat_id) = active_chat_id {
            let _ = append_chat_message(app, chat_id, "assistant", &assistant_content, None).await;
        }
    }
    if let Err(e) = stream_res {
        error!("stream error: {}", e);
    }
}

async fn handle_vision_model(
    model: &Arc<Model>,
    user_msg: &str,
    web_search_opts: Option<WebSearchOptions>,
    vision_ctx: &mut VisionContext<'_>,
    params: &mut HandlerParams<'_>,
) {
    let socket = &mut *params.socket;
    let app = params.app;
    let streaming = &mut *params.streaming;
    let active_chat_id = params.active_chat_id;
    // Track the exact set of messages that will be sent *this* turn.
    let msgs_for_stream: Option<VisionMessages>;
    // --- Vision input routing ---
    if let Ok(val) = serde_json::from_str::<Value>(user_msg) {
        // Case 1a: pure image payload => buffer it and wait for a prompt
        if let Some(url) = val.get("image").and_then(|v| v.as_str()) {
            match validate_image_path(url) {
                Ok(safe_path) => {
                    // load & decode
                    match tokio::fs::read(&safe_path).await {
                        Ok(bytes) => match image::load_from_memory(&bytes) {
                            Ok(img) => {
                                vision_ctx.image_buffer.push(img);
                            }
                            Err(e) => {
                                error!("image decode error: {}", e);
                                let _ = socket
                                    .send(Message::Text(format!("Error: {e}").into()))
                                    .await;
                            }
                        },
                        Err(e) => {
                            error!("image read error: {}", e);
                            let _ = socket
                                .send(Message::Text(format!("Error: {e}").into()))
                                .await;
                        }
                    }
                }
                Err(e) => {
                    error!("Invalid image path: {}", e);
                    let _ = socket
                        .send(Message::Text(
                            format!("Error: Invalid image path - {e}").into(),
                        ))
                        .await;
                }
            }
            // Skip sending to model until we get a prompt
            return;
        // Case 1b: pure audio payload => buffer and wait for prompt
        } else if let Some(url) = val.get("audio").and_then(|v| v.as_str()) {
            match validate_audio_path(url) {
                Ok(safe_path) => match tokio::fs::read(&safe_path).await {
                    Ok(bytes) => match AudioInput::from_bytes(&bytes) {
                        Ok(audio) => {
                            vision_ctx.audio_buffer.push(audio);
                        }
                        Err(e) => {
                            error!("audio decode error: {}", e);
                            let _ = socket
                                .send(Message::Text(format!("Error: {e}").into()))
                                .await;
                        }
                    },
                    Err(e) => {
                        error!("audio read error: {}", e);
                        let _ = socket
                            .send(Message::Text(format!("Error: {e}").into()))
                            .await;
                    }
                },
                Err(e) => {
                    error!("Invalid audio path: {}", e);
                    let _ = socket
                        .send(Message::Text(
                            format!("Error: Invalid audio path - {e}").into(),
                        ))
                        .await;
                }
            }
            return;
        } else {
            // Fallback: treat whole JSON as text
            *vision_ctx.msgs = vision_ctx
                .msgs
                .clone()
                .add_message(TextMessageRole::User, user_msg);
            msgs_for_stream = Some(vision_ctx.msgs.clone());
            if let Some(chat_id) = active_chat_id {
                if let Err(e) = append_chat_message(app, chat_id, "user", user_msg, None).await {
                    error!("chat save error: {}", e);
                }
            }
        }
    } else {
        // Plain-text prompt arrives here
        if vision_ctx.image_buffer.is_empty() && vision_ctx.audio_buffer.is_empty() {
            *vision_ctx.msgs = vision_ctx
                .msgs
                .clone()
                .add_message(TextMessageRole::User, user_msg);
            // Send the text‑only context to the model
            msgs_for_stream = Some(vision_ctx.msgs.clone());
            if let Some(chat_id) = active_chat_id {
                if let Err(e) = append_chat_message(app, chat_id, "user", user_msg, None).await {
                    error!("chat save error: {}", e);
                }
            }
        } else {
            // Prepare multimodal message with images and/or audios
            let temp_msgs = vision_ctx.msgs.clone().add_multimodal_message(
                TextMessageRole::User,
                user_msg,
                vision_ctx.image_buffer.clone(),
                vision_ctx.audio_buffer.clone(),
            );
            // Keep the *text‑only* conversation in our long‑term state,
            // but build a one‑off request that includes images.
            *vision_ctx.msgs = vision_ctx
                .msgs
                .clone()
                .add_message(TextMessageRole::User, user_msg);
            msgs_for_stream = Some(temp_msgs.clone());
            // ---- persist user message with images ----
            let mut imgs_b64 = Vec::new();
            for img in vision_ctx.image_buffer.iter() {
                let mut buf = Vec::new();
                if img
                    .write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                    .is_ok()
                {
                    imgs_b64.push(format!("data:image/png;base64,{}", BASE64.encode(&buf)));
                }
            }
            if let Some(chat_id) = active_chat_id {
                if let Err(e) = append_chat_message(
                    app,
                    chat_id,
                    "user",
                    user_msg,
                    if imgs_b64.is_empty() {
                        None
                    } else {
                        Some(imgs_b64)
                    },
                )
                .await
                {
                    error!("chat save error: {}", e);
                }
            }
            vision_ctx.image_buffer.clear();
            vision_ctx.audio_buffer.clear();
        }
    }

    let msgs = msgs_for_stream.expect("msgs_for_stream must be set");
    let mut request_builder = RequestBuilder::from(msgs);

    // Apply generation parameters
    request_builder = apply_gen_params(request_builder, &params.gen_params, &app.default_params);

    if let Some(opts) = web_search_opts {
        request_builder = request_builder.with_web_search_options(opts);
    }
    *streaming = true;
    let mut assistant_content = String::new();
    let stream_res = stream_and_forward(
        model,
        request_builder,
        socket,
        |tok| {
            assistant_content = tok.to_string();
            let cur = mem::take(vision_ctx.msgs);
            *vision_ctx.msgs = cur.add_message(TextMessageRole::Assistant, tok);
        },
        || *streaming = false,
    )
    .await;
    if !assistant_content.is_empty() {
        if let Some(chat_id) = active_chat_id {
            let _ = append_chat_message(app, chat_id, "assistant", &assistant_content, None).await;
        }
    }
    if let Err(e) = stream_res {
        error!("stream error: {}", e);
    }
}
