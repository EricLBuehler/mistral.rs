use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::Extension,
    response::IntoResponse,
};
use futures_util::stream::StreamExt;
use mistralrs::{
    AudioInput, Model, RequestBuilder, TextMessageRole, TextMessages, VisionMessages,
    WebSearchOptions,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::ui::chat::append_chat_message;
use crate::ui::types::{AppState, GenerationParams};
use crate::ui::utils::get_cache_dir;

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

#[derive(Debug, Clone, Deserialize)]
pub struct RestoreMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub images: Vec<String>,
}

const CLEAR_CMD: &str = "__CLEAR__";

fn apply_gen_params(
    mut builder: RequestBuilder,
    params: &Option<MessageGenerationParams>,
    defaults: &GenerationParams,
) -> RequestBuilder {
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

pub async fn ws_handler(
    ws: WebSocketUpgrade,
    Extension(app): Extension<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_socket(socket, app))
}

async fn stream_and_forward<Msgs>(
    model: &Model,
    msgs: Msgs,
    socket: &mut WebSocket,
    model_id: &str,
) -> Result<String, anyhow::Error>
where
    Msgs: mistralrs::RequestLike + Send + 'static,
{
    match model
        .stream_chat_request_with_model(msgs, Some(model_id))
        .await
    {
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
            Ok(assistant_reply)
        }
        Err(e) => {
            let _ = socket
                .send(Message::Text(format!("Error: {e}").into()))
                .await;
            Err(e.into())
        }
    }
}

fn validate_image_path(path: &str) -> Result<String, &'static str> {
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

fn validate_audio_path(path: &str) -> Result<String, &'static str> {
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

fn uploads_dir() -> PathBuf {
    get_cache_dir().join("uploads")
}

fn upload_path_to_url(path: &str) -> Option<String> {
    let uploads_dir = uploads_dir();
    let canonical_uploads = uploads_dir.canonicalize().ok()?;
    let canonical_path = Path::new(path).canonicalize().ok()?;
    if !canonical_path.starts_with(&canonical_uploads) {
        return None;
    }
    let rel = canonical_path.strip_prefix(&canonical_uploads).ok()?;
    let rel_str = rel.to_string_lossy().replace('\\', "/");
    Some(format!("uploads/{rel_str}"))
}

fn resolve_restore_image(path_or_url: &str) -> Option<PathBuf> {
    let uploads_dir = uploads_dir();
    let candidate = if Path::new(path_or_url).is_absolute() {
        PathBuf::from(path_or_url)
    } else {
        let cleaned = path_or_url
            .trim_start_matches("/ui/")
            .trim_start_matches('/');
        let rel = cleaned.strip_prefix("uploads/")?;
        uploads_dir.join(rel)
    };
    let canonical_uploads = uploads_dir.canonicalize().ok()?;
    let canonical_path = candidate.canonicalize().ok()?;
    if canonical_path.starts_with(&canonical_uploads) {
        Some(canonical_path)
    } else {
        None
    }
}

pub async fn handle_socket(mut socket: WebSocket, app: Arc<AppState>) {
    let mut text_msgs = TextMessages::new();
    let mut vision_msgs = VisionMessages::new();
    let mut image_buffer: Vec<image::DynamicImage> = Vec::new();
    let mut image_path_buffer: Vec<String> = Vec::new();
    let mut audio_buffer: Vec<AudioInput> = Vec::new();
    let mut active_chat_id: Option<String> = None;
    let mut system_prompt: Option<String> = app.default_params.system_prompt.clone();
    let mut system_prompt_added = false;

    while let Some(Ok(Message::Text(user_msg))) = socket.next().await {
        if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
            if let Some(id) = val.get("chat_id").and_then(|v| v.as_str()) {
                if active_chat_id.as_deref() != Some(id) {
                    active_chat_id = Some(id.to_string());
                    text_msgs = TextMessages::new();
                    vision_msgs = VisionMessages::new();
                    image_buffer.clear();
                    image_path_buffer.clear();
                    audio_buffer.clear();
                    system_prompt_added = false;
                }
                continue;
            }
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
            if let Some(restore) = val.get("restore") {
                if let Ok(msg) = serde_json::from_value::<RestoreMessage>(restore.clone()) {
                    if !system_prompt_added {
                        if let Some(prompt) = &system_prompt {
                            text_msgs = text_msgs.add_message(TextMessageRole::System, prompt);
                            vision_msgs = vision_msgs.add_message(TextMessageRole::System, prompt);
                        }
                        system_prompt_added = true;
                    }

                    let role = if msg.role == "assistant" {
                        TextMessageRole::Assistant
                    } else {
                        TextMessageRole::User
                    };
                    let is_user = role == TextMessageRole::User;
                    text_msgs = text_msgs.add_message(role.clone(), msg.content.clone());

                    let model_id = {
                        let cur = app.current.read().await;
                        cur.clone()
                    };
                    let model_kind = model_id
                        .as_ref()
                        .and_then(|id| app.models.get(id))
                        .map(|m| m.kind.as_str())
                        .unwrap_or("text");

                    if model_kind == "vision" {
                        if is_user && !msg.images.is_empty() {
                            let mut images = Vec::new();
                            for src in &msg.images {
                                if let Some(path) = resolve_restore_image(src) {
                                    if let Ok(bytes) = std::fs::read(path) {
                                        if let Ok(img) = image::load_from_memory(&bytes) {
                                            images.push(img);
                                        }
                                    }
                                }
                            }
                            if !images.is_empty() {
                                vision_msgs = vision_msgs.clone().add_multimodal_message(
                                    role.clone(),
                                    msg.content.clone(),
                                    images,
                                    Vec::new(),
                                );
                            } else {
                                vision_msgs =
                                    vision_msgs.add_message(role.clone(), msg.content.clone());
                            }
                        } else {
                            vision_msgs = vision_msgs.add_message(role, msg.content.clone());
                        }
                    }
                }
                continue;
            }
        }

        if user_msg.trim() == CLEAR_CMD {
            text_msgs = TextMessages::new();
            vision_msgs = VisionMessages::new();
            image_buffer.clear();
            image_path_buffer.clear();
            audio_buffer.clear();
            system_prompt_added = false;
            let _ = socket.send(Message::Text("[Context cleared]".into())).await;
            continue;
        }

        let model_id = {
            let cur = app.current.read().await;
            if let Some(name) = &*cur {
                name.clone()
            } else {
                let _ = socket.send(Message::Text("No model selected".into())).await;
                continue;
            }
        };
        let model_kind = app
            .models
            .get(&model_id)
            .map(|m| m.kind.clone())
            .unwrap_or_else(|| "text".to_string());

        let mut gen_params: Option<MessageGenerationParams> = None;
        let mut web_search_opts: Option<WebSearchOptions> = None;
        let mut content = user_msg.to_string();

        if let Ok(val) = serde_json::from_str::<Value>(&user_msg) {
            if let Some(c) = val.get("content").and_then(|v| v.as_str()) {
                content = c.to_string();
            }
            if let Some(params) = val.get("generation_params") {
                if let Ok(p) = serde_json::from_value(params.clone()) {
                    gen_params = Some(p);
                }
            }
            if let Some(opts) = val.get("web_search_options") {
                if let Ok(o) = serde_json::from_value(opts.clone()) {
                    web_search_opts = Some(o);
                }
            }
            if let Some(image) = val.get("image").and_then(|v| v.as_str()) {
                if let Ok(path) = validate_image_path(image) {
                    if let Ok(bytes) = std::fs::read(&path) {
                        if let Ok(img) = image::load_from_memory(&bytes) {
                            image_buffer.push(img);
                        }
                    }
                    image_path_buffer.push(path);
                }
                continue;
            }
            if let Some(audio) = val.get("audio").and_then(|v| v.as_str()) {
                if let Ok(path) = validate_audio_path(audio) {
                    if let Ok(bytes) = std::fs::read(path) {
                        if let Ok(audio) = AudioInput::from_bytes(&bytes) {
                            audio_buffer.push(audio);
                        }
                    }
                }
                continue;
            }
        }

        let image_paths = mem::take(&mut image_path_buffer);
        let images_for_chat = if image_paths.is_empty() {
            None
        } else {
            let urls: Vec<String> = image_paths
                .iter()
                .filter_map(|path| upload_path_to_url(path))
                .collect();
            if urls.is_empty() {
                None
            } else {
                Some(urls)
            }
        };
        if (!content.is_empty() || images_for_chat.is_some()) && active_chat_id.is_some() {
            if let Some(id) = &active_chat_id {
                let _ = append_chat_message(&app, id, "user", &content, images_for_chat).await;
            }
        }

        if model_kind == "speech" {
            let _ = socket
                .send(Message::Text("Selected model is speech-only.".into()))
                .await;
            continue;
        }

        if !system_prompt_added {
            if let Some(prompt) = &system_prompt {
                text_msgs = text_msgs.add_message(TextMessageRole::System, prompt);
                vision_msgs = vision_msgs.add_message(TextMessageRole::System, prompt);
            }
            system_prompt_added = true;
        }

        if model_kind == "vision" {
            if !image_buffer.is_empty() || !audio_buffer.is_empty() {
                let images = mem::take(&mut image_buffer);
                let audios = mem::take(&mut audio_buffer);
                vision_msgs = vision_msgs.clone().add_multimodal_message(
                    TextMessageRole::User,
                    content.clone(),
                    images,
                    audios,
                );
            } else {
                vision_msgs = vision_msgs.add_message(TextMessageRole::User, content.clone());
            }
            let mut builder =
                apply_gen_params(vision_msgs.clone().into(), &gen_params, &app.default_params);
            if app.search_enabled {
                if let Some(opts) = web_search_opts {
                    builder = builder.with_web_search_options(opts);
                }
            }

            let reply = stream_and_forward(&app.model, builder, &mut socket, &model_id).await;
            match reply {
                Ok(text) => {
                    if !text.is_empty() {
                        let cur = mem::take(&mut vision_msgs);
                        vision_msgs = cur.add_message(TextMessageRole::Assistant, &text);
                        if let Some(id) = &active_chat_id {
                            let _ = append_chat_message(&app, id, "assistant", &text, None).await;
                        }
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        } else {
            image_buffer.clear();
            audio_buffer.clear();
            text_msgs = text_msgs.add_message(TextMessageRole::User, content.clone());
            let mut builder =
                apply_gen_params(text_msgs.clone().into(), &gen_params, &app.default_params);
            if app.search_enabled {
                if let Some(opts) = web_search_opts {
                    builder = builder.with_web_search_options(opts);
                }
            }
            let reply = stream_and_forward(&app.model, builder, &mut socket, &model_id).await;
            match reply {
                Ok(text) => {
                    if !text.is_empty() {
                        let cur = mem::take(&mut text_msgs);
                        text_msgs = cur.add_message(TextMessageRole::Assistant, &text);
                        if let Some(id) = &active_chat_id {
                            let _ = append_chat_message(&app, id, "assistant", &text, None).await;
                        }
                    }
                }
                Err(_) => {
                    continue;
                }
            }
        }
    }
}
