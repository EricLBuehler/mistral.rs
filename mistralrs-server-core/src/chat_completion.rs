//! ## Chat Completions functionality and route handler.

use std::{
    collections::HashMap, io::Cursor, ops::Deref, pin::Pin, sync::Arc, task::Poll, time::Duration,
};

use anyhow::{Context, Result};
use axum::{
    extract::{Json, State},
    http::{self},
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
    Extension,
};
use base64::{engine::general_purpose::STANDARD, Engine};
use either::Either;
use image::DynamicImage;
use indexmap::IndexMap;
use itertools::Itertools;
use mistralrs_core::{
    AgentPermission, AgentToolApprovalHandler, AgentToolApprovalNotifier, AgenticToolCallData,
    AgenticToolCallPhase, AgenticToolCallRecord, ChatCompletionChunkResponse,
    ChatCompletionResponse, Constraint, MistralRs, ModelCategory, NormalRequest, ReasoningEffort,
    Request, RequestMessage, Response, SamplingParams,
};
use serde_json::{json, Value};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    completion_core::{
        convert_stop_tokens, get_dry_sampling_params, handle_completion_error,
        BaseCompletionResponder,
    },
    handler_core::{
        create_response_channel, send_request_with_model, BaseJsonModelError, ErrorToResponse,
        JsonError, ModelErrorMessage,
    },
    input_files::{resolve_input_file, InputFileSpec},
    mistralrs_server_router_builder::AgenticDefaults,
    openai::{
        normalize_chat_completion_tools, normalize_responses_tools, validate_openai_tool_choice,
        ChatCompletionRequest, Grammar, JsonSchemaResponseFormat, MessageInnerContent,
        OpenAiToolSurface, ResponseFormat,
    },
    skills::SkillStore,
    streaming::{base_create_streamer, get_keep_alive_interval, BaseStreamer, DoneState},
    types::{ExtractedMistralRsState, OnChunkCallback, OnDoneCallback, SharedMistralRsState},
    util::{
        parse_audio_url_for_server, parse_image_url_for_server, sanitize_error_message,
        validate_model_name,
    },
    video::parse_video_url_for_server,
};

/// A callback function that processes streaming response chunks before they are sent to the client.
///
/// This hook allows modification of each chunk in the streaming response, enabling features like
/// content filtering, transformation, or logging. The callback receives a chunk and must return
/// a (potentially modified) chunk.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::chat_completion::ChatCompletionOnChunkCallback;
///
/// let on_chunk: ChatCompletionOnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
/// Max files surfaced on a single chat completion response body. Additional files
/// produced by the agentic loop are still reachable via `GET /v1/files/{id}` but
/// are not embedded in the response JSON to keep the body bounded.
const MAX_FILES_PER_RESPONSE: usize = 64;

pub type ChatCompletionOnChunkCallback = OnChunkCallback<ChatCompletionChunkResponse>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::chat_completion::ChatCompletionOnDoneCallback;
///
/// let on_done: ChatCompletionOnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type ChatCompletionOnDoneCallback = OnDoneCallback<ChatCompletionChunkResponse>;

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub type ChatCompletionStreamer = BaseStreamer<
    ChatCompletionChunkResponse,
    ChatCompletionOnChunkCallback,
    ChatCompletionOnDoneCallback,
>;

fn encode_agentic_tool_images(images: &[DynamicImage]) -> Vec<String> {
    images
        .iter()
        .filter_map(|image| {
            let mut buffer = Vec::new();
            match image.write_to(&mut Cursor::new(&mut buffer), image::ImageFormat::Png) {
                Ok(()) => Some(STANDARD.encode(buffer)),
                Err(e) => {
                    tracing::warn!("failed to encode agentic tool image: {e}");
                    None
                }
            }
        })
        .collect()
}

pub(crate) fn serialize_agentic_progress(
    round: usize,
    tool_name: &str,
    phase: &AgenticToolCallPhase,
) -> Value {
    let (phase_str, data) = match phase {
        AgenticToolCallPhase::Calling(data) => ("calling", serialize_agentic_data(data)),
        AgenticToolCallPhase::Complete(data) => ("complete", serialize_agentic_data(data)),
    };
    json!({
        "type": "agentic_tool_call_progress",
        "round": round,
        "tool_name": tool_name,
        "phase": phase_str,
        "data": data,
    })
}

fn serialize_agentic_data(data: &AgenticToolCallData) -> Value {
    match data {
        AgenticToolCallData::CodeExecution {
            code,
            stdout,
            stderr,
            exception,
            images,
            video_frames,
            video_frame_count,
            working_directory,
            execution_time_ms,
        } => {
            let mut v = json!({"tool_type": "code_execution"});
            if let Some(c) = code {
                v["code"] = json!(c);
            }
            if let Some(s) = stdout {
                v["stdout"] = json!(s);
            }
            if let Some(s) = stderr {
                v["stderr"] = json!(s);
            }
            if let Some(e) = exception {
                v["exception"] = json!(e);
            }
            if !images.is_empty() {
                v["images_base64"] = json!(encode_agentic_tool_images(images));
            }
            if !video_frames.is_empty() {
                v["video_frames_base64"] = json!(encode_agentic_tool_images(video_frames));
            }
            if let Some(n) = video_frame_count {
                v["video_frame_count"] = json!(n);
            }
            if let Some(d) = working_directory {
                v["working_directory"] = json!(d);
            }
            if let Some(ms) = execution_time_ms {
                v["execution_time_ms"] = json!(ms);
            }
            v
        }
        AgenticToolCallData::WebSearch {
            query,
            results_count,
            sources,
        } => {
            let mut v = json!({"tool_type": "web_search"});
            if let Some(q) = query {
                v["query"] = json!(q);
            }
            if let Some(n) = results_count {
                v["results_count"] = json!(n);
            }
            if !sources.is_empty() {
                v["sources"] = json!(sources);
            }
            v
        }
        AgenticToolCallData::Shell {
            commands,
            stdout,
            stderr,
            exit_code,
            status,
            working_directory,
            timed_out,
        } => {
            let mut v = json!({"tool_type": "shell", "commands": commands});
            if let Some(s) = stdout {
                v["stdout"] = json!(s);
            }
            if let Some(s) = stderr {
                v["stderr"] = json!(s);
            }
            if let Some(code) = exit_code {
                v["exit_code"] = json!(code);
            }
            if let Some(s) = status {
                v["status"] = json!(s);
            }
            if let Some(d) = working_directory {
                v["working_directory"] = json!(d);
            }
            if let Some(t) = timed_out {
                v["timed_out"] = json!(t);
            }
            v
        }
        AgenticToolCallData::Custom { arguments, content } => {
            let mut v = json!({"tool_type": "custom"});
            if !arguments.is_empty() {
                v["arguments"] = json!(arguments);
            }
            if !content.is_empty() {
                v["content"] = json!(content);
            }
            v
        }
    }
}

/// Arguments string from a Calling-phase `AgenticToolCallData`.
fn extract_arguments(data: &AgenticToolCallData) -> String {
    match data {
        AgenticToolCallData::CodeExecution {
            code: Some(code), ..
        } => serde_json::json!({"code": code}).to_string(),
        AgenticToolCallData::WebSearch {
            query: Some(query), ..
        } => serde_json::json!({"query": query}).to_string(),
        AgenticToolCallData::Shell { commands, .. } => {
            serde_json::json!({"commands": commands}).to_string()
        }
        AgenticToolCallData::Custom { arguments, .. } => arguments.clone(),
        _ => String::new(),
    }
}

/// Fold progress events into `AgenticToolCallRecord` for non-streaming responses. `pending_args` keeps Calling-phase args keyed by (round, tool_name).
fn record_agentic_progress(
    records: &mut Vec<AgenticToolCallRecord>,
    pending_args: &mut HashMap<(usize, String), String>,
    round: usize,
    tool_name: &str,
    phase: &AgenticToolCallPhase,
) {
    match phase {
        AgenticToolCallPhase::Calling(data) => {
            pending_args.insert((round, tool_name.to_string()), extract_arguments(data));
        }
        AgenticToolCallPhase::Complete(data) => {
            let arguments = pending_args
                .remove(&(round, tool_name.to_string()))
                .unwrap_or_default();

            let (result_content, result_images_base64) = match data {
                AgenticToolCallData::CodeExecution {
                    stdout,
                    stderr,
                    exception,
                    images,
                    ..
                } => {
                    let mut content_parts = Vec::new();
                    if let Some(s) = stdout {
                        content_parts.push(format!("stdout: {s}"));
                    }
                    if let Some(s) = stderr {
                        content_parts.push(format!("stderr: {s}"));
                    }
                    if let Some(e) = exception {
                        content_parts.push(format!("exception: {e}"));
                    }
                    (content_parts.join("\n"), encode_agentic_tool_images(images))
                }
                AgenticToolCallData::WebSearch {
                    results_count,
                    sources,
                    ..
                } => {
                    let mut parts = Vec::new();
                    if let Some(n) = results_count {
                        parts.push(format!("{n} results"));
                    }
                    if !sources.is_empty() {
                        parts.push(format!("sources: {}", sources.join(", ")));
                    }
                    let msg = parts.join("\n");
                    (msg, vec![])
                }
                AgenticToolCallData::Shell {
                    stdout,
                    stderr,
                    exit_code,
                    status,
                    working_directory,
                    timed_out,
                    ..
                } => {
                    let mut content = json!({
                        "stdout": stdout,
                        "stderr": stderr,
                        "exit_code": exit_code,
                        "status": status,
                        "working_directory": working_directory,
                        "timed_out": timed_out,
                    });
                    if let Some(obj) = content.as_object_mut() {
                        obj.retain(|_, value| !value.is_null());
                    }
                    (content.to_string(), vec![])
                }
                AgenticToolCallData::Custom { content, .. } => (content.clone(), vec![]),
            };
            records.push(AgenticToolCallRecord {
                round,
                name: tool_name.to_string(),
                arguments,
                result_content,
                result_images_base64,
                file_ids: Vec::new(),
            });
        }
    }
}

fn attach_agentic_tool_calls(
    mut response: ChatCompletionResponse,
    records: Vec<AgenticToolCallRecord>,
) -> ChatCompletionResponse {
    if !records.is_empty() {
        response.agentic_tool_calls = Some(records);
    }
    response
}

/// Fill each record's `file_ids` from files whose `source.round` and `source.tool` match.
fn stamp_file_ids(records: &mut [AgenticToolCallRecord], files: &[mistralrs_core::File]) {
    for r in records.iter_mut() {
        let matched: Vec<String> = files
            .iter()
            .filter(|f| f.source.round == r.round && f.source.tool == r.name)
            .map(|f| f.id.clone())
            .collect();
        if !matched.is_empty() {
            r.file_ids = matched;
        }
    }
}

impl futures::Stream for ChatCompletionStreamer {
    type Item = Result<Event, axum::Error>;

    /// Polls the stream for the next Server-Sent Event.
    ///
    /// This method implements the core streaming logic:
    /// 1. Handles stream completion by sending `[DONE]` and executing callbacks
    /// 2. Processes incoming model responses and converts them to SSE events
    /// 3. Applies chunk modifications if a callback is provided
    /// 4. Stores chunks if completion callback is configured
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
                // https://platform.openai.com/docs/api-reference/completions/create
                // If true, returns a stream of events that happen during the Run as server-sent events, terminating when the Run enters a terminal state with a data: [DONE] message.
                self.done_state = DoneState::Done;
                return Poll::Ready(Some(Ok(Event::default().data("[DONE]"))));
            }
            DoneState::Done => {
                if let Some(on_done) = &self.on_done {
                    on_done(&self.chunks);
                }
                return Poll::Ready(None);
            }
            DoneState::Running => (),
        }

        match self.rx.poll_recv(cx) {
            Poll::Ready(Some(resp)) => match resp {
                Response::ModelError(msg, _) => {
                    MistralRs::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.to_string()),
                    );
                    // Done now, just need to send the [DONE]
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(Event::default().data(msg))))
                }
                Response::ValidationError(e) => {
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(
                        Event::default().data(sanitize_error_message(e.as_ref()))
                    )))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(
                        Event::default().data(sanitize_error_message(e.as_ref()))
                    )))
                }
                Response::Chunk(mut response) => {
                    if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                        self.done_state = DoneState::SendingDone;
                    }
                    // Done now, just need to send the [DONE]
                    MistralRs::maybe_log_response(self.state.clone(), &response);

                    if let Some(on_chunk) = &self.on_chunk {
                        response = on_chunk(response);
                    }

                    if self.store_chunks {
                        self.chunks.push(response.clone());
                    }

                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                Response::AgenticToolCallProgress {
                    round,
                    tool_name,
                    phase,
                } => {
                    let payload = serialize_agentic_progress(round, &tool_name, &phase);
                    Poll::Ready(Some(
                        Event::default()
                            .event("agentic_tool_call_progress")
                            .json_data(payload),
                    ))
                }
                Response::AgenticToolApprovalRequired {
                    approval_id,
                    session_id,
                    round,
                    tool,
                    arguments,
                } => {
                    let payload = json!({
                        "type": "agentic_tool_approval_required",
                        "approval_id": approval_id,
                        "session_id": session_id,
                        "round": round,
                        "tool": tool,
                        "arguments": arguments,
                    });
                    Poll::Ready(Some(
                        Event::default()
                            .event("agentic_tool_approval_required")
                            .json_data(payload),
                    ))
                }
                Response::BlockDenoisingProgress(_) => {
                    cx.waker().wake_by_ref();
                    Poll::Pending
                }
                Response::File(file) => Poll::Ready(Some(
                    Event::default().event("file_produced").json_data(file),
                )),
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            },
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

/// Represents different types of chat completion responses.
pub type ChatCompletionResponder =
    BaseCompletionResponder<ChatCompletionResponse, KeepAliveStream<ChatCompletionStreamer>>;

type JsonModelError = BaseJsonModelError<ChatCompletionResponse>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for ChatCompletionResponder {
    /// Converts the chat completion responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::Json(s) => Json(s).into_response(),
            ChatCompletionResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatCompletionResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatCompletionResponder::ModelError(msg, response) => {
                JsonModelError::new(msg, response)
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
}

/// Parse reasoning_effort string to ReasoningEffort enum
fn parse_reasoning_effort(effort: &Option<String>) -> Option<ReasoningEffort> {
    effort
        .as_ref()
        .and_then(|e| match e.to_lowercase().as_str() {
            "low" => Some(ReasoningEffort::Low),
            "medium" => Some(ReasoningEffort::Medium),
            "high" => Some(ReasoningEffort::High),
            _ => None,
        })
}

pub struct ChatCompletionParseContext {
    pub state: SharedMistralRsState,
    pub tx: Sender<Response>,
    pub tool_dispatch_url: Option<String>,
    pub agent_approval_handler: Option<AgentToolApprovalHandler>,
    pub agent_approval_notifier: Option<Arc<AgentToolApprovalNotifier>>,
    pub tool_surface: OpenAiToolSurface,
    pub skill_store: Option<Arc<SkillStore>>,
}

/// Parses and validates a chat completion request.
///
/// This function transforms an OpenAI-compatible chat completion request into the
/// request format used by mistral.rs.
pub async fn parse_request(
    oairequest: ChatCompletionRequest,
    ctx: ChatCompletionParseContext,
) -> Result<(Request, bool)> {
    let ChatCompletionParseContext {
        state,
        tx,
        tool_dispatch_url,
        agent_approval_handler,
        agent_approval_notifier,
        tool_surface,
        skill_store,
    } = ctx;
    let repr = serde_json::to_string(&oairequest)
        .context("Failed to serialize chat completion request for logging")?;
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    // Parse reasoning effort for Harmony-format models
    let reasoning_effort = parse_reasoning_effort(&oairequest.reasoning_effort);

    let mut normalized_tools = match tool_surface {
        OpenAiToolSurface::ChatCompletions => {
            normalize_chat_completion_tools(oairequest.tools, oairequest.web_search_options)?
        }
        OpenAiToolSurface::Responses => normalize_responses_tools(oairequest.tools)?,
    };
    normalized_tools.enable_shell |= oairequest.enable_shell;
    normalized_tools
        .shell_skill_references
        .extend(oairequest.shell_skill_references);
    validate_openai_tool_choice(oairequest.tool_choice.as_ref(), &normalized_tools)?;
    let shell_options = if normalized_tools.shell_skill_references.is_empty() {
        None
    } else {
        let store = skill_store
            .as_ref()
            .context("tools[].type=\"shell\" skill references require a configured skill store.")?;
        Some(store.resolve_references(&normalized_tools.shell_skill_references)?)
    };

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs);
    let mut input_files = Vec::new();

    let messages = match oairequest.messages {
        Either::Left(req_messages) => {
            let mut messages = Vec::new();
            let mut image_urls = Vec::new();
            let mut audio_urls = Vec::new();
            let mut video_urls = Vec::new();
            for message in req_messages {
                let content = match message.content.as_deref() {
                    Some(content) => content.clone(),
                    None => {
                        // Handle tool call
                        let calls = message
                            .tool_calls
                            .as_ref()
                            .context(
                                "No content was provided, expected tool calls to be provided.",
                            )?
                            .iter()
                            .map(|call| &call.function)
                            .collect::<Vec<_>>();

                        Either::Left(serde_json::to_string(&calls)?)
                    }
                };

                match &content {
                    Either::Left(content) => {
                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role.clone()));
                        message_map.insert("content".to_string(), Either::Left(content.clone()));

                        // Add tool_calls for assistant messages that have them
                        if let Some(ref tool_calls) = message.tool_calls {
                            // Convert tool_calls to Vec<IndexMap<String, Value>> for Jinja template
                            let tool_calls_vec: Vec<IndexMap<String, Value>> = tool_calls
                                .iter()
                                .map(|tc| {
                                    let mut tc_map = IndexMap::new();
                                    // Use provided ID or fallback to function name
                                    let id =
                                        tc.id.clone().unwrap_or_else(|| tc.function.name.clone());
                                    tc_map.insert("id".to_string(), Value::String(id));
                                    tc_map.insert(
                                        "type".to_string(),
                                        Value::String("function".to_string()),
                                    );
                                    let mut function_map = serde_json::Map::new();
                                    function_map.insert(
                                        "name".to_string(),
                                        Value::String(tc.function.name.clone()),
                                    );
                                    function_map.insert(
                                        "arguments".to_string(),
                                        Value::String(tc.function.arguments.clone()),
                                    );
                                    tc_map.insert(
                                        "function".to_string(),
                                        Value::Object(function_map),
                                    );
                                    tc_map
                                })
                                .collect();
                            message_map
                                .insert("tool_calls".to_string(), Either::Right(tool_calls_vec));
                        }

                        // Add tool_call_id for tool messages
                        if let Some(ref tool_call_id) = message.tool_call_id {
                            message_map.insert(
                                "tool_call_id".to_string(),
                                Either::Left(tool_call_id.clone()),
                            );
                        }

                        // Add name for tool messages
                        if let Some(ref name) = message.name {
                            message_map.insert("name".to_string(), Either::Left(name.clone()));
                        }

                        messages.push(message_map);
                    }
                    Either::Right(image_messages) => {
                        // If there is only one message, it is possible a text message
                        // found when rig is used as client. In this case, we need to check if
                        // the message is a text message or an image message.
                        if image_messages.len() == 1 && !image_messages[0].contains_key("type") {
                            if !image_messages[0].contains_key("text") {
                                anyhow::bail!("Expected `text` key in input message.");
                            }
                            let content = match image_messages[0]["text"].deref() {
                                Either::Left(left) => left.to_string(),
                                Either::Right(right) => format!("{right:?}"),
                            };
                            let mut message_map: IndexMap<
                                String,
                                Either<String, Vec<IndexMap<String, Value>>>,
                            > = IndexMap::new();
                            message_map.insert("role".to_string(), Either::Left(message.role));
                            message_map.insert("content".to_string(), Either::Left(content));
                            messages.push(message_map);
                            continue;
                        }
                        if message.role != "user" {
                            anyhow::bail!(
                                "Role for an image message must be `user`, but it is {}",
                                message.role
                            );
                        }

                        enum ContentPart {
                            Text { text: String },
                            Image { image_url: String },
                            Audio { audio_url: String },
                            Video { video_url: String },
                            File { spec: InputFileSpec },
                        }

                        let mut items = Vec::new();
                        for image_message in image_messages {
                            match image_message.get("type") {
                                Some(MessageInnerContent(Either::Left(x))) if x == "text" => {
                                    items.push(ContentPart::Text {
                                        text: image_message
                                            .get("text").as_ref()
                                            .context("Text sub-content must have `text` key.")?.as_ref()
                                            .left().context("Text sub-content `text` key must be a string.")?.clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "image_url" => {
                                    items.push(ContentPart::Image {
                                        image_url: image_message
                                            .get("image_url")
                                            .as_ref()
                                            .context("Image sub-content must have `image_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Image sub-content `image_url` key must be an object.")?
                                            .get("url")
                                            .context("Image sub-content `image_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "audio_url" => {
                                    items.push(ContentPart::Audio {
                                        audio_url: image_message
                                            .get("audio_url")
                                            .as_ref()
                                            .context("Audio sub-content must have `audio_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Audio sub-content `audio_url` key must be an object.")?
                                            .get("url")
                                            .context("Audio sub-content `audio_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "video_url" => {
                                    items.push(ContentPart::Video {
                                        video_url: image_message
                                            .get("video_url")
                                            .as_ref()
                                            .context("Video sub-content must have `video_url` key.")?
                                            .as_ref()
                                            .right()
                                            .context("Video sub-content `video_url` key must be an object.")?
                                            .get("url")
                                            .context("Video sub-content `video_url` object must have a `url` key.")?
                                            .clone(),
                                    });
                                }
                                Some(MessageInnerContent(Either::Left(x))) if x == "file" => {
                                    let file = image_message
                                        .get("file")
                                        .as_ref()
                                        .context("File sub-content must have `file` key.")?
                                        .as_ref()
                                        .right()
                                        .context("File sub-content `file` key must be an object.")?;
                                    let spec = InputFileSpec {
                                        file_id: file.get("file_id").cloned(),
                                        file_data: file.get("file_data").cloned(),
                                        file_url: file.get("file_url").cloned(),
                                        filename: file.get("filename").cloned(),
                                    };
                                    if spec.file_url.is_some()
                                        && matches!(tool_surface, OpenAiToolSurface::ChatCompletions)
                                    {
                                        anyhow::bail!(
                                            "Chat Completions file content does not support `file_url`; use Responses `input_file`."
                                        );
                                    }
                                    items.push(ContentPart::File { spec });
                                }
                                _ => anyhow::bail!("Expected array content sub-content to be one of `text`, `image_url`, `audio_url`, `video_url`, or `file`.")
                            }
                        }

                        let text_content = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Text { text } => Some(text),
                                _ => None,
                            })
                            .join(" ");
                        let image_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Image { image_url } => Some(image_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();

                        let audio_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Audio { audio_url } => Some(audio_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();

                        let video_urls_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::Video { video_url } => Some(video_url.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        let file_specs_iter = items
                            .iter()
                            .filter_map(|item| match item {
                                ContentPart::File { spec } => Some(spec.clone()),
                                _ => None,
                            })
                            .collect::<Vec<_>>();
                        for spec in file_specs_iter {
                            let file =
                                resolve_input_file(state.clone(), spec, "input_file").await?;
                            state.insert_file(None, file.clone(), None)?;
                            input_files.push(file);
                        }

                        // Apply prefixer to text content if this is a multimodal model with images/audio/video
                        // This matches the behavior of interactive mode which auto-inserts media tokens
                        let text_content = if !image_urls_iter.is_empty()
                            || !audio_urls_iter.is_empty()
                            || !video_urls_iter.is_empty()
                        {
                            if let Ok(ModelCategory::Multimodal { prefixer }) =
                                state.get_model_category(None)
                            {
                                let mut prefixed = text_content;

                                // Apply image prefixer
                                if !image_urls_iter.is_empty() {
                                    let start_idx = image_urls.len();
                                    let image_indices: Vec<usize> =
                                        (start_idx..start_idx + image_urls_iter.len()).collect();
                                    prefixed = prefixer.prefix_image(image_indices, &prefixed);
                                }

                                // Apply audio prefixer
                                if !audio_urls_iter.is_empty() {
                                    let start_idx = audio_urls.len();
                                    let audio_indices: Vec<usize> =
                                        (start_idx..start_idx + audio_urls_iter.len()).collect();
                                    prefixed = prefixer.prefix_audio(audio_indices, &prefixed);
                                }

                                // Apply video prefixer
                                if !video_urls_iter.is_empty() {
                                    let start_idx = video_urls.len();
                                    let video_indices: Vec<usize> =
                                        (start_idx..start_idx + video_urls_iter.len()).collect();
                                    prefixed = prefixer.prefix_video(video_indices, &prefixed);
                                }

                                prefixed
                            } else {
                                text_content
                            }
                        } else {
                            text_content
                        };

                        let mut message_map: IndexMap<
                            String,
                            Either<String, Vec<IndexMap<String, Value>>>,
                        > = IndexMap::new();
                        message_map.insert("role".to_string(), Either::Left(message.role));

                        let mut content_map: Vec<IndexMap<String, Value>> = Vec::new();
                        for _ in &image_urls_iter {
                            let mut content_image_map = IndexMap::new();
                            content_image_map
                                .insert("type".to_string(), Value::String("image".to_string()));
                            content_map.push(content_image_map);
                        }
                        for _ in &audio_urls_iter {
                            let mut content_audio_map = IndexMap::new();
                            content_audio_map
                                .insert("type".to_string(), Value::String("audio".to_string()));
                            content_map.push(content_audio_map);
                        }
                        for _ in &video_urls_iter {
                            let mut content_video_map = IndexMap::new();
                            content_video_map
                                .insert("type".to_string(), Value::String("video".to_string()));
                            content_map.push(content_video_map);
                        }
                        {
                            let mut content_text_map = IndexMap::new();
                            content_text_map
                                .insert("type".to_string(), Value::String("text".to_string()));
                            content_text_map
                                .insert("text".to_string(), Value::String(text_content));
                            content_map.push(content_text_map);
                        }

                        message_map.insert("content".to_string(), Either::Right(content_map));
                        messages.push(message_map);
                        image_urls.extend(image_urls_iter);
                        audio_urls.extend(audio_urls_iter);
                        video_urls.extend(video_urls_iter);
                    }
                }
            }
            if !image_urls.is_empty() || !audio_urls.is_empty() || !video_urls.is_empty() {
                // Parse images
                let mut images = Vec::new();
                for url_unparsed in image_urls {
                    let image = parse_image_url_for_server(&url_unparsed)
                        .await
                        .context(format!("Failed to parse image resource: {url_unparsed}"))?;
                    images.push(image);
                }

                // Parse audios
                let mut audios = Vec::new();
                for url_unparsed in audio_urls {
                    let audio = parse_audio_url_for_server(&url_unparsed)
                        .await
                        .context(format!("Failed to parse audio resource: {url_unparsed}"))?;
                    audios.push(audio);
                }

                // Parse videos
                let mut videos = Vec::new();
                for url_unparsed in video_urls {
                    let video = parse_video_url_for_server(&url_unparsed, None)
                        .await
                        .context(format!("Failed to parse video resource: {url_unparsed}"))?;
                    videos.push(video);
                }

                RequestMessage::MultimodalChat {
                    messages,
                    images,
                    audios,
                    videos,
                    enable_thinking: oairequest.enable_thinking,
                    reasoning_effort,
                }
            } else {
                RequestMessage::Chat {
                    messages,
                    enable_thinking: oairequest.enable_thinking,
                    reasoning_effort,
                }
            }
        }
        Either::Right(prompt) => {
            let mut messages = Vec::new();
            let mut message_map: IndexMap<String, Either<String, Vec<IndexMap<String, Value>>>> =
                IndexMap::new();
            message_map.insert("role".to_string(), Either::Left("user".to_string()));
            message_map.insert("content".to_string(), Either::Left(prompt));
            messages.push(message_map);
            RequestMessage::Chat {
                messages,
                enable_thinking: oairequest.enable_thinking,
                reasoning_effort,
            }
        }
    };

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    if oairequest.max_tokens == Some(0) {
        anyhow::bail!("max_tokens must be at least 1.");
    }

    let is_streaming = oairequest.stream.unwrap_or(false);

    if oairequest.grammar.is_some() && oairequest.response_format.is_some() {
        anyhow::bail!("Request `grammar` and `response_format` were both provided but are mutually exclusive.")
    }

    let constraint = match oairequest.grammar {
        Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
        Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
        Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
        Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
        None => match oairequest.response_format {
            Some(ResponseFormat::JsonSchema {
                json_schema: JsonSchemaResponseFormat { name: _, schema },
            }) => Constraint::JsonSchema(schema),
            Some(ResponseFormat::JsonObject) => Constraint::JsonSchema(json!({"type": "object"})),
            Some(ResponseFormat::Text) => Constraint::None,
            None => Constraint::None,
        },
    };

    Ok((
        Request::Normal(Box::new(NormalRequest {
            id: state.next_request_id(),
            messages,
            sampling_params: SamplingParams {
                temperature: oairequest.temperature,
                top_k: oairequest.top_k,
                top_p: oairequest.top_p,
                min_p: oairequest.min_p,
                top_n_logprobs: oairequest.top_logprobs.unwrap_or(1),
                frequency_penalty: oairequest.frequency_penalty,
                presence_penalty: oairequest.presence_penalty,
                repetition_penalty: oairequest.repetition_penalty,
                max_len: oairequest.max_tokens,
                stop_toks,
                logits_bias: oairequest.logit_bias,
                n_choices: oairequest.n_choices,
                dry_params,
            },
            response: tx,
            return_logprobs: oairequest.logprobs,
            is_streaming,
            suffix: None,
            constraint,
            tool_choice: oairequest.tool_choice,
            tools: normalized_tools.tools,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: normalized_tools.web_search_options,
            enable_code_execution: normalized_tools.enable_code_execution,
            enable_shell: normalized_tools.enable_shell,
            shell_options,
            code_execution_permission: oairequest.code_execution_permission,
            code_execution_approval_notifier: None,
            agent_permission: oairequest.agent_permission,
            agent_approval_handler,
            agent_approval_notifier,
            session_id: oairequest.session_id,
            files: oairequest.files,
            input_files,
            max_tool_rounds: oairequest.max_tool_rounds,
            tool_dispatch_url,
            model_id: if oairequest.model == "default" {
                None
            } else {
                Some(oairequest.model.clone())
            },
            truncate_sequence: oairequest.truncate_sequence.unwrap_or(false),
        })),
        is_streaming,
    ))
}

/// OpenAI-compatible chat completions endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/chat/completions",
    request_body = ChatCompletionRequest,
    responses((status = 200, description = "Chat completions"))
)]
pub async fn chatcompletions(
    State(state): ExtractedMistralRsState,
    Extension(agentic_defaults): Extension<AgenticDefaults>,
    Extension(skill_store): Extension<Arc<SkillStore>>,
    Json(mut oairequest): Json<ChatCompletionRequest>,
) -> ChatCompletionResponder {
    let (tx, mut rx) = create_response_channel(None);

    // Apply server-level default for max_tool_rounds (per-request value takes priority)
    oairequest.max_tool_rounds = oairequest
        .max_tool_rounds
        .or(agentic_defaults.max_tool_rounds);

    let request_permission = oairequest
        .agent_permission
        .or_else(|| oairequest.code_execution_permission.map(Into::into));
    oairequest.agent_permission = match (agentic_defaults.agent_permission, request_permission) {
        (Some(server_permission), Some(request_permission)) => {
            Some(server_permission.strictest(request_permission))
        }
        (Some(server_permission), None) => Some(server_permission),
        (None, permission) => permission,
    };
    oairequest.code_execution_permission = None;

    let is_streaming = oairequest.stream.unwrap_or(false);
    if matches!(oairequest.agent_permission, Some(AgentPermission::Ask)) && !is_streaming {
        return ChatCompletionResponder::ValidationError(Box::new(JsonError::new(
            "agent_permission \"ask\" requires stream=true over HTTP; approve or deny emitted requests with POST /v1/agent/approvals/{approval_id}.".to_string(),
        )));
    }

    let agent_approval_handler = matches!(oairequest.agent_permission, Some(AgentPermission::Ask))
        .then(|| AgentToolApprovalHandler::from_async(agentic_defaults.approval_broker.callback()));
    let agent_approval_notifier =
        if is_streaming && matches!(oairequest.agent_permission, Some(AgentPermission::Ask)) {
            Some(agentic_defaults.approval_broker.notifier(tx.clone()))
        } else {
            None
        };

    // Extract model_id for routing before parsing
    let model_id = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    // tool_dispatch_url is server-level only (not settable per-request via HTTP API) for security
    let (request, is_streaming) = match parse_request(
        oairequest,
        ChatCompletionParseContext {
            state: state.clone(),
            tx,
            tool_dispatch_url: agentic_defaults.tool_dispatch_url,
            agent_approval_handler,
            agent_approval_notifier,
            tool_surface: OpenAiToolSurface::ChatCompletions,
            skill_store: Some(skill_store),
        },
    )
    .await
    {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        ChatCompletionResponder::Sse(create_streamer(rx, state, None, None))
    } else {
        process_non_streaming_response(&mut rx, state).await
    }
}

/// Handle route / generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ChatCompletionResponder {
    handle_completion_error(state, e)
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<ChatCompletionOnChunkCallback>,
    on_done: Option<ChatCompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<ChatCompletionStreamer>> {
    let streamer = base_create_streamer(rx, state, on_chunk, on_done);
    let keep_alive_interval = get_keep_alive_interval();

    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// Process non-streaming chat completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> ChatCompletionResponder {
    let mut tool_call_records = Vec::new();
    let mut pending_args = std::collections::HashMap::new();
    let mut files: Vec<mistralrs_core::File> = Vec::new();

    loop {
        match rx.recv().await {
            Some(Response::AgenticToolCallProgress {
                round,
                tool_name,
                phase,
            }) => record_agentic_progress(
                &mut tool_call_records,
                &mut pending_args,
                round,
                &tool_name,
                &phase,
            ),
            Some(Response::AgenticToolApprovalRequired { .. }) => {
                return ChatCompletionResponder::ValidationError(Box::new(JsonError::new(
                    "code execution approval requires a streaming HTTP request.".to_string(),
                )));
            }
            Some(Response::BlockDenoisingProgress(_)) => continue,
            Some(Response::File(file)) => {
                if files.len() < MAX_FILES_PER_RESPONSE {
                    files.push(file);
                } else {
                    tracing::warn!(
                        "MAX_FILES_PER_RESPONSE ({MAX_FILES_PER_RESPONSE}) reached; remaining files are fetchable via /v1/files/{{id}}",
                    );
                }
            }
            Some(Response::Done(response)) => {
                if !files.is_empty() {
                    stamp_file_ids(&mut tool_call_records, &files);
                }
                let mut response = attach_agentic_tool_calls(response, tool_call_records);
                if !files.is_empty() {
                    response.files = Some(files);
                }
                return match_responses(state, Response::Done(response));
            }
            Some(Response::ModelError(msg, response)) => {
                if !files.is_empty() {
                    stamp_file_ids(&mut tool_call_records, &files);
                }
                let mut response = attach_agentic_tool_calls(response, tool_call_records);
                if !files.is_empty() {
                    response.files = Some(files);
                }
                return match_responses(state, Response::ModelError(msg, response));
            }
            Some(response) => return match_responses(state, response),
            None => {
                let error = anyhow::Error::msg("No response received from the model.");
                return handle_error(state, error.into());
            }
        }
    }
}

/// Matches and processes different types of model responses into appropriate chat completion responses.
pub fn match_responses(state: SharedMistralRsState, response: Response) -> ChatCompletionResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            ChatCompletionResponder::InternalError(e)
        }
        Response::ModelError(msg, response) => {
            MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            MistralRs::maybe_log_response(state, &response);
            ChatCompletionResponder::ModelError(msg, response)
        }
        Response::ValidationError(e) => ChatCompletionResponder::ValidationError(e),
        Response::Done(response) => {
            MistralRs::maybe_log_response(state, &response);
            ChatCompletionResponder::Json(response)
        }
        Response::Chunk(_) => unreachable!(),
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionModelError(_, _) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
        Response::AgenticToolCallProgress { .. } => unreachable!(),
        Response::BlockDenoisingProgress(_) => unreachable!(),
        Response::AgenticToolApprovalRequired { .. } => unreachable!(),
        Response::File(_) => unreachable!(),
    }
}
