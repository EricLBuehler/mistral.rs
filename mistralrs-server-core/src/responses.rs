//! ## Responses API functionality and route handlers.

use std::{pin::Pin, task::Poll, time::Duration};

use anyhow::Result;
use axum::{
    extract::{Json, Path, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use either::Either;
use mistralrs_core::{ChatCompletionResponse, MistralRs, Request, Response};
use serde_json::Value;
use tokio::sync::mpsc::Sender;
use uuid::Uuid;

use crate::{
    cached_responses::get_response_cache,
    chat_completion::parse_request as parse_chat_request,
    completion_core::{handle_completion_error, BaseCompletionResponder},
    handler_core::{
        create_response_channel, send_request_with_model, BaseJsonModelError, ErrorToResponse,
        JsonError, ModelErrorMessage,
    },
    openai::{
        ChatCompletionRequest, Message, MessageContent, ResponsesChunk, ResponsesContent,
        ResponsesCreateRequest, ResponsesDelta, ResponsesDeltaContent, ResponsesDeltaOutput,
        ResponsesError, ResponsesObject, ResponsesOutput, ResponsesUsage,
    },
    streaming::{get_keep_alive_interval, BaseStreamer, DoneState},
    types::{ExtractedMistralRsState, OnChunkCallback, OnDoneCallback, SharedMistralRsState},
    util::sanitize_error_message,
};

/// Response streamer for the Responses API
pub type ResponsesStreamer =
    BaseStreamer<ResponsesChunk, OnChunkCallback<ResponsesChunk>, OnDoneCallback<ResponsesChunk>>;

impl futures::Stream for ResponsesStreamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        match self.done_state {
            DoneState::SendingDone => {
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
                    self.done_state = DoneState::SendingDone;
                    Poll::Ready(Some(Ok(Event::default().data(msg))))
                }
                Response::ValidationError(e) => Poll::Ready(Some(Ok(
                    Event::default().data(sanitize_error_message(e.as_ref()))
                ))),
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    Poll::Ready(Some(Ok(
                        Event::default().data(sanitize_error_message(e.as_ref()))
                    )))
                }
                Response::Chunk(chat_chunk) => {
                    // Convert ChatCompletionChunkResponse to ResponsesChunk
                    let mut delta_outputs = vec![];

                    // Check if all choices are finished
                    let all_finished = chat_chunk.choices.iter().all(|c| c.finish_reason.is_some());

                    for choice in &chat_chunk.choices {
                        let mut delta_content_items = Vec::new();

                        // Handle text content in delta
                        if let Some(content) = &choice.delta.content {
                            delta_content_items.push(ResponsesDeltaContent {
                                content_type: "output_text".to_string(),
                                text: Some(content.clone()),
                            });
                        }

                        // Handle tool calls in delta
                        if let Some(tool_calls) = &choice.delta.tool_calls {
                            for tool_call in tool_calls {
                                let tool_text = format!(
                                    "Tool: {} args: {}",
                                    tool_call.function.name, tool_call.function.arguments
                                );
                                delta_content_items.push(ResponsesDeltaContent {
                                    content_type: "tool_use".to_string(),
                                    text: Some(tool_text),
                                });
                            }
                        }

                        if !delta_content_items.is_empty() {
                            delta_outputs.push(ResponsesDeltaOutput {
                                id: format!("msg_{}", Uuid::new_v4()),
                                output_type: "message".to_string(),
                                content: Some(delta_content_items),
                            });
                        }
                    }

                    let mut response_chunk = ResponsesChunk {
                        id: chat_chunk.id.clone(),
                        object: "response.chunk",
                        created_at: chat_chunk.created as f64,
                        model: chat_chunk.model.clone(),
                        chunk_type: "delta".to_string(),
                        delta: Some(ResponsesDelta {
                            output: if delta_outputs.is_empty() {
                                None
                            } else {
                                Some(delta_outputs)
                            },
                            status: if all_finished {
                                Some("completed".to_string())
                            } else {
                                None
                            },
                        }),
                        usage: None,
                        metadata: None,
                    };

                    if all_finished {
                        self.done_state = DoneState::SendingDone;
                    }

                    MistralRs::maybe_log_response(self.state.clone(), &chat_chunk);

                    if let Some(on_chunk) = &self.on_chunk {
                        response_chunk = on_chunk(response_chunk);
                    }

                    if self.store_chunks {
                        self.chunks.push(response_chunk.clone());
                    }

                    Poll::Ready(Some(Event::default().json_data(response_chunk)))
                }
                _ => unreachable!(),
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

/// Response responder types
pub type ResponsesResponder = BaseCompletionResponder<ResponsesObject, ResponsesStreamer>;

type JsonModelError = BaseJsonModelError<ResponsesObject>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for ResponsesResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            ResponsesResponder::Sse(s) => s.into_response(),
            ResponsesResponder::Json(s) => Json(s).into_response(),
            ResponsesResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ResponsesResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ResponsesResponder::ModelError(msg, response) => JsonModelError::new(msg, response)
                .to_response(http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}

/// Convert chat completion response to responses object
fn chat_response_to_responses_object(
    chat_resp: &ChatCompletionResponse,
    request_id: String,
    metadata: Option<Value>,
) -> ResponsesObject {
    let mut outputs = Vec::new();
    let mut output_text_parts = Vec::new();

    for choice in &chat_resp.choices {
        let mut content_items = Vec::new();
        let mut has_content = false;

        // Handle text content
        if let Some(text) = &choice.message.content {
            output_text_parts.push(text.clone());
            content_items.push(ResponsesContent {
                content_type: "output_text".to_string(),
                text: Some(text.clone()),
                annotations: None,
            });
            has_content = true;
        }

        // Handle tool calls
        if let Some(tool_calls) = &choice.message.tool_calls {
            for tool_call in tool_calls {
                let tool_text = format!(
                    "Tool call: {} with args: {}",
                    tool_call.function.name, tool_call.function.arguments
                );
                content_items.push(ResponsesContent {
                    content_type: "tool_use".to_string(),
                    text: Some(tool_text),
                    annotations: None,
                });
                has_content = true;
            }
        }

        // Only add output if we have content
        if has_content {
            outputs.push(ResponsesOutput {
                id: format!("msg_{}", Uuid::new_v4()),
                output_type: "message".to_string(),
                role: choice.message.role.clone(),
                status: None,
                content: content_items,
            });
        }
    }

    ResponsesObject {
        id: request_id,
        object: "response",
        created_at: chat_resp.created as f64,
        model: chat_resp.model.clone(),
        status: "completed".to_string(),
        output: outputs,
        output_text: if output_text_parts.is_empty() {
            None
        } else {
            Some(output_text_parts.join(" "))
        },
        usage: Some(ResponsesUsage {
            input_tokens: chat_resp.usage.prompt_tokens,
            output_tokens: chat_resp.usage.completion_tokens,
            total_tokens: chat_resp.usage.total_tokens,
            input_tokens_details: None,
            output_tokens_details: None,
        }),
        error: None,
        metadata,
        instructions: None,
        incomplete_details: None,
    }
}

/// Parse responses request into internal format
async fn parse_responses_request(
    oairequest: ResponsesCreateRequest,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<(Request, bool, Option<Vec<Message>>)> {
    if oairequest.instructions.is_some() {
        return Err(anyhow::anyhow!(
            "The 'instructions' field is not supported in the Responses API"
        ));
    }
    // If previous_response_id is provided, get the full conversation history from cache
    let previous_messages = if let Some(prev_id) = &oairequest.previous_response_id {
        let cache = get_response_cache();
        cache.get_conversation_history(prev_id)?
    } else {
        None
    };

    // Get messages from either messages or input field
    let messages = oairequest.input.into_either();

    // Convert to ChatCompletionRequest for reuse
    let mut chat_request = ChatCompletionRequest {
        messages: messages.clone(),
        model: oairequest.model,
        logit_bias: oairequest.logit_bias,
        logprobs: oairequest.logprobs,
        top_logprobs: oairequest.top_logprobs,
        max_tokens: oairequest.max_tokens,
        n_choices: oairequest.n_choices,
        presence_penalty: oairequest.presence_penalty,
        frequency_penalty: oairequest.frequency_penalty,
        repetition_penalty: oairequest.repetition_penalty,
        stop_seqs: oairequest.stop_seqs,
        temperature: oairequest.temperature,
        top_p: oairequest.top_p,
        stream: oairequest.stream,
        tools: oairequest.tools,
        tool_choice: oairequest.tool_choice,
        response_format: oairequest.response_format,
        web_search_options: oairequest.web_search_options,
        top_k: oairequest.top_k,
        grammar: oairequest.grammar,
        min_p: oairequest.min_p,
        dry_multiplier: oairequest.dry_multiplier,
        dry_base: oairequest.dry_base,
        dry_allowed_length: oairequest.dry_allowed_length,
        dry_sequence_breakers: oairequest.dry_sequence_breakers,
        enable_thinking: oairequest.enable_thinking,
    };

    // Prepend previous messages if available
    if let Some(prev_msgs) = previous_messages {
        match &mut chat_request.messages {
            Either::Left(msgs) => {
                let mut combined = prev_msgs;
                combined.extend(msgs.clone());
                chat_request.messages = Either::Left(combined);
            }
            Either::Right(_) => {
                // If it's a prompt string, convert to messages and prepend
                let prompt = chat_request.messages.as_ref().right().unwrap().clone();
                let mut combined = prev_msgs;
                combined.push(Message {
                    content: Some(MessageContent::from_text(prompt)),
                    role: "user".to_string(),
                    name: None,
                    tool_calls: None,
                });
                chat_request.messages = Either::Left(combined);
            }
        }
    }

    // Get all messages for prompt_details
    let all_messages = match &chat_request.messages {
        Either::Left(msgs) => msgs.clone(),
        Either::Right(prompt) => vec![Message {
            content: Some(MessageContent::from_text(prompt.clone())),
            role: "user".to_string(),
            name: None,
            tool_calls: None,
        }],
    };

    let (request, is_streaming) = parse_chat_request(chat_request, state, tx).await?;
    Ok((request, is_streaming, Some(all_messages)))
}

/// Create response endpoint
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/responses",
    request_body = ResponsesCreateRequest,
    responses((status = 200, description = "Response created"))
)]
pub async fn create_response(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<ResponsesCreateRequest>,
) -> ResponsesResponder {
    let (tx, mut rx) = create_response_channel(None);
    let request_id = format!("resp_{}", Uuid::new_v4());
    let metadata = oairequest.metadata.clone();
    let store = oairequest.store.unwrap_or(true);

    // Extract model_id for routing
    let model_id = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let (request, is_streaming, conversation_history) =
        match parse_responses_request(oairequest, state.clone(), tx).await {
            Ok(x) => x,
            Err(e) => return handle_error(state, e.into()),
        };

    if let Err(e) = send_request_with_model(&state, request, model_id.as_deref()).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        let streamer = ResponsesStreamer {
            rx,
            done_state: DoneState::Running,
            state: state.clone(),
            on_chunk: None,
            on_done: None,
            chunks: Vec::new(),
            store_chunks: store,
        };

        // Store chunks for later retrieval if requested
        if store {
            let cache = get_response_cache();
            let id = request_id.clone();
            let chunks_cache = cache.clone();

            // Create a wrapper that stores chunks and conversation history
            let history_for_streaming = conversation_history.clone();
            let on_done: OnDoneCallback<ResponsesChunk> = Box::new(move |chunks| {
                let _ = chunks_cache.store_chunks(id.clone(), chunks.to_vec());

                // Reconstruct the assistant's message from chunks and store conversation history
                if let Some(history) = history_for_streaming.clone() {
                    let mut history = history;
                    let mut assistant_message = String::new();

                    // Collect all text from chunks
                    for chunk in chunks {
                        if let Some(delta) = &chunk.delta {
                            if let Some(outputs) = &delta.output {
                                for output in outputs {
                                    if let Some(contents) = &output.content {
                                        for content in contents {
                                            if let Some(text) = &content.text {
                                                assistant_message.push_str(text);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    // Add the complete assistant message to history
                    if !assistant_message.is_empty() {
                        history.push(Message {
                            content: Some(MessageContent::from_text(assistant_message)),
                            role: "assistant".to_string(),
                            name: None,
                            tool_calls: None,
                        });
                    }

                    let _ = chunks_cache.store_conversation_history(id.clone(), history);
                }
            });

            ResponsesResponder::Sse(create_streamer(streamer, Some(on_done)))
        } else {
            ResponsesResponder::Sse(create_streamer(streamer, None))
        }
    } else {
        // Non-streaming response
        match rx.recv().await {
            Some(Response::Done(chat_resp)) => {
                let response_obj =
                    chat_response_to_responses_object(&chat_resp, request_id.clone(), metadata);

                // Store if requested
                if store {
                    let cache = get_response_cache();
                    let _ = cache.store_response(request_id.clone(), response_obj.clone());

                    // Create complete conversation history including the assistant's response
                    if let Some(mut history) = conversation_history.clone() {
                        // Add the assistant's response to the conversation history
                        for choice in &chat_resp.choices {
                            if let Some(content) = &choice.message.content {
                                history.push(Message {
                                    content: Some(MessageContent::from_text(content.clone())),
                                    role: choice.message.role.clone(),
                                    name: None,
                                    tool_calls: None, // TODO: Convert ToolCallResponse to ToolCall if needed
                                });
                            }
                        }
                        let _ = cache.store_conversation_history(request_id, history);
                    }
                }

                ResponsesResponder::Json(response_obj)
            }
            Some(Response::ModelError(msg, partial_resp)) => {
                let mut response_obj =
                    chat_response_to_responses_object(&partial_resp, request_id.clone(), metadata);
                response_obj.error = Some(ResponsesError {
                    error_type: "model_error".to_string(),
                    message: msg.to_string(),
                });
                response_obj.status = "failed".to_string();

                if store {
                    let cache = get_response_cache();
                    let _ = cache.store_response(request_id.clone(), response_obj.clone());

                    // Even on error, store conversation history with partial response
                    if let Some(mut history) = conversation_history.clone() {
                        // Add any partial response to the conversation history
                        for choice in &partial_resp.choices {
                            if let Some(content) = &choice.message.content {
                                history.push(Message {
                                    content: Some(MessageContent::from_text(content.clone())),
                                    role: choice.message.role.clone(),
                                    name: None,
                                    tool_calls: None, // TODO: Convert ToolCallResponse to ToolCall if needed
                                });
                            }
                        }
                        let _ = cache.store_conversation_history(request_id, history);
                    }
                }
                ResponsesResponder::ModelError(msg.to_string(), response_obj)
            }
            Some(Response::ValidationError(e)) => ResponsesResponder::ValidationError(e),
            Some(Response::InternalError(e)) => ResponsesResponder::InternalError(e),
            _ => ResponsesResponder::InternalError(
                anyhow::anyhow!("Unexpected response type").into(),
            ),
        }
    }
}

/// Get response by ID endpoint
#[utoipa::path(
    get,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to retrieve")),
    responses((status = 200, description = "Response object"))
)]
pub async fn get_response(
    State(_state): ExtractedMistralRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    let cache = get_response_cache();

    match cache.get_response(&response_id) {
        Ok(Some(response)) => (StatusCode::OK, Json(response)).into_response(),
        Ok(None) => JsonError::new(format!("Response with ID '{response_id}' not found"))
            .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error retrieving response: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Delete response by ID endpoint
#[utoipa::path(
    delete,
    tag = "Mistral.rs",
    path = "/v1/responses/{response_id}",
    params(("response_id" = String, Path, description = "The ID of the response to delete")),
    responses((status = 200, description = "Response deleted"))
)]
pub async fn delete_response(
    State(_state): ExtractedMistralRsState,
    Path(response_id): Path<String>,
) -> impl IntoResponse {
    let cache = get_response_cache();

    match cache.delete_response(&response_id) {
        Ok(true) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "deleted": true,
                "id": response_id,
                "object": "response.deleted"
            })),
        )
            .into_response(),
        Ok(false) => JsonError::new(format!("Response with ID '{response_id}' not found"))
            .to_response(StatusCode::NOT_FOUND),
        Err(e) => JsonError::new(format!(
            "Error deleting response: {}",
            sanitize_error_message(&*e)
        ))
        .to_response(StatusCode::INTERNAL_SERVER_ERROR),
    }
}

/// Handle errors
fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ResponsesResponder {
    handle_completion_error(state, e)
}

/// Create SSE streamer
fn create_streamer(
    streamer: ResponsesStreamer,
    on_done: Option<OnDoneCallback<ResponsesChunk>>,
) -> Sse<ResponsesStreamer> {
    let keep_alive_interval = get_keep_alive_interval();

    let streamer_with_callback = ResponsesStreamer {
        on_done,
        ..streamer
    };

    Sse::new(streamer_with_callback)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}
