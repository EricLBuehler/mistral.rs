//! ## Chat Completions functionality and route handler.

use std::{ops::Deref, pin::Pin, task::Poll, time::Duration};

use anyhow::{Context, Result};
use axum::{
    extract::{Json, State},
    http::{self},
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
};
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Constraint, MistralRs, ModelCategory,
    NormalRequest, ReasoningEffort, Request, RequestMessage, Response, SamplingParams,
};
use serde_json::Value;
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    completion_core::{
        convert_stop_tokens, get_dry_sampling_params, handle_completion_error,
        BaseCompletionResponder,
    },
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request_with_model,
        BaseJsonModelError, ErrorToResponse, JsonError, ModelErrorMessage,
    },
    openai::{
        ChatCompletionRequest, Grammar, JsonSchemaResponseFormat, MessageInnerContent,
        ResponseFormat,
    },
    streaming::{base_create_streamer, get_keep_alive_interval, BaseStreamer, DoneState},
    types::{ExtractedMistralRsState, OnChunkCallback, OnDoneCallback, SharedMistralRsState},
    util::{parse_audio_url, parse_image_url, sanitize_error_message, validate_model_name},
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
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::CompletionModelError(_, _) => unreachable!(),
                Response::CompletionChunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
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

/// Parses and validates a chat completion request.
///
/// This function transforms an OpenAI-compatible chat completion request into the
/// request format used by mistral.rs.
pub async fn parse_request(
    oairequest: ChatCompletionRequest,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    // Parse reasoning effort for Harmony-format models
    let reasoning_effort = parse_reasoning_effort(&oairequest.reasoning_effort);

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs);

    let messages = match oairequest.messages {
        Either::Left(req_messages) => {
            let mut messages = Vec::new();
            let mut image_urls = Vec::new();
            let mut audio_urls = Vec::new();
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
                        if image_messages.len() == 1 {
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
                                _ => anyhow::bail!("Expected array content sub-content to be of format {{`type`: `text`, `text`: ...}} and {{`type`: `url`, `image_url`: {{`url`: ...}}}}")
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

                        // Apply prefixer to text content if this is a vision model with images/audio
                        // This matches the behavior of interactive mode which auto-inserts media tokens
                        let text_content = if !image_urls_iter.is_empty()
                            || !audio_urls_iter.is_empty()
                        {
                            if let Ok(ModelCategory::Vision { prefixer }) =
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
                    }
                }
            }
            if !image_urls.is_empty() || !audio_urls.is_empty() {
                // Parse images
                let mut images = Vec::new();
                for url_unparsed in image_urls {
                    let image = parse_image_url(&url_unparsed)
                        .await
                        .context(format!("Failed to parse image resource: {url_unparsed}"))?;
                    images.push(image);
                }

                // Parse audios
                let mut audios = Vec::new();
                for url_unparsed in audio_urls {
                    let audio = parse_audio_url(&url_unparsed)
                        .await
                        .context(format!("Failed to parse audio resource: {url_unparsed}"))?;
                    audios.push(audio);
                }

                RequestMessage::VisionChat {
                    messages,
                    images,
                    audios,
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
            tools: oairequest.tools,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: oairequest.web_search_options,
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
    Json(oairequest): Json<ChatCompletionRequest>,
) -> ChatCompletionResponder {
    let (tx, mut rx) = create_response_channel(None);

    // Extract model_id for routing before parsing
    let model_id = if oairequest.model == "default" {
        None
    } else {
        Some(oairequest.model.clone())
    };

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx).await {
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
    base_process_non_streaming_response(rx, state, match_responses, handle_error).await
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
    }
}
