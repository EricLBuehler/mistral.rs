//! ## Chat Completions functionality and route handler.

use std::{env, error::Error, ops::Deref, pin::Pin, task::Poll, time::Duration};

use anyhow::{Context, Result};
use axum::{
    extract::{Json, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use mistralrs_core::{
    ChatCompletionChunkResponse, ChatCompletionResponse, Constraint, DrySamplingParams, MistralRs,
    NormalRequest, Request, RequestMessage, Response, SamplingParams,
    StopTokens as InternalStopTokens,
};
use serde::Serialize;
use serde_json::Value;
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::{
    openai::{
        ChatCompletionRequest, Grammar, JsonSchemaResponseFormat, MessageInnerContent,
        ResponseFormat, StopTokens,
    },
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util,
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
/// use mistralrs_server_core::chat_completion::OnChunkCallback;
///
/// let on_chunk: OnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type OnChunkCallback =
    Box<dyn Fn(ChatCompletionChunkResponse) -> ChatCompletionChunkResponse + Send + Sync>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::chat_completion::OnDoneCallback;
///
/// let on_done: OnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type OnDoneCallback = Box<dyn Fn(&[ChatCompletionChunkResponse]) + Send + Sync>;

/// Default buffer size for the response channel used in streaming operations.
///
/// This constant defines the maximum number of response messages that can be buffered
/// in the channel before backpressure is applied. A larger buffer reduces the likelihood
/// of blocking but uses more memory.
pub const DEFAULT_CHANNEL_BUFFER_SIZE: usize = 10_000;

/// Default keep-alive interval for Server-Sent Events (SSE) streams in milliseconds.
pub const DEFAULT_KEEP_ALIVE_INTERVAL_MS: u64 = 10_000;

/// Internal error type for model-related errors with a descriptive message.
///
/// This struct wraps error messages from the underlying model and implements
/// the standard error traits for proper error handling and display.
#[derive(Debug)]
struct ModelErrorMessage(String);
impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ModelErrorMessage {}

/// Represents the current state of a streaming response.
enum DoneState {
    /// The stream is actively processing and sending response chunks
    Running,
    /// The stream has finished processing and is about to send the `[DONE]` message
    SendingDone,
    /// The stream has completed entirely
    Done,
}

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub struct Streamer {
    /// Channel receiver for incoming model responses
    rx: Receiver<Response>,
    /// Current state of the streaming operation
    done_state: DoneState,
    /// Underlying mistral.rs instance
    state: SharedMistralRsState,
    /// Whether to store chunks for the completion callback
    store_chunks: bool,
    /// All chunks received during streaming (if `store_chunks` is true)
    chunks: Vec<ChatCompletionChunkResponse>,
    /// Optional callback to process each chunk before sending
    on_chunk: Option<OnChunkCallback>,
    /// Optional callback to execute when streaming completes
    on_done: Option<OnDoneCallback>,
}

impl futures::Stream for Streamer {
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
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
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
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

/// Represents different types of chat completion responses.
pub enum ChatCompletionResponder {
    /// Server-Sent Events streaming response
    Sse(Sse<Streamer>),
    /// Complete JSON response for non-streaming requests
    Json(ChatCompletionResponse),
    /// Model error with partial response data
    ModelError(String, ChatCompletionResponse),
    /// Internal server error
    InternalError(Box<dyn Error>),
    /// Request validation error
    ValidationError(Box<dyn Error>),
}

/// Trait for converting errors to HTTP responses with appropriate status codes.
trait ErrorToResponse: Serialize {
    /// Converts the error to an HTTP response with the specified status code.
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

/// Standard JSON error response structure.
#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    /// Creates a new JSON error with the specified message.
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

/// JSON error response structure for model errors.
#[derive(Serialize)]
struct JsonModelError {
    message: String,
    /// Partial response data that was generated before the error occurred
    partial_response: ChatCompletionResponse,
}

impl JsonModelError {
    /// Creates a new JSON model error with message and partial response.
    fn new(message: String, partial_response: ChatCompletionResponse) -> Self {
        Self {
            message,
            partial_response,
        }
    }
}

impl ErrorToResponse for JsonModelError {}

impl IntoResponse for ChatCompletionResponder {
    /// Converts the chat completion responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            ChatCompletionResponder::Sse(s) => s.into_response(),
            ChatCompletionResponder::Json(s) => Json(s).into_response(),
            ChatCompletionResponder::InternalError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ChatCompletionResponder::ValidationError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            ChatCompletionResponder::ModelError(msg, response) => {
                JsonModelError::new(msg, response)
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
        }
    }
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

    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        None => None,
    };
    let messages = match oairequest.messages {
        Either::Left(req_messages) => {
            let mut messages = Vec::new();
            let mut image_urls = Vec::new();
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
                        message_map.insert("role".to_string(), Either::Left(message.role));
                        message_map.insert("content".to_string(), Either::Left(content.clone()));
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
                                Either::Right(right) => format!("{:?}", right),
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
                                        image_url: image_message.get("image_url").as_ref()
                                            .context("Image sub-content must have `image_url` key.")?.as_ref()
                                            .right()
                                            .context("Image sub-content `image_url` key must be an object.")?
                                            .get("url")
                                            .context("Image sub-content `image_url` object must have a `url` key.")?.clone()
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
                    }
                }
            }
            if !image_urls.is_empty() {
                let mut images = Vec::new();
                for url_unparsed in image_urls {
                    let image = util::parse_image_url(&url_unparsed)
                        .await
                        .context(format!("Failed to parse image resource: {}", url_unparsed))?;

                    images.push(image);
                }
                RequestMessage::VisionChat {
                    messages,
                    images,
                    enable_thinking: oairequest.enable_thinking,
                }
            } else {
                RequestMessage::Chat {
                    messages,
                    enable_thinking: oairequest.enable_thinking,
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
            }
        }
    };

    let dry_params = if let Some(dry_multiplier) = oairequest.dry_multiplier {
        Some(DrySamplingParams::new_with_defaults(
            dry_multiplier,
            oairequest.dry_sequence_breakers,
            oairequest.dry_base,
            oairequest.dry_allowed_length,
        )?)
    } else {
        None
    };

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

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx).await {
        Ok(x) => x,
        Err(e) => return handle_chat_completion_error(state, e.into()),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_chat_completion_error(state, e.into());
    }

    if is_streaming {
        ChatCompletionResponder::Sse(create_chat_streamer(rx, state, None, None))
    } else {
        process_non_streaming_chat_response(&mut rx, state).await
    }
}

/// Helper function to handle chat completion errors and logging them.
pub fn handle_chat_completion_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ChatCompletionResponder {
    let e = anyhow::Error::msg(e.to_string());
    MistralRs::maybe_log_error(state, &*e);
    ChatCompletionResponder::InternalError(e.into())
}

/// Creates a channel for response communication.
pub fn create_response_channel(
    buffer_size: Option<usize>,
) -> (Sender<Response>, Receiver<Response>) {
    let channel_buffer_size = buffer_size.unwrap_or(DEFAULT_CHANNEL_BUFFER_SIZE);

    channel(channel_buffer_size)
}

/// Gets the keep-alive interval for SSE streams from environment or default.
pub fn get_keep_alive_interval() -> u64 {
    env::var("KEEP_ALIVE_INTERVAL")
        .map(|val| {
            val.parse::<u64>().unwrap_or_else(|e| {
                tracing::warn!("Failed to parse KEEP_ALIVE_INTERVAL: {}. Using default.", e);
                DEFAULT_KEEP_ALIVE_INTERVAL_MS
            })
        })
        .unwrap_or(DEFAULT_KEEP_ALIVE_INTERVAL_MS)
}

/// Sends a request to the model processing pipeline.
pub async fn send_request(state: &SharedMistralRsState, request: Request) -> Result<()> {
    let sender = state.get_sender().unwrap();
    sender.send(request).await.map_err(|e| e.into())
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_chat_streamer(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<OnChunkCallback>,
    on_done: Option<OnDoneCallback>,
) -> Sse<Streamer> {
    let store_chunks = on_done.is_some();

    let streamer = Streamer {
        rx,
        done_state: DoneState::Running,
        store_chunks,
        state,
        chunks: Vec::new(),
        on_chunk,
        on_done,
    };

    let keep_alive_interval = get_keep_alive_interval();

    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// Processes non-streaming chat completion responses.
pub async fn process_non_streaming_chat_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> ChatCompletionResponder {
    let response = match rx.recv().await {
        Some(response) => response,
        None => {
            let e = anyhow::Error::msg("No response received from the model.");
            return handle_chat_completion_error(state, e.into());
        }
    };

    match_responses(state, response)
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
    }
}
