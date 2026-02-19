//! ## Completions functionality and route handler.

use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use crate::{
    completion_core::{
        convert_stop_tokens, get_dry_sampling_params, handle_completion_error,
        BaseCompletionResponder,
    },
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request,
        BaseJsonModelError, ErrorToResponse, JsonError, ModelErrorMessage,
    },
    openai::{CompletionRequest, Grammar},
    streaming::{base_create_streamer, get_keep_alive_interval, BaseStreamer, DoneState},
    types::{ExtractedMistralRsState, OnChunkCallback, OnDoneCallback, SharedMistralRsState},
    util::{sanitize_error_message, validate_model_name},
};
use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self},
    response::{
        sse::{Event, KeepAlive, KeepAliveStream},
        IntoResponse, Sse,
    },
};
use mistralrs_core::{
    CompletionChunkResponse, CompletionResponse, Constraint, MistralRs, NormalRequest, Request,
    RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

/// A callback function that processes streaming response chunks before they are sent to the client.
///
/// This hook allows modification of each chunk in the streaming response, enabling features like
/// content filtering, transformation, or logging. The callback receives a chunk and must return
/// a (potentially modified) chunk.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::completions::CompletionOnChunkCallback;
///
/// let on_chunk: CompletionOnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type CompletionOnChunkCallback = OnChunkCallback<CompletionChunkResponse>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::completions::CompletionOnDoneCallback;
///
/// let on_done: CompletionOnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type CompletionOnDoneCallback = OnDoneCallback<CompletionChunkResponse>;

/// A streaming response handler.
///
/// It processes incoming response chunks from a model and converts them
/// into Server-Sent Events (SSE) format for real-time streaming to clients.
pub type CompletionStreamer =
    BaseStreamer<CompletionChunkResponse, CompletionOnChunkCallback, CompletionOnDoneCallback>;

impl futures::Stream for CompletionStreamer {
    type Item = Result<Event, axum::Error>;

    /// Polls the stream for the next Server-Sent Event.
    ///
    /// This method implements the core streaming logic:
    /// 1. Handles stream completion by sending `[DONE]` and executing callbacks
    /// 2. Processes incoming model responses and converts them to SSE events
    /// 3. Applies chunk modifications if a callback is provided
    /// 4. Stores chunks if completion callback is configured
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
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
                Response::CompletionModelError(msg, _) => {
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
                Response::CompletionChunk(mut response) => {
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
                Response::Chunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::ModelError(_, _) => unreachable!(),
                Response::Speech { .. } => unreachable!(),
                Response::Raw { .. } => unreachable!(),
                Response::Embeddings { .. } => unreachable!(),
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

/// Represents different types of completion responses.
pub type CompletionResponder =
    BaseCompletionResponder<CompletionResponse, KeepAliveStream<CompletionStreamer>>;

/// JSON error response structure for model errors.
type JsonModelError = BaseJsonModelError<CompletionResponse>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for CompletionResponder {
    /// Converts the completion responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            CompletionResponder::Sse(s) => s.into_response(),
            CompletionResponder::Json(s) => Json(s).into_response(),
            CompletionResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            CompletionResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            CompletionResponder::ModelError(msg, response) => JsonModelError::new(msg, response)
                .to_response(http::StatusCode::INTERNAL_SERVER_ERROR),
        }
    }
}

/// Parses and validates a completion request.
///
/// This function transforms an OpenAI-compatible completion request into the
/// request format used by mistral.rs.
pub fn parse_request(
    oairequest: CompletionRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    let stop_toks = convert_stop_tokens(oairequest.stop_seqs);

    let is_streaming = oairequest.stream.unwrap_or(false);

    let dry_params = get_dry_sampling_params(
        oairequest.dry_multiplier,
        oairequest.dry_sequence_breakers,
        oairequest.dry_base,
        oairequest.dry_allowed_length,
    )?;

    Ok((
        Request::Normal(Box::new(NormalRequest {
            id: state.next_request_id(),
            messages: RequestMessage::Completion {
                text: oairequest.prompt,
                echo_prompt: oairequest.echo_prompt,
                best_of: oairequest.best_of,
            },
            sampling_params: SamplingParams {
                temperature: oairequest.temperature,
                top_k: oairequest.top_k,
                top_p: oairequest.top_p,
                min_p: oairequest.min_p,
                top_n_logprobs: oairequest.logprobs.unwrap_or(1),
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
            return_logprobs: oairequest.logprobs.is_some(),
            is_streaming,
            suffix: oairequest.suffix,
            constraint: match oairequest.grammar {
                Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
                Some(Grammar::Lark(lark)) => Constraint::Lark(lark),
                Some(Grammar::JsonSchema(schema)) => Constraint::JsonSchema(schema),
                Some(Grammar::Llguidance(llguidance)) => Constraint::Llguidance(llguidance),
                None => Constraint::None,
            },
            tool_choice: oairequest.tool_choice,
            tools: oairequest.tools,
            logits_processors: None,
            return_raw_logits: false,
            web_search_options: None,
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

/// OpenAI-compatible completions endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses((status = 200, description = "Completions"))
)]
pub async fn completions(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<CompletionRequest>,
) -> CompletionResponder {
    let (tx, mut rx) = create_response_channel(None);

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    if is_streaming {
        CompletionResponder::Sse(create_streamer(rx, state, None, None))
    } else {
        process_non_streaming_response(&mut rx, state).await
    }
}

/// Handle route / generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> CompletionResponder {
    handle_completion_error(state, e)
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
) -> Sse<KeepAliveStream<CompletionStreamer>> {
    let streamer = base_create_streamer(rx, state, on_chunk, on_done);
    let keep_alive_interval = get_keep_alive_interval();

    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// Process non-streaming completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> CompletionResponder {
    base_process_non_streaming_response(rx, state, match_responses, handle_error).await
}

/// Matches and processes different types of model responses into appropriate completion responses.
pub fn match_responses(state: SharedMistralRsState, response: Response) -> CompletionResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            CompletionResponder::InternalError(e)
        }
        Response::CompletionModelError(msg, response) => {
            MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(msg.to_string()));
            MistralRs::maybe_log_response(state, &response);
            CompletionResponder::ModelError(msg, response)
        }
        Response::ValidationError(e) => CompletionResponder::ValidationError(e),
        Response::CompletionDone(response) => {
            MistralRs::maybe_log_response(state, &response);
            CompletionResponder::Json(response)
        }
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::ImageGeneration(_) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
    }
}
