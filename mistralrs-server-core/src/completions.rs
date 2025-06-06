use std::{
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use mistralrs_core::{
    CompletionChunkResponse, CompletionResponse, Constraint, DrySamplingParams, MistralRs,
    NormalRequest, Request, RequestMessage, Response, SamplingParams,
    StopTokens as InternalStopTokens,
};
use tokio::sync::mpsc::{Receiver, Sender};
use tracing::warn;

use crate::{
    completion_base::{
        base_handle_completion_error, base_process_non_streaming_response, create_response_channel,
        send_model_request, BaseCompletionResponder, BaseJsonModelError, ErrorToResponse,
        JsonError, ModelErrorMessage,
    },
    openai::{CompletionRequest, Grammar, StopTokens},
    streaming::{base_create_streamer, get_keep_alive_interval, BaseStreamer, DoneState},
    types::{ExtractedMistralRsState, SharedMistralRsState},
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
/// use mistralrs_server_core::completion::OnChunkCallback;
///
/// let on_chunk: OnChunkCallback = Box::new(|mut chunk| {
///     // Log the chunk or modify its content
///     println!("Processing chunk: {:?}", chunk);
///     chunk
/// });
/// ```
pub type CompletionOnChunkCallback =
    Box<dyn Fn(CompletionChunkResponse) -> CompletionChunkResponse + Send + Sync>;

/// A callback function that is executed when the streaming response completes.
///
/// This hook receives all chunks that were streamed during the response, allowing for
/// post-processing, analytics, or cleanup operations after the stream finishes.
///
/// ### Examples
///
/// ```no_run
/// use mistralrs_server_core::completion::OnDoneCallback;
///
/// let on_done: OnDoneCallback = Box::new(|chunks| {
///     println!("Stream completed with {} chunks", chunks.len());
///     // Process all chunks for analytics
/// });
/// ```
pub type CompletionOnDoneCallback = Box<dyn Fn(&[CompletionChunkResponse]) + Send + Sync>;

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
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
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
            },
            Poll::Pending | Poll::Ready(None) => Poll::Pending,
        }
    }
}

pub type CompletionResponder = BaseCompletionResponder<CompletionResponse, CompletionStreamer>;

/// JSON error response structure for model errors.
type JsonModelError = BaseJsonModelError<CompletionResponse>;
impl ErrorToResponse for JsonModelError {}

impl IntoResponse for CompletionResponder {
    fn into_response(self) -> axum::response::Response {
        match self {
            CompletionResponder::Sse(s) => s.into_response(),
            CompletionResponder::Json(s) => Json(s).into_response(),
            CompletionResponder::InternalError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            CompletionResponder::ValidationError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
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
fn parse_request(
    oairequest: CompletionRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, bool)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    let stop_toks = match oairequest.stop_seqs {
        Some(StopTokens::Multi(m)) => Some(InternalStopTokens::Seqs(m)),
        Some(StopTokens::Single(s)) => Some(InternalStopTokens::Seqs(vec![s])),
        None => None,
    };

    if oairequest.logprobs.is_some() {
        warn!("Completion requests do not support logprobs.");
    }

    let is_streaming = oairequest.stream.unwrap_or(false);

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
                top_n_logprobs: 1,
                frequency_penalty: oairequest.frequency_penalty,
                presence_penalty: oairequest.presence_penalty,
                max_len: oairequest.max_tokens,
                stop_toks,
                logits_bias: oairequest.logit_bias,
                n_choices: oairequest.n_choices,
                dry_params,
            },
            response: tx,
            return_logprobs: false,
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
        })),
        is_streaming,
    ))
}

/// OpenAI-compatible chat completions endpoint handler.
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

    if oairequest.logprobs.is_some() {
        return CompletionResponder::ValidationError(
            "Completion requests do not support logprobs.".into(),
        );
    }

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_completion_error(state, e.into()),
    };

    if let Err(e) = send_model_request(&state, request).await {
        return handle_completion_error(state, e.into());
    }

    if is_streaming {
        CompletionResponder::Sse(create_streamer(rx, state, None, None))
    } else {
        process_non_streaming_response(&mut rx, state).await
    }
}

/// Helper function to handle chat completion errors and logging them.
pub fn handle_completion_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> CompletionResponder {
    base_handle_completion_error(state, e)
}

/// Creates a SSE streamer for chat completions with optional callbacks.
pub fn create_streamer(
    rx: Receiver<Response>,
    state: SharedMistralRsState,
    on_chunk: Option<CompletionOnChunkCallback>,
    on_done: Option<CompletionOnDoneCallback>,
) -> Sse<CompletionStreamer> {
    let streamer = base_create_streamer(rx, state, on_chunk, on_done);
    let keep_alive_interval = get_keep_alive_interval();

    Sse::new(streamer)
        .keep_alive(KeepAlive::new().interval(Duration::from_millis(keep_alive_interval)))
}

/// Processes non-streaming chat completion responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> CompletionResponder {
    base_process_non_streaming_response(rx, state, match_responses, handle_completion_error).await
}

/// Matches and processes different types of model responses into appropriate chat completion responses.
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
    }
}
