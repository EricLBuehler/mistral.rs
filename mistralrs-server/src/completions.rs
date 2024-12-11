use anyhow::Result;
use std::{
    env,
    error::Error,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};
use tokio::sync::mpsc::{channel, Receiver, Sender};

use crate::openai::{CompletionRequest, Grammar, StopTokens};
use axum::{
    extract::{Json, State},
    http::{self, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use mistralrs_core::{
    CompletionResponse, Constraint, DrySamplingParams, MistralRs, NormalRequest, Request,
    RequestMessage, Response, SamplingParams, StopTokens as InternalStopTokens,
};
use serde::Serialize;
use tracing::warn;

#[derive(Debug)]
struct ModelErrorMessage(String);
impl std::fmt::Display for ModelErrorMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
impl std::error::Error for ModelErrorMessage {}

pub struct Streamer {
    rx: Receiver<Response>,
    is_done: bool,
    state: Arc<MistralRs>,
}

impl futures::Stream for Streamer {
    type Item = Result<Event, axum::Error>;

    fn poll_next(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.is_done {
            return Poll::Ready(None);
        }
        match self.rx.try_recv() {
            Ok(resp) => match resp {
                Response::CompletionModelError(msg, _) => {
                    MistralRs::maybe_log_error(
                        self.state.clone(),
                        &ModelErrorMessage(msg.to_string()),
                    );
                    self.is_done = true;
                    Poll::Ready(Some(Ok(Event::default().data(msg))))
                }
                Response::ValidationError(e) => {
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::InternalError(e) => {
                    MistralRs::maybe_log_error(self.state.clone(), &*e);
                    Poll::Ready(Some(Ok(Event::default().data(e.to_string()))))
                }
                Response::CompletionChunk(response) => {
                    if response.choices.iter().all(|x| x.finish_reason.is_some()) {
                        self.is_done = true;
                    }
                    MistralRs::maybe_log_response(self.state.clone(), &response);
                    Poll::Ready(Some(Event::default().json_data(response)))
                }
                Response::Done(_) => unreachable!(),
                Response::CompletionDone(_) => unreachable!(),
                Response::Chunk(_) => unreachable!(),
                Response::ImageGeneration(_) => unreachable!(),
                Response::ModelError(_, _) => unreachable!(),
                Response::Raw { .. } => unreachable!(),
            },
            Err(_) => Poll::Pending,
        }
    }
}

pub enum CompletionResponder {
    Sse(Sse<Streamer>),
    Json(CompletionResponse),
    ModelError(String, CompletionResponse),
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
}

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}
impl ErrorToResponse for JsonError {}

#[derive(Serialize)]
struct JsonModelError {
    message: String,
    partial_response: CompletionResponse,
}

impl JsonModelError {
    fn new(message: String, partial_response: CompletionResponse) -> Self {
        Self {
            message,
            partial_response,
        }
    }
}

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
        Request::Normal(NormalRequest {
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
            adapters: oairequest.adapters,
            tool_choice: oairequest.tool_choice,
            tools: oairequest.tools,
            logits_processors: None,
            return_raw_logits: false,
        }),
        is_streaming,
    ))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/completions",
    request_body = CompletionRequest,
    responses((status = 200, description = "Completions"))
)]

pub async fn completions(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<CompletionRequest>,
) -> CompletionResponder {
    let (tx, mut rx) = channel(10_000);
    if oairequest.logprobs.is_some() {
        return CompletionResponder::ValidationError(
            "Completion requests do not support logprobs.".into(),
        );
    }

    let (request, is_streaming) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => {
            let e = anyhow::Error::msg(e.to_string());
            MistralRs::maybe_log_error(state, &*e);
            return CompletionResponder::InternalError(e.into());
        }
    };
    let sender = state.get_sender().unwrap();

    if let Err(e) = sender.send(request).await {
        let e = anyhow::Error::msg(e.to_string());
        MistralRs::maybe_log_error(state, &*e);
        return CompletionResponder::InternalError(e.into());
    }

    if is_streaming {
        let streamer = Streamer {
            rx,
            is_done: false,
            state,
        };

        CompletionResponder::Sse(
            Sse::new(streamer).keep_alive(
                KeepAlive::new()
                    .interval(Duration::from_millis(
                        env::var("KEEP_ALIVE_INTERVAL")
                            .map(|val| val.parse::<u64>().unwrap_or(1000))
                            .unwrap_or(1000),
                    ))
                    .text("keep-alive-text"),
            ),
        )
    } else {
        let response = match rx.recv().await {
            Some(response) => response,
            None => {
                let e = anyhow::Error::msg("No response received from the model.");
                MistralRs::maybe_log_error(state, &*e);
                return CompletionResponder::InternalError(e.into());
            }
        };

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
            Response::Raw { .. } => unreachable!(),
        }
    }
}
