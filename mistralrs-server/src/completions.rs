use std::{error::Error, sync::Arc};
use tokio::sync::mpsc::{channel, Sender};

use crate::openai::{CompletionRequest, Grammar, StopTokens};
use axum::{
    extract::{Json, State},
    http::{self, StatusCode},
    response::IntoResponse,
};
use mistralrs_core::{
    CompletionResponse, Constraint, MistralRs, Request, RequestMessage, Response, SamplingParams,
    StopTokens as InternalStopTokens,
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
pub enum CompletionResponder {
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
) -> Request {
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

    if oairequest._stream.is_some_and(|x| x) {
        warn!("Completion requests do not support streaming.");
    }

    Request {
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
            top_n_logprobs: 1,
            frequency_penalty: oairequest.frequency_penalty,
            presence_penalty: oairequest.presence_penalty,
            max_len: oairequest.max_tokens,
            stop_toks,
            logits_bias: oairequest.logit_bias,
            n_choices: oairequest.n_choices,
        },
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: oairequest.suffix,
        constraint: match oairequest.grammar {
            Some(Grammar::Yacc(yacc)) => Constraint::Yacc(yacc),
            Some(Grammar::Regex(regex)) => Constraint::Regex(regex),
            None => Constraint::None,
        },
    }
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
    let request = parse_request(oairequest, state.clone(), tx);
    let is_streaming = request.is_streaming;
    let sender = state.get_sender();

    if request.return_logprobs {
        return CompletionResponder::ValidationError(
            "Completion requests do not support logprobs.".into(),
        );
    }

    if is_streaming {
        return CompletionResponder::ValidationError(
            "Completion requests do not support streaming.".into(),
        );
    }

    sender.send(request).await.unwrap();

    let response = rx.recv().await.unwrap();

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
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
    }
}
