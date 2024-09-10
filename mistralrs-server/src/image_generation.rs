use anyhow::Result;
use std::{error::Error, sync::Arc};
use tokio::sync::mpsc::{channel, Sender};

use crate::openai::ImageGenerationRequest;
use axum::{
    extract::{Json, State},
    http::{self, StatusCode},
    response::IntoResponse,
};
use mistralrs_core::{
    Constraint, ImageGenerationResponse, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams
};
use serde::Serialize;

pub enum CompletionResponder {
    Json(ImageGenerationResponse),
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
        }
    }
}

fn parse_request(
    oairequest: ImageGenerationRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<Request> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    Ok(Request::Normal(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::ImageGeneration {
            prompt: oairequest.prompt,
            format: oairequest.response_format,
        },
        sampling_params: SamplingParams::default(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        adapters: None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
    }))
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/images/generations",
    request_body = ImageGenerationRequest,
    responses((status = 200, description = "Image generation"))
)]

pub async fn image_generation(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<ImageGenerationRequest>,
) -> CompletionResponder {
    let (tx, mut rx) = channel(10_000);

    let request = match parse_request(oairequest, state.clone(), tx) {
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
        Response::ValidationError(e) => CompletionResponder::ValidationError(e),
        Response::ImageGeneration(response) => {
            MistralRs::maybe_log_response(state, &response);
            CompletionResponder::Json(response)
        }
        Response::CompletionModelError(_, _) => unreachable!(),
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
    }
}
