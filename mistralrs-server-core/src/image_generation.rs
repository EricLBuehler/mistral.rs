//! ## Image generation functionality and route handler.

use std::{error::Error, sync::Arc};

use anyhow::Result;
use axum::{
    extract::{Json, State},
    http::{self},
    response::IntoResponse,
};
use mistralrs_core::{
    Constraint, DiffusionGenerationParams, ImageGenerationResponse, MistralRs, NormalRequest,
    Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, send_request,
        ErrorToResponse, JsonError,
    },
    openai::ImageGenerationRequest,
    types::{ExtractedMistralRsState, SharedMistralRsState},
    util::{sanitize_error_message, validate_model_name},
};

/// Represents different types of image generation responses.
pub enum ImageGenerationResponder {
    Json(ImageGenerationResponse),
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
}

impl IntoResponse for ImageGenerationResponder {
    /// Converts the image generation responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            ImageGenerationResponder::Json(s) => Json(s).into_response(),
            ImageGenerationResponder::InternalError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            ImageGenerationResponder::ValidationError(e) => {
                JsonError::new(sanitize_error_message(e.as_ref()))
                    .to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
        }
    }
}

/// Parses and validates a image generation request.
///
/// This function transforms a image generation request into the
/// request format used by mistral.rs.
pub fn parse_request(
    oairequest: ImageGenerationRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<Request> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    Ok(Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::ImageGeneration {
            prompt: oairequest.prompt,
            format: oairequest.response_format,
            generation_params: DiffusionGenerationParams {
                height: oairequest.height,
                width: oairequest.width,
            },
            save_file: None,
        },
        sampling_params: SamplingParams::deterministic(),
        response: tx,
        return_logprobs: false,
        is_streaming: false,
        suffix: None,
        constraint: Constraint::None,
        tool_choice: None,
        tools: None,
        logits_processors: None,
        return_raw_logits: false,
        web_search_options: None,
        model_id: if oairequest.model == "default" {
            None
        } else {
            Some(oairequest.model.clone())
        },
        truncate_sequence: false,
    })))
}

/// Image generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/images/generations",
    request_body = ImageGenerationRequest,
    responses((status = 200, description = "Image generation"))
)]
pub async fn image_generation(
    State(state): ExtractedMistralRsState,
    Json(oairequest): Json<ImageGenerationRequest>,
) -> ImageGenerationResponder {
    let (tx, mut rx) = create_response_channel(None);

    let request = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    process_non_streaming_response(&mut rx, state).await
}

/// Helper function to handle image generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> ImageGenerationResponder {
    let sanitized_msg = sanitize_error_message(&*e);
    let e = anyhow::Error::msg(sanitized_msg);
    MistralRs::maybe_log_error(state, &*e);
    ImageGenerationResponder::InternalError(e.into())
}

/// Process non-streaming image generation responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
) -> ImageGenerationResponder {
    base_process_non_streaming_response(rx, state, match_responses, handle_error).await
}

/// Matches and processes different types of model responses into appropriate image generation responses.
pub fn match_responses(
    state: SharedMistralRsState,
    response: Response,
) -> ImageGenerationResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            ImageGenerationResponder::InternalError(e)
        }
        Response::ValidationError(e) => ImageGenerationResponder::ValidationError(e),
        Response::ImageGeneration(response) => {
            MistralRs::maybe_log_response(state, &response);
            ImageGenerationResponder::Json(response)
        }
        Response::CompletionModelError(m, _) => {
            let e = anyhow::Error::msg(m.to_string());
            MistralRs::maybe_log_error(state, &*e);
            ImageGenerationResponder::InternalError(e.into())
        }
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::Speech { .. } => unreachable!(),
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
    }
}
