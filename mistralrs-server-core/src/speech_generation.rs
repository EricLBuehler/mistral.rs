//! ## Speech generation functionality and route handler.

use std::{error::Error, sync::Arc};

use anyhow::Result;
use axum::{
    body::Bytes,
    extract::{Json, State},
    http::{self, HeaderMap, HeaderValue, StatusCode},
    response::IntoResponse,
};
use mistralrs_core::{
    speech_utils::{self, Sample},
    Constraint, MistralRs, NormalRequest, Request, RequestMessage, Response, SamplingParams,
};
use tokio::sync::mpsc::{Receiver, Sender};

use crate::{
    handler_core::{create_response_channel, send_request, ErrorToResponse, JsonError},
    openai::{AudioResponseFormat, SpeechGenerationRequest},
    types::SharedMistralRsState,
    util::validate_model_name,
};

/// Represents different types of speech generation responses.
pub enum SpeechGenerationResponder {
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
    RawResponse(axum::response::Response),
}

impl IntoResponse for SpeechGenerationResponder {
    /// Converts the speech generation responder into an HTTP response.
    fn into_response(self) -> axum::response::Response {
        match self {
            SpeechGenerationResponder::InternalError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::INTERNAL_SERVER_ERROR)
            }
            SpeechGenerationResponder::ValidationError(e) => {
                JsonError::new(e.to_string()).to_response(http::StatusCode::UNPROCESSABLE_ENTITY)
            }
            SpeechGenerationResponder::RawResponse(resp) => resp,
        }
    }
}

/// Parses and validates a speech generation request.
///
/// This function transforms a speech generation request into the
/// request format used by mistral.rs.
pub fn parse_request(
    oairequest: SpeechGenerationRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, AudioResponseFormat)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

    // Validate that the requested model matches the loaded model
    validate_model_name(&oairequest.model, state.clone())?;

    let request = Request::Normal(Box::new(NormalRequest {
        id: state.next_request_id(),
        messages: RequestMessage::SpeechGeneration {
            prompt: oairequest.input,
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
    }));

    Ok((request, oairequest.response_format))
}

/// Speech generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/speech",
    request_body = SpeechGenerationRequest,
    responses((status = 200, description = "Speech generation"))
)]
pub async fn speech_generation(
    State(state): State<Arc<MistralRs>>,
    Json(oairequest): Json<SpeechGenerationRequest>,
) -> SpeechGenerationResponder {
    let (tx, mut rx) = create_response_channel(None);

    let (request, response_format) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => return handle_error(state, e.into()),
    };

    // Validate response format here
    if !matches!(
        response_format,
        AudioResponseFormat::Wav | AudioResponseFormat::Pcm
    ) {
        return SpeechGenerationResponder::ValidationError(Box::new(JsonError::new(
            "Only support wav/pcm response format.".to_string(),
        )));
    }

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    process_non_streaming_response(&mut rx, state, response_format).await
}

/// Helper function to handle speech generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> SpeechGenerationResponder {
    let e = anyhow::Error::msg(e.to_string());
    MistralRs::maybe_log_error(state, &*e);
    SpeechGenerationResponder::InternalError(e.into())
}

/// Process non-streaming speech generation responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    let response = match rx.recv().await {
        Some(response) => response,
        None => {
            let e = anyhow::Error::msg("No response received from the model.");
            return handle_error(state, e.into());
        }
    };

    match_responses(state, response, response_format)
}

/// Matches and processes different types of model responses into appropriate speech generation responses.
pub fn match_responses(
    state: SharedMistralRsState,
    response: Response,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            SpeechGenerationResponder::InternalError(e)
        }
        Response::ValidationError(e) => SpeechGenerationResponder::ValidationError(e),
        Response::ImageGeneration(_) => unreachable!(),
        Response::CompletionModelError(m, _) => {
            let e = anyhow::Error::msg(m.to_string());
            MistralRs::maybe_log_error(state, &*e);
            SpeechGenerationResponder::InternalError(e.into())
        }
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(_) => unreachable!(),
        Response::ModelError(_, _) => unreachable!(),
        Response::Speech {
            pcm,
            rate,
            channels,
        } => {
            let pcm_endianness = "s16le";

            let content_type = response_format.audio_content_type(rate, channels, pcm_endianness);
            let mut headers = HeaderMap::new();
            headers.insert(
                http::header::CONTENT_TYPE,
                HeaderValue::from_str(&content_type).unwrap(),
            );

            let encoded = match response_format {
                AudioResponseFormat::Pcm => {
                    let samples: &[f32] = &pcm;
                    let mut buf = Vec::with_capacity(samples.len() * std::mem::size_of::<i64>());
                    for &sample in samples {
                        buf.extend_from_slice(&sample.to_i16().to_le_bytes());
                    }
                    buf
                }
                AudioResponseFormat::Wav => {
                    // Write WAV data into an in-memory buffer
                    let mut buf = Vec::new();
                    speech_utils::write_pcm_as_wav(&mut buf, &pcm, rate as u32, channels as u16)
                        .unwrap();
                    buf
                }
                _ => unreachable!("Should be validated above."),
            };

            let bytes = Bytes::from(encoded);

            SpeechGenerationResponder::RawResponse((StatusCode::OK, headers, bytes).into_response())
        }
        Response::Raw { .. } => unreachable!(),
    }
}
