use anyhow::Result;
use std::{error::Error, sync::Arc};
use tokio::sync::mpsc::{channel, Sender};

use crate::openai::{AudioResponseFormat, SpeechGenerationRequest};
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
use serde::Serialize;

pub enum SpeechGenerationResponder {
    InternalError(Box<dyn Error>),
    ValidationError(Box<dyn Error>),
    RawResponse(axum::response::Response),
}

trait ErrorToResponse: Serialize {
    fn to_response(&self, code: StatusCode) -> axum::response::Response {
        let mut r = Json(self).into_response();
        *r.status_mut() = code;
        r
    }
}

#[derive(Serialize, Debug)]
struct JsonError {
    message: String,
}

impl JsonError {
    fn new(message: String) -> Self {
        Self { message }
    }
}

impl std::fmt::Display for JsonError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for JsonError {}

impl ErrorToResponse for JsonError {}

impl IntoResponse for SpeechGenerationResponder {
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

fn parse_request(
    oairequest: SpeechGenerationRequest,
    state: Arc<MistralRs>,
    tx: Sender<Response>,
) -> Result<(Request, AudioResponseFormat)> {
    let repr = serde_json::to_string(&oairequest).expect("Serialization of request failed.");
    MistralRs::maybe_log_request(state.clone(), repr);

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
    }));

    Ok((request, oairequest.response_format))
}

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
    let (tx, mut rx) = channel(10_000);

    let (request, response_format) = match parse_request(oairequest, state.clone(), tx) {
        Ok(x) => x,
        Err(e) => {
            let e = anyhow::Error::msg(e.to_string());
            MistralRs::maybe_log_error(state, &*e);
            return SpeechGenerationResponder::InternalError(e.into());
        }
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

    let sender = state.get_sender().unwrap();

    if let Err(e) = sender.send(request).await {
        let e = anyhow::Error::msg(e.to_string());
        MistralRs::maybe_log_error(state, &*e);
        return SpeechGenerationResponder::InternalError(e.into());
    }

    let response = match rx.recv().await {
        Some(response) => response,
        None => {
            let e = anyhow::Error::msg("No response received from the model.");
            MistralRs::maybe_log_error(state, &*e);
            return SpeechGenerationResponder::InternalError(e.into());
        }
    };

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
