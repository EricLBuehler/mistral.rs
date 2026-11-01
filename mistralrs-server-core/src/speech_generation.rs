//! ## Speech generation functionality and route handler.

use std::{collections::VecDeque, convert::Infallible, error::Error, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{rejection::JsonRejection, Json, State},
    http::{self, HeaderMap, HeaderValue, StatusCode},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Sse,
    },
};
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{
    ChatAudioFormat, Constraint, MistralRs, ModelCategory, NormalRequest, Request, RequestMessage,
    Response, SamplingParams, SupportedModality, Usage,
};
use serde::Serialize;
use tokio::sync::mpsc::{Receiver, Sender};
use utoipa::ToSchema;

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, openai_error_from_error,
        send_request, ApiError, ApiErrorKind, ModelErrorMessage,
    },
    openai::{
        AudioResponseFormat, ChatCompletionRequest, OpenAiToolSurface, SpeechGenerationRequest,
    },
    streaming::openai_error_event,
    types::SharedMistralRsState,
    util::validate_model_name,
};

const MIN_SPEECH_SPEED: f64 = 0.25;
const MAX_SPEECH_SPEED: f64 = 4.0;
const SPEECH_TRUNCATED_ERROR: &str =
    "Speech generation reached the model context limit before finishing";

#[derive(ToSchema)]
#[schema(value_type = String, format = Binary)]
pub struct SpeechAudioFile(pub Vec<u8>);

#[derive(Serialize, ToSchema)]
pub struct SpeechTokenUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
    pub total_tokens: usize,
}

#[derive(Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum SpeechStreamEvent {
    #[serde(rename = "speech.audio.delta")]
    Delta { audio: String },
    #[serde(rename = "speech.audio.done")]
    Done { usage: SpeechTokenUsage },
}

#[derive(Clone, Copy)]
struct SpeechOutputOptions {
    format: AudioResponseFormat,
    speed: f64,
    sse: bool,
}

impl From<&SpeechGenerationRequest> for SpeechOutputOptions {
    fn from(request: &SpeechGenerationRequest) -> Self {
        Self {
            format: request.response_format,
            speed: request.speed.unwrap_or(1.0),
            sse: request.stream_format.as_deref() == Some("sse"),
        }
    }
}

impl SpeechOutputOptions {
    fn progressive(self) -> bool {
        self.sse && self.format == AudioResponseFormat::Pcm && self.speed == 1.0
    }
}

pub(crate) async fn parse_audio_chat_request(
    value: serde_json::Value,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<Request> {
    let request: ChatCompletionRequest = serde_json::from_value(value)?;
    let (request, _) = crate::chat_completion::parse_request(
        request,
        crate::chat_completion::ChatCompletionParseContext {
            state,
            tx,
            tool_dispatch_url: None,
            agent_approval_handler: None,
            agent_approval_notifier: None,
            tool_surface: OpenAiToolSurface::ChatCompletions,
            skill_store: None,
        },
    )
    .await?;
    Ok(request)
}

fn speech_chat_body(request: &SpeechGenerationRequest) -> serde_json::Value {
    let system = mistralrs_core::speech_task_prompt(request.instructions.as_deref());
    let progressive = SpeechOutputOptions::from(request).progressive();
    serde_json::json!({
        "model": request.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": request.input},
        ],
        "modalities": ["audio"],
        "audio": {"format": if progressive { "pcm16" } else { "wav" }, "voice": "default"},
        "temperature": 0.0,
        "enable_thinking": false,
        "stream": progressive,
    })
}

fn validate_speech_request(request: &SpeechGenerationRequest) -> Result<()> {
    anyhow::ensure!(!request.input.trim().is_empty(), "input must not be empty");
    anyhow::ensure!(
        request
            .voice
            .as_deref()
            .is_none_or(|voice| voice == "default"),
        "This model supports only voice=default"
    );
    anyhow::ensure!(
        request.speed.is_none_or(
            |speed| speed.is_finite() && (MIN_SPEECH_SPEED..=MAX_SPEECH_SPEED).contains(&speed)
        ),
        "speed must be between 0.25 and 4.0"
    );
    anyhow::ensure!(
        request
            .stream_format
            .as_deref()
            .is_none_or(|format| matches!(format, "audio" | "sse")),
        "stream_format must be audio or sse"
    );
    Ok(())
}

fn output_format(format: AudioResponseFormat) -> ChatAudioFormat {
    match format {
        AudioResponseFormat::Mp3 => ChatAudioFormat::Mp3,
        AudioResponseFormat::Opus => ChatAudioFormat::Opus,
        AudioResponseFormat::Aac => ChatAudioFormat::Aac,
        AudioResponseFormat::Flac => ChatAudioFormat::Flac,
        AudioResponseFormat::Wav => ChatAudioFormat::Wav,
        AudioResponseFormat::Pcm => ChatAudioFormat::Pcm16,
    }
}

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
                openai_error_from_error(e.as_ref(), ApiErrorKind::Internal)
            }
            SpeechGenerationResponder::ValidationError(e) => {
                openai_error_from_error(e.as_ref(), ApiErrorKind::InvalidRequest)
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
        audio_output: None,
        id: state.next_request_id(),
        queued_at: None,
        messages: RequestMessage::SpeechGeneration {
            prompt: oairequest.input,
        },
        sampling_params: SamplingParams::deterministic(),
        seed: None,
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
        enable_code_execution: false,
        enable_shell: false,
        shell_options: None,
        code_execution_permission: None,
        code_execution_approval_notifier: None,
        agent_permission: None,
        agent_approval_handler: None,
        agent_approval_notifier: None,
        max_tool_rounds: None,
        tool_dispatch_url: None,
        model_id: if oairequest.model == "default" {
            None
        } else {
            Some(oairequest.model.clone())
        },
        adapter: None,
        truncate_sequence: false,
        session_id: None,
        files: None,
        input_files: Vec::new(),
    }));

    Ok((request, oairequest.response_format))
}

/// Speech generation endpoint handler.
#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/speech",
    request_body = SpeechGenerationRequest,
    responses(
        (status = 200, description = "Encoded audio or speech events", content(
            (SpeechAudioFile = "audio/mpeg"),
            (SpeechAudioFile = "audio/ogg"),
            (SpeechAudioFile = "audio/aac"),
            (SpeechAudioFile = "audio/flac"),
            (SpeechAudioFile = "audio/wav"),
            (SpeechAudioFile = "audio/pcm"),
            (SpeechStreamEvent = "text/event-stream")
        )),
        (status = 400, description = "Invalid speech parameters or unsupported model capability")
    )
)]
pub async fn speech_generation(
    State(state): State<Arc<MistralRs>>,
    payload: Result<Json<SpeechGenerationRequest>, JsonRejection>,
) -> SpeechGenerationResponder {
    let oairequest = match payload {
        Ok(Json(request)) => request,
        Err(error) => {
            return SpeechGenerationResponder::ValidationError(Box::new(
                ApiError::from_json_rejection(error),
            ));
        }
    };
    let (tx, mut rx) = create_response_channel(None);
    if let Err(error) = validate_speech_request(&oairequest) {
        return SpeechGenerationResponder::ValidationError(error.into());
    }
    let output = SpeechOutputOptions::from(&oairequest);
    let model = (oairequest.model != "default").then_some(oairequest.model.as_str());
    let config = match state.config(model) {
        Ok(config) => config,
        Err(error) => {
            return SpeechGenerationResponder::ValidationError(anyhow::anyhow!(error).into())
        }
    };
    let request = if matches!(config.category, ModelCategory::Speech) {
        if output.sse {
            return SpeechGenerationResponder::ValidationError(
                anyhow::anyhow!("SSE speech output requires a chat audio model").into(),
            );
        }
        if oairequest
            .instructions
            .as_deref()
            .is_some_and(|text| !text.is_empty())
        {
            return SpeechGenerationResponder::ValidationError(
                anyhow::anyhow!("instructions are not supported by this speech model").into(),
            );
        }
        parse_request(oairequest, state.clone(), tx).map(|(request, _)| request)
    } else if config.modalities.output.contains(&SupportedModality::Audio) {
        parse_audio_chat_request(speech_chat_body(&oairequest), state.clone(), tx).await
    } else {
        Err(anyhow::anyhow!(
            "The requested model does not support speech output"
        ))
    };
    let request = match request {
        Ok(x) => x,
        Err(e) => return SpeechGenerationResponder::ValidationError(e.into()),
    };

    if let Err(e) = send_request(&state, request).await {
        return handle_error(state, e.into());
    }

    if output.progressive() {
        return SpeechGenerationResponder::RawResponse(speech_stream(rx, state));
    }

    let response = base_process_non_streaming_response(
        &mut rx,
        state.clone(),
        |_, response| response,
        |_, error| Response::InternalError(error),
    )
    .await;
    match_response_with_options(state, response, output).await
}

/// Helper function to handle speech generation errors and logging them.
pub fn handle_error(
    state: SharedMistralRsState,
    e: Box<dyn std::error::Error + Send + Sync + 'static>,
) -> SpeechGenerationResponder {
    MistralRs::maybe_log_error(state, e.as_ref());
    SpeechGenerationResponder::InternalError(e)
}

/// Process non-streaming speech generation responses.
pub async fn process_non_streaming_response(
    rx: &mut Receiver<Response>,
    state: SharedMistralRsState,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    let response = base_process_non_streaming_response(
        rx,
        state.clone(),
        |_, response| response,
        |_, error| Response::InternalError(error),
    )
    .await;
    match_responses(state, response, response_format).await
}

/// Matches and processes different types of model responses into appropriate speech generation responses.
pub async fn match_responses(
    state: SharedMistralRsState,
    response: Response,
    response_format: AudioResponseFormat,
) -> SpeechGenerationResponder {
    match_response_with_options(
        state,
        response,
        SpeechOutputOptions {
            format: response_format,
            speed: 1.0,
            sse: false,
        },
    )
    .await
}

async fn match_response_with_options(
    state: SharedMistralRsState,
    response: Response,
    output: SpeechOutputOptions,
) -> SpeechGenerationResponder {
    match response {
        Response::InternalError(e) => {
            MistralRs::maybe_log_error(state, &*e);
            SpeechGenerationResponder::InternalError(e)
        }
        Response::ValidationError(e) => SpeechGenerationResponder::ValidationError(e),
        Response::ImageGeneration(_) => unreachable!(),
        Response::CompletionModelError(message, _) | Response::ModelError(message, _) => {
            MistralRs::maybe_log_error(state, &ModelErrorMessage(message));
            SpeechGenerationResponder::InternalError(Box::new(ApiError::model_error()))
        }
        Response::CompletionDone(_) => unreachable!(),
        Response::CompletionChunk(_) => unreachable!(),
        Response::Chunk(_) => unreachable!(),
        Response::Done(response) => {
            let usage = response.usage;
            let Some(choice) = response.choices.into_iter().next() else {
                return handle_error(
                    state,
                    anyhow::anyhow!("Model returned no speech choice").into(),
                );
            };
            if choice.finish_reason == "length" {
                return SpeechGenerationResponder::ValidationError(Box::new(
                    ApiError::invalid_request(SPEECH_TRUNCATED_ERROR),
                ));
            }
            let decoded = tokio::task::spawn_blocking(move || -> Result<_> {
                let encoded = choice
                    .message
                    .audio
                    .and_then(|audio| audio.data)
                    .context("Model returned no speech audio")?;
                let wav = STANDARD.decode(encoded)?;
                mistralrs_core::decode_generated_wav(&wav)
            })
            .await;
            let result = match decoded {
                Ok(Ok(audio)) => {
                    encode_response(
                        Arc::new(audio.samples),
                        audio.sample_rate,
                        audio.channels,
                        output,
                        Some(&usage),
                    )
                    .await
                }
                Ok(Err(error)) => Err(error),
                Err(error) => Err(error.into()),
            };
            match result {
                Ok(response) => response,
                Err(error) => handle_error(state, error.into()),
            }
        }
        Response::Speech {
            pcm,
            rate,
            channels,
        } => match encode_response(pcm, rate as u32, channels as u16, output, None).await {
            Ok(response) => response,
            Err(error) => handle_error(state, error.into()),
        },
        Response::Raw { .. } => unreachable!(),
        Response::Embeddings { .. } => unreachable!(),
        Response::AgenticToolCallProgress { .. } => unreachable!(),
        Response::BlockDenoisingProgress(_) => unreachable!(),
        Response::AgenticToolApprovalRequired { .. } => unreachable!(),
        Response::File(_) => unreachable!(),
    }
}

async fn encode_response(
    samples: Arc<Vec<f32>>,
    rate: u32,
    channels: u16,
    output: SpeechOutputOptions,
    usage: Option<&Usage>,
) -> Result<SpeechGenerationResponder> {
    let bytes = mistralrs_core::encode_audio_pcm_with_speed_async(
        samples,
        rate,
        channels,
        output_format(output.format),
        output.speed,
    )
    .await?;
    if output.sse {
        let usage = usage.context("Speech streaming requires token usage from the model")?;
        let events = [speech_delta(STANDARD.encode(bytes)), speech_done(usage)];
        return Ok(SpeechGenerationResponder::RawResponse(
            Sse::new(futures::stream::iter(
                events.into_iter().map(Ok::<_, Infallible>),
            ))
            .into_response(),
        ));
    }
    let mut headers = HeaderMap::new();
    headers.insert(
        http::header::CONTENT_TYPE,
        HeaderValue::from_str(&output.format.audio_content_type(
            rate as usize,
            channels as usize,
            "s16le",
        ))?,
    );
    Ok(SpeechGenerationResponder::RawResponse(
        (StatusCode::OK, headers, Bytes::from(bytes)).into_response(),
    ))
}

fn speech_delta(audio: String) -> Event {
    Event::default()
        .json_data(SpeechStreamEvent::Delta { audio })
        .expect("JSON event serialization")
}

fn speech_done(usage: &Usage) -> Event {
    Event::default()
        .json_data(SpeechStreamEvent::Done {
            usage: SpeechTokenUsage {
                input_tokens: usage.prompt_tokens,
                output_tokens: usage.completion_tokens,
                total_tokens: usage.total_tokens,
            },
        })
        .expect("JSON event serialization")
}

fn speech_stream(rx: Receiver<Response>, state: SharedMistralRsState) -> axum::response::Response {
    let stream = futures::stream::unfold(
        (rx, VecDeque::<Event>::new(), false, state),
        |(mut rx, mut pending, mut finished, state)| async move {
            loop {
                if let Some(event) = pending.pop_front() {
                    return Some((Ok::<_, Infallible>(event), (rx, pending, finished, state)));
                }
                if finished {
                    return None;
                }
                match rx.recv().await {
                    Some(Response::Chunk(chunk)) => {
                        for choice in chunk.choices {
                            if let Some(data) = choice
                                .delta
                                .audio
                                .and_then(|audio| audio.data)
                                .filter(|data| !data.is_empty())
                            {
                                pending.push_back(speech_delta(data));
                            }
                            if let Some(reason) = choice.finish_reason {
                                finished = true;
                                let event = if reason == "length" {
                                    openai_error_event(ApiError::invalid_request(
                                        SPEECH_TRUNCATED_ERROR,
                                    ))
                                } else if let Some(usage) = &chunk.usage {
                                    speech_done(usage)
                                } else {
                                    openai_error_event(ApiError::internal())
                                };
                                pending.push_back(event);
                            }
                        }
                    }
                    Some(Response::ValidationError(error)) => {
                        finished = true;
                        pending.push_back(openai_error_event(ApiError::invalid_request(
                            error.to_string(),
                        )));
                    }
                    Some(Response::InternalError(error)) => {
                        MistralRs::maybe_log_error(state.clone(), error.as_ref());
                        finished = true;
                        pending.push_back(openai_error_event(ApiError::internal()));
                    }
                    Some(Response::ModelError(message, _)) => {
                        MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(message));
                        finished = true;
                        pending.push_back(openai_error_event(ApiError::model_error()));
                    }
                    Some(
                        Response::AgenticToolCallProgress { .. }
                        | Response::BlockDenoisingProgress(_)
                        | Response::AgenticToolApprovalRequired { .. }
                        | Response::File(_),
                    ) => {}
                    _ => {
                        finished = true;
                        pending.push_back(openai_error_event(ApiError::internal()));
                    }
                }
            }
        },
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mistralrs_core::AudioInput;

    #[tokio::test]
    async fn speech_sse_uses_documented_events_and_actual_usage() -> Result<()> {
        let usage = Usage {
            prompt_tokens: 7,
            completion_tokens: 11,
            total_tokens: 18,
            prompt_tokens_details: None,
            avg_tok_per_sec: 0.0,
            avg_prompt_tok_per_sec: 0.0,
            avg_compl_tok_per_sec: 0.0,
            total_time_sec: 0.0,
            total_prompt_time_sec: 0.0,
            total_completion_time_sec: 0.0,
        };
        let response = Sse::new(futures::stream::iter(
            [speech_delta("AAABAA==".to_string()), speech_done(&usage)]
                .into_iter()
                .map(Ok::<_, Infallible>),
        ))
        .into_response();
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        let events = std::str::from_utf8(&bytes)?
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            events[0],
            serde_json::json!({"type":"speech.audio.delta", "audio":"AAABAA=="})
        );
        assert_eq!(
            events[1],
            serde_json::json!({"type":"speech.audio.done", "usage":{"input_tokens":7,"output_tokens":11,"total_tokens":18}})
        );
        let request: SpeechGenerationRequest = serde_json::from_value(
            serde_json::json!({"input":"Hello.","response_format":"pcm","stream_format":"sse"}),
        )?;
        validate_speech_request(&request)?;
        let body = speech_chat_body(&request);
        assert_eq!(body["stream"], true);
        assert_eq!(body["audio"]["format"], "pcm16");
        Ok(())
    }

    #[test]
    fn speech_defaults_to_mp3_and_builds_audio_only_chat() -> Result<()> {
        let request: SpeechGenerationRequest =
            serde_json::from_value(serde_json::json!({"input": "Hello."}))?;
        assert_eq!(request.response_format, AudioResponseFormat::Mp3);
        validate_speech_request(&request)?;
        let body = speech_chat_body(&request);
        assert_eq!(body["modalities"], serde_json::json!(["audio"]));
        assert_eq!(body["audio"]["format"], "wav");
        assert_eq!(body["messages"][1]["content"], "Hello.");
        assert!(body.get("max_tokens").is_none());
        let _: ChatCompletionRequest = serde_json::from_value(body)?;
        Ok(())
    }

    #[test]
    fn speech_rejects_unsupported_controls() -> Result<()> {
        for extra in [
            serde_json::json!({"voice":"alloy"}),
            serde_json::json!({"speed":5.0}),
            serde_json::json!({"stream_format":"invalid"}),
        ] {
            let mut value = serde_json::json!({"input":"Hello."});
            value
                .as_object_mut()
                .unwrap()
                .extend(extra.as_object().unwrap().clone());
            let request: SpeechGenerationRequest = serde_json::from_value(value)?;
            assert!(validate_speech_request(&request).is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn wav_response_contains_audio_file() -> Result<()> {
        let response = encode_response(
            Arc::new(vec![0.0, 0.5, -0.5]),
            24_000,
            1,
            SpeechOutputOptions {
                format: AudioResponseFormat::Wav,
                speed: 1.0,
                sse: false,
            },
            None,
        )
        .await?
        .into_response();
        assert!(response.headers()[http::header::CONTENT_TYPE]
            .to_str()?
            .starts_with("audio/wav"));
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        let decoded = AudioInput::from_bytes(&bytes)?;
        assert_eq!(decoded.sample_rate, 24_000);
        assert_eq!(decoded.samples.len(), 3);
        Ok(())
    }
}
