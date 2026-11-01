use std::{
    collections::{HashSet, VecDeque},
    convert::Infallible,
};

use anyhow::{Context, Result};
use axum::{
    extract::{multipart::MultipartRejection, Multipart, State},
    response::{
        sse::{Event, KeepAlive},
        IntoResponse, Response as HttpResponse, Sse,
    },
    Json,
};
use either::Either;
use mistralrs_core::{
    AudioInput, MistralRs, ModelCategory, Request, RequestMessage, Response, SupportedModality,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc::{Receiver, Sender};
use utoipa::ToSchema;

use crate::{
    handler_core::{
        base_process_non_streaming_response, create_response_channel, openai_error_from_error,
        openai_error_response, send_request, ApiError, ApiErrorKind, ModelErrorMessage,
    },
    media_source::MAX_MEDIA_BYTES,
    speech_generation::parse_audio_chat_request,
    streaming::openai_error_event,
    types::SharedMistralRsState,
    util::validate_model_name,
};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, ToSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TranscriptionResponseFormat {
    #[default]
    Json,
    Text,
}

#[derive(ToSchema)]
pub struct AudioTranscriptionRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
    pub model: String,
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub response_format: Option<TranscriptionResponseFormat>,
    pub temperature: Option<f64>,
    pub stream: Option<bool>,
}

#[derive(ToSchema)]
pub struct AudioTranslationRequest {
    #[schema(value_type = String, format = Binary)]
    pub file: Vec<u8>,
    pub model: String,
    pub prompt: Option<String>,
    pub response_format: Option<TranscriptionResponseFormat>,
    pub temperature: Option<f64>,
}

#[derive(Serialize, ToSchema)]
pub struct AudioTranscriptionResponse {
    pub text: String,
}

#[derive(Serialize, ToSchema)]
#[serde(tag = "type")]
pub enum TranscriptionStreamEvent {
    #[serde(rename = "transcript.text.delta")]
    Delta { delta: String },
    #[serde(rename = "transcript.text.done")]
    Done { text: String },
}

#[derive(Default)]
struct TranscriptionUpload {
    file: Option<Vec<u8>>,
    model: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
    format: TranscriptionResponseFormat,
    temperature: f64,
    stream: bool,
}

impl TranscriptionUpload {
    fn set_field(&mut self, name: &str, value: String, translate: bool) -> Result<()> {
        match name {
            "model" => self.model = Some(value),
            "prompt" => self.prompt = Some(value),
            "language" if !translate => self.language = Some(value),
            "response_format" => {
                self.format = match value.as_str() {
                    "json" => TranscriptionResponseFormat::Json,
                    "text" => TranscriptionResponseFormat::Text,
                    _ => anyhow::bail!("Only response_format=json or text is supported; timestamps and diarization are unavailable"),
                };
            }
            "temperature" => {
                self.temperature = value.parse().context("temperature must be a number")?;
                anyhow::ensure!(self.temperature.is_finite() && (0.0..=1.0).contains(&self.temperature), "temperature must be between 0 and 1");
            }
            "stream" if !translate => self.stream = value.parse().context("stream must be true or false")?,
            _ => anyhow::bail!("Unsupported audio parameter: {name}; timestamp and diarization controls are not supported"),
        }
        Ok(())
    }

    fn chat_body(&self, translate: bool) -> Result<serde_json::Value> {
        anyhow::ensure!(
            !self.stream || self.format == TranscriptionResponseFormat::Json,
            "Streaming transcription requires response_format=json"
        );
        let model = self
            .model
            .as_deref()
            .filter(|model| !model.is_empty())
            .context("model is required")?;
        let file = self
            .file
            .as_deref()
            .filter(|file| !file.is_empty())
            .context("file is required and must not be empty")?;
        anyhow::ensure!(
            file.len() <= MAX_MEDIA_BYTES,
            "Audio file exceeds the {MAX_MEDIA_BYTES} byte limit."
        );
        let mut instruction = if translate {
            "Translate the speech in the audio into English. Return only the English translation."
                .to_string()
        } else {
            match self.language.as_deref() {
                Some("en") => "Perform ASR in English.".to_string(),
                Some(language) => format!("Perform ASR in the language identified by ISO 639-1 code {language}. Return only the transcript."),
                None => "Perform ASR. Return only the transcript in the language spoken.".to_string(),
            }
        };
        if let Some(prompt) = self.prompt.as_deref().filter(|prompt| !prompt.is_empty()) {
            instruction.push_str("\nSpelling and vocabulary context: ");
            instruction.push_str(prompt);
        }
        Ok(serde_json::json!({
            "model": model,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": ""},
            ],
            "temperature": self.temperature,
            "enable_thinking": false,
            "stream": self.stream,
        }))
    }
}

pub(crate) async fn parse_transcription_request(
    body: serde_json::Value,
    audio: AudioInput,
    state: SharedMistralRsState,
    tx: Sender<Response>,
) -> Result<Request> {
    let Request::Normal(mut request) = parse_audio_chat_request(body, state.clone(), tx).await?
    else {
        unreachable!()
    };
    let prefix = match state.get_model_category(request.model_id.as_deref())? {
        ModelCategory::Multimodal { prefixer, .. } => prefixer.prefix_audio(vec![0], ""),
        _ => String::new(),
    };
    request.messages = transcript_messages(request.messages, audio, prefix);
    Ok(Request::Normal(request))
}

fn transcript_messages(
    messages: RequestMessage,
    audio: AudioInput,
    prefix: String,
) -> RequestMessage {
    let RequestMessage::Chat {
        mut messages,
        enable_thinking,
        reasoning_effort,
    } = messages
    else {
        unreachable!()
    };
    messages
        .last_mut()
        .expect("transcription user message")
        .insert(
            "content".to_string(),
            Either::Right(vec![
                indexmap::IndexMap::from([("type".to_string(), serde_json::json!("audio"))]),
                indexmap::IndexMap::from([
                    ("type".to_string(), serde_json::json!("text")),
                    ("text".to_string(), serde_json::json!(prefix)),
                ]),
            ]),
        );
    RequestMessage::MultimodalChat {
        messages,
        images: Vec::new(),
        audios: vec![audio],
        videos: Vec::new(),
        enable_thinking,
        reasoning_effort,
    }
}

async fn parse_upload(mut multipart: Multipart, translate: bool) -> Result<TranscriptionUpload> {
    let mut upload = TranscriptionUpload::default();
    let mut seen = HashSet::new();
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field
            .name()
            .context("Each multipart field requires a name")?
            .to_string();
        anyhow::ensure!(
            seen.insert(name.clone()),
            "Duplicate audio parameter: {name}"
        );
        if name == "file" {
            upload.file = Some(field.bytes().await.map_err(multipart_error)?.to_vec());
        } else {
            upload.set_field(
                &name,
                field.text().await.map_err(multipart_error)?,
                translate,
            )?;
        }
    }
    Ok(upload)
}

fn multipart_error(error: axum::extract::multipart::MultipartError) -> ApiError {
    ApiError::from_status(error.status(), error.body_text())
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/transcriptions",
    request_body(content = AudioTranscriptionRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "Transcript as JSON, text, or streaming events", content(
            (AudioTranscriptionResponse = "application/json"),
            (String = "text/plain"),
            (TranscriptionStreamEvent = "text/event-stream")
        )),
        (status = 400, description = "Invalid audio or unsupported parameters")
    )
)]
pub async fn audio_transcriptions(
    State(state): State<SharedMistralRsState>,
    payload: Result<Multipart, MultipartRejection>,
) -> HttpResponse {
    transcribe(state, payload, false).await
}

#[utoipa::path(
    post,
    tag = "Mistral.rs",
    path = "/v1/audio/translations",
    request_body(content = AudioTranslationRequest, content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "English translation as JSON or text", content(
            (AudioTranscriptionResponse = "application/json"),
            (String = "text/plain")
        )),
        (status = 400, description = "Invalid audio or unsupported parameters")
    )
)]
pub async fn audio_translations(
    State(state): State<SharedMistralRsState>,
    payload: Result<Multipart, MultipartRejection>,
) -> HttpResponse {
    transcribe(state, payload, true).await
}

async fn transcribe(
    state: SharedMistralRsState,
    payload: Result<Multipart, MultipartRejection>,
    translate: bool,
) -> HttpResponse {
    let multipart = match payload {
        Ok(multipart) => multipart,
        Err(error) => {
            return openai_error_response(ApiError::from_status(error.status(), error.body_text()))
        }
    };
    let (tx, mut rx) = create_response_channel(None);
    let parsed = async {
        let mut upload = parse_upload(multipart, translate).await?;
        let body = upload.chat_body(translate)?;
        let model = upload.model.as_ref().expect("validated model");
        validate_model_name(model, state.clone())?;
        let config = state
            .config((model != "default").then_some(model.as_str()))
            .map_err(anyhow::Error::msg)?;
        anyhow::ensure!(
            config.modalities.input.contains(&SupportedModality::Audio),
            "The requested model does not support audio input"
        );
        let file = upload.file.take().expect("validated audio file");
        let audio = tokio::task::spawn_blocking(move || AudioInput::from_bytes(&file))
            .await?
            .context("Could not decode the uploaded audio")?;
        let request = parse_transcription_request(body, audio, state.clone(), tx).await?;
        Ok::<_, anyhow::Error>((request, upload.format, upload.stream))
    }
    .await;
    let (request, format, streaming) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => return openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest),
    };
    if let Err(error) = send_request(&state, request).await {
        return openai_error_from_error(&error, ApiErrorKind::Internal);
    }
    if streaming {
        return transcription_stream(rx, state);
    }
    base_process_non_streaming_response(
        &mut rx,
        state,
        |state, response| match response {
            Response::Done(response) => {
                MistralRs::maybe_log_response(state, &response);
                match response.choices.first() {
                    Some(choice) if choice.finish_reason == "length" => {
                        openai_error_response(ApiError::invalid_request(
                            "Transcription reached the model context limit before finishing",
                        ))
                    }
                    Some(choice) => transcript_response(
                        choice.message.content.clone().unwrap_or_default(),
                        format,
                    ),
                    None => openai_error_response(ApiError::model_error()),
                }
            }
            Response::ValidationError(error) => {
                openai_error_from_error(error.as_ref(), ApiErrorKind::InvalidRequest)
            }
            Response::InternalError(error) => {
                MistralRs::maybe_log_error(state, error.as_ref());
                openai_error_from_error(error.as_ref(), ApiErrorKind::Internal)
            }
            Response::ModelError(message, _) => {
                MistralRs::maybe_log_error(state, &ModelErrorMessage(message));
                openai_error_response(ApiError::model_error())
            }
            _ => openai_error_response(ApiError::model_error()),
        },
        |state, error| {
            MistralRs::maybe_log_error(state, error.as_ref());
            openai_error_from_error(error.as_ref(), ApiErrorKind::Internal)
        },
    )
    .await
}

#[derive(Default)]
struct TranscriptEvents {
    text: String,
    pending: VecDeque<Event>,
    finished: bool,
}

impl TranscriptEvents {
    fn push(&mut self, delta: Option<String>, finish_reason: Option<String>) {
        if let Some(delta) = delta.filter(|delta| !delta.is_empty()) {
            self.text.push_str(&delta);
            self.pending.push_back(
                Event::default()
                    .json_data(TranscriptionStreamEvent::Delta { delta })
                    .expect("JSON event serialization"),
            );
        }
        if let Some(reason) = finish_reason {
            self.finished = true;
            let event = if reason == "length" {
                openai_error_event(ApiError::invalid_request(
                    "Transcription reached the model context limit before finishing",
                ))
            } else {
                Event::default()
                    .json_data(TranscriptionStreamEvent::Done {
                        text: self.text.clone(),
                    })
                    .expect("JSON event serialization")
            };
            self.pending.push_back(event);
        }
    }

    fn error(&mut self, error: ApiError) {
        self.finished = true;
        self.pending.push_back(openai_error_event(error));
    }
}

fn transcription_stream(rx: Receiver<Response>, state: SharedMistralRsState) -> HttpResponse {
    let stream = futures::stream::unfold(
        (rx, TranscriptEvents::default(), state),
        |(mut rx, mut events, state)| async move {
            loop {
                if let Some(event) = events.pending.pop_front() {
                    return Some((Ok::<_, Infallible>(event), (rx, events, state)));
                }
                if events.finished {
                    return None;
                }
                match rx.recv().await {
                    Some(Response::Chunk(chunk)) => {
                        for choice in chunk.choices {
                            events.push(choice.delta.content, choice.finish_reason);
                        }
                    }
                    Some(Response::ValidationError(error)) => {
                        events.error(ApiError::invalid_request(error.to_string()))
                    }
                    Some(Response::InternalError(error)) => {
                        MistralRs::maybe_log_error(state.clone(), error.as_ref());
                        events.error(ApiError::internal());
                    }
                    Some(Response::ModelError(message, _)) => {
                        MistralRs::maybe_log_error(state.clone(), &ModelErrorMessage(message));
                        events.error(ApiError::model_error());
                    }
                    Some(
                        Response::AgenticToolCallProgress { .. }
                        | Response::BlockDenoisingProgress(_)
                        | Response::AgenticToolApprovalRequired { .. }
                        | Response::File(_),
                    ) => {}
                    _ => events.error(ApiError::internal()),
                }
            }
        },
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn transcript_response(text: String, format: TranscriptionResponseFormat) -> HttpResponse {
    match format {
        TranscriptionResponseFormat::Json => {
            Json(AudioTranscriptionResponse { text }).into_response()
        }
        TranscriptionResponseFormat::Text => text.into_response(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn streamed_transcript_has_deltas_and_complete_text() -> Result<()> {
        let mut events = TranscriptEvents::default();
        events.push(Some("Hello".to_string()), None);
        events.push(Some(".".to_string()), Some("stop".to_string()));
        assert!(events.finished);
        let response = Sse::new(futures::stream::iter(
            events.pending.into_iter().map(Ok::<_, Infallible>),
        ))
        .into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        let text = std::str::from_utf8(&body)?;
        let values = text
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .map(serde_json::from_str::<serde_json::Value>)
            .collect::<std::result::Result<Vec<_>, _>>()?;
        assert_eq!(
            values[0],
            serde_json::json!({"type":"transcript.text.delta", "delta":"Hello"})
        );
        assert_eq!(
            values[2],
            serde_json::json!({"type":"transcript.text.done", "text":"Hello."})
        );
        let mut limited = TranscriptEvents::default();
        limited.push(None, Some("length".to_string()));
        let response = Sse::new(futures::stream::iter(
            limited.pending.into_iter().map(Ok::<_, Infallible>),
        ))
        .into_response();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX).await?;
        assert!(!std::str::from_utf8(&body)?.contains("transcript.text.done"));
        Ok(())
    }

    #[tokio::test]
    async fn multipart_upload_requires_unique_fields_and_preserves_file_bytes() -> Result<()> {
        use axum::{body::Body, extract::FromRequest};
        for duplicate in [false, true] {
            let mut body = b"--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\ndefault\r\n--test\r\nContent-Disposition: form-data; name=\"file\"; filename=\"speech.wav\"\r\nContent-Type: audio/wav\r\n\r\n".to_vec();
            body.extend_from_slice(&[0, 1, 2, 255]);
            body.extend_from_slice(b"\r\n");
            if duplicate {
                body.extend_from_slice(
                    b"--test\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nsecond\r\n",
                );
            }
            body.extend_from_slice(b"--test--\r\n");
            let request = axum::http::Request::builder()
                .header("content-type", "multipart/form-data; boundary=test")
                .body(Body::from(body))?;
            let multipart = Multipart::from_request(request, &()).await?;
            let parsed = parse_upload(multipart, false).await;
            if duplicate {
                assert!(parsed.is_err());
            } else {
                let parsed = parsed?;
                assert_eq!(parsed.model.as_deref(), Some("default"));
                assert_eq!(parsed.file.as_deref(), Some([0, 1, 2, 255].as_slice()));
            }
        }
        Ok(())
    }
}
