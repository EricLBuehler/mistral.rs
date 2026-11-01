use std::sync::Arc;

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use either::Either;
use indexmap::IndexMap;

use crate::{
    ChatAudioFormat, ChatAudioMode, ChatAudioRequest, NormalRequest, RequestMessage, Response,
};

const SPEECH_RESPONSE_CAPACITY: usize = 1;
const TTS_SYSTEM_PROMPT: &str = "Perform TTS in American English using a female voice.";

pub fn speech_task_prompt(instructions: Option<&str>) -> String {
    match instructions.filter(|text| !text.is_empty()) {
        Some(instructions) => format!("{TTS_SYSTEM_PROMPT}\n{instructions}"),
        None => TTS_SYSTEM_PROMPT.to_string(),
    }
}

pub(super) fn adapt_request(request: &mut NormalRequest) {
    let RequestMessage::SpeechGeneration { prompt } = &request.messages else {
        return;
    };
    request.messages = RequestMessage::Chat {
        messages: [
            ("system", speech_task_prompt(None)),
            ("user", prompt.clone()),
        ]
        .into_iter()
        .map(|(role, content)| {
            IndexMap::from([
                ("role".to_string(), Either::Left(role.to_string())),
                ("content".to_string(), Either::Left(content)),
            ])
        })
        .collect(),
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    request.audio_output = Some(ChatAudioRequest {
        mode: ChatAudioMode::Audio,
        format: ChatAudioFormat::Wav,
    });
    let destination = request.response.clone();
    let (sender, mut receiver) = tokio::sync::mpsc::channel(SPEECH_RESPONSE_CAPACITY);
    request.response = sender;
    tokio::spawn(async move {
        loop {
            let response = tokio::select! {
                _ = destination.closed() => return,
                response = receiver.recv() => response,
            };
            let Some(response) = response else { return };
            let terminal = matches!(
                response,
                Response::Done(_)
                    | Response::ValidationError(_)
                    | Response::InternalError(_)
                    | Response::ModelError(_, _)
            );
            let response = match response {
                Response::Done(response) => {
                    let decode = tokio::task::spawn_blocking(move || decode_speech(response));
                    let result = tokio::select! {
                        _ = destination.closed() => return,
                        result = decode => result,
                    };
                    match result {
                        Ok(Ok(response)) => response,
                        Ok(Err(error)) => Response::InternalError(error.into()),
                        Err(error) => Response::InternalError(error.into()),
                    }
                }
                response => response,
            };
            if destination.send(response).await.is_err() || terminal {
                return;
            }
        }
    });
}

fn decode_speech(response: crate::ChatCompletionResponse) -> Result<Response> {
    anyhow::ensure!(
        response.choices.len() == 1,
        "Speech generation requires exactly one choice"
    );
    let choice = response.choices.into_iter().next().expect("one choice");
    anyhow::ensure!(
        choice.finish_reason != "length",
        "Speech generation reached the model context limit before finishing"
    );
    let data = choice
        .message
        .audio
        .and_then(|audio| audio.data)
        .context("Model returned no speech audio")?;
    let audio = crate::audio_output::decode_generated_wav(&STANDARD.decode(data)?)?;
    Ok(Response::Speech {
        pcm: Arc::new(audio.samples),
        rate: audio.sample_rate as usize,
        channels: audio.channels as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    const TEST_TIMEOUT: Duration = Duration::from_secs(5);
    const TEST_RATE: u32 = 24_000;

    fn chat_speech_response(finish_reason: &str) -> Result<crate::ChatCompletionResponse> {
        Ok(crate::ChatCompletionResponse {
            id: "test".to_string(),
            choices: vec![crate::Choice {
                finish_reason: finish_reason.to_string(),
                stop_sequence: None,
                index: 0,
                message: crate::ResponseMessage {
                    audio: Some(crate::ChatAudioResponse::from_pcm(
                        &[0.0, 0.5, -0.5],
                        TEST_RATE,
                        ChatAudioFormat::Wav,
                    )?),
                    content: None,
                    role: "assistant".to_string(),
                    tool_calls: None,
                    reasoning_content: None,
                },
                logprobs: None,
            }],
            created: 0,
            model: "test".to_string(),
            system_fingerprint: "test".to_string(),
            object: "chat.completion".to_string(),
            usage: crate::Usage {
                completion_tokens: 1,
                prompt_tokens: 2,
                total_tokens: 3,
                prompt_tokens_details: None,
                avg_tok_per_sec: 0.0,
                avg_prompt_tok_per_sec: 0.0,
                avg_compl_tok_per_sec: 0.0,
                total_time_sec: 0.0,
                total_prompt_time_sec: 0.0,
                total_completion_time_sec: 0.0,
            },
            adapter_generation: None,
            agentic_tool_calls: None,
            files: None,
            session_id: None,
        })
    }

    #[tokio::test]
    async fn speech_task_becomes_audio_chat_and_preserves_errors() -> Result<()> {
        let (destination, mut responses) = tokio::sync::mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            RequestMessage::SpeechGeneration {
                prompt: "Hello.".to_string(),
            },
            crate::SamplingParams::deterministic(),
            destination,
            0,
            None,
            None,
        );
        adapt_request(&mut request);
        let RequestMessage::Chat {
            messages,
            enable_thinking,
            ..
        } = &request.messages
        else {
            panic!("speech task must use chat for an audio chat model");
        };
        assert_eq!(*enable_thinking, Some(false));
        assert_eq!(
            messages[0]["content"],
            Either::Left(speech_task_prompt(None))
        );
        assert_eq!(messages[1]["content"], Either::Left("Hello.".to_string()));
        assert_eq!(request.audio_output.unwrap().mode, ChatAudioMode::Audio);
        request
            .response
            .send(Response::ValidationError("test rejection".into()))
            .await?;
        assert!(matches!(
            tokio::time::timeout(TEST_TIMEOUT, responses.recv()).await?,
            Some(Response::ValidationError(_))
        ));
        drop(responses);
        tokio::time::timeout(TEST_TIMEOUT, request.response.closed()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn speech_adapter_returns_pcm_and_closes_on_cancellation() -> Result<()> {
        let (destination, mut responses) = tokio::sync::mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            RequestMessage::SpeechGeneration {
                prompt: "Hello.".to_string(),
            },
            crate::SamplingParams::deterministic(),
            destination,
            0,
            None,
            None,
        );
        adapt_request(&mut request);
        request
            .response
            .send(Response::Done(chat_speech_response("stop")?))
            .await?;
        let Some(Response::Speech {
            pcm,
            rate,
            channels,
        }) = tokio::time::timeout(TEST_TIMEOUT, responses.recv()).await?
        else {
            anyhow::bail!("Expected PCM speech response");
        };
        assert_eq!(rate, TEST_RATE as usize);
        assert_eq!(channels, 1);
        assert_eq!(pcm.as_ref(), &[0.0, 0.5, -0.5]);
        tokio::time::timeout(TEST_TIMEOUT, request.response.closed()).await?;

        let (destination, responses) = tokio::sync::mpsc::channel(1);
        request.messages = RequestMessage::SpeechGeneration {
            prompt: "Hello.".to_string(),
        };
        request.response = destination;
        adapt_request(&mut request);
        drop(responses);
        tokio::time::timeout(TEST_TIMEOUT, request.response.closed()).await?;
        Ok(())
    }

    #[test]
    fn speech_adapter_rejects_truncated_audio() -> Result<()> {
        assert!(decode_speech(chat_speech_response("length")?)
            .err()
            .expect("truncated speech must fail")
            .to_string()
            .contains("context limit"));
        assert_eq!(
            speech_task_prompt(Some("Speak slowly.")),
            format!("{TTS_SYSTEM_PROMPT}\nSpeak slowly.")
        );
        Ok(())
    }
}
