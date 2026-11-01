use std::{process::Output, sync::Arc, time::Duration};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use tokio::sync::{mpsc, Semaphore};

use crate::{ChatAudioFormat, NormalRequest, Response};

mod encoding;
pub use encoding::{decode_generated_wav, encode_audio_pcm, encode_audio_pcm_with_speed};
use encoding::{finish_ffmpeg_output, prepare_ffmpeg_encoding};

const MAX_PENDING_AUDIO_ENCODINGS: usize = 32;
const MAX_AUDIO_ENCODING_WORKERS: usize = 2;
const AUDIO_ENCODING_TIMEOUT: Duration = Duration::from_secs(120);
const AUDIO_RESPONSE_BUFFER: usize = 1;
static PENDING_ENCODINGS: Semaphore = Semaphore::const_new(MAX_PENDING_AUDIO_ENCODINGS);
static ENCODING_JOBS: Semaphore = Semaphore::const_new(MAX_PENDING_AUDIO_ENCODINGS);
static ENCODING_WORKERS: Semaphore = Semaphore::const_new(MAX_AUDIO_ENCODING_WORKERS);

pub(crate) fn prepare_response(request: &mut NormalRequest) -> Result<()> {
    let Some(output) = &mut request.audio_output else {
        return Ok(());
    };
    if matches!(output.format, ChatAudioFormat::Wav | ChatAudioFormat::Pcm16) {
        return Ok(());
    }
    let permit = PENDING_ENCODINGS
        .try_acquire()
        .context("Audio encoding queue is full; retry after an earlier request finishes")?;
    let format = output.format;
    output.format = ChatAudioFormat::Wav;
    let (tx, mut rx) = mpsc::channel(AUDIO_RESPONSE_BUFFER);
    let target = std::mem::replace(&mut request.response, tx);
    tokio::spawn(async move {
        let _permit = permit;
        loop {
            let Some(response) = (tokio::select! {
                _ = target.closed() => return,
                response = rx.recv() => response,
            }) else {
                return;
            };
            let terminal = matches!(
                response,
                Response::Done(_)
                    | Response::InternalError(_)
                    | Response::ValidationError(_)
                    | Response::ModelError(_, _)
            );
            let response = tokio::select! {
                _ = target.closed() => return,
                result = encode_response(response, format) => match result {
                    Ok(response) => response,
                    Err(error) => {
                        let _ = target.send(Response::InternalError(error.into())).await;
                        return;
                    }
                },
            };
            if target.send(response).await.is_err() || terminal {
                return;
            }
        }
    });
    Ok(())
}

async fn encode_response(mut response: Response, format: ChatAudioFormat) -> Result<Response> {
    if let Response::Done(completion) = &mut response {
        for choice in &mut completion.choices {
            let Some(audio) = &mut choice.message.audio else {
                continue;
            };
            let Some(wav) = audio.data.take() else {
                continue;
            };
            let decoded =
                tokio::task::spawn_blocking(move || decode_generated_wav(&STANDARD.decode(wav)?))
                    .await??;
            let encoded = encode_audio_pcm_with_speed_async(
                Arc::new(decoded.samples),
                decoded.sample_rate,
                decoded.channels,
                format,
                1.0,
            )
            .await?;
            audio.data = Some(STANDARD.encode(encoded));
        }
    }
    Ok(response)
}

/// Encodes audio outside inference with bounded workers and cancellable FFmpeg processes.
pub async fn encode_audio_pcm_with_speed_async(
    samples: Arc<Vec<f32>>,
    sample_rate: u32,
    channels: u16,
    format: ChatAudioFormat,
    speed: f64,
) -> Result<Vec<u8>> {
    let job = ENCODING_JOBS
        .try_acquire()
        .context("Audio encoding queue is full; retry after an earlier request finishes")?;
    let permit = ENCODING_WORKERS.acquire().await?;
    if speed == 1.0 && matches!(format, ChatAudioFormat::Wav | ChatAudioFormat::Pcm16) {
        return tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _job = job;
            crate::encode_audio_pcm(&samples, sample_rate, channels, format)
        })
        .await?;
    }
    let (permit, job, _input, command) = tokio::task::spawn_blocking(move || {
        let (input, command) =
            prepare_ffmpeg_encoding(&samples, sample_rate, channels, format, speed)?;
        Ok::<_, anyhow::Error>((permit, job, input, command))
    })
    .await??;
    let output = run_audio_command(command, AUDIO_ENCODING_TIMEOUT).await?;
    if format == ChatAudioFormat::Wav {
        return tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let _job = job;
            finish_ffmpeg_output(output, sample_rate, channels, format)
        })
        .await?;
    }
    finish_ffmpeg_output(output, sample_rate, channels, format)
}

async fn run_audio_command(command: std::process::Command, timeout: Duration) -> Result<Output> {
    let mut command = tokio::process::Command::from(command);
    command.kill_on_drop(true);
    tokio::time::timeout(timeout, command.output())
        .await
        .context("Audio encoding exceeded its time limit")?
        .context("Compressed audio and speed adjustment require FFmpeg on PATH")
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_RATE: u32 = 24_000;
    const TEST_TIMEOUT: Duration = Duration::from_secs(2);
    #[cfg(unix)]
    const TEST_POLL: Duration = Duration::from_millis(10);
    #[cfg(unix)]
    const TEST_PROCESS_TIMEOUT: Duration = Duration::from_millis(100);
    const TEST_TONE_HZ: f32 = 440.0;
    const TEST_AMPLITUDE: f32 = 0.5;

    #[tokio::test]
    async fn compressed_response_defers_encoding_and_cancels_generation_on_disconnect() -> Result<()>
    {
        let (target, mut responses) = mpsc::channel(1);
        let mut request = NormalRequest::new_simple(
            crate::RequestMessage::Chat {
                messages: Vec::new(),
                enable_thinking: None,
                reasoning_effort: None,
            },
            crate::SamplingParams::deterministic(),
            target,
            0,
            None,
            None,
        );
        request.audio_output = Some(crate::ChatAudioRequest {
            mode: crate::ChatAudioMode::Audio,
            format: ChatAudioFormat::Mp3,
        });
        prepare_response(&mut request)?;
        assert_eq!(request.audio_output.unwrap().format, ChatAudioFormat::Wav);
        request
            .response
            .send(Response::ValidationError("invalid input".into()))
            .await?;
        assert!(matches!(
            responses.recv().await,
            Some(Response::ValidationError(_))
        ));
        tokio::time::timeout(TEST_TIMEOUT, request.response.closed()).await?;

        let (target, responses) = mpsc::channel(1);
        request.response = target;
        request.audio_output.as_mut().unwrap().format = ChatAudioFormat::Mp3;
        prepare_response(&mut request)?;
        drop(responses);
        tokio::time::timeout(TEST_TIMEOUT, request.response.closed()).await?;
        Ok(())
    }

    #[tokio::test]
    async fn native_audio_encoding_preserves_pcm_and_wav() -> Result<()> {
        let samples = Arc::new(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        for format in [ChatAudioFormat::Wav, ChatAudioFormat::Pcm16] {
            let expected = crate::encode_audio_pcm(&samples, TEST_RATE, 1, format)?;
            let actual =
                encode_audio_pcm_with_speed_async(samples.clone(), TEST_RATE, 1, format, 1.0)
                    .await?;
            assert_eq!(actual, expected);
        }
        Ok(())
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn dropped_or_timed_out_encoding_kills_the_child() -> Result<()> {
        for cancel in [true, false] {
            let dir = tempfile::tempdir()?;
            let pid_file = dir.path().join("pid");
            let mut command = std::process::Command::new("sh");
            command
                .args(["-c", "echo $$ > \"$1\"; exec sleep 30", "sh"])
                .arg(&pid_file);
            let timeout = if cancel {
                TEST_TIMEOUT
            } else {
                TEST_PROCESS_TIMEOUT
            };
            let task = tokio::spawn(run_audio_command(command, timeout));
            let pid = tokio::time::timeout(TEST_TIMEOUT, async {
                loop {
                    if let Ok(pid) = tokio::fs::read_to_string(&pid_file).await {
                        if !pid.trim().is_empty() {
                            break pid.trim().to_string();
                        }
                    }
                    tokio::time::sleep(TEST_POLL).await;
                }
            })
            .await?;
            if cancel {
                task.abort();
                assert!(task.await.unwrap_err().is_cancelled());
            } else {
                assert!(task.await?.is_err());
            }
            tokio::time::timeout(TEST_TIMEOUT, async {
                loop {
                    let status = tokio::process::Command::new("kill")
                        .args(["-0", &pid])
                        .stderr(std::process::Stdio::null())
                        .status()
                        .await
                        .unwrap();
                    if !status.success() {
                        break;
                    }
                    tokio::time::sleep(TEST_POLL).await;
                }
            })
            .await?;
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires FFmpeg with audio encoders and atempo"]
    #[allow(clippy::cast_precision_loss)] // One second of 24 kHz indices is exactly representable.
    async fn async_encoders_match_native_and_compressed_outputs() -> Result<()> {
        async fn decoded_pcm(bytes: Vec<u8>, format: ChatAudioFormat) -> Result<Vec<u8>> {
            if format == ChatAudioFormat::Pcm16 {
                return Ok(bytes);
            }
            let input = tempfile::NamedTempFile::new()?;
            tokio::fs::write(input.path(), bytes).await?;
            let mut command = std::process::Command::new("ffmpeg");
            command
                .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
                .arg(input.path())
                .args(["-f", "s16le", "-acodec", "pcm_s16le", "-ar"])
                .arg(TEST_RATE.to_string())
                .args(["-ac", "1", "pipe:1"]);
            let output = run_audio_command(command, TEST_TIMEOUT).await?;
            anyhow::ensure!(output.status.success(), "Could not decode {format:?}");
            Ok(output.stdout)
        }
        let samples = Arc::new(
            (0..TEST_RATE)
                .map(|index| {
                    (std::f32::consts::TAU * TEST_TONE_HZ * index as f32 / TEST_RATE as f32).sin()
                        * TEST_AMPLITUDE
                })
                .collect::<Vec<_>>(),
        );
        for format in [
            ChatAudioFormat::Wav,
            ChatAudioFormat::Pcm16,
            ChatAudioFormat::Mp3,
            ChatAudioFormat::Flac,
            ChatAudioFormat::Opus,
            ChatAudioFormat::Aac,
        ] {
            for speed in [1.0, 2.0] {
                let expected =
                    crate::encode_audio_pcm_with_speed(&samples, TEST_RATE, 1, format, speed)?;
                let actual =
                    encode_audio_pcm_with_speed_async(samples.clone(), TEST_RATE, 1, format, speed)
                        .await?;
                let actual = decoded_pcm(actual, format).await?;
                let expected = decoded_pcm(expected, format).await?;
                assert!(
                    actual == expected,
                    "{format:?} at speed {speed}: PCM differs"
                );
            }
        }
        Ok(())
    }
}
