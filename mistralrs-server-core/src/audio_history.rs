use std::{
    collections::{HashMap, HashSet},
    future::Future,
    io::{BufReader, Read, Seek, Write},
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use either::Either;
use mistralrs_core::{
    AudioInput, ChatAudioFormat, ChatAudioResponse, ChatCompletionChunkResponse, Response,
    MAX_INPUT_AUDIO_BYTES,
};
use serde_json::json;
use tokio::sync::{
    mpsc::{Receiver, Sender},
    OwnedSemaphorePermit, Semaphore,
};

use crate::{
    audio_transcription::parse_transcription_request,
    handler_core::{create_response_channel, send_request_with_model},
    openai::{Message, MessageContent},
    types::SharedMistralRsState,
};

const AUDIO_HISTORY_TTL_SECONDS: u64 = 3600;
const AUDIO_HISTORY_MAX_BYTES: usize = 16 * 1024 * 1024;
const AUDIO_HISTORY_MAX_ENTRIES: usize = 1024;
const PCM_SAMPLE_RATE: u32 = 24_000;
const PCM_CHANNELS: u16 = 1;
const PCM_SCALE: f32 = 32768.0;
const TRANSCRIPT_MAX_TOKENS: usize = 8192;
const TRANSCRIPT_CONCURRENCY: usize = 2;
const AUDIO_DECODE_TIMEOUT: Duration = Duration::from_secs(60);
const PCM_SAMPLE_BYTES: usize = std::mem::size_of::<i16>();
const PCM_READ_BUFFER_BYTES: usize = 8192;

struct HistoryEntry {
    transcript: String,
    expires_at: u64,
}

pub struct AudioHistoryStore {
    entries: Mutex<HashMap<String, HistoryEntry>>,
    transcription_slots: Arc<Semaphore>,
}

impl Default for AudioHistoryStore {
    fn default() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            transcription_slots: Arc::new(Semaphore::new(TRANSCRIPT_CONCURRENCY)),
        }
    }
}

impl AudioHistoryStore {
    fn insert(&self, id: String, transcript: String, expires_at: u64) {
        let mut entries = self.entries.lock().expect("audio history lock poisoned");
        let now = now();
        entries.retain(|_, entry| entry.expires_at > now);
        if transcript.len() > AUDIO_HISTORY_MAX_BYTES {
            return;
        }
        entries.remove(&id);
        let mut stored_bytes = entries
            .values()
            .map(|entry| entry.transcript.len())
            .sum::<usize>();
        while entries.len() >= AUDIO_HISTORY_MAX_ENTRIES
            || stored_bytes + transcript.len() > AUDIO_HISTORY_MAX_BYTES
        {
            let oldest = entries
                .iter()
                .min_by_key(|(_, entry)| entry.expires_at)
                .map(|(id, _)| id.clone());
            if let Some(entry) = oldest.and_then(|id| entries.remove(&id)) {
                stored_bytes -= entry.transcript.len();
            } else {
                break;
            }
        }
        entries.insert(
            id,
            HistoryEntry {
                transcript,
                expires_at,
            },
        );
    }

    pub(crate) fn resolve_messages(
        &self,
        messages: &mut Either<Vec<Message>, String>,
    ) -> Result<()> {
        let Either::Left(messages) = messages else {
            return Ok(());
        };
        let entries = self.entries.lock().expect("audio history lock poisoned");
        for message in messages {
            let Some(audio) = message.audio.take() else {
                continue;
            };
            anyhow::ensure!(
                message.role == "assistant",
                "audio.id is only valid on assistant messages"
            );
            let entry = entries
                .get(&audio.id)
                .filter(|entry| entry.expires_at > now())
                .context(
                    "Unknown or expired audio.id; resend the assistant transcript as content",
                )?;
            if message.content.is_none() {
                message.content = Some(MessageContent::from_text(entry.transcript.clone()));
            }
        }
        Ok(())
    }
}

fn now() -> u64 {
    chrono::Utc::now().timestamp() as u64
}

struct AudioSpool {
    file: tempfile::NamedTempFile,
    len: usize,
}

impl AudioSpool {
    fn append(spool: Option<Self>, data: &str, limit: usize) -> Result<Self> {
        let mut spool = match spool {
            Some(spool) => spool,
            None => Self {
                file: tempfile::NamedTempFile::new()?,
                len: 0,
            },
        };
        let remaining = limit.saturating_sub(spool.len);
        anyhow::ensure!(
            data.len() <= remaining.div_ceil(3) * 4,
            "Generated audio transcript capture exceeds the {limit} byte limit"
        );
        let bytes = STANDARD.decode(data)?;
        anyhow::ensure!(
            bytes.len() <= remaining,
            "Generated audio transcript capture exceeds the {limit} byte limit"
        );
        spool.file.write_all(&bytes)?;
        spool.len += bytes.len();
        Ok(spool)
    }
}

#[derive(Default)]
struct AudioCapture {
    id: Option<String>,
    spool: Option<AudioSpool>,
    capture_failed: bool,
    text: String,
    transcript: String,
}

impl AudioCapture {
    fn add_text(&mut self, text: &str) {
        self.text.push_str(text);
        if !self.text.is_empty() {
            self.spool = None;
        }
    }

    async fn add_audio(&mut self, audio: &mut ChatAudioResponse) {
        let id = self.id.get_or_insert_with(|| {
            audio
                .id
                .clone()
                .unwrap_or_else(|| format!("audio_{}", uuid::Uuid::new_v4().simple()))
        });
        audio.id = Some(id.clone());
        if let Some(transcript) = &audio.transcript {
            self.transcript.push_str(transcript);
        }
        if !self.text.is_empty() || !self.transcript.is_empty() {
            self.spool = None;
        } else if !self.capture_failed {
            if let Some(data) = &audio.data {
                let data = data.clone();
                let spool = self.spool.take();
                let captured = tokio::task::spawn_blocking(move || {
                    AudioSpool::append(spool, &data, MAX_INPUT_AUDIO_BYTES)
                })
                .await
                .map_err(anyhow::Error::from)
                .flatten();
                match captured {
                    Ok(spool) => self.spool = Some(spool),
                    Err(error) => {
                        self.capture_failed = true;
                        tracing::warn!(audio_id = ?self.id, %error, "Could not capture generated audio transcript");
                    }
                }
            }
        }
    }

    fn expire(&self, audio: &mut ChatAudioResponse) {
        audio.transcript = Some(String::new());
        audio.expires_at = Some(now());
    }

    async fn finish(&mut self, audio: &mut ChatAudioResponse, ctx: &AudioResponseContext) {
        self.add_audio(audio).await;
        let transcript = if !self.transcript.is_empty() {
            self.transcript.clone()
        } else if !self.text.is_empty() {
            self.text.clone()
        } else if self.capture_failed {
            self.expire(audio);
            return;
        } else if let Some(spool) = self.spool.take().filter(|spool| spool.len > 0) {
            match transcribe(spool, ctx).await {
                Ok(transcript) => transcript,
                Err(error) => {
                    tracing::warn!(audio_id = ?self.id, %error, "Could not retain generated audio transcript");
                    self.expire(audio);
                    return;
                }
            }
        } else {
            String::new()
        };
        let expires_at = now() + AUDIO_HISTORY_TTL_SECONDS;
        audio.expires_at = Some(expires_at);
        audio.transcript = Some(transcript.clone());
        ctx.store.insert(
            audio.id.clone().expect("audio id assigned"),
            transcript,
            expires_at,
        );
    }
}

pub(crate) struct AudioResponseOptions {
    pub format: ChatAudioFormat,
    pub choices: usize,
}

struct StreamingAudioCapture {
    captures: HashMap<usize, AudioCapture>,
    finished: HashSet<usize>,
    final_chunk: Option<ChatCompletionChunkResponse>,
    choices: usize,
}

impl StreamingAudioCapture {
    fn new(choices: usize) -> Self {
        Self {
            captures: HashMap::new(),
            finished: HashSet::new(),
            final_chunk: None,
            choices,
        }
    }

    async fn stage(
        &mut self,
        mut chunk: ChatCompletionChunkResponse,
    ) -> Vec<ChatCompletionChunkResponse> {
        let choices = std::mem::take(&mut chunk.choices);
        for mut choice in choices {
            let capture = self.captures.entry(choice.index).or_default();
            if let Some(text) = &choice.delta.content {
                capture.add_text(text);
            }
            if choice.finish_reason.is_some() {
                self.finished.insert(choice.index);
                self.final_chunk
                    .get_or_insert_with(|| {
                        let mut final_chunk = chunk.clone();
                        final_chunk.choices.clear();
                        final_chunk
                    })
                    .choices
                    .push(choice);
            } else {
                if let Some(audio) = &mut choice.delta.audio {
                    capture.add_audio(audio).await;
                }
                chunk.choices.push(choice);
            }
        }
        if let Some(final_chunk) = &mut self.final_chunk {
            if chunk.usage.is_some() {
                final_chunk.usage = chunk.usage.take();
            }
            if chunk.session_id.is_some() {
                final_chunk.session_id = chunk.session_id.take();
            }
        }
        let mut output = Vec::new();
        if !chunk.choices.is_empty() {
            output.push(chunk);
        }
        // Original generation must finish before ASR uses the same engine; terminal choices also end the SSE stream.
        if self.finished.len() == self.choices {
            if let Some(mut final_chunk) = self.final_chunk.take() {
                final_chunk.choices.sort_by_key(|choice| choice.index);
                output.push(final_chunk);
            }
        }
        output
    }
}

struct AudioResponseContext {
    state: SharedMistralRsState,
    model: Option<String>,
    format: ChatAudioFormat,
    store: Arc<AudioHistoryStore>,
}

async fn while_connected<T>(tx: &Sender<Response>, future: impl Future<Output = T>) -> Option<T> {
    tokio::select! {
        _ = tx.closed() => None,
        result = future => Some(result),
    }
}

pub(crate) fn decorate_responses(
    mut rx: Receiver<Response>,
    state: SharedMistralRsState,
    model: Option<String>,
    options: AudioResponseOptions,
    store: Arc<AudioHistoryStore>,
) -> Receiver<Response> {
    let (tx, output) = create_response_channel(None);
    tokio::spawn(async move {
        let ctx = AudioResponseContext {
            state,
            model,
            format: options.format,
            store,
        };
        let mut streaming = StreamingAudioCapture::new(options.choices);
        loop {
            let Some(response) = (tokio::select! {
                _ = tx.closed() => return,
                response = rx.recv() => response,
            }) else {
                break;
            };
            let responses = match response {
                Response::Chunk(chunk) => {
                    let Some(chunks) = while_connected(&tx, streaming.stage(chunk)).await else {
                        return;
                    };
                    chunks.into_iter().map(Response::Chunk).collect::<Vec<_>>()
                }
                response => vec![response],
            };
            for mut response in responses {
                let enrich = async {
                    match &mut response {
                        Response::Done(response) => {
                            for choice in &mut response.choices {
                                if let Some(audio) = &mut choice.message.audio {
                                    let mut capture = AudioCapture {
                                        text: choice.message.content.clone().unwrap_or_default(),
                                        ..Default::default()
                                    };
                                    capture.finish(audio, &ctx).await;
                                }
                            }
                        }
                        Response::Chunk(response) => {
                            for choice in &mut response.choices {
                                if choice.finish_reason.is_some() {
                                    let mut capture = streaming
                                        .captures
                                        .remove(&choice.index)
                                        .unwrap_or_default();
                                    if capture.id.is_some() || choice.delta.audio.is_some() {
                                        let audio =
                                            choice.delta.audio.get_or_insert_with(Default::default);
                                        capture.finish(audio, &ctx).await;
                                    }
                                }
                            }
                        }
                        _ => (),
                    }
                };
                if while_connected(&tx, enrich).await.is_none() {
                    return;
                }
                if tx.send(response).await.is_err() {
                    return;
                }
            }
        }
    });
    output
}

async fn transcribe(spool: AudioSpool, ctx: &AudioResponseContext) -> Result<String> {
    let permit = Arc::new(
        ctx.store
            .transcription_slots
            .clone()
            .acquire_owned()
            .await?,
    );
    let audio = decode_spool(spool, ctx.format, permit.clone()).await?;
    let body = json!({
        "model": ctx.model.as_deref().unwrap_or("default"),
        "messages": [
            {"role":"system", "content":"Transcribe the spoken words exactly. Output only the transcription, in the original language."},
            {"role":"user", "content":""}
        ],
        "max_tokens":TRANSCRIPT_MAX_TOKENS,
        "temperature":0,
        "enable_thinking":false
    });
    let (tx, mut rx) = create_response_channel(None);
    let request = parse_transcription_request(body, audio, ctx.state.clone(), tx).await?;
    send_request_with_model(&ctx.state, request, ctx.model.as_deref()).await?;
    while let Some(response) = rx.recv().await {
        match response {
            Response::Done(response) => {
                let choice = response
                    .choices
                    .into_iter()
                    .next()
                    .context("Missing audio transcript")?;
                anyhow::ensure!(
                    choice.finish_reason != "length",
                    "Audio transcript exceeded the generation limit"
                );
                return choice.message.content.context("Missing audio transcript");
            }
            Response::InternalError(error) | Response::ValidationError(error) => {
                anyhow::bail!(error.to_string())
            }
            Response::ModelError(error, _) => anyhow::bail!(error),
            _ => (),
        }
    }
    anyhow::bail!("Audio transcription ended without a response")
}

async fn decode_spool(
    spool: AudioSpool,
    format: ChatAudioFormat,
    permit: Arc<OwnedSemaphorePermit>,
) -> Result<AudioInput> {
    if matches!(format, ChatAudioFormat::Pcm16 | ChatAudioFormat::Wav) {
        return tokio::task::spawn_blocking(move || {
            let _permit = permit;
            decode_audio_file(spool.file, format)
        })
        .await?;
    }
    let output = tokio::task::spawn_blocking(tempfile::NamedTempFile::new).await??;
    let mut command = tokio::process::Command::new("ffmpeg");
    command
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i"])
        .arg(spool.file.path())
        .args(["-f", "s16le", "-acodec", "pcm_s16le", "-ar"])
        .arg(PCM_SAMPLE_RATE.to_string())
        .arg("-ac")
        .arg(PCM_CHANNELS.to_string())
        .arg("-fs")
        .arg((MAX_INPUT_AUDIO_BYTES + PCM_SAMPLE_BYTES).to_string())
        .arg(output.path())
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .kill_on_drop(true);
    let result = tokio::time::timeout(AUDIO_DECODE_TIMEOUT, command.output())
        .await
        .context("Generated audio decoding timed out")?
        .context("FFmpeg is required to transcribe compressed speech output")?;
    anyhow::ensure!(
        result.status.success(),
        "FFmpeg could not decode generated speech: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    tokio::task::spawn_blocking(move || {
        let _permit = permit;
        decode_audio_file(output, ChatAudioFormat::Pcm16)
    })
    .await?
}

fn decode_audio_file(
    mut file: tempfile::NamedTempFile,
    format: ChatAudioFormat,
) -> Result<AudioInput> {
    file.rewind()?;
    let len = usize::try_from(file.as_file().metadata()?.len())?;
    anyhow::ensure!(
        len <= MAX_INPUT_AUDIO_BYTES,
        "Decoded audio exceeds the {MAX_INPUT_AUDIO_BYTES} byte transcript limit"
    );
    if format == ChatAudioFormat::Wav {
        let mut reader = hound::WavReader::new(BufReader::new(file.as_file_mut()))?;
        let spec = reader.spec();
        anyhow::ensure!(
            spec.bits_per_sample == 16 && spec.sample_format == hound::SampleFormat::Int,
            "Generated WAV must use PCM16 samples"
        );
        anyhow::ensure!(
            reader.len() as usize <= MAX_INPUT_AUDIO_BYTES / PCM_SAMPLE_BYTES,
            "Decoded audio exceeds the {MAX_INPUT_AUDIO_BYTES} byte transcript limit"
        );
        let samples = reader
            .samples::<i16>()
            .map(|sample| sample.map(|sample| f32::from(sample) / PCM_SCALE))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        return Ok(AudioInput {
            samples,
            sample_rate: spec.sample_rate,
            channels: spec.channels,
        });
    }
    anyhow::ensure!(
        len.is_multiple_of(PCM_SAMPLE_BYTES),
        "Generated PCM16 has an incomplete sample"
    );
    let mut reader = BufReader::new(file.as_file_mut());
    let mut samples = Vec::with_capacity(len / PCM_SAMPLE_BYTES);
    let mut bytes = [0; PCM_READ_BUFFER_BYTES];
    let mut remaining = len;
    while remaining > 0 {
        let count = remaining.min(bytes.len());
        reader.read_exact(&mut bytes[..count])?;
        samples.extend(
            bytes[..count]
                .as_chunks::<PCM_SAMPLE_BYTES>()
                .0
                .iter()
                .map(|value| f32::from(i16::from_le_bytes(*value)) / PCM_SCALE),
        );
        remaining -= count;
    }
    Ok(AudioInput {
        samples,
        sample_rate: PCM_SAMPLE_RATE,
        channels: PCM_CHANNELS,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn captured_bytes(capture: &AudioCapture) -> Result<Vec<u8>> {
        Ok(std::fs::read(capture.spool.as_ref().unwrap().file.path())?)
    }

    fn audio_choice(index: usize, finished: bool, data: &[u8]) -> mistralrs_core::ChunkChoice {
        mistralrs_core::ChunkChoice {
            index,
            finish_reason: finished.then(|| "stop".into()),
            stop_sequence: None,
            logprobs: None,
            delta: mistralrs_core::Delta {
                content: None,
                role: "assistant".into(),
                tool_calls: None,
                reasoning_content: None,
                audio: Some(ChatAudioResponse {
                    id: Some(format!("audio_{index}")),
                    data: Some(STANDARD.encode(data)),
                    ..Default::default()
                }),
            },
        }
    }

    fn audio_chunk(choices: Vec<mistralrs_core::ChunkChoice>) -> ChatCompletionChunkResponse {
        ChatCompletionChunkResponse {
            id: "test".into(),
            choices,
            created: 0,
            model: "test".into(),
            system_fingerprint: "test".into(),
            object: "chat.completion.chunk".into(),
            usage: None,
            adapter_generation: None,
            session_id: None,
        }
    }

    #[tokio::test]
    async fn streaming_choices_finish_together_without_blocking_other_audio() -> Result<()> {
        let mut stream = StreamingAudioCapture::new(2);
        let first = stream
            .stage(audio_chunk(vec![audio_choice(0, false, &[1, 2])]))
            .await;
        assert_eq!(first.len(), 1);
        assert!(first[0].choices[0].finish_reason.is_none());

        let middle = stream
            .stage(audio_chunk(vec![
                audio_choice(1, false, &[5, 6]),
                audio_choice(0, true, &[3, 4]),
            ]))
            .await;
        assert_eq!(middle.len(), 1);
        assert_eq!(middle[0].choices.len(), 1);
        assert_eq!(middle[0].choices[0].index, 1);
        assert!(middle[0].choices[0].finish_reason.is_none());
        assert_eq!(captured_bytes(&stream.captures[&0])?, [1, 2]);
        assert_eq!(captured_bytes(&stream.captures[&1])?, [5, 6]);

        let mut last = audio_chunk(vec![audio_choice(1, true, &[7, 8])]);
        last.session_id = Some("session".into());
        let mut finished = stream.stage(last).await;
        assert_eq!(finished.len(), 1);
        let final_chunk = &mut finished[0];
        assert_eq!(final_chunk.session_id.as_deref(), Some("session"));
        assert_eq!(final_chunk.choices.len(), 2);
        for (index, choice) in final_chunk.choices.iter_mut().enumerate() {
            assert_eq!(choice.index, index);
            assert!(choice.finish_reason.is_some());
            stream
                .captures
                .get_mut(&index)
                .unwrap()
                .add_audio(choice.delta.audio.as_mut().unwrap())
                .await;
        }
        assert_eq!(captured_bytes(&stream.captures[&0])?, [1, 2, 3, 4]);
        assert_eq!(captured_bytes(&stream.captures[&1])?, [5, 6, 7, 8]);
        Ok(())
    }

    #[tokio::test]
    async fn text_or_transcript_releases_unneeded_pcm_capture() -> Result<()> {
        for transcript in [false, true] {
            let mut capture = AudioCapture::default();
            let mut first = audio_choice(0, false, &[1, 2]).delta.audio.unwrap();
            capture.add_audio(&mut first).await;
            assert_eq!(capture.spool.as_ref().unwrap().len, 2);
            let path = capture.spool.as_ref().unwrap().file.path().to_path_buf();
            let mut next = audio_choice(0, false, &[3, 4]).delta.audio.unwrap();
            if transcript {
                next.transcript = Some("hello".into());
            } else {
                capture.add_text("hello");
            }
            capture.add_audio(&mut next).await;
            assert!(capture.spool.is_none());
            assert!(!path.exists());
            assert_eq!(next.data, Some(STANDARD.encode([3, 4])));
            capture.add_audio(&mut first).await;
            assert!(capture.spool.is_none());
        }
        Ok(())
    }

    #[test]
    fn assistant_audio_references_restore_transcripts_and_validate_expiry() -> Result<()> {
        let store = AudioHistoryStore::default();
        store.insert(
            "audio_valid".into(),
            "The answer is blue.".into(),
            now() + AUDIO_HISTORY_TTL_SECONDS,
        );
        store.insert("audio_expired".into(), "Old text".into(), now() - 1);
        let mut messages = Either::Left(serde_json::from_value(json!([
            {"role":"assistant", "audio":{"id":"audio_valid"}},
            {"role":"user", "content":"What color did you say?"}
        ]))?);
        store.resolve_messages(&mut messages)?;
        assert_eq!(
            messages.left().unwrap()[0]
                .content
                .as_ref()
                .unwrap()
                .as_ref()
                .left()
                .unwrap(),
            "The answer is blue."
        );
        for (role, id) in [
            ("user", "audio_valid"),
            ("assistant", "audio_expired"),
            ("assistant", "missing"),
        ] {
            let mut messages = Either::Left(serde_json::from_value(
                json!([{"role":role,"audio":{"id":id}}]),
            )?);
            assert!(store.resolve_messages(&mut messages).is_err());
        }
        Ok(())
    }

    #[tokio::test]
    async fn audio_chunks_preserve_bytes_samples_and_one_id() -> Result<()> {
        let samples = [0.0, 0.25, -0.5];
        for format in [ChatAudioFormat::Pcm16, ChatAudioFormat::Wav] {
            let bytes =
                mistralrs_core::encode_audio_pcm(&samples, PCM_SAMPLE_RATE, PCM_CHANNELS, format)?;
            if format == ChatAudioFormat::Pcm16 {
                assert_eq!(bytes, [0, 0, 0, 32, 0, 192]);
            }
            let mut capture = AudioCapture::default();
            let mut id = None;
            for bytes in bytes.chunks(bytes.len() / 2) {
                let mut chunk = ChatAudioResponse {
                    data: Some(STANDARD.encode(bytes)),
                    ..Default::default()
                };
                capture.add_audio(&mut chunk).await;
                if id.is_some() {
                    assert_eq!(chunk.id, id);
                }
                id = chunk.id;
            }
            assert!(id.is_some());
            assert_eq!(captured_bytes(&capture)?, bytes);
            let spool = capture.spool.take().unwrap();
            let path = spool.file.path().to_path_buf();
            let audio = decode_audio_file(spool.file, format)?;
            assert_eq!(audio.sample_rate, PCM_SAMPLE_RATE);
            assert_eq!(audio.channels, PCM_CHANNELS);
            assert_eq!(audio.samples, samples);
            assert!(!path.exists());
        }
        Ok(())
    }

    #[test]
    fn capture_limit_removes_spool_without_accepting_partial_audio() -> Result<()> {
        const LIMIT: usize = 4;
        let spool = AudioSpool::append(None, &STANDARD.encode([1, 2]), LIMIT)?;
        let path = spool.file.path().to_path_buf();
        let spool = AudioSpool::append(Some(spool), &STANDARD.encode([3, 4]), LIMIT)?;
        assert_eq!(std::fs::read(&path)?, [1, 2, 3, 4]);
        assert!(AudioSpool::append(Some(spool), &STANDARD.encode([5]), LIMIT).is_err());
        assert!(!path.exists());
        Ok(())
    }

    #[test]
    fn decoded_audio_limit_rejects_expansion_before_loading_samples() -> Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        file.as_file()
            .set_len((MAX_INPUT_AUDIO_BYTES + PCM_SAMPLE_BYTES) as u64)?;
        let path = file.path().to_path_buf();
        let result = decode_audio_file(file, ChatAudioFormat::Pcm16);
        assert!(result.unwrap_err().to_string().contains("transcript limit"));
        assert!(!path.exists());
        Ok(())
    }
}
