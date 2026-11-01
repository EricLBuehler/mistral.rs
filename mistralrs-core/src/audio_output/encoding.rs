use std::{
    io::Write,
    process::{Command, Output, Stdio},
};

use anyhow::Context;

use crate::{AudioInput, ChatAudioFormat};

const PCM16_MAX: f32 = i16::MAX as f32;
const PCM16_MIN: f32 = i16::MIN as f32;
const PCM16_SCALE: f32 = 32768.0;
const PCM16_BITS: u16 = 16;
const PCM_SAMPLE_BYTES: usize = std::mem::size_of::<i16>();
const MIN_AUDIO_SPEED: f64 = 0.25;
const MAX_AUDIO_SPEED: f64 = 4.0;
const MIN_TEMPO_STAGE: f64 = 0.5;
const MAX_TEMPO_STAGE: f64 = 2.0;

/// Decodes PCM16 WAV produced by the audio encoder without external-input size limits.
pub fn decode_generated_wav(wav: &[u8]) -> anyhow::Result<AudioInput> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(wav))?;
    let spec = reader.spec();
    let samples = reader
        .samples::<i16>()
        .map(|sample| sample.map(|sample| f32::from(sample) / PCM16_SCALE))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    Ok(AudioInput {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

/// Encodes interleaved PCM samples; compressed formats require FFmpeg on PATH.
pub fn encode_audio_pcm(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    format: ChatAudioFormat,
) -> anyhow::Result<Vec<u8>> {
    encode_audio_pcm_with_speed(samples, sample_rate, channels, format, 1.0)
}

/// Encodes interleaved PCM with pitch-preserving speed adjustment through FFmpeg when needed.
pub fn encode_audio_pcm_with_speed(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    format: ChatAudioFormat,
    speed: f64,
) -> anyhow::Result<Vec<u8>> {
    let pcm = quantize_audio(samples, sample_rate, channels, speed)?;
    if speed == 1.0 {
        match format {
            ChatAudioFormat::Pcm16 => {
                return Ok(pcm.iter().flat_map(|sample| sample.to_le_bytes()).collect());
            }
            ChatAudioFormat::Wav => return encode_pcm16_wav(pcm, sample_rate, channels),
            _ => {}
        }
    }
    let (_input, mut command) = ffmpeg_audio_command(&pcm, sample_rate, channels, format, speed)?;
    let output = command
        .output()
        .context("Compressed audio and speed adjustment require FFmpeg on PATH")?;
    finish_ffmpeg_output(output, sample_rate, channels, format)
}

fn encode_pcm16_wav(
    samples: impl IntoIterator<Item = i16>,
    sample_rate: u32,
    channels: u16,
) -> anyhow::Result<Vec<u8>> {
    let mut buffer = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(
        &mut buffer,
        hound::WavSpec {
            channels,
            sample_rate,
            bits_per_sample: PCM16_BITS,
            sample_format: hound::SampleFormat::Int,
        },
    )?;
    for sample in samples {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;
    Ok(buffer.into_inner())
}

pub(super) fn finish_ffmpeg_output(
    output: Output,
    sample_rate: u32,
    channels: u16,
    format: ChatAudioFormat,
) -> anyhow::Result<Vec<u8>> {
    anyhow::ensure!(
        output.status.success(),
        "FFmpeg could not encode {format:?} audio: {}",
        String::from_utf8_lossy(&output.stderr).trim()
    );
    if format == ChatAudioFormat::Wav {
        return encode_pcm16_wav(
            output
                .stdout
                .as_chunks::<PCM_SAMPLE_BYTES>()
                .0
                .iter()
                .map(|bytes| i16::from_le_bytes(*bytes)),
            sample_rate,
            channels,
        );
    }
    Ok(output.stdout)
}

#[allow(clippy::cast_possible_truncation)] // Samples are rounded and clamped to the PCM16 range.
fn quantize_audio(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    speed: f64,
) -> anyhow::Result<Vec<i16>> {
    anyhow::ensure!(
        sample_rate > 0 && channels > 0,
        "Audio rate and channels must be nonzero"
    );
    anyhow::ensure!(
        samples.len().is_multiple_of(usize::from(channels)),
        "Interleaved audio samples must contain complete channel frames"
    );
    anyhow::ensure!(
        speed.is_finite() && (MIN_AUDIO_SPEED..=MAX_AUDIO_SPEED).contains(&speed),
        "Audio speed must be between {MIN_AUDIO_SPEED} and {MAX_AUDIO_SPEED}"
    );
    Ok(samples
        .iter()
        .map(|sample| (sample * PCM16_SCALE).round().clamp(PCM16_MIN, PCM16_MAX) as i16)
        .collect())
}

pub(super) fn prepare_ffmpeg_encoding(
    samples: &[f32],
    sample_rate: u32,
    channels: u16,
    format: crate::ChatAudioFormat,
    speed: f64,
) -> anyhow::Result<(tempfile::NamedTempFile, Command)> {
    let pcm = quantize_audio(samples, sample_rate, channels, speed)?;
    ffmpeg_audio_command(&pcm, sample_rate, channels, format, speed)
}

fn ffmpeg_audio_command(
    pcm: &[i16],
    sample_rate: u32,
    channels: u16,
    format: crate::ChatAudioFormat,
    speed: f64,
) -> anyhow::Result<(tempfile::NamedTempFile, Command)> {
    let (codec, container) = match format {
        crate::ChatAudioFormat::Mp3 => ("libmp3lame", "mp3"),
        crate::ChatAudioFormat::Flac => ("flac", "flac"),
        crate::ChatAudioFormat::Opus => ("libopus", "opus"),
        crate::ChatAudioFormat::Aac => ("aac", "adts"),
        crate::ChatAudioFormat::Wav | crate::ChatAudioFormat::Pcm16 => ("pcm_s16le", "s16le"),
    };
    let mut input = tempfile::NamedTempFile::new()?;
    {
        let mut writer = std::io::BufWriter::new(input.as_file_mut());
        for sample in pcm {
            writer.write_all(&sample.to_le_bytes())?;
        }
        writer.flush()?;
    }
    let mut command = Command::new("ffmpeg");
    command
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-f",
            "s16le",
            "-ar",
        ])
        .arg(sample_rate.to_string())
        .arg("-ac")
        .arg(channels.to_string())
        .arg("-i")
        .arg(input.path());
    if speed != 1.0 {
        let tempo = if speed < MIN_TEMPO_STAGE {
            format!(
                "atempo={MIN_TEMPO_STAGE},atempo={}",
                speed / MIN_TEMPO_STAGE
            )
        } else if speed > MAX_TEMPO_STAGE {
            format!(
                "atempo={MAX_TEMPO_STAGE},atempo={}",
                speed / MAX_TEMPO_STAGE
            )
        } else {
            format!("atempo={speed}")
        };
        command.args(["-af", &tempo]);
    }
    command
        .args([
            "-vn",
            "-map_metadata",
            "-1",
            "-c:a",
            codec,
            "-f",
            container,
            "pipe:1",
        ])
        .stdin(Stdio::null());
    Ok((input, command))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode_audio_pcm_with_speed_async;
    use std::sync::Arc;

    const TEST_SAMPLE_RATE: u32 = 24_000;
    const TEST_CHANNELS: u16 = 2;
    const TEST_TONE_FREQUENCY: f32 = 440.0;
    const TEST_TONE_AMPLITUDE: f32 = 0.5;
    const TEST_DURATION_DIVISOR: usize = 5;
    const LOSSY_PADDING_DIVISOR: usize = 10;

    #[tokio::test]
    async fn native_audio_preserves_pcm_limits_channels_and_wav_metadata() -> anyhow::Result<()> {
        let samples = Arc::new(vec![-2., -1., 0., 0.5, 1., 2.]);
        let expected = [i16::MIN, i16::MIN, 0, 16_384, i16::MAX, i16::MAX];
        for channels in [1, TEST_CHANNELS] {
            for format in [ChatAudioFormat::Pcm16, ChatAudioFormat::Wav] {
                let bytes = encode_audio_pcm_with_speed_async(
                    samples.clone(),
                    TEST_SAMPLE_RATE,
                    channels,
                    format,
                    1.0,
                )
                .await?;
                let pcm = if format == ChatAudioFormat::Wav {
                    let mut wav = hound::WavReader::new(std::io::Cursor::new(bytes))?;
                    assert_eq!(wav.spec().sample_rate, TEST_SAMPLE_RATE);
                    assert_eq!(wav.spec().channels, channels);
                    wav.samples::<i16>().collect::<Result<Vec<_>, _>>()?
                } else {
                    bytes
                        .as_chunks::<PCM_SAMPLE_BYTES>()
                        .0
                        .iter()
                        .map(|sample| i16::from_le_bytes(*sample))
                        .collect()
                };
                assert_eq!(pcm, expected);
            }
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires FFmpeg with the atempo filter"]
    #[allow(clippy::cast_precision_loss)] // One second of 24 kHz indices is exactly representable.
    async fn audio_speed_changes_duration_without_changing_wav_rate() -> anyhow::Result<()> {
        let samples = (0..TEST_SAMPLE_RATE)
            .map(|index| {
                TEST_TONE_AMPLITUDE
                    * (std::f32::consts::TAU * TEST_TONE_FREQUENCY * index as f32
                        / TEST_SAMPLE_RATE as f32)
                        .sin()
            })
            .collect::<Vec<_>>();
        for speed in [
            MIN_AUDIO_SPEED,
            MIN_TEMPO_STAGE,
            MAX_TEMPO_STAGE,
            MAX_AUDIO_SPEED,
        ] {
            let bytes = encode_audio_pcm_with_speed(
                &samples,
                TEST_SAMPLE_RATE,
                1,
                crate::ChatAudioFormat::Wav,
                speed,
            )?;
            if speed == MAX_TEMPO_STAGE {
                assert_eq!(
                    encode_audio_pcm_with_speed_async(
                        Arc::new(samples.clone()),
                        TEST_SAMPLE_RATE,
                        1,
                        ChatAudioFormat::Wav,
                        speed,
                    )
                    .await?,
                    bytes
                );
            }
            let wav = hound::WavReader::new(std::io::Cursor::new(bytes))?;
            assert_eq!(wav.spec().sample_rate, TEST_SAMPLE_RATE);
            assert_eq!(wav.spec().channels, 1);
            let expected = f64::from(TEST_SAMPLE_RATE) / speed;
            let relative_error = (f64::from(wav.len()) - expected).abs() / expected;
            assert!(
                relative_error < 0.1,
                "speed {speed}: duration error {relative_error}"
            );
        }
        Ok(())
    }

    #[tokio::test]
    #[ignore = "requires FFmpeg with MP3, FLAC, Opus and AAC encoders"]
    #[allow(clippy::cast_precision_loss)] // The short tone's sample indices fit in the f32 mantissa.
    async fn compressed_audio_signatures_and_decode_roundtrips() -> anyhow::Result<()> {
        let samples = (0..TEST_SAMPLE_RATE as usize / TEST_DURATION_DIVISOR)
            .map(|index| {
                TEST_TONE_AMPLITUDE
                    * (std::f32::consts::TAU * TEST_TONE_FREQUENCY * index as f32
                        / TEST_SAMPLE_RATE as f32)
                        .sin()
            })
            .collect::<Vec<_>>();
        let expected =
            encode_audio_pcm(&samples, TEST_SAMPLE_RATE, 1, crate::ChatAudioFormat::Pcm16)?;
        for format in [
            crate::ChatAudioFormat::Mp3,
            crate::ChatAudioFormat::Flac,
            crate::ChatAudioFormat::Opus,
            crate::ChatAudioFormat::Aac,
        ] {
            let encoded = encode_audio_pcm(&samples, TEST_SAMPLE_RATE, 1, format)?;
            if format == ChatAudioFormat::Flac {
                assert_eq!(
                    encode_audio_pcm_with_speed_async(
                        Arc::new(samples.clone()),
                        TEST_SAMPLE_RATE,
                        1,
                        format,
                        1.0,
                    )
                    .await?,
                    encoded
                );
            }
            let valid_header = match format {
                crate::ChatAudioFormat::Mp3 => encoded.starts_with(b"ID3"),
                crate::ChatAudioFormat::Flac => encoded.starts_with(b"fLaC"),
                crate::ChatAudioFormat::Opus => {
                    encoded.starts_with(b"OggS")
                        && encoded
                            .windows(b"OpusHead".len())
                            .any(|bytes| bytes == b"OpusHead")
                }
                crate::ChatAudioFormat::Aac => {
                    encoded.len() > 1 && encoded[0] == 0xff && encoded[1] & 0xf0 == 0xf0
                }
                _ => unreachable!(),
            };
            assert!(valid_header, "{format:?}: invalid container header");
            let mut input = tempfile::NamedTempFile::new()?;
            input.write_all(&encoded)?;
            let decoded = Command::new("ffmpeg")
                .args(["-hide_banner", "-loglevel", "error", "-nostdin", "-i"])
                .arg(input.path())
                .args(["-f", "s16le", "-ar"])
                .arg(TEST_SAMPLE_RATE.to_string())
                .args(["-ac", "1", "pipe:1"])
                .stdin(Stdio::null())
                .output()?;
            anyhow::ensure!(
                decoded.status.success(),
                "{format:?}: decoding failed: {}",
                String::from_utf8_lossy(&decoded.stderr)
            );
            if format == crate::ChatAudioFormat::Flac {
                assert_eq!(decoded.stdout, expected);
            } else {
                assert!(decoded.stdout.len() >= expected.len());
                assert!(
                    decoded.stdout.len()
                        <= expected.len() + 2 * TEST_SAMPLE_RATE as usize / LOSSY_PADDING_DIVISOR
                );
                assert!(decoded
                    .stdout
                    .as_chunks::<PCM_SAMPLE_BYTES>()
                    .0
                    .iter()
                    .any(|bytes| {
                        i16::from_le_bytes(*bytes).unsigned_abs() > i16::MAX as u16 / 4
                    }));
            }
        }
        Ok(())
    }
}
