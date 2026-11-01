use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine};

pub const MAX_INPUT_AUDIO_BYTES: usize = 64 * 1024 * 1024;
const MAX_INPUT_AUDIO_BASE64_BYTES: usize = MAX_INPUT_AUDIO_BYTES.div_ceil(3) * 4;

pub fn input_audio_data_url(data: &str, format: &str) -> Result<String> {
    let mime = match format {
        "wav" => "audio/wav",
        "mp3" => "audio/mpeg",
        _ => anyhow::bail!("input_audio.format must be wav or mp3"),
    };
    anyhow::ensure!(
        data.len() <= MAX_INPUT_AUDIO_BASE64_BYTES,
        "input_audio.data exceeds the {MAX_INPUT_AUDIO_BYTES} byte audio limit"
    );
    let decoded_len = STANDARD
        .decode(data)
        .context("input_audio.data must be valid base64")?
        .len();
    anyhow::ensure!(decoded_len > 0, "input_audio.data must not be empty");
    anyhow::ensure!(
        decoded_len <= MAX_INPUT_AUDIO_BYTES,
        "input_audio.data exceeds the {MAX_INPUT_AUDIO_BYTES} byte audio limit"
    );
    Ok(format!("data:{mime};base64,{data}"))
}
