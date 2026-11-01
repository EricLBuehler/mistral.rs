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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_input_audio_preserves_bytes_and_selects_the_media_type() -> Result<()> {
        let data = STANDARD.encode(b"audio bytes");
        assert_eq!(
            input_audio_data_url(&data, "wav")?,
            format!("data:audio/wav;base64,{data}")
        );
        assert_eq!(
            input_audio_data_url(&data, "mp3")?,
            format!("data:audio/mpeg;base64,{data}")
        );
        Ok(())
    }

    #[test]
    fn invalid_inline_audio_is_rejected() {
        assert!(input_audio_data_url("YWJj", "flac").is_err());
        assert!(input_audio_data_url("not base64", "wav").is_err());
        assert!(input_audio_data_url("", "mp3").is_err());
    }
}
