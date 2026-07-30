use anyhow::Context;
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{File, FileSource, FILE_PURPOSE_USER_DATA};

use crate::media_source::{
    data_url_mime, decode_data_url_limited, fetch_remote_limited, MAX_MEDIA_BYTES,
};
use crate::types::SharedMistralRsState;

const MAX_INPUT_FILE_BYTES: usize = MAX_MEDIA_BYTES;
const BASE64_FILE_DATA_ALLOWANCE: usize = 4096;

#[derive(Debug, Clone, Default)]
pub struct InputFileSpec {
    pub file_id: Option<String>,
    pub file_data: Option<String>,
    pub file_url: Option<String>,
    pub filename: Option<String>,
}

pub async fn resolve_input_file(
    state: SharedMistralRsState,
    spec: InputFileSpec,
    source: &str,
) -> anyhow::Result<File> {
    if let Some(file_id) = spec.file_id {
        return state
            .find_file(&file_id)
            .map(|file| (*file).clone())
            .ok_or_else(|| anyhow::anyhow!("Input file `{file_id}` was not found."));
    }

    if let Some(file_data) = spec.file_data {
        let (bytes, mime_type) = decode_file_data(&file_data)?;
        let filename = spec.filename.unwrap_or_else(|| "input_file".to_string());
        return Ok(file_from_bytes(filename, mime_type, source, bytes));
    }

    if let Some(file_url) = spec.file_url {
        let (bytes, mime_type, filename) = fetch_file_url(&file_url).await?;
        let filename = spec
            .filename
            .or(filename)
            .unwrap_or_else(|| "input_file".to_string());
        return Ok(file_from_bytes(filename, mime_type, source, bytes));
    }

    Err(anyhow::anyhow!(
        "Input file requires one of `file_id`, `file_data`, or `file_url`."
    ))
}

fn file_from_bytes(
    filename: String,
    mime_type: Option<String>,
    source: &str,
    bytes: Vec<u8>,
) -> File {
    File::from_bytes(
        File::make_upload_id(),
        filename,
        mime_type,
        FILE_PURPOSE_USER_DATA.to_string(),
        FileSource {
            tool: source.to_string(),
            round: 0,
            turn: 0,
        },
        bytes,
    )
}

fn decode_file_data(file_data: &str) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    if file_data.starts_with("data:") {
        let mime_type = data_url_mime(file_data);
        let bytes =
            decode_data_url_limited(file_data, MAX_INPUT_FILE_BYTES, "input_file.file_data")?;
        ensure_size(bytes, mime_type)
    } else {
        let encoded_limit = MAX_INPUT_FILE_BYTES
            .saturating_mul(4)
            .saturating_div(3)
            .saturating_add(BASE64_FILE_DATA_ALLOWANCE);
        if file_data.len() > encoded_limit {
            anyhow::bail!("input_file.file_data exceeds the {MAX_INPUT_FILE_BYTES} byte limit.");
        }
        let bytes = STANDARD
            .decode(file_data)
            .context("input_file.file_data must be base64 or a data URL")?;
        ensure_size(bytes, None)
    }
}

async fn fetch_file_url(url: &str) -> anyhow::Result<(Vec<u8>, Option<String>, Option<String>)> {
    let parsed = url::Url::parse(url).context("input_file.file_url must be a valid URL")?;
    let media = fetch_remote_limited(parsed, MAX_INPUT_FILE_BYTES, "input_file.file_url").await?;
    let filename = media
        .final_url
        .as_ref()
        .and_then(|url| {
            url.path_segments()
                .and_then(|mut segments| segments.next_back())
        })
        .filter(|segment| !segment.is_empty())
        .map(ToString::to_string);
    let (bytes, mime_type) = ensure_size(media.bytes, media.mime_type)?;
    Ok((bytes, mime_type, filename))
}

fn ensure_size(
    bytes: Vec<u8>,
    mime_type: Option<String>,
) -> anyhow::Result<(Vec<u8>, Option<String>)> {
    if bytes.len() > MAX_INPUT_FILE_BYTES {
        anyhow::bail!("Input file exceeds the {MAX_INPUT_FILE_BYTES} byte limit.");
    }
    Ok((bytes, mime_type))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_data_url_with_mime_type() {
        let (bytes, mime_type) = decode_file_data("data:text/plain;base64,aGVsbG8=").unwrap();
        assert_eq!(bytes, b"hello");
        assert_eq!(mime_type.as_deref(), Some("text/plain"));
    }

    #[tokio::test]
    async fn rejects_private_literal_file_urls() {
        assert!(fetch_file_url("http://127.0.0.1/file.txt").await.is_err());
    }

    #[test]
    fn rejects_oversized_file_data_before_decode() {
        let source = "a".repeat(BASE64_FILE_DATA_ALLOWANCE + MAX_INPUT_FILE_BYTES * 2);
        assert!(decode_file_data(&source).is_err());
    }
}
