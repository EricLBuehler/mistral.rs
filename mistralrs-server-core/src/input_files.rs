use std::{net::IpAddr, time::Duration};

use anyhow::Context;
use base64::{engine::general_purpose::STANDARD, Engine};
use mistralrs_core::{File, FileSource, FILE_PURPOSE_USER_DATA};
use url::Url;

use crate::types::SharedMistralRsState;

const MAX_INPUT_FILE_BYTES: usize = 64 * 1024 * 1024;
const FILE_URL_TIMEOUT: Duration = Duration::from_secs(30);
const FILE_URL_REDIRECTS: usize = 3;

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
        let data_url = data_url::DataUrl::process(file_data)?;
        let bytes = data_url.decode_to_vec()?.0;
        ensure_size(bytes, mime_type)
    } else {
        let bytes = STANDARD
            .decode(file_data)
            .context("input_file.file_data must be base64 or a data URL")?;
        ensure_size(bytes, None)
    }
}

fn data_url_mime(file_data: &str) -> Option<String> {
    let header = file_data.strip_prefix("data:")?.split_once(',')?.0;
    let mime = header.split(';').next().unwrap_or_default();
    (!mime.is_empty()).then(|| mime.to_string())
}

async fn fetch_file_url(url: &str) -> anyhow::Result<(Vec<u8>, Option<String>, Option<String>)> {
    let parsed = Url::parse(url).context("input_file.file_url must be a valid URL")?;
    if parsed.scheme() != "http" && parsed.scheme() != "https" {
        anyhow::bail!("input_file.file_url only supports http and https URLs.");
    }
    reject_private_host(&parsed)?;

    let client = reqwest::Client::builder()
        .timeout(FILE_URL_TIMEOUT)
        .redirect(reqwest::redirect::Policy::limited(FILE_URL_REDIRECTS))
        .build()?;
    let response = client
        .get(parsed.clone())
        .send()
        .await?
        .error_for_status()?;
    if response
        .content_length()
        .is_some_and(|len| len > MAX_INPUT_FILE_BYTES as u64)
    {
        anyhow::bail!("input_file.file_url exceeds the {MAX_INPUT_FILE_BYTES} byte limit.");
    }
    let mime_type = response
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(|v| v.split(';').next().unwrap_or(v).trim().to_string())
        .filter(|v| !v.is_empty());
    let bytes = response.bytes().await?.to_vec();
    let (bytes, mime_type) = ensure_size(bytes, mime_type)?;
    let filename = parsed
        .path_segments()
        .and_then(|mut segments| segments.next_back())
        .filter(|segment| !segment.is_empty())
        .map(ToString::to_string);
    Ok((bytes, mime_type, filename))
}

fn reject_private_host(url: &Url) -> anyhow::Result<()> {
    let Some(host) = url.host_str() else {
        anyhow::bail!("input_file.file_url must include a host.");
    };
    let host_lower = host.to_ascii_lowercase();
    if host_lower == "localhost"
        || host_lower.ends_with(".localhost")
        || host_lower.ends_with(".local")
    {
        anyhow::bail!("input_file.file_url must not target local hosts.");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        let blocked = match ip {
            IpAddr::V4(ip) => {
                ip.is_private()
                    || ip.is_loopback()
                    || ip.is_link_local()
                    || ip.is_broadcast()
                    || ip.is_unspecified()
            }
            IpAddr::V6(ip) => {
                ip.is_loopback()
                    || ip.is_unspecified()
                    || ip.is_unique_local()
                    || ip.is_unicast_link_local()
            }
        };
        if blocked {
            anyhow::bail!("input_file.file_url must not target private or local IP addresses.");
        }
    }
    Ok(())
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

    #[test]
    fn rejects_private_literal_file_urls() {
        let url = Url::parse("http://127.0.0.1/file.txt").unwrap();
        assert!(reject_private_host(&url).is_err());
    }
}
