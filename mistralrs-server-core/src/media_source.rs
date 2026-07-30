use std::{
    path::{Path, PathBuf},
    sync::OnceLock,
    time::Duration,
};

use anyhow::{Context, Result};
use mistralrs_core::network_security::{fetch_public_url_limited, PublicHttpFetchOptions};
use reqwest::{header, redirect::Policy};
use tokio::{fs::File, io::AsyncReadExt};
use url::Url;

pub(crate) const MAX_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MEDIA_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
pub(crate) const MEDIA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const MEDIA_FETCH_REDIRECTS: usize = 3;
pub(crate) const SERVER_VIDEO_FRAME_LIMIT: usize = 32;

const DATA_URL_HEADER_ALLOWANCE: usize = 4096;
const UI_UPLOAD_SCHEME: &str = "mistralrs-upload";
const IMAGE_UPLOAD_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "gif", "webp", "bmp", "svg"];
const VIDEO_UPLOAD_EXTENSIONS: &[&str] = &["mp4", "avi", "mov", "mkv", "webm", "m4v", "gif"];
const AUDIO_UPLOAD_EXTENSIONS: &[&str] =
    &["wav", "mp3", "ogg", "flac", "m4a", "aac", "opus", "webm"];

static UI_UPLOAD_DIR: OnceLock<PathBuf> = OnceLock::new();

#[derive(Clone, Copy)]
pub(crate) enum MediaSourcePolicy {
    ServerRequest,
    Local,
}

pub(crate) struct LoadedMedia {
    pub bytes: Vec<u8>,
    pub mime_type: Option<String>,
    pub final_url: Option<Url>,
}

pub async fn configure_ui_upload_dir(path: impl AsRef<Path>) -> Result<()> {
    let path = tokio::fs::canonicalize(path.as_ref())
        .await
        .context("Failed to resolve UI uploads directory")?;
    if !tokio::fs::metadata(&path).await?.is_dir() {
        anyhow::bail!("UI uploads path is not a directory.");
    }
    if let Some(configured) = UI_UPLOAD_DIR.get() {
        if configured != &path {
            anyhow::bail!("UI uploads directory is already configured.");
        }
        return Ok(());
    }
    match UI_UPLOAD_DIR.set(path) {
        Ok(()) => Ok(()),
        Err(path) if UI_UPLOAD_DIR.get() == Some(&path) => Ok(()),
        Err(_) => anyhow::bail!("UI uploads directory was configured concurrently"),
    }
}

pub(crate) async fn load_media_source(
    source: &str,
    policy: MediaSourcePolicy,
    kind: &str,
) -> Result<LoadedMedia> {
    match Url::parse(source) {
        Ok(url) => load_media_url(url, policy, kind).await,
        Err(err) => {
            if matches!(policy, MediaSourcePolicy::Local) {
                let path = PathBuf::from(source);
                let bytes = read_local_file_limited(&path, MAX_MEDIA_BYTES)
                    .await
                    .with_context(|| format!("Failed to read local {kind} path"))?;
                Ok(LoadedMedia {
                    bytes,
                    mime_type: None,
                    final_url: None,
                })
            } else {
                anyhow::bail!("Invalid {kind} source: expected http, https, or data URL: {err}");
            }
        }
    }
}

async fn load_media_url(url: Url, policy: MediaSourcePolicy, kind: &str) -> Result<LoadedMedia> {
    match url.scheme() {
        "http" | "https" => {
            fetch_remote_limited_inner(
                url,
                MAX_MEDIA_BYTES,
                kind,
                matches!(policy, MediaSourcePolicy::ServerRequest),
            )
            .await
        }
        "data" => {
            let bytes = decode_data_url_limited(url.as_str(), MAX_MEDIA_BYTES, kind)?;
            Ok(LoadedMedia {
                bytes,
                mime_type: data_url_mime(url.as_str()),
                final_url: Some(url),
            })
        }
        "file" if matches!(policy, MediaSourcePolicy::Local) => {
            let path = url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid local {kind} file URL."))?;
            let bytes = read_local_file_limited(&path, MAX_MEDIA_BYTES).await?;
            Ok(LoadedMedia {
                bytes,
                mime_type: None,
                final_url: Some(url),
            })
        }
        "file" => anyhow::bail!("Server request {kind} sources do not support file URLs."),
        UI_UPLOAD_SCHEME if matches!(policy, MediaSourcePolicy::ServerRequest) => {
            load_ui_upload(url, kind).await
        }
        scheme => anyhow::bail!("Unsupported {kind} URL scheme: {scheme}"),
    }
}

async fn load_ui_upload(url: Url, kind: &str) -> Result<LoadedMedia> {
    let root = UI_UPLOAD_DIR
        .get()
        .ok_or_else(|| anyhow::anyhow!("Unsupported {kind} URL scheme: {UI_UPLOAD_SCHEME}"))?;
    load_ui_upload_from_root(url, kind, root, MAX_MEDIA_BYTES).await
}

async fn load_ui_upload_from_root(
    url: Url,
    kind: &str,
    root: &Path,
    max_bytes: usize,
) -> Result<LoadedMedia> {
    let filename = ui_upload_filename(&url, kind)?;
    let path = resolve_ui_upload_path(root, filename).await?;
    let bytes = read_local_file_limited(&path, max_bytes).await?;
    Ok(LoadedMedia {
        bytes,
        mime_type: None,
        final_url: Some(url),
    })
}

fn ui_upload_filename<'a>(url: &'a Url, kind: &str) -> Result<&'a str> {
    if url.scheme() != UI_UPLOAD_SCHEME
        || url.host().is_some()
        || url.query().is_some()
        || url.fragment().is_some()
    {
        anyhow::bail!("Invalid UI upload reference.");
    }

    let filename = url.path();
    if url.as_str() != format!("{UI_UPLOAD_SCHEME}:{filename}") {
        anyhow::bail!("Invalid UI upload reference.");
    }
    let (id, extension) = filename
        .rsplit_once('.')
        .ok_or_else(|| anyhow::anyhow!("Invalid UI upload filename."))?;
    let uuid = uuid::Uuid::parse_str(id).context("Invalid UI upload filename")?;
    if uuid.get_version_num() != 4 || uuid.hyphenated().to_string() != id {
        anyhow::bail!("Invalid UI upload filename.");
    }

    let extensions = match kind {
        "image" => IMAGE_UPLOAD_EXTENSIONS,
        "video" => VIDEO_UPLOAD_EXTENSIONS,
        "audio" => AUDIO_UPLOAD_EXTENSIONS,
        _ => anyhow::bail!("UI uploads do not support {kind} sources."),
    };
    if !extensions.contains(&extension) {
        anyhow::bail!("Unsupported UI upload format for {kind}.");
    }
    Ok(filename)
}

async fn resolve_ui_upload_path(root: &Path, filename: &str) -> Result<PathBuf> {
    let candidate = root.join(filename);
    let metadata = tokio::fs::symlink_metadata(&candidate)
        .await
        .with_context(|| format!("Could not read UI upload metadata: {}", candidate.display()))?;
    if metadata.file_type().is_symlink() {
        anyhow::bail!("UI upload references may not target symlinks.");
    }
    let path = tokio::fs::canonicalize(&candidate)
        .await
        .with_context(|| format!("Could not resolve UI upload: {}", candidate.display()))?;
    if path.parent() != Some(root) {
        anyhow::bail!("UI upload resolved outside the uploads directory.");
    }
    Ok(path)
}

pub(crate) async fn fetch_remote_limited(
    url: Url,
    max_bytes: usize,
    kind: &str,
) -> Result<LoadedMedia> {
    fetch_remote_limited_inner(url, max_bytes, kind, true).await
}

async fn fetch_remote_limited_inner(
    url: Url,
    max_bytes: usize,
    kind: &str,
    validate_network: bool,
) -> Result<LoadedMedia> {
    if validate_network {
        let response = fetch_public_url_limited(
            url,
            PublicHttpFetchOptions {
                max_bytes,
                connect_timeout: MEDIA_CONNECT_TIMEOUT,
                total_timeout: MEDIA_FETCH_TIMEOUT,
                max_redirects: MEDIA_FETCH_REDIRECTS,
            },
            None,
        )
        .await
        .with_context(|| format!("Failed to fetch {kind} from a public URL"))?;
        return Ok(LoadedMedia {
            mime_type: response.media_type().map(str::to_owned),
            bytes: response.bytes,
            final_url: Some(response.final_url),
        });
    }

    tokio::time::timeout(
        MEDIA_FETCH_TIMEOUT,
        fetch_trusted_remote_limited(url, max_bytes, kind),
    )
    .await
    .map_err(|_| anyhow::anyhow!("Fetching {kind} exceeded the total timeout."))?
}

async fn fetch_trusted_remote_limited(
    mut url: Url,
    max_bytes: usize,
    kind: &str,
) -> Result<LoadedMedia> {
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Remote media URLs must use http or https.");
    }
    let client = reqwest::Client::builder()
        .connect_timeout(MEDIA_CONNECT_TIMEOUT)
        .redirect(Policy::none())
        .no_proxy()
        .build()?;

    for redirect_idx in 0..=MEDIA_FETCH_REDIRECTS {
        let response = client
            .get(url.clone())
            .send()
            .await
            .with_context(|| format!("Failed to fetch {kind}: {url}"))?;

        if response.status().is_redirection() {
            if redirect_idx == MEDIA_FETCH_REDIRECTS {
                anyhow::bail!("{kind} URL exceeded the redirect limit.");
            }
            url = redirect_target(&url, &response, kind)?;
            continue;
        }

        response
            .error_for_status_ref()
            .map_err(|err| anyhow::anyhow!("Failed to fetch {kind}: {url}: {err}"))?;
        if response
            .content_length()
            .is_some_and(|len| len > max_bytes as u64)
        {
            anyhow::bail!("{kind} response exceeds the {max_bytes} byte limit.");
        }
        let mime_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(|value| value.split(';').next().unwrap_or(value).trim().to_string())
            .filter(|value| !value.is_empty());
        let bytes = read_response_limited(response, max_bytes, kind).await?;
        return Ok(LoadedMedia {
            bytes,
            mime_type,
            final_url: Some(url),
        });
    }

    unreachable!("redirect loop returns or bails")
}

fn redirect_target(current: &Url, response: &reqwest::Response, kind: &str) -> Result<Url> {
    let Some(location) = response.headers().get(header::LOCATION) else {
        anyhow::bail!("{kind} redirect response is missing Location.");
    };
    let location = location
        .to_str()
        .context("redirect Location header is not valid UTF-8")?;
    let target = current
        .join(location)
        .with_context(|| format!("Invalid {kind} redirect target."))?;
    if target.scheme() != "http" && target.scheme() != "https" {
        anyhow::bail!("{kind} redirects may only target http or https URLs.");
    }
    Ok(target)
}

async fn read_response_limited(
    mut response: reqwest::Response,
    max_bytes: usize,
    kind: &str,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        if bytes.len().saturating_add(chunk.len()) > max_bytes {
            anyhow::bail!("{kind} response exceeds the {max_bytes} byte limit.");
        }
        bytes.extend_from_slice(&chunk);
    }
    Ok(bytes)
}

pub(crate) async fn read_local_file_limited(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = tokio::fs::metadata(path)
        .await
        .with_context(|| format!("Could not read local file metadata: {}", path.display()))?;
    if !metadata.is_file() {
        anyhow::bail!("Local media source is not a file.");
    }
    if metadata.len() > max_bytes as u64 {
        anyhow::bail!("Local file exceeds the {max_bytes} byte limit.");
    }
    let file = File::open(path)
        .await
        .with_context(|| format!("Could not open local file: {}", path.display()))?;
    let mut reader = file.take(max_bytes as u64 + 1);
    let mut bytes = Vec::new();
    reader.read_to_end(&mut bytes).await?;
    if bytes.len() > max_bytes {
        anyhow::bail!("Local file exceeds the {max_bytes} byte limit.");
    }
    Ok(bytes)
}

pub(crate) fn decode_data_url_limited(
    source: &str,
    max_bytes: usize,
    kind: &str,
) -> Result<Vec<u8>> {
    let encoded_limit = max_bytes
        .saturating_mul(4)
        .saturating_div(3)
        .saturating_add(DATA_URL_HEADER_ALLOWANCE);
    if source.len() > encoded_limit {
        anyhow::bail!("{kind} data URL exceeds the {max_bytes} byte limit.");
    }
    let data_url = data_url::DataUrl::process(source)?;
    let bytes = data_url.decode_to_vec()?.0;
    if bytes.len() > max_bytes {
        anyhow::bail!("{kind} data URL exceeds the {max_bytes} byte limit.");
    }
    Ok(bytes)
}

pub(crate) fn data_url_mime(source: &str) -> Option<String> {
    let header = source.strip_prefix("data:")?.split_once(',')?.0;
    let mime = header.split(';').next().unwrap_or_default();
    (!mime.is_empty()).then(|| mime.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn server_policy_rejects_private_remote_hosts() {
        for source in [
            "http://127.0.0.1/image.png",
            "http://127.1/image.png",
            "http://2130706433/image.png",
            "http://0x7f000001/image.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/image.png",
            "http://localhost/image.png",
            "http://service.local/image.png",
        ] {
            assert!(
                load_media_source(source, MediaSourcePolicy::ServerRequest, "test")
                    .await
                    .is_err(),
                "{source}"
            );
        }
    }

    #[tokio::test]
    async fn local_policy_keeps_trusted_private_http_compatibility() {
        use std::net::Ipv4Addr;
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut request = [0; 1024];
            let _ = stream.read(&mut request).await.unwrap();
            stream
                .write_all(b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 5\r\nConnection: close\r\n\r\nlocal")
                .await
                .unwrap();
        });

        let media = load_media_source(
            &format!("http://{address}/media"),
            MediaSourcePolicy::Local,
            "test",
        )
        .await
        .unwrap();
        server.await.unwrap();
        assert_eq!(media.bytes, b"local");
        assert_eq!(media.mime_type.as_deref(), Some("text/plain"));
    }

    #[test]
    fn rejects_oversized_data_urls_before_decode() {
        let source = format!("data:text/plain;base64,{}", "a".repeat(128));
        assert!(decode_data_url_limited(&source, 4, "test").is_err());
    }

    #[test]
    fn validates_ui_upload_references() {
        let image =
            Url::parse("mistralrs-upload:550e8400-e29b-41d4-a716-446655440000.png").unwrap();
        assert_eq!(
            ui_upload_filename(&image, "image").unwrap(),
            "550e8400-e29b-41d4-a716-446655440000.png"
        );

        let video =
            Url::parse("mistralrs-upload:550e8400-e29b-41d4-a716-446655440000.webm").unwrap();
        assert!(ui_upload_filename(&video, "video").is_ok());
        assert!(ui_upload_filename(&video, "image").is_err());

        for source in [
            "mistralrs-upload:../../etc/passwd.png",
            "mistralrs-upload:%2e%2e%2fetc%2fpasswd.png",
            "mistralrs-upload://host/550e8400-e29b-41d4-a716-446655440000.png",
            "mistralrs-upload:6ba7b810-9dad-11d1-80b4-00c04fd430c8.png",
            "mistralrs-upload:550e8400-e29b-41d4-a716-446655440000.PNG",
            "mistralrs-upload:550e8400-e29b-41d4-a716-446655440000.png?x=1",
        ] {
            let url = Url::parse(source).unwrap();
            assert!(ui_upload_filename(&url, "image").is_err(), "{source}");
        }
    }

    #[tokio::test]
    async fn reads_ui_upload_with_size_limit() {
        let dir = tempfile::tempdir().unwrap();
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        let filename = "550e8400-e29b-41d4-a716-446655440000.png";
        tokio::fs::write(root.join(filename), b"image")
            .await
            .unwrap();
        let url = Url::parse(&format!("mistralrs-upload:{filename}")).unwrap();

        let media = load_ui_upload_from_root(url.clone(), "image", &root, 5)
            .await
            .unwrap();
        assert_eq!(media.bytes, b"image");
        assert!(load_ui_upload_from_root(url, "image", &root, 4)
            .await
            .is_err());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn rejects_ui_upload_symlinks() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let root = tokio::fs::canonicalize(dir.path()).await.unwrap();
        let outside = tempfile::NamedTempFile::new().unwrap();
        let filename = "550e8400-e29b-41d4-a716-446655440000.png";
        symlink(outside.path(), root.join(filename)).unwrap();

        assert!(resolve_ui_upload_path(&root, filename).await.is_err());
    }
}
