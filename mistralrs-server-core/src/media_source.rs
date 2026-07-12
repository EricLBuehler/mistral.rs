use std::{
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr},
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::{Context, Result};
use reqwest::{header, redirect::Policy};
use tokio::{fs::File, io::AsyncReadExt};
use url::Url;

pub(crate) const MAX_MEDIA_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MEDIA_FETCH_TIMEOUT: Duration = Duration::from_secs(30);
pub(crate) const MEDIA_FETCH_REDIRECTS: usize = 3;
pub(crate) const SERVER_VIDEO_FRAME_LIMIT: usize = 32;

const DATA_URL_HEADER_ALLOWANCE: usize = 4096;

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
        scheme => anyhow::bail!("Unsupported {kind} URL scheme: {scheme}"),
    }
}

pub(crate) async fn fetch_remote_limited(
    url: Url,
    max_bytes: usize,
    kind: &str,
) -> Result<LoadedMedia> {
    fetch_remote_limited_inner(url, max_bytes, kind, true).await
}

async fn fetch_remote_limited_inner(
    mut url: Url,
    max_bytes: usize,
    kind: &str,
    validate_network: bool,
) -> Result<LoadedMedia> {
    for redirect_idx in 0..=MEDIA_FETCH_REDIRECTS {
        let mut client = reqwest::Client::builder()
            .timeout(MEDIA_FETCH_TIMEOUT)
            .redirect(Policy::none())
            .no_proxy();
        if validate_network {
            let addrs = validate_remote_url(&url).await?;
            client =
                client.resolve_to_addrs(url.host_str().expect("validated URL has host"), &addrs);
        } else if url.scheme() != "http" && url.scheme() != "https" {
            anyhow::bail!("Remote media URLs must use http or https.");
        }
        let client = client.build()?;
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

async fn validate_remote_url(url: &Url) -> Result<Vec<SocketAddr>> {
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Remote media URLs must use http or https.");
    }
    let Some(host) = url.host_str() else {
        anyhow::bail!("Remote media URL must include a host.");
    };
    reject_private_hostname(host)?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("Remote media URL must include a valid port."))?;
    let addrs = tokio::net::lookup_host((host, port))
        .await
        .with_context(|| format!("Failed to resolve remote media host `{host}`"))?
        .collect::<Vec<_>>();
    if addrs.is_empty() {
        anyhow::bail!("Remote media host `{host}` did not resolve to any addresses.");
    }
    for addr in &addrs {
        reject_private_ip(addr.ip())?;
    }
    Ok(addrs)
}

fn reject_private_hostname(host: &str) -> Result<()> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        anyhow::bail!("Remote media URLs must not target local hosts.");
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        reject_private_ip(ip)?;
    }
    Ok(())
}

fn reject_private_ip(ip: IpAddr) -> Result<()> {
    if !is_global_ip(ip) {
        anyhow::bail!("Remote media URLs must not target private or local IP addresses.");
    }
    Ok(())
}

fn is_global_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => is_global_ipv4(ip),
        IpAddr::V6(ip) => {
            if let Some(ip) = ip.to_ipv4_mapped() {
                return is_global_ipv4(ip);
            }
            is_global_ipv6(ip)
        }
    }
}

fn is_global_ipv4(ip: Ipv4Addr) -> bool {
    let octets = ip.octets();
    !(ip.is_private()
        || ip.is_loopback()
        || ip.is_link_local()
        || ip.is_broadcast()
        || ip.is_unspecified()
        || ip.is_multicast()
        || matches!(octets, [100, 64..=127, _, _])
        || matches!(octets, [192, 0, 0, _])
        || matches!(octets, [192, 0, 2, _])
        || matches!(octets, [198, 18 | 19, _, _])
        || matches!(octets, [198, 51, 100, _])
        || matches!(octets, [203, 0, 113, _])
        || octets[0] >= 240)
}

fn is_global_ipv6(ip: Ipv6Addr) -> bool {
    let segments = ip.segments();
    !(ip.is_loopback()
        || ip.is_unspecified()
        || ip.is_unique_local()
        || ip.is_unicast_link_local()
        || ip.is_multicast()
        || segments[0] == 0x2001 && segments[1] == 0x0db8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn rejects_private_remote_hosts() {
        for source in [
            "http://127.0.0.1/image.png",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/image.png",
            "http://localhost/image.png",
            "http://service.local/image.png",
        ] {
            let url = Url::parse(source).unwrap();
            assert!(validate_remote_url(&url).await.is_err(), "{source}");
        }
    }

    #[test]
    fn rejects_oversized_data_urls_before_decode() {
        let source = format!("data:text/plain;base64,{}", "a".repeat(128));
        assert!(decode_data_url_limited(&source, 4, "test").is_err());
    }
}
