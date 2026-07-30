//! Network primitives for fetching attacker-controlled URLs.
//!
//! Public fetches deliberately use only validated IPv4 A records. Each
//! request is pinned to those addresses, redirects are followed manually,
//! and the entire operation (DNS, redirects, and body) shares one deadline.

use std::{
    collections::HashSet,
    future::Future,
    net::{Ipv4Addr, SocketAddr},
    time::Duration,
};

use anyhow::{Context, Result};
use encoding_rs::{Encoding, UTF_8};
use reqwest::{header, redirect::Policy, Url};

/// Limits applied to a fetch of an untrusted public URL.
#[derive(Clone, Copy, Debug)]
pub struct PublicHttpFetchOptions {
    pub max_bytes: usize,
    pub connect_timeout: Duration,
    pub total_timeout: Duration,
    pub max_redirects: usize,
}

/// A bounded response from an untrusted public URL.
#[derive(Debug)]
pub struct PublicHttpResponse {
    pub bytes: Vec<u8>,
    pub content_type: Option<String>,
    pub final_url: Url,
}

impl PublicHttpResponse {
    /// Decode the bounded body using the response's declared charset.
    ///
    /// This mirrors reqwest's UTF-8 fallback while allowing the body to be
    /// size-limited before decoding it.
    pub fn text(&self) -> String {
        let encoding = self
            .content_type
            .as_deref()
            .and_then(content_type_charset)
            .and_then(|charset| Encoding::for_label(charset.as_bytes()))
            .unwrap_or(UTF_8);
        encoding.decode(&self.bytes).0.into_owned()
    }

    /// Return the media type without Content-Type parameters.
    pub fn media_type(&self) -> Option<&str> {
        self.content_type
            .as_deref()
            .and_then(|value| value.split(';').next())
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }
}

/// Fetch an untrusted HTTP(S) URL using only pinned, public IPv4 addresses.
///
/// Proxies and automatic redirects are disabled. Redirect destinations are
/// resolved and validated again before the next connection.
pub async fn fetch_public_url_limited(
    url: Url,
    options: PublicHttpFetchOptions,
    user_agent: Option<&str>,
) -> Result<PublicHttpResponse> {
    fetch_public_url_limited_with_resolver(
        url,
        options,
        user_agent,
        |url, dns_timeout| async move { resolve_public_ipv4(&url, dns_timeout).await },
    )
    .await
}

async fn fetch_public_url_limited_with_resolver<R, F>(
    url: Url,
    options: PublicHttpFetchOptions,
    user_agent: Option<&str>,
    resolver: R,
) -> Result<PublicHttpResponse>
where
    R: Fn(Url, Duration) -> F,
    F: Future<Output = Result<(String, Vec<SocketAddr>)>>,
{
    tokio::time::timeout(
        options.total_timeout,
        fetch_public_url_limited_inner(url, options, user_agent, resolver),
    )
    .await
    .map_err(|_| {
        anyhow::anyhow!(
            "Public URL fetch exceeded the {:?} total timeout.",
            options.total_timeout
        )
    })?
}

async fn fetch_public_url_limited_inner<R, F>(
    mut url: Url,
    options: PublicHttpFetchOptions,
    user_agent: Option<&str>,
    resolver: R,
) -> Result<PublicHttpResponse>
where
    R: Fn(Url, Duration) -> F,
    F: Future<Output = Result<(String, Vec<SocketAddr>)>>,
{
    for redirect_idx in 0..=options.max_redirects {
        let (host, addrs) = resolver(url.clone(), options.connect_timeout).await?;
        let client = reqwest::Client::builder()
            .connect_timeout(options.connect_timeout)
            .redirect(Policy::none())
            .no_proxy()
            .resolve_to_addrs(&host, &addrs)
            .build()?;

        let mut request = client.get(url.clone());
        if let Some(user_agent) = user_agent {
            request = request.header(header::USER_AGENT, user_agent);
        }
        let response = request
            .send()
            .await
            .with_context(|| format!("Failed to fetch public URL: {url}"))?;

        if response.status().is_redirection() {
            if redirect_idx == options.max_redirects {
                anyhow::bail!("Public URL exceeded the redirect limit.");
            }
            url = response_redirect_target(&url, &response)?;
            continue;
        }

        response
            .error_for_status_ref()
            .with_context(|| format!("Failed to fetch public URL: {url}"))?;
        if response
            .content_length()
            .is_some_and(|len| len > options.max_bytes as u64)
        {
            anyhow::bail!(
                "Public URL response exceeds the {} byte limit.",
                options.max_bytes
            );
        }
        let content_type = response
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .map(str::to_owned);
        let bytes = read_response_limited(response, options.max_bytes).await?;
        return Ok(PublicHttpResponse {
            bytes,
            content_type,
            final_url: url,
        });
    }

    unreachable!("redirect loop returns or bails")
}

async fn resolve_public_ipv4(
    url: &Url,
    dns_timeout: Duration,
) -> Result<(String, Vec<SocketAddr>)> {
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Public URLs must use http or https.");
    }
    let host = url
        .host_str()
        .ok_or_else(|| anyhow::anyhow!("Public URL must include a host."))?;
    reject_local_hostname(host)?;
    let port = url
        .port_or_known_default()
        .ok_or_else(|| anyhow::anyhow!("Public URL must include a valid port."))?;

    // URL parsing canonicalizes WHATWG IPv4 forms (for example 127.1 and
    // 0x7f000001), so literal addresses are classified before any request.
    if let Ok(ip) = host.parse::<Ipv4Addr>() {
        if !is_public_ipv4(ip) {
            anyhow::bail!("Public URLs must not target private or special-use IP addresses.");
        }
        return Ok((host.to_owned(), vec![SocketAddr::from((ip, port))]));
    }

    let resolved = tokio::time::timeout(dns_timeout, tokio::net::lookup_host((host, port)))
        .await
        .map_err(|_| anyhow::anyhow!("Resolving public URL host `{host}` timed out."))?
        .with_context(|| format!("Failed to resolve public URL host `{host}`"))?;
    let addrs = select_public_ipv4(resolved);
    if addrs.is_empty() {
        anyhow::bail!("Public URL host `{host}` has no public IPv4 addresses.");
    }
    Ok((host.to_owned(), addrs))
}

fn select_public_ipv4(addrs: impl IntoIterator<Item = SocketAddr>) -> Vec<SocketAddr> {
    let mut seen = HashSet::new();
    addrs
        .into_iter()
        .filter_map(|addr| match addr {
            SocketAddr::V4(addr) if is_public_ipv4(*addr.ip()) => {
                let addr = SocketAddr::V4(addr);
                seen.insert(addr).then_some(addr)
            }
            // Deliberately ignore AAAA and non-public A records. Supplying only
            // this vetted set to reqwest prevents resolver fallback/rebinding.
            _ => None,
        })
        .collect()
}

fn reject_local_hostname(host: &str) -> Result<()> {
    let host = host.trim_end_matches('.').to_ascii_lowercase();
    if host == "localhost" || host.ends_with(".localhost") || host.ends_with(".local") {
        anyhow::bail!("Public URLs must not target local hosts.");
    }
    Ok(())
}

/// Whether an IPv4 address is publicly routable rather than private,
/// loopback, link-local, documentation, benchmarking, multicast, or reserved.
pub fn is_public_ipv4(ip: Ipv4Addr) -> bool {
    let [a, b, c, _] = ip.octets();
    !matches!(
        (a, b, c),
        (0, _, _)
            | (10, _, _)
            | (100, 64..=127, _)
            | (127, _, _)
            | (169, 254, _)
            | (172, 16..=31, _)
            | (192, 0, 0)
            | (192, 0, 2)
            | (192, 88, 99)
            | (192, 168, _)
            | (198, 18..=19, _)
            | (198, 51, 100)
            | (203, 0, 113)
            | (224..=255, _, _)
    )
}

fn response_redirect_target(current: &Url, response: &reqwest::Response) -> Result<Url> {
    let location = response
        .headers()
        .get(header::LOCATION)
        .ok_or_else(|| anyhow::anyhow!("Public URL redirect is missing Location."))?
        .to_str()
        .context("Public URL redirect Location is not valid UTF-8")?;
    redirect_target(current, location)
}

fn redirect_target(current: &Url, location: &str) -> Result<Url> {
    let target = current
        .join(location)
        .context("Invalid public URL redirect target.")?;
    if target.scheme() != "http" && target.scheme() != "https" {
        anyhow::bail!("Public URL redirects may only target http or https URLs.");
    }
    Ok(target)
}

async fn read_response_limited(
    mut response: reqwest::Response,
    max_bytes: usize,
) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        append_limited(&mut bytes, &chunk, max_bytes)?;
    }
    Ok(bytes)
}

fn append_limited(bytes: &mut Vec<u8>, chunk: &[u8], max_bytes: usize) -> Result<()> {
    if bytes.len().saturating_add(chunk.len()) > max_bytes {
        anyhow::bail!("Public URL response exceeds the {max_bytes} byte limit.");
    }
    bytes.extend_from_slice(chunk);
    Ok(())
}

fn content_type_charset(content_type: &str) -> Option<&str> {
    content_type.split(';').skip(1).find_map(|parameter| {
        let (name, value) = parameter.split_once('=')?;
        name.trim()
            .eq_ignore_ascii_case("charset")
            .then(|| value.trim().trim_matches(['\'', '"']))
            .filter(|value| !value.is_empty())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_special_use_ipv4_ranges() {
        for ip in [
            "0.0.0.0",
            "0.255.255.255",
            "10.0.0.1",
            "100.64.0.1",
            "100.127.255.254",
            "127.0.0.1",
            "169.254.169.254",
            "172.16.0.1",
            "172.31.255.254",
            "192.0.0.9",
            "192.0.2.1",
            "192.88.99.1",
            "192.168.0.1",
            "198.18.0.1",
            "198.19.255.254",
            "198.51.100.1",
            "203.0.113.1",
            "224.0.0.1",
            "239.255.255.255",
            "240.0.0.1",
            "255.255.255.255",
        ] {
            assert!(!is_public_ipv4(ip.parse().unwrap()), "{ip}");
        }
        for ip in [
            "1.1.1.1",
            "8.8.8.8",
            "100.63.255.255",
            "100.128.0.0",
            "172.15.255.255",
            "172.32.0.0",
            "198.17.255.255",
            "198.20.0.0",
            "198.51.99.255",
            "198.51.101.0",
            "203.0.112.255",
            "203.0.114.0",
            "223.255.255.254",
        ] {
            assert!(is_public_ipv4(ip.parse().unwrap()), "{ip}");
        }
    }

    #[tokio::test]
    async fn rejects_alternate_private_ipv4_spellings() {
        for raw in [
            "http://127.1/",
            "http://2130706433/",
            "http://0x7f000001/",
            "http://017700000001/",
        ] {
            let url = Url::parse(raw).unwrap();
            assert_eq!(url.host_str(), Some("127.0.0.1"), "{raw}");
            assert!(
                resolve_public_ipv4(&url, Duration::from_millis(50))
                    .await
                    .is_err(),
                "{raw}"
            );
        }
    }

    #[test]
    fn selects_only_public_ipv4_from_mixed_dns_answers() {
        let public = SocketAddr::from(([93, 184, 216, 34], 443));
        let selected = select_public_ipv4([
            SocketAddr::from(([127, 0, 0, 1], 443)),
            "[2606:2800:220:1:248:1893:25c8:1946]:443".parse().unwrap(),
            SocketAddr::from(([10, 0, 0, 1], 443)),
            public,
            public,
        ]);
        assert_eq!(selected, [public]);
    }

    #[tokio::test]
    async fn redirect_gets_a_fresh_pinned_resolution() {
        use std::sync::{Arc, Mutex};
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind((Ipv4Addr::LOCALHOST, 0))
            .await
            .unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            for response in [
                format!(
                    "HTTP/1.1 302 Found\r\nLocation: http://second.test:{}/next\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    address.port()
                ),
                "HTTP/1.1 200 OK\r\nContent-Length: 4\r\nConnection: close\r\n\r\ndone"
                    .to_string(),
            ] {
                let (mut stream, _) = listener.accept().await.unwrap();
                let mut request = [0; 1024];
                let _ = stream.read(&mut request).await.unwrap();
                stream.write_all(response.as_bytes()).await.unwrap();
            }
        });

        let seen = Arc::new(Mutex::new(Vec::new()));
        let resolver_seen = Arc::clone(&seen);
        let resolver = move |url: Url, _dns_timeout: Duration| {
            let seen = Arc::clone(&resolver_seen);
            async move {
                let host = url.host_str().unwrap().to_string();
                seen.lock().unwrap().push(host.clone());
                Ok((host, vec![address]))
            }
        };
        let response = fetch_public_url_limited_with_resolver(
            Url::parse(&format!("http://first.test:{}/start", address.port())).unwrap(),
            test_options(64),
            None,
            resolver,
        )
        .await
        .unwrap();
        server.await.unwrap();

        assert_eq!(response.bytes, b"done");
        assert_eq!(response.final_url.host_str(), Some("second.test"));
        assert_eq!(&*seen.lock().unwrap(), &["first.test", "second.test"]);
    }

    #[tokio::test]
    async fn rejects_streamed_body_after_crossing_limit() {
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
                .write_all(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n3\r\nabc\r\n3\r\ndef\r\n0\r\n\r\n")
                .await
                .unwrap();
        });
        let resolver = move |url: Url, _dns_timeout: Duration| async move {
            Ok((url.host_str().unwrap().to_string(), vec![address]))
        };
        let result = fetch_public_url_limited_with_resolver(
            Url::parse(&format!("http://stream.test:{}/", address.port())).unwrap(),
            test_options(5),
            None,
            resolver,
        )
        .await;
        server.await.unwrap();
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("exceeds the 5 byte limit"));
    }

    fn test_options(max_bytes: usize) -> PublicHttpFetchOptions {
        PublicHttpFetchOptions {
            max_bytes,
            connect_timeout: Duration::from_secs(1),
            total_timeout: Duration::from_secs(2),
            max_redirects: 3,
        }
    }

    #[test]
    fn bounds_response_bytes() {
        let mut bytes = vec![1, 2];
        append_limited(&mut bytes, &[3, 4], 4).unwrap();
        assert_eq!(bytes, [1, 2, 3, 4]);
        assert!(append_limited(&mut bytes, &[5], 4).is_err());
        assert_eq!(bytes, [1, 2, 3, 4]);
    }

    #[test]
    fn decodes_bounded_body_using_declared_charset() {
        let response = PublicHttpResponse {
            bytes: vec![0x63, 0x61, 0x66, 0xe9],
            content_type: Some("text/html; Charset=\"windows-1252\"".to_string()),
            final_url: Url::parse("https://example.com").unwrap(),
        };
        assert_eq!(response.text(), "café");
        assert_eq!(response.media_type(), Some("text/html"));
    }

    #[test]
    fn redirects_are_http_only() {
        let current = Url::parse("https://example.com/a/b").unwrap();
        assert_eq!(
            redirect_target(&current, "../next").unwrap().as_str(),
            "https://example.com/next"
        );
        assert!(redirect_target(&current, "file:///etc/passwd").is_err());
        assert!(redirect_target(&current, "data:text/plain,secret").is_err());
    }
}
