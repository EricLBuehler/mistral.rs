//! Resolve `oci://` model ids to a local directory.
//!
//! Models published as [CNCF ModelPack](https://github.com/modelpack/model-spec) artifacts live in
//! ordinary container registries, so they reuse the registry, credentials, mirroring and air-gap
//! tooling a deployment already has. Pulling is delegated to a running
//! [`llmman serve`](https://github.com/llmmanorg/llmman), which already implements the ModelPack
//! media types, registry auth and a content-addressed store.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::info;

/// Required rather than sniffed: a bare `registry/name:tag` is indistinguishable from an HF repo id.
const SCHEME: &str = "oci://";

const DEFAULT_BIN: &str = "llmman";
const BIN_ENV: &str = "MISTRALRS_LLMMAN_BIN";
const HOST_ENV: &str = "LLMMAN_HOST";
const DEFAULT_HOST: &str = "127.0.0.1";
const DEFAULT_PORT: u16 = 17434;
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);
const STATUS_SUCCESS: &str = "success";

/// Case-insensitive, since URI schemes are per RFC 3986.
fn is_oci_ref(value: &str) -> bool {
    value.len() > SCHEME.len() && value[..SCHEME.len()].eq_ignore_ascii_case(SCHEME)
}

fn strip_scheme(value: &str) -> &str {
    if is_oci_ref(value) {
        &value[SCHEME.len()..]
    } else {
        value
    }
}

fn llmman_bin() -> String {
    std::env::var(BIN_ENV)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| DEFAULT_BIN.to_string())
}

fn endpoint() -> String {
    endpoint_from(std::env::var(HOST_ENV).ok().as_deref())
}

/// Parses `LLMMAN_HOST` as llmman's own clients do. Split out so it is testable without the env.
fn endpoint_from(value: Option<&str>) -> String {
    let raw = value.unwrap_or("").trim().trim_matches(['"', '\'']);
    if raw.is_empty() {
        return format!("http://{DEFAULT_HOST}:{DEFAULT_PORT}");
    }

    let after_scheme = raw.split_once("://").map(|(_, rest)| rest).unwrap_or(raw);
    let hostport = after_scheme.split('/').next().unwrap_or(after_scheme);

    let (host, port) = if let Some(rest) = hostport.strip_prefix('[') {
        match rest.split_once(']') {
            Some((inner, tail)) => (
                format!("[{inner}]"),
                tail.strip_prefix(':')
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(DEFAULT_PORT),
            ),
            None => (hostport.to_string(), DEFAULT_PORT),
        }
    } else {
        match hostport.rsplit_once(':') {
            Some((h, p)) if !h.is_empty() && p.chars().all(|c| c.is_ascii_digit()) => {
                (h.to_string(), p.parse().unwrap_or(DEFAULT_PORT))
            }
            _ => (hostport.to_string(), DEFAULT_PORT),
        }
    };

    let host = if host.is_empty() {
        DEFAULT_HOST.to_string()
    } else {
        connectable_host(&host)
    };
    format!("http://{host}:{port}")
}

/// Wildcard binds (`0.0.0.0`, `[::]`) are not connectable; match by value so `[0:0:0:0:0:0:0:0]` counts.
fn connectable_host(host: &str) -> String {
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<std::net::IpAddr>() {
        Ok(ip) if ip.is_unspecified() => {
            if ip.is_ipv4() {
                "127.0.0.1".to_string()
            } else {
                "[::1]".to_string()
            }
        }
        Ok(ip) if ip.is_ipv6() => format!("[{bare}]"),
        _ => host.to_string(),
    }
}

/// One NDJSON object from `/api/pull`.
#[derive(Deserialize, Default)]
struct PullLine {
    #[serde(default)]
    status: String,
    #[serde(default)]
    error: String,
    #[serde(default)]
    total: u64,
    #[serde(default)]
    completed: u64,
}

/// The subset of `llmman resolve`'s output we depend on; unknown fields are ignored so it can grow.
#[derive(Deserialize)]
struct ResolveOutput {
    path: String,
}

/// Probes reachability and identity: something else on the port is worth distinguishing.
fn check_daemon(client: &reqwest::blocking::Client, base: &str) -> Result<()> {
    let resp = client
        .get(format!("{base}/api/version"))
        .timeout(PROBE_TIMEOUT)
        .send()
        .with_context(|| {
            format!(
                "no llmman daemon reachable at {base}. Start one with `llmman serve`, \
                 or point {HOST_ENV} at an existing daemon."
            )
        })?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "llmman daemon at {base} answered /api/version with {}",
            resp.status()
        );
    }

    let body: serde_json::Value = resp
        .json()
        .with_context(|| format!("the server at {base} is not an llmman daemon"))?;
    if body.get("version").and_then(|v| v.as_str()).is_none() {
        anyhow::bail!("the server at {base} is not an llmman daemon (no version in /api/version)");
    }
    Ok(())
}

/// Returns whether the line reported success. Non-JSON is tolerated rather than aborting a live pull.
fn handle_pull_line(line: &str, reference: &str, succeeded: &mut bool) -> Result<bool> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Ok(false);
    }
    let Ok(parsed) = serde_json::from_str::<PullLine>(trimmed) else {
        return Ok(false);
    };

    if !parsed.error.is_empty() {
        anyhow::bail!("llmman pull of '{reference}' failed: {}", parsed.error);
    }
    if parsed.status == STATUS_SUCCESS {
        *succeeded = true;
        return Ok(true);
    }
    if !parsed.status.is_empty() {
        if parsed.total > 0 {
            info!(
                "llmman: {} ({}/{} bytes)",
                parsed.status, parsed.completed, parsed.total
            );
        } else {
            info!("llmman: {}", parsed.status);
        }
    }
    Ok(false)
}

/// Streams NDJSON so a multi-gigabyte fetch is not silent; errors arrive in-band at HTTP 200.
fn pull(client: &reqwest::blocking::Client, base: &str, reference: &str) -> Result<()> {
    use std::io::BufRead;

    let resp = client
        .post(format!("{base}/api/pull"))
        .json(&serde_json::json!({ "model": reference }))
        .send()
        .with_context(|| format!("llmman pull of '{reference}' failed"))?;

    if !resp.status().is_success() {
        anyhow::bail!("llmman pull of '{reference}' failed: {}", resp.status());
    }

    let mut succeeded = false;
    let reader = std::io::BufReader::new(resp);
    for line in reader.lines() {
        let line = line.with_context(|| format!("reading llmman pull stream for '{reference}'"))?;
        handle_pull_line(&line, reference, &mut succeeded)?;
    }

    if !succeeded {
        anyhow::bail!("llmman pull of '{reference}' ended without reporting success");
    }
    Ok(())
}

/// Takes the last non-empty line, so a diagnostic leaked onto stdout does not break resolution.
fn parse_resolve_output(stdout: &str, reference: &str) -> Result<PathBuf> {
    let line = stdout
        .lines()
        .map(str::trim)
        .rfind(|l| !l.is_empty())
        .with_context(|| format!("llmman resolve '{reference}': no output on stdout"))?;

    let parsed: ResolveOutput = serde_json::from_str(line).with_context(|| {
        format!("llmman resolve '{reference}': could not parse output as JSON: {line}")
    })?;

    if parsed.path.trim().is_empty() {
        anyhow::bail!("llmman resolve '{reference}': returned an empty path");
    }

    let path = PathBuf::from(parsed.path);
    if !path.exists() {
        anyhow::bail!(
            "llmman resolve '{reference}': reported path '{}' does not exist",
            path.display()
        );
    }
    Ok(path)
}

/// The daemon exposes no local path, so ask the CLI. `--no-pull` keeps the daemon the only thing
/// that touches the network.
fn resolve(reference: &str) -> Result<PathBuf> {
    let bin = llmman_bin();
    let output = std::process::Command::new(&bin)
        .arg("resolve")
        .arg("--no-pull")
        .arg(reference)
        .stderr(std::process::Stdio::inherit())
        .stdout(std::process::Stdio::piped())
        .stdin(std::process::Stdio::null())
        .output()
        .with_context(|| {
            format!(
                "failed to run '{bin} resolve --no-pull {reference}'. Install llmman \
                 (https://github.com/llmmanorg/llmman) and put it on PATH, or point \
                 {BIN_ENV} at it."
            )
        })?;

    if !output.status.success() {
        anyhow::bail!(
            "'{bin} resolve --no-pull {reference}' failed with {}. See the error above.",
            output.status
        );
    }

    let stdout = String::from_utf8(output.stdout)
        .with_context(|| format!("llmman resolve '{reference}': stdout was not valid UTF-8"))?;
    parse_resolve_output(&stdout, reference)
}

fn cache() -> &'static Mutex<HashMap<String, PathBuf>> {
    static CACHE: OnceLock<Mutex<HashMap<String, PathBuf>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn pull_and_resolve(reference: &str) -> Result<PathBuf> {
    let base = endpoint();
    let client = reqwest::blocking::Client::new();
    check_daemon(&client, &base)?;

    info!("Pulling OCI model '{reference}' via llmman daemon at {base}");
    pull(&client, &base, reference)?;
    resolve(reference)
}

/// Pull an `oci://` id (once per process) and return its extracted directory; any other shape
/// returns `Ok(None)` untouched, so local dirs and HF ids contact nothing.
pub(crate) fn maybe_resolve(model_id: &Path) -> Result<Option<PathBuf>> {
    let Some(raw) = model_id.to_str() else {
        return Ok(None);
    };
    if !is_oci_ref(raw) {
        return Ok(None);
    }

    let reference = strip_scheme(raw).trim().to_string();
    if reference.is_empty() {
        anyhow::bail!("empty OCI model reference: '{raw}'");
    }

    if let Some(hit) = cache().lock().unwrap().get(&reference) {
        return Ok(Some(hit.clone()));
    }

    // Several callers are async, and a reqwest blocking client panics when its runtime is dropped
    // inside an async context, so keep the whole pull on a thread of its own.
    let path = match std::thread::scope(|s| s.spawn(|| pull_and_resolve(&reference)).join()) {
        Ok(res) => res?,
        Err(_) => anyhow::bail!("llmman resolution of '{reference}' panicked"),
    };

    cache().lock().unwrap().insert(reference, path.clone());
    Ok(Some(path))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_the_oci_scheme() {
        assert!(is_oci_ref("oci://ghcr.io/org/model:tag"));
        assert!(is_oci_ref("OCI://ghcr.io/org/model:tag"));
    }

    #[test]
    fn leaves_every_other_reference_shape_alone() {
        assert!(!is_oci_ref("meta-llama/Llama-3.1-8B"));
        assert!(!is_oci_ref("ghcr.io/org/model:tag"));
        assert!(!is_oci_ref("/local/path/to/model"));
        assert!(!is_oci_ref("s3://bucket/key"));
        assert!(!is_oci_ref(""));
        assert!(!is_oci_ref("oci://"));
    }

    #[test]
    fn strips_the_scheme_only_when_present() {
        assert_eq!(
            strip_scheme("oci://ghcr.io/org/model:tag"),
            "ghcr.io/org/model:tag"
        );
        assert_eq!(
            strip_scheme("meta-llama/Llama-3.1-8B"),
            "meta-llama/Llama-3.1-8B"
        );
    }

    #[test]
    fn maybe_resolve_ignores_non_oci_paths() {
        assert!(maybe_resolve(Path::new("meta-llama/Llama-3.1-8B"))
            .unwrap()
            .is_none());
        assert!(maybe_resolve(Path::new("/tmp")).unwrap().is_none());
    }

    #[test]
    fn maybe_resolve_rejects_an_empty_reference() {
        assert!(maybe_resolve(Path::new("oci://   ")).is_err());
    }

    #[test]
    fn endpoint_defaults_and_parses_every_host_form() {
        assert_eq!(endpoint_from(None), "http://127.0.0.1:17434");
        assert_eq!(endpoint_from(Some("")), "http://127.0.0.1:17434");
        assert_eq!(endpoint_from(Some("1.2.3.4:9999")), "http://1.2.3.4:9999");
        assert_eq!(endpoint_from(Some("1.2.3.4")), "http://1.2.3.4:17434");
        assert_eq!(
            endpoint_from(Some("http://1.2.3.4:9999/ignored")),
            "http://1.2.3.4:9999"
        );
        assert_eq!(
            endpoint_from(Some("\"1.2.3.4:9999\"")),
            "http://1.2.3.4:9999"
        );
    }

    #[test]
    fn endpoint_rewrites_a_wildcard_bind_to_loopback() {
        assert_eq!(endpoint_from(Some("0.0.0.0:9999")), "http://127.0.0.1:9999");
        assert_eq!(endpoint_from(Some("[::]:9999")), "http://[::1]:9999");
        assert_eq!(
            endpoint_from(Some("[0:0:0:0:0:0:0:0]:9999")),
            "http://[::1]:9999"
        );
    }

    // Build via serde_json, never format!: a Windows path is full of backslashes.
    #[test]
    fn parses_the_documented_resolve_contract() {
        let dir = tempfile::tempdir().unwrap();
        let line = serde_json::json!({
            "reference": "r",
            "path": dir.path(),
            "format": "safetensors",
        })
        .to_string();
        assert_eq!(parse_resolve_output(&line, "r").unwrap(), dir.path());
    }

    #[test]
    fn tolerates_leaked_diagnostics_and_unknown_fields() {
        let dir = tempfile::tempdir().unwrap();
        let line = serde_json::json!({ "path": dir.path(), "mmproj": "/x" }).to_string();
        let out = format!("pulling blobs...\n{line}\n");
        assert_eq!(parse_resolve_output(&out, "r").unwrap(), dir.path());
    }

    #[test]
    fn rejects_malformed_resolve_output() {
        assert!(parse_resolve_output("", "r").is_err());
        assert!(parse_resolve_output("   \n\n", "r").is_err());
        assert!(parse_resolve_output("not json", "r").is_err());
        assert!(parse_resolve_output("{\"no_path\":1}", "r").is_err());
        assert!(parse_resolve_output(r#"{"path":""}"#, "r").is_err());
        assert!(parse_resolve_output(r#"{"path":"/nonexistent/xyzzy"}"#, "r").is_err());
    }

    #[test]
    fn pull_line_reports_an_in_band_error() {
        let mut ok = false;
        let err = handle_pull_line(r#"{"error":"unauthorized"}"#, "r", &mut ok).unwrap_err();
        assert!(err.to_string().contains("unauthorized"));
        assert!(!ok);
    }

    #[test]
    fn pull_line_marks_success_and_tolerates_noise() {
        let mut ok = false;
        assert!(!handle_pull_line("not json", "r", &mut ok).unwrap());
        assert!(!handle_pull_line("", "r", &mut ok).unwrap());
        assert!(!handle_pull_line(
            r#"{"status":"pulling","completed":5,"total":10}"#,
            "r",
            &mut ok
        )
        .unwrap());
        assert!(!ok);
        assert!(handle_pull_line(r#"{"status":"success"}"#, "r", &mut ok).unwrap());
        assert!(ok);
    }
}
