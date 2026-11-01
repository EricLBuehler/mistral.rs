use std::{
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc, Mutex,
    },
    thread::{self, JoinHandle},
    time::Duration,
};

use super::{AsyncHubRepo, HubClient};
use crate::{pipeline::hf, pipeline::TokenSource, utils::tokens::get_token};

const CHILD_CASE: &str = "MISTRALRS_HUB_TEST_CASE";
const CHILD_ROOT: &str = "MISTRALRS_HUB_TEST_ROOT";
const CHILD_TEST: &str = "pipeline::hf::client::tests::hub_environment_child";
const COMMIT: &str = "1234567890123456789012345678901234567890";
const XET_HASH: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
const CONTENT: &[u8] = b"download fixture";
const SERVER_POLL: Duration = Duration::from_millis(5);
const SOCKET_TIMEOUT: Duration = Duration::from_secs(5);
const MAX_REQUEST_BYTES: usize = 16384;

struct TestServer {
    endpoint: String,
    requests: Arc<Mutex<Vec<String>>>,
    stopped: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl TestServer {
    fn new() -> Self {
        Self::with_xet(false)
    }

    fn with_xet(advertise_xet: bool) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let endpoint = format!("http://{}", listener.local_addr().unwrap());
        listener.set_nonblocking(true).unwrap();
        let requests = Arc::new(Mutex::new(Vec::new()));
        let stopped = Arc::new(AtomicBool::new(false));
        let worker_requests = requests.clone();
        let worker_stopped = stopped.clone();
        let worker = thread::spawn(move || {
            while !worker_stopped.load(Ordering::Relaxed) {
                let (mut stream, _) = match listener.accept() {
                    Ok(connection) => connection,
                    Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                        thread::sleep(SERVER_POLL);
                        continue;
                    }
                    Err(error) => panic!("{error}"),
                };
                stream.set_read_timeout(Some(SOCKET_TIMEOUT)).unwrap();
                let mut buffer = [0; MAX_REQUEST_BYTES];
                let mut length = 0;
                while !buffer[..length].ends_with(b"\r\n\r\n") {
                    let read = stream.read(&mut buffer[length..]).unwrap();
                    assert_ne!(read, 0);
                    length += read;
                }
                let request = String::from_utf8(buffer[..length].to_vec()).unwrap();
                worker_requests.lock().unwrap().push(request.clone());
                let mut request_line = request.lines().next().unwrap().split_whitespace();
                let method = request_line.next().unwrap();
                let path = request_line.next().unwrap();
                let (status, body) = if path.contains("missing.bin") {
                    ("404 Not Found", Vec::new())
                } else if path.contains("private.bin") || path.contains("/xet-") {
                    ("401 Unauthorized", Vec::new())
                } else if path.contains("/api/models/") {
                    (
                        "200 OK",
                        serde_json::to_vec(&serde_json::json!({
                            "id":"org/model", "sha":COMMIT,
                            "siblings":[{"rfilename":"model.bin"}]
                        }))
                        .unwrap(),
                    )
                } else if path.contains("/resolve/") {
                    ("200 OK", CONTENT.to_vec())
                } else {
                    ("400 Bad Request", Vec::new())
                };
                let xet_header = if advertise_xet {
                    format!("X-Xet-Hash: {XET_HASH}\r\n")
                } else {
                    String::new()
                };
                let response = format!(
                    "HTTP/1.1 {status}\r\nContent-Length: {}\r\nContent-Type: application/json\r\nETag: \"fixture\"\r\nX-Repo-Commit: {COMMIT}\r\n{xet_header}Connection: close\r\n\r\n",
                    body.len()
                );
                stream.write_all(response.as_bytes()).unwrap();
                if method != "HEAD" {
                    stream.write_all(&body).unwrap();
                }
            }
        });
        Self {
            endpoint,
            requests,
            stopped,
            worker: Some(worker),
        }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().unwrap().clone()
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        self.stopped.store(true, Ordering::Relaxed);
        self.worker.take().unwrap().join().unwrap();
    }
}

fn run_child(case: &str, server: &TestServer) {
    let root = tempfile::tempdir().unwrap();
    let token_path = root.path().join("custom-token");
    fs::write(&token_path, "saved-token\n").unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap());
    child
        .args(["--exact", CHILD_TEST, "--nocapture"])
        .env(CHILD_CASE, case)
        .env(CHILD_ROOT, root.path())
        .env("HF_HOME", root.path().join("ambient-home"))
        .env("HF_HUB_CACHE", root.path().join("ambient-cache"))
        .env("HF_TOKEN_PATH", token_path)
        .env("HF_ENDPOINT", &server.endpoint)
        .env("HF_TOKEN", "ambient-token")
        .env_remove("HF_HUB_TOKEN")
        .env_remove("HF_HUB_OFFLINE")
        .env_remove("HF_HUB_DISABLE_IMPLICIT_TOKEN");
    if case == "offline" {
        child.env("HF_HUB_OFFLINE", "true");
    }
    if case == "token_path" {
        child.env_remove("HF_TOKEN");
    }
    if case == "empty_token" {
        child.env("HF_TOKEN", " \n").env("HF_HUB_TOKEN", " ");
    }
    if case == "xdg_cache" {
        child
            .env_remove("HF_HOME")
            .env_remove("HF_HUB_CACHE")
            .env("XDG_CACHE_HOME", root.path().join("xdg"));
    }
    if case == "disable_implicit" {
        child.env("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1");
    }
    let output = child.output().unwrap();
    assert!(
        output.status.success(),
        "child {case} failed:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn explicit_auth_ignores_ambient_credentials() {
    let server = TestServer::new();
    run_child("auth", &server);
    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    assert!(!requests[0].to_ascii_lowercase().contains("authorization:"));
    assert!(requests[1].contains("Bearer explicit-token"));
    assert!(requests[2].contains("Bearer ambient-token"));
}

#[test]
fn token_path_and_implicit_auth_opt_out() {
    let server = TestServer::new();
    run_child("token_path", &server);
    run_child("empty_token", &server);
    run_child("disable_implicit", &server);
    run_child("xdg_cache", &server);
    assert!(server.requests().is_empty());
}

#[test]
fn offline_uses_client_cache_override_for_sync_and_async() {
    let server = TestServer::new();
    run_child("offline", &server);
    assert!(server.requests().is_empty());
}

#[test]
fn xet_bootstrap_uses_explicit_auth_and_preserves_denials() {
    let server = TestServer::with_xet(true);
    run_child("xet_auth", &server);
    let requests = server.requests();
    let token_requests = requests
        .iter()
        .filter(|request| request.contains("/xet-read-token/"))
        .collect::<Vec<_>>();
    assert_eq!(token_requests.len(), 2);
    assert!(!token_requests[0]
        .to_ascii_lowercase()
        .contains("authorization:"));
    assert!(token_requests[1].contains("Bearer explicit-token"));
    assert!(!requests
        .iter()
        .any(|request| request.starts_with("GET /org/model/resolve/")));
}

#[test]
fn http_download_reuses_shared_cache() {
    let server = TestServer::new();
    run_child("http", &server);
    let requests = server.requests();
    assert!(requests
        .iter()
        .any(|request| request.starts_with("GET /org/model/resolve/")));
    assert!(requests
        .iter()
        .any(|request| request.contains("feature%2Ftest")));
    assert!(!requests.iter().any(|request| request.contains("/xet-")));
    assert!(!requests
        .iter()
        .any(|request| request.to_ascii_lowercase().contains("authorization:")));
    let branch_requests = requests
        .iter()
        .filter(|request| {
            request
                .lines()
                .next()
                .unwrap()
                .contains("feature%2Ftest/model.bin")
        })
        .count();
    assert_eq!(
        branch_requests, 2,
        "warm cache should not send another request"
    );
}

fn cached_fixture(root: &Path, content: &[u8]) -> PathBuf {
    let repository = root.join("models--org--model");
    fs::create_dir_all(repository.join("refs/feature")).unwrap();
    fs::write(repository.join("refs/feature/test"), COMMIT).unwrap();
    let snapshot = repository.join("snapshots").join(COMMIT);
    fs::create_dir_all(snapshot.join("nested")).unwrap();
    fs::write(snapshot.join("nested/model.bin"), content).unwrap();
    snapshot.join("nested/model.bin")
}

#[test]
fn hub_environment_child() {
    let Ok(case) = std::env::var(CHILD_CASE) else {
        return;
    };
    let root = PathBuf::from(std::env::var_os(CHILD_ROOT).unwrap());
    let model = Path::new("org/model");
    let cache = root.join("explicit-cache");
    match case.as_str() {
        "auth" => {
            for source in [
                TokenSource::None,
                TokenSource::Literal("explicit-token".into()),
                TokenSource::CacheToken,
            ] {
                let client = HubClient::new(&source, false, Some(cache.clone())).unwrap();
                assert_eq!(
                    client.repo("org/model", "main").files().unwrap(),
                    ["model.bin"]
                );
            }
        }
        "token_path" | "empty_token" => assert_eq!(
            get_token(&TokenSource::CacheToken).unwrap().as_deref(),
            Some("saved-token")
        ),
        "disable_implicit" => {
            assert_eq!(get_token(&TokenSource::CacheToken).unwrap(), None);
            assert_eq!(
                get_token(&TokenSource::Literal("explicit".into()))
                    .unwrap()
                    .as_deref(),
                Some("explicit")
            );
        }
        "xdg_cache" => {
            assert_eq!(hf::hf_home_dir().unwrap(), root.join("xdg/huggingface"));
            assert_eq!(
                hf::hf_hub_cache_dir().unwrap(),
                root.join("xdg/huggingface/hub")
            );
        }
        "xet_auth" => {
            for source in [
                TokenSource::None,
                TokenSource::Literal("explicit-token".into()),
            ] {
                let client = HubClient::new(&source, false, Some(cache.clone())).unwrap();
                let error = client
                    .repo("org/model", "main")
                    .get("model.bin")
                    .unwrap_err();
                assert_eq!(hf::api_error_status_code(&error), Some(401));
            }
        }
        "offline" => {
            let global_cache = root.join("global-cache");
            cached_fixture(&global_cache, b"wrong cache");
            crate::set_hf_cache_path(global_cache);
            let expected = cached_fixture(&cache, CONTENT);
            let client = HubClient::new(&TokenSource::None, false, Some(cache.clone())).unwrap();
            let repo = client.repo("org/model", "feature/test");
            assert_eq!(
                hf::get_file(&repo, model, "nested/model.bin", "feature/test").unwrap(),
                expected
            );
            assert_eq!(fs::read(&expected).unwrap(), CONTENT);
            assert_eq!(repo.files().unwrap(), ["nested/model.bin"]);
            assert_eq!(
                hf::list_repo_files(&repo, model, true, "feature/test").unwrap(),
                ["nested/model.bin"]
            );
            assert!(hf::try_get_file(&repo, model, "missing.bin")
                .unwrap()
                .is_none());
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async {
                let repo = AsyncHubRepo::new(
                    &TokenSource::None,
                    Some(cache.clone()),
                    "org/model",
                    "feature/test",
                )
                .unwrap();
                assert_eq!(repo.get("nested/model.bin").await.unwrap(), expected);
                assert_eq!(repo.files().await.unwrap(), ["nested/model.bin"]);
                assert_eq!(
                    hf::list_repo_files_async(&repo, model, true, "feature/test")
                        .await
                        .unwrap(),
                    ["nested/model.bin"]
                );
                assert!(hf::try_get_file_async(&repo, model, "missing.bin")
                    .await
                    .unwrap()
                    .is_none());
            });
            let invalid_ref = cache.join("models--org--model/refs/broken");
            fs::create_dir(&invalid_ref).unwrap();
            assert!(client.repo("org/model", "broken").files().is_err());
        }
        "http" => {
            let client = HubClient::new(&TokenSource::None, false, Some(cache.clone())).unwrap();
            let repo = client.repo("org/model", "feature/test");
            let downloaded = repo.get("model.bin").unwrap();
            assert_eq!(fs::read(&downloaded).unwrap(), CONTENT);
            assert_eq!(repo.get("model.bin").unwrap(), downloaded);
            let independent =
                HubClient::new(&TokenSource::None, false, Some(cache.clone())).unwrap();
            assert_eq!(
                independent
                    .repo("org/model", "feature/test")
                    .get("model.bin")
                    .unwrap(),
                downloaded
            );
            assert!(hf::try_get_file(&repo, model, "missing.bin")
                .unwrap()
                .is_none());
            let denied = hf::try_get_file(&repo, model, "private.bin").unwrap_err();
            assert_eq!(hf::api_error_status_code(&denied), Some(401));
        }
        _ => panic!("unknown child case"),
    }
}
