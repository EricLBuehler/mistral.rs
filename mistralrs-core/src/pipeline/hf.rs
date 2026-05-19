use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use hf_hub::{
    api::sync::{ApiError, ApiRepo},
    Cache, Repo, RepoType,
};
use tracing::{info, warn};

use super::FileListCache;

/// Env variable that, when set to a truthy value, disables all network calls
/// to the Hugging Face Hub. Only cached files are used.
pub const HF_HUB_OFFLINE_ENV: &str = "HF_HUB_OFFLINE";

/// Returns true when the user has requested fully-offline operation via
/// `HF_HUB_OFFLINE`. Accepted truthy values: `1`, `true`, `yes`, `on`
/// (case-insensitive). Anything else, or unset, is treated as online.
pub fn is_hf_hub_offline() -> bool {
    matches!(
        env::var(HF_HUB_OFFLINE_ENV)
            .ok()
            .map(|v| v.trim().to_ascii_lowercase()),
        Some(ref v) if matches!(v.as_str(), "1" | "true" | "yes" | "on")
    )
}

fn offline_repo(model_id: &Path, revision: &str) -> Repo {
    Repo::with_revision(
        model_id.display().to_string(),
        RepoType::Model,
        revision.to_string(),
    )
}

pub(crate) fn offline_cache_repo(model_id: &Path, revision: &str) -> hf_hub::CacheRepo {
    let cache = hf_hub_cache_dir().map(Cache::new).unwrap_or_default();
    cache.repo(offline_repo(model_id, revision))
}

pub(crate) fn offline_missing_file_error(
    model_id: &Path,
    file: &str,
    revision: &str,
) -> anyhow::Error {
    anyhow!(
        "`{HF_HUB_OFFLINE_ENV}` is set but `{file}` for `{}` (revision `{revision}`) was not found in the local Hugging Face cache. \
         Unset `{HF_HUB_OFFLINE_ENV}` or pre-download the file (e.g. via `huggingface-cli download`).",
        model_id.display()
    )
}

fn offline_snapshot_files(model_id: &Path, revision: &str) -> Vec<String> {
    fn walk(root: &Path, dir: &Path, out: &mut Vec<String>) -> std::io::Result<()> {
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                walk(root, &path, out)?;
            } else if let Ok(rel) = path.strip_prefix(root) {
                out.push(rel.to_string_lossy().replace('\\', "/"));
            }
        }
        Ok(())
    }

    let Some(cache_dir) = hf_hub_cache_dir() else {
        return Vec::new();
    };
    let repo = offline_repo(model_id, revision);
    let folder = repo.folder_name();
    let ref_path = cache_dir.join(&folder).join("refs").join(revision);
    // Refs file is typically a branch/tag name; fall back to treating revision as a literal commit.
    let commit = fs::read_to_string(&ref_path)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|_| revision.to_string());
    let snapshot_dir = cache_dir.join(&folder).join("snapshots").join(&commit);
    if !snapshot_dir.is_dir() {
        return Vec::new();
    }
    let mut files = Vec::new();
    let _ = walk(&snapshot_dir, &snapshot_dir, &mut files);
    files
}

#[derive(Clone, Debug)]
pub(crate) struct RemoteAccessIssue {
    pub status_code: Option<u16>,
    pub message: String,
}

/// Resolve the Hugging Face home directory.
///
/// Precedence:
/// 1. HF_HOME
/// 2. ~/.cache/huggingface
pub fn hf_home_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HOME")
        .ok()
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".cache").join("huggingface")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face home directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face Hub cache directory.
///
/// Precedence:
/// 1. HF_HUB_CACHE
/// 2. HF_HOME/hub
/// 3. ~/.cache/huggingface/hub
pub fn hf_hub_cache_dir() -> Option<PathBuf> {
    let dir = env::var("HF_HUB_CACHE")
        .ok()
        .map(PathBuf::from)
        .or_else(|| hf_home_dir().map(|home| home.join("hub")));

    if let Some(ref dir) = dir {
        if let Err(err) = fs::create_dir_all(dir) {
            warn!(
                "Could not create Hugging Face hub cache directory `{}`: {err}",
                dir.display()
            );
        }
    }

    dir
}

/// Resolve the Hugging Face token file path.
pub fn hf_token_path() -> Option<PathBuf> {
    hf_home_dir().map(|home| home.join("token"))
}

fn cache_dir() -> PathBuf {
    hf_hub_cache_dir().unwrap_or_else(|| PathBuf::from("./"))
}

fn cache_file_for_model(model_id: &Path) -> PathBuf {
    let sanitized_id = model_id.display().to_string().replace('/', "-");
    cache_dir().join(format!("{sanitized_id}_repo_list.json"))
}

fn read_cached_repo_files(cache_file: &Path) -> Option<Vec<String>> {
    if !cache_file.exists() {
        return None;
    }

    let mut file = match fs::File::open(cache_file) {
        Ok(file) => file,
        Err(err) => {
            warn!(
                "Could not open Hugging Face repo cache file `{}`: {err}",
                cache_file.display()
            );
            return None;
        }
    };

    let mut contents = String::new();
    if let Err(err) = file.read_to_string(&mut contents) {
        warn!(
            "Could not read Hugging Face repo cache file `{}`: {err}",
            cache_file.display()
        );
        return None;
    }

    match serde_json::from_str::<FileListCache>(&contents) {
        Ok(cache) => {
            info!("Read from cache file `{}`", cache_file.display());
            Some(cache.files)
        }
        Err(err) => {
            warn!(
                "Could not parse Hugging Face repo cache file `{}`: {err}",
                cache_file.display()
            );
            None
        }
    }
}

fn write_cached_repo_files(cache_file: &Path, files: &[String]) {
    let cache = FileListCache {
        files: files.to_vec(),
    };
    match serde_json::to_string_pretty(&cache) {
        Ok(json) => {
            if let Err(err) = fs::write(cache_file, json) {
                warn!(
                    "Could not write Hugging Face repo cache file `{}`: {err}",
                    cache_file.display()
                );
            } else {
                info!("Write to cache file `{}`", cache_file.display());
            }
        }
        Err(err) => warn!(
            "Could not serialize Hugging Face repo cache for `{}`: {err}",
            cache_file.display()
        ),
    }
}

pub(crate) fn parse_status_code(message: &str) -> Option<u16> {
    let marker = "status code ";
    let (_, tail) = message.split_once(marker)?;
    let digits = tail
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>();
    digits.parse().ok()
}

pub(crate) fn api_error_status_code(err: &ApiError) -> Option<u16> {
    match err {
        ApiError::TooManyRetries(inner) => api_error_status_code(inner),
        _ => parse_status_code(&err.to_string()),
    }
}

pub(crate) fn should_propagate_api_error(err: &ApiError) -> bool {
    matches!(api_error_status_code(err), Some(401 | 403 | 404))
}

pub(crate) fn remote_issue_from_api_error(
    model_id: &Path,
    file: Option<&str>,
    err: &ApiError,
) -> RemoteAccessIssue {
    let target = match file {
        Some(file) => format!("`{file}` for `{}`", model_id.display()),
        None => format!("`{}`", model_id.display()),
    };
    RemoteAccessIssue {
        status_code: api_error_status_code(err),
        message: format!("Failed to access {target}: {err}"),
    }
}

pub(crate) fn hf_access_error(model_id: &Path, issue: &RemoteAccessIssue) -> anyhow::Error {
    match issue.status_code {
        Some(code @ (401 | 403)) => anyhow!(
            "Could not access `{}` on Hugging Face (HTTP {code}). You may need to run `mistralrs login` or set HF_TOKEN.",
            model_id.display()
        ),
        Some(404) => anyhow!(
            "Model `{}` was not found or is not accessible on Hugging Face (HTTP 404). Check the model ID and your access token.",
            model_id.display()
        ),
        Some(code) => anyhow!(
            "Failed to access `{}` on Hugging Face (HTTP {code}): {}",
            model_id.display(),
            issue.message
        ),
        None => anyhow!(
            "Failed to access `{}` on Hugging Face: {}",
            model_id.display(),
            issue.message
        ),
    }
}

pub(crate) fn hf_api_error(model_id: &Path, file: Option<&str>, err: &ApiError) -> anyhow::Error {
    let status_code = api_error_status_code(err);
    let file_context = file
        .map(|f| format!(" while fetching `{f}`"))
        .unwrap_or_default();
    match status_code {
        Some(code @ (401 | 403)) => anyhow!(
            "Could not access `{}` on Hugging Face (HTTP {code}){file_context}. You may need to run `mistralrs login` or set HF_TOKEN.",
            model_id.display()
        ),
        Some(404) => anyhow!(
            "Model `{}` was not found or is not accessible on Hugging Face (HTTP 404){file_context}. Check the model ID and your access token.",
            model_id.display()
        ),
        Some(code) => anyhow!(
            "Failed to access `{}` on Hugging Face (HTTP {code}){file_context}: {err}",
            model_id.display()
        ),
        None => anyhow!(
            "Failed to access `{}` on Hugging Face{file_context}: {err}",
            model_id.display()
        ),
    }
}

pub(crate) fn local_file_missing_error(model_id: &Path, file: &str) -> anyhow::Error {
    anyhow!(
        "File `{file}` was not found at local model path `{}`.",
        model_id.display()
    )
}

pub(crate) fn list_repo_files(
    api: &ApiRepo,
    model_id: &Path,
    should_error: bool,
    revision: &str,
) -> Result<Vec<String>> {
    if model_id.exists() {
        let listing = fs::read_dir(model_id).map_err(|err| {
            anyhow!(
                "Cannot list local model directory `{}`: {err}",
                model_id.display()
            )
        })?;
        let files = listing
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                entry
                    .path()
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(std::string::ToString::to_string)
            })
            .collect::<Vec<_>>();
        return Ok(files);
    }

    let cache_file = cache_file_for_model(model_id);
    if let Some(files) = read_cached_repo_files(&cache_file) {
        return Ok(files);
    }

    if is_hf_hub_offline() {
        let files = offline_snapshot_files(model_id, revision);
        if !files.is_empty() {
            write_cached_repo_files(&cache_file, &files);
            return Ok(files);
        }
        if should_error {
            return Err(anyhow!(
                "`{HF_HUB_OFFLINE_ENV}` is set but no cached file list or snapshot was found for `{}` (revision `{revision}`).",
                model_id.display()
            ));
        }
        warn!(
            "`{HF_HUB_OFFLINE_ENV}` is set and no local Hugging Face cache was found for `{}` (revision `{revision}`)",
            model_id.display()
        );
        return Ok(Vec::new());
    }

    match api.info() {
        Ok(repo) => {
            let files = repo
                .siblings
                .iter()
                .map(|x| x.rfilename.clone())
                .collect::<Vec<_>>();
            write_cached_repo_files(&cache_file, &files);
            Ok(files)
        }
        Err(err) => {
            if should_error || should_propagate_api_error(&err) {
                Err(hf_api_error(model_id, None, &err))
            } else {
                warn!(
                    "Could not get directory listing from Hugging Face for `{}`: {err}",
                    model_id.display()
                );
                Ok(Vec::new())
            }
        }
    }
}

pub(crate) fn get_file(
    api: &ApiRepo,
    model_id: &Path,
    file: &str,
    revision: &str,
) -> Result<PathBuf> {
    if model_id.exists() {
        let path = model_id.join(file);
        if !path.exists() {
            return Err(local_file_missing_error(model_id, file));
        }
        info!("Loading `{file}` locally at `{}`", path.display());
        return Ok(path);
    }

    if is_hf_hub_offline() {
        if let Some(path) = offline_cache_repo(model_id, revision).get(file) {
            info!(
                "Loading `{file}` from local Hugging Face cache at `{}` (offline mode)",
                path.display()
            );
            return Ok(path);
        }
        return Err(offline_missing_file_error(model_id, file, revision));
    }

    api.get(file)
        .map_err(|err| hf_api_error(model_id, Some(file), &err))
}

/// Like [`get_file`] but returns `Ok(None)` (instead of an error) when the file is genuinely missing, and used with `HF_HUB_OFFLINE`.
pub(crate) fn try_get_file(
    api: &ApiRepo,
    model_id: &Path,
    file: &str,
    revision: &str,
) -> std::result::Result<Option<PathBuf>, ApiError> {
    if model_id.exists() {
        let path = model_id.join(file);
        if path.exists() {
            info!("Loading `{file}` locally at `{}`", path.display());
            return Ok(Some(path));
        }
        return Ok(None);
    }

    if is_hf_hub_offline() {
        return Ok(offline_cache_repo(model_id, revision).get(file));
    }

    match api.get(file) {
        Ok(p) => Ok(Some(p)),
        Err(err) => match api_error_status_code(&err) {
            Some(404) => Ok(None),
            _ => Err(err),
        },
    }
}

/// Best-effort file listing for a HF repo. Returns `None` on 404, API failure,
/// or offline-without-cache. Quiet by design: callers choose what to log.
pub fn probe_hf_repo_files(
    model_id: &str,
    revision: &str,
    token_source: &crate::pipeline::TokenSource,
) -> Option<Vec<String>> {
    use hf_hub::api::sync::ApiBuilder;

    if is_hf_hub_offline() {
        let files = offline_snapshot_files(Path::new(model_id), revision);
        return (!files.is_empty()).then_some(files);
    }

    let token = crate::utils::tokens::get_token(token_source).ok().flatten();
    let cache = hf_hub_cache_dir()
        .map(Cache::new)
        .unwrap_or_else(Cache::from_env);
    let mut api = ApiBuilder::from_cache(cache)
        .with_progress(false)
        .with_token(token);
    if let Some(cache_dir) = hf_hub_cache_dir() {
        api = api.with_cache_dir(cache_dir);
    }
    let repo = api.build().ok()?.repo(Repo::with_revision(
        model_id.to_string(),
        RepoType::Model,
        revision.to_string(),
    ));
    repo.info()
        .ok()
        .map(|info| info.siblings.into_iter().map(|s| s.rfilename).collect())
}
