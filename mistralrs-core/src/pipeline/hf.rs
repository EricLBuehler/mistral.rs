use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

use anyhow::{anyhow, Result};
use hf_hub::api::sync::{ApiError, ApiRepo};
use tracing::{info, warn};

use super::FileListCache;

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

pub(crate) fn get_file(api: &ApiRepo, model_id: &Path, file: &str) -> Result<PathBuf> {
    if model_id.exists() {
        let path = model_id.join(file);
        if !path.exists() {
            return Err(local_file_missing_error(model_id, file));
        }
        info!("Loading `{file}` locally at `{}`", path.display());
        return Ok(path);
    }

    api.get(file)
        .map_err(|err| hf_api_error(model_id, Some(file), &err))
}
