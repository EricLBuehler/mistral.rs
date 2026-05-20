use std::path::{Path, PathBuf};

use hf_hub::{
    api::sync::{ApiBuilder, ApiRepo},
    Cache, Repo, RepoType,
};

use crate::{
    pipeline::{
        hf::{get_file, hf_hub_cache_dir, list_repo_files, try_get_file},
        TokenSource,
    },
    utils::tokens::get_token,
    GLOBAL_HF_CACHE,
};

#[derive(Clone, Debug)]
pub enum SpeculativeConfig {
    Off,
    Mtp(MtpConfig),
}

#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub model: ModelSource,
    pub n_predict: Option<usize>,
}

impl MtpConfig {
    pub fn new(model: ModelSource, n_predict: Option<usize>) -> Self {
        Self { model, n_predict }
    }

    pub fn from_cli(model: impl Into<String>, n_predict: Option<usize>) -> Self {
        Self {
            model: ModelSource::from_cli(model),
            n_predict,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ModelSource {
    Hf {
        id: String,
        revision: Option<String>,
    },
    Path {
        path: PathBuf,
    },
}

impl ModelSource {
    pub fn hf(id: impl Into<String>) -> Self {
        Self::Hf {
            id: id.into(),
            revision: None,
        }
    }

    pub fn hf_revision(id: impl Into<String>, revision: impl Into<String>) -> Self {
        Self::Hf {
            id: id.into(),
            revision: Some(revision.into()),
        }
    }

    pub fn path(path: impl Into<PathBuf>) -> Self {
        Self::Path { path: path.into() }
    }

    pub fn from_cli(value: impl Into<String>) -> Self {
        let value = value.into();
        let path = PathBuf::from(&value);
        if path.exists() || value.starts_with('.') || value.starts_with('/') {
            Self::Path { path }
        } else {
            Self::Hf {
                id: value,
                revision: None,
            }
        }
    }

    pub fn resolve_path(&self) -> candle_core::Result<PathBuf> {
        match self {
            Self::Path { path } => Ok(path.clone()),
            Self::Hf { id, revision } => resolve_hf_mtp_path(id, revision.as_deref()),
        }
    }
}

impl From<PathBuf> for ModelSource {
    fn from(path: PathBuf) -> Self {
        Self::Path { path }
    }
}

impl From<&Path> for ModelSource {
    fn from(path: &Path) -> Self {
        Self::Path {
            path: path.to_path_buf(),
        }
    }
}

fn build_hf_api(id: &str, revision: &str) -> candle_core::Result<ApiRepo> {
    let cache = GLOBAL_HF_CACHE
        .get()
        .cloned()
        .unwrap_or_else(|| hf_hub_cache_dir().map(Cache::new).unwrap_or_default());
    let mut api = ApiBuilder::from_cache(cache)
        .with_progress(true)
        .with_token(get_token(&TokenSource::CacheToken).map_err(candle_core::Error::msg)?);
    if let Some(cache_dir) = hf_hub_cache_dir() {
        api = api.with_cache_dir(cache_dir);
    }
    Ok(api
        .build()
        .map_err(candle_core::Error::msg)?
        .repo(Repo::with_revision(
            id.to_string(),
            RepoType::Model,
            revision.to_string(),
        )))
}

fn resolve_hf_mtp_path(id: &str, revision: Option<&str>) -> candle_core::Result<PathBuf> {
    let revision = revision.unwrap_or("main");
    let api = build_hf_api(id, revision)?;
    let model_id = Path::new(id);

    let config_path =
        get_file(&api, model_id, "config.json", revision).map_err(candle_core::Error::msg)?;
    let files = list_repo_files(&api, model_id, true, revision).map_err(candle_core::Error::msg)?;
    let mut weight_files = files
        .iter()
        .filter(|file| file.ends_with(".safetensors"))
        .cloned()
        .collect::<Vec<_>>();
    weight_files.sort();
    if weight_files.is_empty() {
        candle_core::bail!("MTP HF model `{id}` does not contain safetensors weights");
    }
    for file in weight_files {
        get_file(&api, model_id, &file, revision).map_err(candle_core::Error::msg)?;
    }

    try_get_file(&api, model_id, "generation_config.json", revision)
        .map_err(|err| candle_core::Error::Msg(err.to_string()))?;

    config_path.parent().map(Path::to_path_buf).ok_or_else(|| {
        candle_core::Error::Msg(format!("config path has no parent: {config_path:?}"))
    })
}
