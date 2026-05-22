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
    pub model: String,
    pub n_predict: Option<usize>,
}

impl MtpConfig {
    pub fn new(model: impl Into<String>, n_predict: Option<usize>) -> Self {
        Self {
            model: model.into(),
            n_predict,
        }
    }

    pub fn resolve_path(&self) -> candle_core::Result<PathBuf> {
        let path = PathBuf::from(&self.model);
        if path.exists() || self.model.starts_with('.') || self.model.starts_with('/') {
            Ok(path)
        } else {
            resolve_hf_mtp_path(&self.model)
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

fn resolve_hf_mtp_path(id: &str) -> candle_core::Result<PathBuf> {
    let revision = "main";
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
        candle_core::bail!("MTP model `{id}` does not contain safetensors weights");
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
