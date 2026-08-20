use std::path::{Path, PathBuf};

use hf_hub::{api::sync::ApiRepo, Repo, RepoType};

use crate::pipeline::{
    hf::{build_api, get_file, list_repo_files, try_get_file},
    TokenSource,
};

#[derive(Clone, Debug)]
pub enum SpeculativeConfig {
    Off,
    Mtp(MtpConfig),
}

/// MTP proposer configuration; `model: None` uses the head built into the target checkpoint.
#[derive(Clone, Debug)]
pub struct MtpConfig {
    pub model: Option<String>,
    pub n_predict: Option<usize>,
    /// ISQ type for a draft-only copy of `lm_head`, so drafting skips the promoted (wider)
    /// sensitive-tensor type; the target still verifies with the promoted head.
    pub draft_lm_head_isq: Option<crate::IsqType>,
}

impl MtpConfig {
    pub fn new(model: impl Into<String>, n_predict: Option<usize>) -> Self {
        Self {
            model: Some(model.into()),
            n_predict,
            draft_lm_head_isq: None,
        }
    }

    pub fn builtin(n_predict: Option<usize>) -> Self {
        Self {
            model: None,
            n_predict,
            draft_lm_head_isq: None,
        }
    }

    pub fn with_draft_lm_head_isq(mut self, isq: Option<crate::IsqType>) -> Self {
        self.draft_lm_head_isq = isq;
        self
    }

    pub fn is_builtin(&self) -> bool {
        self.model.is_none()
    }

    pub fn resolve_path(&self) -> candle_core::Result<PathBuf> {
        let Some(model) = &self.model else {
            candle_core::bail!("this MTP proposer requires a separate assistant model (`--mtp-model`), not the built-in head");
        };
        let path = PathBuf::from(model);
        if path.exists() || model.starts_with('.') || model.starts_with('/') {
            Ok(path)
        } else {
            resolve_hf_mtp_path(model)
        }
    }
}

fn build_hf_api(id: &str, revision: &str) -> candle_core::Result<ApiRepo> {
    let api = build_api(&TokenSource::CacheToken, true).map_err(candle_core::Error::msg)?;
    Ok(api.repo(Repo::with_revision(
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
