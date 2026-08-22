use std::{
    fs,
    path::{Path, PathBuf},
};

use candle_core::{DType, Device};
use hf_hub::{api::sync::ApiRepo, Repo, RepoType};

use crate::{
    paged_attention::PagedAttentionConfig,
    pipeline::{
        hf::{build_api, get_file, list_repo_files, try_get_file},
        TokenSource,
    },
    utils::normal::TryIntoDType,
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

    /// Returns a conservative runtime weight footprint for an external assistant checkpoint.
    pub fn external_weight_size_in_bytes(&self, target_dtype: DType) -> candle_core::Result<usize> {
        if self.is_builtin() {
            return Ok(0);
        }
        let path = self.resolve_path()?;
        let mut weight_paths = fs::read_dir(&path)
            .map_err(|err| {
                candle_core::Error::msg(format!("failed to list {}: {err}", path.display()))
            })?
            .filter_map(|entry| entry.ok().map(|entry| entry.path()))
            .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
            .collect::<Vec<_>>();
        weight_paths.sort();
        crate::pipeline::checkpoint_runtime_size(&weight_paths, target_dtype)
            .map_err(candle_core::Error::msg)?
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "MTP model directory {} has no safetensors weights",
                    path.display()
                ))
            })
    }
}

/// Adds an external assistant's runtime weight footprint to a paged-cache memory reservation.
pub fn reserve_external_mtp_memory(
    cache_config: Option<PagedAttentionConfig>,
    mtp_config: Option<&MtpConfig>,
    dtype: &dyn TryIntoDType,
    device: &Device,
) -> anyhow::Result<Option<PagedAttentionConfig>> {
    let Some(cache_config) = cache_config else {
        return Ok(None);
    };
    let Some(mtp_config) = mtp_config else {
        return Ok(Some(cache_config));
    };
    if mtp_config.is_builtin() {
        return Ok(Some(cache_config));
    }
    let dtype = dtype.try_into_dtype(&[device])?;
    let bytes = mtp_config.external_weight_size_in_bytes(dtype)?;
    Ok(Some(
        cache_config.with_base_device_memory_reservation(bytes)?,
    ))
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use safetensors::{serialize_to_file, tensor::Dtype as SafeDtype, tensor::TensorView};
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn external_weight_size_uses_runtime_dtype() -> anyhow::Result<()> {
        let dir = tempdir()?;
        let path = dir.path().join("model.safetensors");
        let data = vec![0u8; 8];
        serialize_to_file(
            HashMap::from([(
                "layers.0.weight",
                TensorView::new(SafeDtype::BF16, vec![4], &data)?,
            )]),
            None,
            &path,
        )?;
        let config = MtpConfig::new(dir.path().to_string_lossy().into_owned(), None);

        assert_eq!(config.external_weight_size_in_bytes(DType::F32)?, 16);
        let cache_config = crate::PagedAttentionConfig::new(
            None,
            crate::MemoryGpuConfig::Utilization(0.9),
            crate::PagedCacheType::Auto,
        )?
        .with_base_device_memory_reservation(usize::MAX - 16)?;
        let cache_config = reserve_external_mtp_memory(
            Some(cache_config),
            Some(&config),
            &DType::F32,
            &Device::Cpu,
        )?
        .expect("cache config missing");
        let error = reserve_external_mtp_memory(
            Some(cache_config),
            Some(&config),
            &DType::F32,
            &Device::Cpu,
        )
        .expect_err("adding the checkpoint twice should overflow");
        assert!(error
            .to_string()
            .contains("paged attention device memory reservation overflow"));
        Ok(())
    }
}
