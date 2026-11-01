use std::{path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use mistralrs_quant::{safetensors::MmapedSafetensors, GgufArchive, ShardedVarBuilder};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub enum HostBacking {
    Owned,
    FileMapped,
}

/// Storage outside ordinary weight bindings, with its own placement decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerResource {
    pub name: String,
    pub layer: usize,
    pub device_bytes: usize,
    pub host_backing: Option<HostBacking>,
}

impl LayerResource {
    pub fn can_offload(&self, unified: bool) -> bool {
        self.host_backing
            .is_some_and(|backing| !unified || backing == HostBacking::FileMapped)
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub struct ResourceSource<'a> {
    pub weights: Option<&'a ShardedVarBuilder>,
    pub paths: &'a [PathBuf],
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl<'a> ResourceSource<'a> {
    pub fn checkpoint_tensors(&self) -> Result<CheckpointTensors<'a>> {
        Ok(CheckpointTensors {
            weights: self.weights,
            safetensors: self.safetensors()?,
        })
    }

    pub fn safetensors(&self) -> Result<Option<Arc<MmapedSafetensors>>> {
        if let Some(raw) = self.weights.and_then(ShardedVarBuilder::raw_safetensors) {
            return Ok(Some(raw));
        }
        if self.paths.is_empty()
            || self.paths.iter().any(|path| {
                path.extension().and_then(|extension| extension.to_str()) != Some("safetensors")
            })
        {
            return Ok(None);
        }
        Ok(Some(Arc::new(unsafe {
            MmapedSafetensors::multi_unique(self.paths)?
        })))
    }

    pub fn gguf(&self) -> Option<Arc<GgufArchive>> {
        self.weights.and_then(ShardedVarBuilder::raw_gguf)
    }
}

pub struct CheckpointTensors<'a> {
    weights: Option<&'a ShardedVarBuilder>,
    safetensors: Option<Arc<MmapedSafetensors>>,
}

impl CheckpointTensors<'_> {
    pub fn contains(&self, name: &str) -> bool {
        self.weights
            .is_some_and(|weights| crate::layers::contains_tensor_or_weight_source(weights, name))
            || self
                .safetensors
                .as_ref()
                .is_some_and(|raw| raw.get(name).is_ok())
    }
}

pub(crate) fn add_resource_bytes(
    layers: &mut [usize],
    total: usize,
    resources: &[LayerResource],
) -> Result<usize> {
    resources.iter().try_fold(total, |total, resource| {
        let layer = layers
            .get_mut(resource.layer)
            .context("resource owner is outside the mapped layer stack")?;
        *layer = layer
            .checked_add(resource.device_bytes)
            .context("layer resource size overflow")?;
        total
            .checked_add(resource.device_bytes)
            .context("model resource size overflow")
    })
}
