use std::fmt::{self, Display};

use crate::memory_plan::{device_memory_budget, MemoryGoal};
use crate::paged_attention::{
    requested_cache_blocks, CacheConfig, ModelConfigLike, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

use super::DeviceMappedModelLoader;

const MIN_MAPPING_CACHE_BLOCKS: usize = 2;

fn checked_memory_sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
}

#[derive(Clone, Debug)]
pub(crate) enum NonMappedSubModel {
    Vision,
    Audio,
}

impl Display for NonMappedSubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NonMappedSubModel::Vision => write!(f, "vision"),
            NonMappedSubModel::Audio => write!(f, "audio"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum AutoDeviceMapParams {
    Text {
        max_seq_len: usize,
        max_batch_size: usize,
    },
    Multimodal {
        max_seq_len: usize,
        max_batch_size: usize,
        max_image_shape: (usize, usize),
        max_num_images: usize,
    },
}

impl AutoDeviceMapParams {
    pub fn maybe_promote_to_multimodal(&self) -> Self {
        match *self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape: (
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                ),
                max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            },
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            },
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Text { max_seq_len, .. } | Self::Multimodal { max_seq_len, .. } => *max_seq_len,
        }
    }

    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Text { max_batch_size, .. } | Self::Multimodal { max_batch_size, .. } => {
                *max_batch_size
            }
        }
    }
}

impl Display for AutoDeviceMapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => write!(
                f,
                "text[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}]"
            ),
            Self::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => write!(
                f,
                "multimodal[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}, max_image_shape: {max_image_shape:?}, max_num_images: {max_num_images}]"
            ),
        }
    }
}

impl AutoDeviceMapParams {
    // Default max sequence length for memory estimation when not specified
    pub const DEFAULT_MAX_SEQ_LEN: usize = 4 * 1024;
    pub const DEFAULT_MAX_BATCH_SIZE: usize = 1;
    pub const DEFAULT_MAX_NUM_IMAGES: usize = 1;
    pub const DEFAULT_MAX_IMAGE_LENGTH: usize = 1024;

    pub fn default_text() -> Self {
        Self::Text {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
        }
    }

    pub fn default_multimodal() -> Self {
        Self::Multimodal {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            max_image_shape: (
                Self::DEFAULT_MAX_IMAGE_LENGTH,
                Self::DEFAULT_MAX_IMAGE_LENGTH,
            ),
        }
    }
}

fn paged_layer_cache_bytes(
    model_config: &dyn ModelConfigLike,
    layer_idx: usize,
    model_dtype: DType,
    cache: &CacheConfig,
) -> Result<usize> {
    let cache_dtype = cache.cache_type.to_dtype(model_dtype);
    let elements = model_config
        .layer_kv_cache_elements_per_token(layer_idx)
        .unwrap_or(0);
    elements
        .checked_mul(cache.block_size)
        .and_then(|bytes| bytes.checked_mul(cache.num_gpu_blocks))
        .and_then(|bytes| bytes.checked_mul(cache_dtype.size_in_bytes()))
        .context("paged KV cache size overflow")
}

fn mapping_cache_config(
    model: &dyn ModelConfigLike,
    dtype: DType,
    config: &PagedAttentionConfig,
) -> Result<CacheConfig> {
    let block_size = config
        .block_size
        .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    let cache_dtype = config.cache_type.to_dtype(dtype);
    let total_bytes_per_block = (0..model.num_layers())
        .try_fold(0usize, |total, idx| {
            total.checked_add(model.layer_kv_cache_elements_per_token(idx).unwrap_or(0))
        })
        .and_then(|elements| elements.checked_mul(block_size))
        .and_then(|elements| elements.checked_mul(cache_dtype.size_in_bytes()))
        .context("paged KV block size overflow")?;
    anyhow::ensure!(
        total_bytes_per_block != 0,
        "The model has no paged KV cache storage"
    );
    let num_gpu_blocks =
        requested_cache_blocks(config.memory_goal, block_size, total_bytes_per_block)?
            .unwrap_or(MIN_MAPPING_CACHE_BLOCKS);
    Ok(CacheConfig {
        block_size,
        num_gpu_blocks,
        cache_type: config.cache_type,
        kv_cache_group_ids: model.kv_cache_group_ids(),
    })
}

macro_rules! b_to_mb {
    ($x:expr) => {
        $x / (1024 * 1024)
    };
}

#[allow(
    clippy::too_many_arguments,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss
)]
/// Core logic for automatic device mapping
pub fn get_device_layers(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    mut layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &AutoDeviceMapParams,
    mut paged_attn_config: Option<&mut PagedAttentionConfig>,
) -> Result<DeviceMapMetadata> {
    let (mapped_max, non_mapped_max) = if let Some(cfg) = paged_attn_config.as_deref_mut() {
        let estimate = super::reserve_model_activation_memory(
            loader,
            config,
            cfg,
            dtype,
            &devices[0],
            params,
        )?;
        (estimate.mapped_bytes, estimate.non_mapped_bytes)
    } else {
        (
            loader
                .mapped_max_act_size_elems(config, params)?
                .checked_mul(dtype.size_in_bytes())
                .context("mapped activation memory size overflow")?,
            loader
                .non_mapped_max_act_size_elems(config, params)?
                .checked_mul(dtype.size_in_bytes())
                .context("non-mapped activation memory size overflow")?,
        )
    };
    let goal = paged_attn_config
        .as_ref()
        .map_or(MemoryGoal::Auto, |cfg| cfg.memory_goal);

    let mut layer_sizes_backup = if paged_attn_config.is_some() {
        Some(layer_sizes_in_bytes.clone())
    } else {
        None
    };

    let mut remaining = total_model_size_in_bytes;
    let max_seq_len = match params {
        AutoDeviceMapParams::Text { max_seq_len, .. }
        | AutoDeviceMapParams::Multimodal { max_seq_len, .. } => *max_seq_len,
    };
    let max_batch_size = match params {
        AutoDeviceMapParams::Text { max_batch_size, .. }
        | AutoDeviceMapParams::Multimodal { max_batch_size, .. } => *max_batch_size,
    };

    let model_cfg = loader.model_config(config)?;
    let has_paged_attn = paged_attn_config.is_some();
    let base_device_memory_reservation_bytes = paged_attn_config
        .as_ref()
        .map_or(0, |config| config.base_device_memory_reservation_bytes);
    let kv_cache_bytes = match paged_attn_config.as_deref_mut() {
        Some(cfg) => {
            info!(
                "Reserving {} MB on the primary device and {} MB on mapped devices for activations (predicted).",
                b_to_mb!(non_mapped_max.max(mapped_max)),
                b_to_mb!(mapped_max),
            );
            let cache = mapping_cache_config(&*model_cfg, dtype, cfg)?;
            (0..model_cfg.num_layers())
                .map(|idx| paged_layer_cache_bytes(&*model_cfg, idx, dtype, &cache))
                .collect::<Result<Vec<_>>>()?
        }
        None => (0..model_cfg.num_layers())
            .map(|idx| {
                model_cfg
                    .layer_kv_cache_elements_per_token(idx)
                    .unwrap_or(0)
                    .checked_mul(max_batch_size)
                    .and_then(|bytes| bytes.checked_mul(max_seq_len))
                    .and_then(|bytes| bytes.checked_mul(dtype.size_in_bytes()))
                    .context("dense KV cache size overflow")
            })
            .collect::<Result<Vec<_>>>()?,
    };
    let kv_bytes_for_layer = |idx: usize| kv_cache_bytes[idx];
    // Paged layers past the mapped stack (an MTP head) are charged to the non-mapped device.
    let extra_kv_bytes = (num_layers..model_cfg.num_layers())
        .map(kv_bytes_for_layer)
        .try_fold(0usize, |total, bytes| total.checked_add(bytes))
        .context("extra-layer KV cache size overflow")?;

    // prepare available memory per device, CPU fallback last (unless unified memory)
    let has_unified_memory = devices.iter().any(crate::utils::normal::is_integrated_gpu);

    let mut avail = Vec::new();
    for dev in devices {
        let memory = MemoryUsage.query(dev)?;
        let capacity = if dev.is_cpu() {
            memory.available()
        } else if let Some(config) = paged_attn_config.as_ref() {
            config.allocation_budget(dev, memory)?
        } else {
            device_memory_budget(goal, memory)?
        };
        avail.push((capacity, dev.clone()));
    }
    // On unified memory systems (iGPUs), GPU and CPU share the same physical RAM.
    // Don't add CPU as a fallback device since it would double-count memory.
    if !has_unified_memory {
        let a = MemoryUsage.query(&Device::Cpu)?.available();
        avail.push((a, Device::Cpu));
    }

    avail.reverse();
    layer_sizes_in_bytes.reverse();

    let mut mappings = Vec::new();
    info!("Using automatic device mapping parameters: {params}.");
    if let Some(subs) = loader.non_mapped_sub_models_for_config(config)? {
        let (_, last) = avail.last().unwrap();
        info!(
            "The following sub-models will not be device mapped and will be loaded on {}: {}",
            last.device_pretty_repr(),
            subs.iter().map(|x| x.to_string()).join(", ")
        );
    }

    let mut ordinal = 0;
    let mut layer = 0;
    let avail_copy = avail.clone();
    let mut includes_cpu = false;
    while remaining > 0 && !avail.is_empty() {
        let (avail_bytes, dev) = avail
            .pop()
            .context("No more devices to map to. The model does not fit on this system.")?;

        let cap = avail_bytes;
        if ordinal == 0
            && checked_memory_sum([
                non_mapped_max.max(mapped_max),
                non_mapped_size_in_bytes,
                base_device_memory_reservation_bytes,
            ])
            .is_none_or(|required| required > cap)
        {
            anyhow::bail!(
                "Primary device {} cannot fit its fixed model components, activations, and post-load reservation within {} MB of usable capacity.",
                dev.device_pretty_repr(),
                b_to_mb!(cap),
            );
        }

        // Algorithm is to check the following:
        // 1) (no mapping) if *everything* fits on the first dev (non mapped and mapped)
        // 2) if the mapped activations plus remaining fits on the nth device
        // 3) common case, iteratively find the optimal amount of layers to put on the nth device
        //   - if this is the first dev: must hold the non-mapped act and non-mapped model
        //   - otherwise, must hold the mapped act
        let remaining_kv_bytes = (layer..num_layers)
            .map(kv_bytes_for_layer)
            .try_fold(0usize, |total, bytes| total.checked_add(bytes))
            .context("mapped KV cache size overflow")?;
        let required_whole_capacity = if ordinal == 0 {
            checked_memory_sum([
                remaining,
                non_mapped_max.max(mapped_max),
                remaining_kv_bytes,
                extra_kv_bytes,
                base_device_memory_reservation_bytes,
            ])
        } else {
            checked_memory_sum([remaining, mapped_max, remaining_kv_bytes])
        };

        let layers_on_dev = if required_whole_capacity.is_some_and(|required| cap >= required) {
            remaining = 0;
            num_layers - layer
        } else {
            let mut used = mapped_max;
            let mut used_weight_bytes = 0usize;
            let mut count = 0;
            if ordinal == 0 {
                used = checked_memory_sum([
                    used.max(non_mapped_max),
                    non_mapped_size_in_bytes,
                    extra_kv_bytes,
                    base_device_memory_reservation_bytes,
                ])
                .unwrap_or(usize::MAX);
                used_weight_bytes = used_weight_bytes.saturating_add(non_mapped_size_in_bytes);
            }
            while let Some(&sz) = layer_sizes_in_bytes.last() {
                let Some(delta) = sz.checked_add(kv_bytes_for_layer(layer + count)) else {
                    break;
                };
                let Some(next_used) = used.checked_add(delta) else {
                    break;
                };
                if next_used > cap {
                    break;
                }
                layer_sizes_in_bytes.pop();
                used = next_used;
                used_weight_bytes = used_weight_bytes.saturating_add(sz);
                count += 1;
            }
            if count > 0 {
                remaining = remaining.saturating_sub(used_weight_bytes);
            } else {
                warn!(
                    "Device {} can fit 0 layers after model, execution, and KV memory reservations",
                    dev.device_pretty_repr(),
                );
                ordinal += 1;
                continue;
            }
            count
        };
        if !dev.is_cpu() {
            mappings.push(DeviceLayerMapMetadata {
                ordinal,
                layers: layers_on_dev,
            });
            ordinal += 1;
        } else {
            includes_cpu = true;
        }
        layer += layers_on_dev;
    }
    if remaining > 0 {
        let over = b_to_mb!(remaining);
        anyhow::bail!(
            "This model does not fit on the devices {:?}, and exceeds total capacity by {}MB. Auto device mapping params: {params}",
            avail_copy.iter().rev().map(|(a, d)| format!("{} (avail: {}MB)", d.device_pretty_repr(), b_to_mb!(a))).collect::<Vec<_>>(),
            over
        );
    }
    if has_paged_attn && includes_cpu {
        if goal.is_capacity() {
            anyhow::bail!(
                "The requested KV capacity cannot fit with the model on the accelerator devices; reduce the capacity or change device placement."
            );
        }
        let original_layers = layer_sizes_backup
            .take()
            .expect("layer sizes backup missing for paged attention fallback");
        // The original vector was in forward order, but `get_device_layers` handles
        // reversing internally, so we can pass it along unchanged.
        return get_device_layers(
            loader,
            config,
            num_layers,
            original_layers,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            devices,
            dtype,
            params,
            None,
        );
    }
    Ok(DeviceMapMetadata::from_num_device_layers(mappings))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{KvCacheLayout, ModelConfigMetadata, PagedCacheType};

    fn cache_test_model() -> ModelConfigMetadata {
        ModelConfigMetadata {
            max_seq_len: 4096,
            num_layers: 4,
            hidden_size: 256,
            num_kv_heads: 1,
            num_attn_heads: 4,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: KvCacheLayout::StandardNoFlashInfer,
        }
    }

    #[test]
    fn fp8_mapping_charges_cache_dtype_bytes() {
        let model = cache_test_model();
        let mut cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 1024,
            cache_type: PagedCacheType::F8E4M3,
            kv_cache_group_ids: vec![0],
        };
        assert_eq!(
            paged_layer_cache_bytes(&model, 0, DType::BF16, &cache).unwrap(),
            4 * 1024 * 1024
        );
        cache.cache_type = PagedCacheType::Auto;
        assert_eq!(
            paged_layer_cache_bytes(&model, 0, DType::BF16, &cache).unwrap(),
            8 * 1024 * 1024
        );
    }

    #[test]
    fn explicit_mapping_capacity_is_not_clamped_to_a_device_snapshot() -> Result<()> {
        let model = cache_test_model();
        for (goal, expected_blocks) in [
            (MemoryGoal::Auto, 2),
            (MemoryGoal::Utilization(0.95), 2),
            (MemoryGoal::KvTokens(65), 4),
            (MemoryGoal::KvBytes(1024 * 1024), 32),
        ] {
            let config = PagedAttentionConfig::new(Some(32), goal, PagedCacheType::Auto)?;
            let cache = mapping_cache_config(&model, DType::BF16, &config)?;
            assert_eq!(cache.num_gpu_blocks, expected_blocks, "{goal:?}");
            assert_eq!(config.memory_goal, goal);
        }
        Ok(())
    }

    #[test]
    fn heterogeneous_mapping_uses_each_layers_physical_cache() -> Result<()> {
        let config = serde_json::json!({
            "architectures": ["Gemma4ForCausalLM"],
            "text_config": {
                "hidden_size": 256,
                "intermediate_size": 512,
                "num_hidden_layers": 4,
                "num_attention_heads": 8,
                "num_key_value_heads": 2,
                "num_global_key_value_heads": 4,
                "head_dim": 64,
                "global_head_dim": 128,
                "sliding_window": 16,
                "num_kv_shared_layers": 2,
                "layer_types": ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
            }
        }).to_string();
        let model = super::super::Gemma4Loader.model_config(&config)?;
        let bytes_per_block = (1024 + 256 + 1024 + 256) * 32 * 2;
        let config = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::KvBytes(bytes_per_block * 3),
            PagedCacheType::Auto,
        )?;
        let cache = mapping_cache_config(&*model, DType::BF16, &config)?;
        assert_eq!(cache.num_gpu_blocks, 3);
        let per_layer = (0..4)
            .map(|idx| paged_layer_cache_bytes(&*model, idx, DType::BF16, &cache))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            per_layer,
            vec![
                1024 * 32 * 2 * 3,
                256 * 32 * 2 * 3,
                1024 * 32 * 2 * 3,
                256 * 32 * 2 * 3
            ]
        );
        assert_eq!(per_layer.iter().sum::<usize>(), bytes_per_block * 3);
        assert!(model.has_kv_cache_sharing());
        assert!(!model.uses_own_kv_cache_for_layer(2));
        Ok(())
    }

    #[test]
    fn hybrid_mapping_skips_recurrent_layers() -> Result<()> {
        let model = crate::paged_attention::HybridPagedKvCacheConfig::new(
            cache_test_model(),
            vec![true, false, true, false],
        );
        let config =
            PagedAttentionConfig::new(Some(32), MemoryGoal::KvTokens(32), PagedCacheType::F8E4M3)?;
        let cache = mapping_cache_config(&model, DType::BF16, &config)?;
        assert_eq!(
            paged_layer_cache_bytes(&model, 0, DType::BF16, &cache)?,
            8192
        );
        assert_eq!(paged_layer_cache_bytes(&model, 1, DType::BF16, &cache)?, 0);
        Ok(())
    }

    #[test]
    fn text_params_promote_to_multimodal_defaults_after_detection() {
        let params = AutoDeviceMapParams::Text {
            max_seq_len: 4096,
            max_batch_size: 7,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(max_batch_size, 7);
                assert_eq!(
                    max_image_shape,
                    (
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                        AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
                    )
                );
                assert_eq!(max_num_images, AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn multimodal_params_preserve_explicit_limits_after_detection() {
        let params = AutoDeviceMapParams::Multimodal {
            max_seq_len: 8192,
            max_batch_size: 3,
            max_image_shape: (1536, 1024),
            max_num_images: 5,
        };

        match params.maybe_promote_to_multimodal() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1536, 1024));
                assert_eq!(max_num_images, 5);
            }
            AutoDeviceMapParams::Text { .. } => panic!("expected multimodal parameters"),
        }
    }

    #[test]
    fn checked_memory_sum_distinguishes_max_from_overflow() {
        assert_eq!(checked_memory_sum([usize::MAX - 1, 1]), Some(usize::MAX));
        assert_eq!(checked_memory_sum([usize::MAX, 1]), None);
    }
}
