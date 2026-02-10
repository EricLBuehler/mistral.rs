use std::fmt::{self, Display};

use crate::paged_attention::{
    calculate_cache_config, MemoryGpuConfig, ModelConfigLike, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

use super::DeviceMappedModelLoader;

const GPU_RESERVE_FRACTION: f64 = 0.02;
const GPU_MIN_RESERVE_BYTES: usize = 512 * 1024 * 1024; // 512MB safety buffer

/// Usable device capacity after subtracting a small safety reserve for GPUs.
/// CPU devices return `avail_bytes` unchanged.
#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn device_cap(avail_bytes: usize, dev: &Device) -> usize {
    if dev.is_cpu() {
        avail_bytes
    } else {
        let reserve_frac = (avail_bytes as f64 * GPU_RESERVE_FRACTION) as usize;
        let reserve = reserve_frac.max(GPU_MIN_RESERVE_BYTES).min(avail_bytes);
        avail_bytes.saturating_sub(reserve)
    }
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
    Vision {
        max_seq_len: usize,
        max_batch_size: usize,
        max_image_shape: (usize, usize),
        max_num_images: usize,
    },
}

impl AutoDeviceMapParams {
    pub fn maybe_promote_to_vision(&self) -> Self {
        match *self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => Self::Vision {
                max_seq_len,
                max_batch_size,
                max_image_shape: (
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                    Self::DEFAULT_MAX_IMAGE_LENGTH,
                ),
                max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            },
            Self::Vision {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => Self::Vision {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            },
        }
    }

    pub fn max_seq_len(&self) -> usize {
        match self {
            Self::Text { max_seq_len, .. } | Self::Vision { max_seq_len, .. } => *max_seq_len,
        }
    }

    pub fn max_batch_size(&self) -> usize {
        match self {
            Self::Text { max_batch_size, .. } | Self::Vision { max_batch_size, .. } => {
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
            Self::Vision {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => write!(
                f,
                "vision[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}, max_image_shape: {max_image_shape:?}, max_num_images: {max_num_images}]"
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

    pub fn default_vision() -> Self {
        Self::Vision {
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

fn calculate_key_block_shape(
    model_config: &dyn ModelConfigLike,
    dtype: DType,
    block_size: usize,
) -> (usize, usize, usize, usize) {
    let element_size = dtype.size_in_bytes();
    let x = 16 / element_size;
    (
        model_config.num_kv_heads(),
        model_config.k_head_dim() / x,
        block_size,
        x,
    )
}

fn calculate_value_block_shape(
    model_config: &dyn ModelConfigLike,
    block_size: usize,
) -> (usize, usize, usize) {
    (
        model_config.num_kv_heads(),
        model_config.v_head_dim(),
        block_size,
    )
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
    paged_attn_config: Option<&PagedAttentionConfig>,
) -> Result<DeviceMapMetadata> {
    let mapped_max = loader.mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();
    let non_mapped_max =
        loader.non_mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();

    let mut layer_sizes_backup = if paged_attn_config.is_some() {
        Some(layer_sizes_in_bytes.clone())
    } else {
        None
    };

    let mut remaining = total_model_size_in_bytes;
    let max_seq_len = match params {
        AutoDeviceMapParams::Text { max_seq_len, .. }
        | AutoDeviceMapParams::Vision { max_seq_len, .. } => *max_seq_len,
    };
    let max_batch_size = match params {
        AutoDeviceMapParams::Text { max_batch_size, .. }
        | AutoDeviceMapParams::Vision { max_batch_size, .. } => *max_batch_size,
    };

    let model_cfg = loader.model_config(config)?;
    let kv_cache_elems = match paged_attn_config {
        Some(cfg) => {
            // For MbAmount, clamp to available memory so the capacity check
            // below stays consistent. Utilization and ContextSize pass through
            // to calculate_cache_config which handles model weight subtraction.
            let effective_mem_gpu = match cfg.mem_gpu {
                MemoryGpuConfig::MbAmount(user_mb) => {
                    // Clamp user's KV budget to available memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.get_memory_available(primary_dev)?;
                    let cap = device_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap.saturating_sub(act_overhead) / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb.min(user_mb))
                }
                MemoryGpuConfig::Utilization(f) => {
                    // Prevent overallocation when total_memory > available_memory
                    // (e.g., unified memory systems, other GPU processes using VRAM).
                    // Cap the KV budget so model + activations + KV fits within
                    // the device capacity derived from *available* memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.get_memory_available(primary_dev)?;
                    let cap = device_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = ((cap as f64 * f as f64) as usize)
                        .saturating_sub(remaining + act_overhead)
                        / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb)
                }
                // ContextSize passes through to calculate_cache_config.
                other => other,
            };

            let cache = calculate_cache_config(
                effective_mem_gpu,
                Some(cfg.block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE)),
                dtype,
                paged_attn_config
                    .map(|cfg| cfg.cache_type)
                    .unwrap_or_default(),
                &*model_cfg,
                &devices[0],
                &devices.iter().map(|d| Some(d.clone())).collect::<Vec<_>>(),
                true,
                Some(total_model_size_in_bytes),
                Some(max_seq_len * max_batch_size),
            )?;
            let key_shape = calculate_key_block_shape(&*model_cfg, dtype, cache.block_size);
            let key_sz =
                cache.num_gpu_blocks * key_shape.0 * key_shape.1 * key_shape.2 * key_shape.3;
            let val_shape = calculate_value_block_shape(&*model_cfg, cache.block_size);
            let val_sz = cache.num_gpu_blocks * val_shape.0 * val_shape.1 * val_shape.2;
            key_sz + val_sz
        }
        None => {
            let key_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.k_head_dim(),
            ];
            let val_shape = [
                max_batch_size,
                model_cfg.num_kv_heads(),
                max_seq_len,
                model_cfg.v_head_dim(),
            ];
            key_shape.iter().product::<usize>() + val_shape.iter().product::<usize>()
        }
    };
    let kv_cache_bytes = kv_cache_elems * dtype.size_in_bytes();

    // prepare available memory per device, CPU fallback last (unless unified memory)
    let has_unified_memory = devices.iter().any(crate::utils::normal::is_integrated_gpu);

    let mut avail = Vec::new();
    for dev in devices {
        let a = MemoryUsage.get_memory_available(dev)?;
        avail.push((a, dev.clone()));
    }
    // On unified memory systems (iGPUs), GPU and CPU share the same physical RAM.
    // Don't add CPU as a fallback device since it would double-count memory.
    if !has_unified_memory {
        let a = MemoryUsage.get_memory_available(&Device::Cpu)?;
        avail.push((a, Device::Cpu));
    }

    avail.reverse();
    layer_sizes_in_bytes.reverse();

    let mut mappings = Vec::new();
    info!("Using automatic device mapping parameters: {params}.");
    if let Some(subs) = loader.non_mapped_sub_models() {
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

        // For GPU/accelerators: keep a small dynamic safety reserve to avoid OOMs
        let cap = device_cap(avail_bytes, &dev);

        // Algorithm is to check the following:
        // 1) (no mapping) if *everything* fits on the first dev (non mapped and mapped)
        // 2) if the mapped activations plus remaining fits on the nth device
        // 3) common case, iteratively find the optimal amount of layers to put on the nth device
        //   - if this is the first dev: must hold the non-mapped act and non-mapped model
        //   - otherwise, must hold the mapped act
        let required_whole_capacity = if ordinal == 0 {
            remaining + non_mapped_max.max(mapped_max) + kv_cache_bytes * (num_layers - layer)
        } else {
            remaining + mapped_max + kv_cache_bytes * (num_layers - layer)
        };

        let layers_on_dev = if cap >= required_whole_capacity {
            remaining = 0;
            num_layers - layer
        } else {
            let mut used = mapped_max;
            let mut used_weight_bytes = 0;
            let mut count = 0;
            if ordinal == 0 {
                used = used.max(non_mapped_max) + non_mapped_size_in_bytes;
                used_weight_bytes += non_mapped_size_in_bytes;
            }
            while let Some(&sz) = layer_sizes_in_bytes.last() {
                let delta = sz + kv_cache_bytes;
                if used + delta > cap {
                    break;
                }
                layer_sizes_in_bytes.pop();
                used += delta;
                used_weight_bytes += sz;
                count += 1;
            }
            if count > 0 {
                remaining = remaining.saturating_sub(used_weight_bytes);
            } else {
                warn!(
                    "Device {} can fit 0 layers. Consider reducing auto map params from current: {params} (ex. reducing max seq len or max num images)",
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
    if paged_attn_config.is_some_and(|_| includes_cpu) {
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
