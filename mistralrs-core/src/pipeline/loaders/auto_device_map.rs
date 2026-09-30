use std::fmt::{self, Display};

use crate::paged_attention::{
    calculate_cache_config, device_memory_cap, CacheConfig, CacheMemoryReservations,
    MemoryGpuConfig, ModelConfigLike, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

use super::DeviceMappedModelLoader;

fn saturating_memory_sum<const N: usize>(parts: [usize; N]) -> usize {
    parts
        .into_iter()
        .fold(0usize, |total, part| total.saturating_add(part))
}

fn checked_memory_sum<const N: usize>(parts: [usize; N]) -> Option<usize> {
    parts
        .into_iter()
        .try_fold(0usize, |total, part| total.checked_add(part))
}

fn post_load_memory_config(
    requested: MemoryGpuConfig,
    pre_load_budget: MemoryGpuConfig,
    resolve_utilization_after_load: bool,
) -> MemoryGpuConfig {
    match requested {
        MemoryGpuConfig::Utilization(_) if !resolve_utilization_after_load => pre_load_budget,
        _ => requested,
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

fn paged_kv_bytes_per_layer(
    model_config: &dyn ModelConfigLike,
    cache: &CacheConfig,
    dtype: DType,
) -> Vec<usize> {
    let bytes_per_element =
        cache.num_gpu_blocks * cache.block_size * cache.cache_type.to_dtype(dtype).size_in_bytes();
    (0..model_config.num_layers())
        .map(|idx| {
            model_config
                .layer_kv_cache_elements_per_token(idx)
                .unwrap_or(0)
                * bytes_per_element
        })
        .collect()
}

const BYTES_PER_GIB: f64 = 1_073_741_824.0;

#[allow(clippy::cast_precision_loss)]
fn gib(bytes: usize) -> f64 {
    bytes as f64 / BYTES_PER_GIB
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
    paged_attn_config: Option<&mut PagedAttentionConfig>,
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
    let unified_device = devices
        .first()
        .filter(|dev| crate::utils::normal::is_integrated_gpu(dev));
    // Unified memory has no fallback device, so a model that can't fit fails here with the real numbers
    if let Some(dev) = unified_device {
        let required = saturating_memory_sum([
            total_model_size_in_bytes,
            non_mapped_max.max(mapped_max),
            base_device_memory_reservation_bytes,
        ]);
        let available = device_memory_cap(MemoryUsage.query(dev)?.available(), dev);
        if required > available {
            anyhow::bail!(
                "Model weights, activations, and runtime reservations need {:.1} GiB, but only {:.1} GiB is available within the unified-memory budget. Close other applications, pick a smaller quantization, reduce the expected context or batch size, or adjust MISTRALRS_IGPU_MEMORY_FRACTION.",
                gib(required),
                gib(available),
            );
        }
    }
    let kv_cache_bytes = match paged_attn_config {
        Some(cfg) => {
            // The mapping estimate is bounded independently from the post-load memory mode.
            let requested_mem_gpu = cfg.mem_gpu;
            let effective_mem_gpu = match requested_mem_gpu {
                MemoryGpuConfig::MbAmount(user_mb) => {
                    // Clamp user's KV budget to available memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb.min(user_mb))
                }
                MemoryGpuConfig::BestEffortMbAmount { target_mb, min_mb } => {
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let budget_mb = cap
                        .saturating_sub(act_overhead)
                        .saturating_sub(base_device_memory_reservation_bytes)
                        / (1024 * 1024);
                    MemoryGpuConfig::BestEffortMbAmount {
                        target_mb: budget_mb.min(target_mb),
                        min_mb,
                    }
                }
                MemoryGpuConfig::Utilization(f) => {
                    // Prevent overallocation when total_memory > available_memory
                    // (e.g., unified memory systems, other GPU processes using VRAM).
                    // Cap the KV budget so model + activations + KV fits within
                    // the device capacity derived from *available* memory.
                    let primary_dev = &devices[0];
                    let avail_bytes = MemoryUsage.query(primary_dev)?.available();
                    let cap = device_memory_cap(avail_bytes, primary_dev);
                    let act_overhead = non_mapped_max.max(mapped_max);
                    let occupied = saturating_memory_sum([
                        remaining,
                        act_overhead,
                        base_device_memory_reservation_bytes,
                    ]);
                    // On unified memory the KV cache is capped to the context instead of a share of memory
                    let fraction = if unified_device.is_some() {
                        1.0
                    } else {
                        f64::from(f)
                    };
                    let budget_mb =
                        ((cap as f64 * fraction) as usize).saturating_sub(occupied) / (1024 * 1024);
                    MemoryGpuConfig::MbAmount(budget_mb)
                }
                // ContextSize passes through to calculate_cache_config.
                other => other,
            };
            info!(
                "Reserving {} MB on the primary device and {} MB on mapped devices for activations (predicted).",
                b_to_mb!(non_mapped_max.max(mapped_max)),
                b_to_mb!(mapped_max),
            );
            cfg.reserve_activation_memory(non_mapped_max.max(mapped_max), mapped_max);
            if base_device_memory_reservation_bytes > 0 {
                info!(
                    "Reserving {} MB on the primary device for post-load model components.",
                    base_device_memory_reservation_bytes.div_ceil(1024 * 1024)
                );
            }
            // Re-resolve utilization after recurrent serving state is allocated.
            cfg.mem_gpu = post_load_memory_config(
                requested_mem_gpu,
                effective_mem_gpu,
                cfg.resolve_memory_utilization_after_load,
            );

            let cache = calculate_cache_config(
                effective_mem_gpu,
                CacheMemoryReservations::default(),
                Some(cfg.block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE)),
                dtype,
                cfg.cache_type,
                &*model_cfg,
                &devices[0],
                &devices.iter().map(|d| Some(d.clone())).collect::<Vec<_>>(),
                true,
                Some(total_model_size_in_bytes),
                Some(max_seq_len * max_batch_size),
            )?;
            paged_kv_bytes_per_layer(&*model_cfg, &cache, dtype)
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
            let bytes = (key_shape.iter().product::<usize>() + val_shape.iter().product::<usize>())
                * dtype.size_in_bytes();
            (0..model_cfg.num_layers())
                .map(|idx| {
                    if model_cfg.layer_has_paged_kv_cache(idx) {
                        bytes
                    } else {
                        0
                    }
                })
                .collect()
        }
    };
    // Per paged layer; hybrid models leave the recurrent/linear layers out of the cache entirely.
    let kv_bytes_for_layer = |idx: usize| kv_cache_bytes[idx];
    // Paged layers past the mapped stack (an MTP head) are charged to the non-mapped device.
    let extra_kv_bytes = (num_layers..model_cfg.num_layers())
        .map(kv_bytes_for_layer)
        .sum::<usize>();

    // prepare available memory per device, CPU fallback last (unless unified memory)
    let has_unified_memory = devices.iter().any(crate::utils::normal::is_integrated_gpu);

    let mut avail = Vec::new();
    for dev in devices {
        let a = MemoryUsage.query(dev)?.available();
        avail.push((a, dev.clone()));
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

        // For GPU/accelerators: keep a small dynamic safety reserve to avoid OOMs
        let cap = device_memory_cap(avail_bytes, &dev);
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
        let remaining_kv_bytes = (layer..num_layers).map(kv_bytes_for_layer).sum::<usize>();
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
    if has_paged_attn && includes_cpu {
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
    use crate::vision_models::qwen4_exp::{
        config::{LayerType, QsaConfig},
        Qwen4ExpPagedConfig,
    };

    #[test]
    fn paged_reservations_include_aux_buffers_and_cache_dtype() {
        let base = ModelConfigMetadata {
            max_seq_len: 128,
            num_layers: 3,
            hidden_size: 2560,
            num_attn_heads: 24,
            num_kv_heads: 2,
            sliding_window: None,
            k_head_dim: 256,
            v_head_dim: 256,
            kv_cache_layout: KvCacheLayout::Standard,
        };
        let qsa = QsaConfig {
            n_heads: 4,
            head_dim: 128,
            budget: 2048,
            compress_ratio: 4,
        };
        let rot_dim = 64;
        let model = Qwen4ExpPagedConfig::new(
            base.clone(),
            &[
                LayerType::FullAttention,
                LayerType::LinearAttention,
                LayerType::FullAttention,
            ],
            qsa.aux_cache_elements_per_token(rot_dim),
            qsa.max_selected_tokens(),
        );
        let mut cache = CacheConfig {
            block_size: 32,
            num_gpu_blocks: 5,
            cache_type: PagedCacheType::Auto,
            kv_cache_group_ids: vec![0],
        };
        let slots = cache.block_size * cache.num_gpu_blocks;
        let kv = slots * base.num_kv_heads * (base.k_head_dim + base.v_head_dim);
        let raw_aux = slots * (qsa.head_dim + rot_dim);
        let block_keys = slots / qsa.compress_ratio * qsa.head_dim;
        let expected = (kv + raw_aux + block_keys) * DType::BF16.size_in_bytes();
        assert_eq!(
            paged_kv_bytes_per_layer(&model, &cache, DType::BF16),
            [expected, 0, expected]
        );

        cache.cache_type = PagedCacheType::F8E4M3;
        assert_eq!(
            paged_kv_bytes_per_layer(&base, &cache, DType::BF16),
            [kv; 3]
        );
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
    fn memory_sum_saturates_capacity_accounting_overflow() {
        assert_eq!(saturating_memory_sum([usize::MAX - 2, 1, 1]), usize::MAX);
        assert_eq!(saturating_memory_sum([usize::MAX, 1]), usize::MAX);
    }

    #[test]
    fn checked_memory_sum_distinguishes_max_from_overflow() {
        assert_eq!(checked_memory_sum([usize::MAX - 1, 1]), Some(usize::MAX));
        assert_eq!(checked_memory_sum([usize::MAX, 1]), None);
    }

    #[test]
    fn post_load_utilization_is_not_frozen_to_the_inventory_estimate() {
        let requested = MemoryGpuConfig::Utilization(0.85);
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), true),
            MemoryGpuConfig::Utilization(value) if value == 0.85
        ));
        assert!(matches!(
            post_load_memory_config(requested, MemoryGpuConfig::MbAmount(45_000), false,),
            MemoryGpuConfig::MbAmount(45_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::MbAmount(60_000),
                MemoryGpuConfig::MbAmount(45_000),
                true,
            ),
            MemoryGpuConfig::MbAmount(60_000)
        ));
        assert!(matches!(
            post_load_memory_config(
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 60_000,
                    min_mb: Some(40_000),
                },
                MemoryGpuConfig::BestEffortMbAmount {
                    target_mb: 45_000,
                    min_mb: Some(40_000),
                },
                true,
            ),
            MemoryGpuConfig::BestEffortMbAmount {
                target_mb: 60_000,
                min_mb: Some(40_000),
            }
        ));
    }
}
