use super::{
    calculate_key_block_shape, calculate_value_block_shape, AutoDeviceMapParams,
    DeviceMappedModelLoader,
};
use crate::paged_attention::{calculate_cache_config, DEFAULT_PAGED_ATTENTION_BLOCK_SIZE};
use crate::utils::debug::DeviceRepr;
use crate::{DeviceLayerMapMetadata, DeviceMapMetadata, MemoryUsage, PagedAttentionConfig};
use anyhow::{Context, Result};
use candle_core::{DType, Device};
use itertools::Itertools;
use tracing::{info, warn};

macro_rules! b_to_mb {
    ($x:expr) => {
        $x / (1024 * 1024)
    };
}

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
    prompt_chunksize: usize,
    paged_attn_config: Option<&PagedAttentionConfig>,
) -> Result<DeviceMapMetadata> {
    let mapped_max =
        loader.mapped_max_act_size_elems(config, params, prompt_chunksize)? * dtype.size_in_bytes();
    let non_mapped_max =
        loader.non_mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();

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
            let cache = calculate_cache_config(
                cfg.mem_gpu,
                cfg.mem_cpu,
                Some(cfg.block_size.unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE)),
                dtype,
                &*model_cfg,
                &devices[0],
                &devices.iter().map(|d| Some(d.clone())).collect::<Vec<_>>(),
                true,
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

    // prepare available memory per device, CPU fallback last
    let mut avail = Vec::new();
    for dev in [devices, &[Device::Cpu]].concat() {
        let a = MemoryUsage.get_memory_available(&dev)?;
        avail.push((a, dev));
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
        let (cap, dev) = avail
            .pop()
            .context("No more devices to map to. The model does not fit on this system.")?;
        let cap = (cap as f64 * 0.90) as usize;
        let layers_on_dev = if ordinal == 0
            && cap
                >= remaining
                    + non_mapped_max.max(mapped_max)
                    + non_mapped_size_in_bytes
                    + kv_cache_bytes * (num_layers - layer)
        {
            remaining = 0;
            num_layers - layer
        } else if ordinal != 0
            && cap >= remaining + mapped_max + kv_cache_bytes * (num_layers - layer)
        {
            remaining = 0;
            num_layers - layer
        } else {
            let mut used = mapped_max;
            let mut used_no_act = 0;
            let mut count = 0;
            if ordinal == 0 {
                used = used.max(non_mapped_max) + non_mapped_size_in_bytes;
                used_no_act += non_mapped_size_in_bytes;
            }
            while let Some(&sz) = layer_sizes_in_bytes.last() {
                let delta = sz + kv_cache_bytes;
                if used + delta > cap {
                    break;
                }
                layer_sizes_in_bytes.pop();
                used += delta;
                used_no_act += delta;
                count += 1;
            }
            if count > 0 {
                remaining = remaining.saturating_sub(used_no_act);
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
        return get_device_layers(
            loader,
            config,
            num_layers,
            layer_sizes_in_bytes,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            devices,
            dtype,
            params,
            prompt_chunksize,
            None,
        );
    }
    Ok(DeviceMapMetadata::from_num_device_layers(mappings))
}
