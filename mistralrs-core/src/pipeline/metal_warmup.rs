use anyhow::{Context, Result};
use candle_core::{DType, Device};
use mistralrs_paged_attn::{
    prepare_metal_paged_attention, MetalPagedAttentionWarmup, MetalPagedPrefillShape,
};
use mistralrs_quant::TrackedModule;

use crate::{
    device_map::DeviceMapper,
    paged_attention::{
        configured_metal_cache_layout, KvCacheLayout, ModelConfigLike, PagedAttentionConfig,
        DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
    },
};

pub(super) struct MetalModelWarmup<'a> {
    pub device: &'a Device,
    pub mapper: &'a dyn DeviceMapper,
    pub model: &'a dyn ModelConfigLike,
    pub paged: Option<&'a PagedAttentionConfig>,
    pub dtype: DType,
    pub tracked_modules: &'a [TrackedModule],
    pub packed_prefill: bool,
}

pub(super) fn prepare_model(ctx: MetalModelWarmup<'_>) -> Result<()> {
    let mut devices = ctx.mapper.get_unique_devices();
    if !devices.iter().any(|device| device.same_device(ctx.device)) {
        devices.push(ctx.device.clone());
    }
    devices.retain(Device::is_metal);
    if devices.is_empty() {
        return Ok(());
    }

    tracing::info!("Preparing Metal kernels for the loaded model");
    for device in &devices {
        crate::metal::warmup_common_kernels(device)?;
    }
    for module in ctx.tracked_modules {
        module
            .ct
            .resolve()?
            .prepare_metal_kernels(ctx.dtype)
            .with_context(|| format!("preparing Metal kernels for {}", module.key))?;
    }

    if let Some(paged) = ctx.paged {
        let hnd = configured_metal_cache_layout()? == KvCacheLayout::MetalHnd;
        let block_size = paged
            .block_size
            .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
        let limits = paged.execution_limits;
        let mut prepared = Vec::new();
        for layer in 0..ctx.model.num_layers() {
            if !ctx.model.layer_has_paged_kv_cache(layer)
                || matches!(
                    ctx.model.kv_cache_layout_for_layer(layer),
                    KvCacheLayout::Mla { .. }
                )
            {
                continue;
            }
            let device = ctx.mapper.device_for(layer, false).unwrap_or(ctx.device);
            let Device::Metal(metal) = device else {
                continue;
            };
            let features = ctx
                .model
                .prefix_prefill_attention_features(layer)
                .unwrap_or_default();
            for packed_queries in [false, true] {
                if packed_queries && !ctx.packed_prefill {
                    continue;
                }
                let shape = MetalPagedPrefillShape {
                    num_seqs: limits.max_num_seqs,
                    num_heads: ctx.model.num_attn_heads_for_layer(layer),
                    num_kv_heads: ctx.model.num_kv_heads_for_layer(layer),
                    max_query_len: limits
                        .max_prefill_chunk_tokens
                        .min(limits.max_num_batched_tokens)
                        .min(ctx.model.max_seq_len()),
                    total_query_tokens: limits.max_num_batched_tokens,
                    max_context_len: ctx.model.max_seq_len(),
                    key_head_dim: ctx.model.k_head_dim_for_layer(layer),
                    value_head_dim: ctx.model.v_head_dim_for_layer(layer),
                    block_size,
                    num_multimodal_ranges: 0,
                    query_dtype: ctx.dtype,
                    cache_dtype: paged.cache_type.to_dtype(ctx.dtype),
                    packed_queries,
                    hnd,
                };
                if !shape.supports() {
                    continue;
                }
                for feature_dtype in [DType::F32, ctx.dtype] {
                    let spec = MetalPagedAttentionWarmup {
                        shape,
                        causal: true,
                        sliding_window: ctx.model.sliding_window_for_layer(layer),
                        window_right: None,
                        attention_chunk_size: None,
                        alibi_dtype: features.has_alibi.then_some(DType::F32),
                        sink_dtype: features.has_sinks.then_some(feature_dtype),
                        softcap: features.has_softcap,
                    };
                    let key = (device.location(), spec);
                    if prepared.contains(&key) {
                        continue;
                    }
                    prepare_metal_paged_attention(metal, spec)?;
                    prepared.push(key);
                }
            }
        }
    }
    tracing::info!("Metal kernels prepared");
    Ok(())
}
