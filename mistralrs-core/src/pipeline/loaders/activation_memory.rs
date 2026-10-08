use anyhow::{Context, Result};
use candle_core::{DType, Device};

use crate::{
    attention::{flash_backend_supports_sdpa, ATTENTION_CHUNK_SIZE},
    memory_plan::ExecutionMemoryLimits,
    paged_attention::{
        requested_cache_blocks, ModelConfigLike, PrefixPrefillAttentionFeatures,
        DEFAULT_PAGED_ATTENTION_BLOCK_SIZE,
    },
    MemoryGoal, PagedAttentionConfig,
};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};

const EAGER_F32_SCORE_BUFFERS: usize = 2;
const RESIDUAL_HIDDEN_BUFFERS: usize = 4;
const MLP_INTERMEDIATE_BUFFERS: usize = 3;
const PROJECTION_BUFFERS: usize = 2;
const PAGED_PARTITION_TOKENS: usize = 512;
const MAX_NATIVE_FLASH_GQA_GROUPS: usize = 8;
const METAL_D512_SMALL_QUERY_TOKENS: usize = 8;
// Metal rounds each pooled buffer up to a power of two.
const METAL_BUFFER_ROUNDING_BOUND: usize = 2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActivationModelConfig {
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub extra_hidden_elements_per_token: usize,
    pub custom_prefill_mask: bool,
    pub metal_prefill_mask_is_structured: bool,
    pub attention_features: Option<PrefixPrefillAttentionFeatures>,
}

pub struct ActivationMemoryContext<'a> {
    pub params: &'a AutoDeviceMapParams,
    pub limits: ExecutionMemoryLimits,
    pub dtype: DType,
    pub device: &'a Device,
    pub paged: &'a PagedAttentionConfig,
    pub cache_capacity_tokens: Option<usize>,
}

impl ActivationMemoryContext<'_> {
    pub fn query_tokens(&self, max_context: usize) -> Result<usize> {
        self.limits.validate()?;
        Ok(max_context
            .checked_mul(self.limits.max_num_seqs)
            .context("activation token capacity overflow")?
            .min(self.limits.max_num_batched_tokens)
            .max(self.limits.max_num_seqs))
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ActivationMemoryEstimate {
    pub mapped_bytes: usize,
    pub non_mapped_bytes: usize,
    pub mapped_non_attention_bytes: usize,
    pub non_mapped_non_attention_bytes: usize,
    pub planned_query_tokens: usize,
    pub logits_bytes_per_query_token: usize,
    pub custom_prefill_mask: bool,
    pub metal_prefill_mask_is_structured: bool,
}

impl ActivationMemoryEstimate {
    fn reserve_attention_inventory(mut self, cache_write_workspace: usize) -> Result<Self> {
        self.mapped_bytes = sum(&[self.mapped_bytes, cache_write_workspace])?;
        let attention = self
            .mapped_bytes
            .saturating_sub(self.mapped_non_attention_bytes);
        // Primary admission combines its non-attention inventory with each layer's attention workspace.
        self.non_mapped_bytes = self
            .non_mapped_bytes
            .max(sum(&[self.non_mapped_non_attention_bytes, attention])?);
        Ok(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AttentionWorkspaceKind {
    Eager,
    Fused,
    Paged,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AttentionMemoryShape {
    pub batch: usize,
    pub query_len: usize,
    pub kv_len: usize,
    pub query_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub dtype: DType,
    pub backend: AttentionWorkspaceKind,
    pub materialized_mask: bool,
}

fn product(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(1usize, |total, part| total.checked_mul(*part))
        .context("activation memory size overflow")
}

fn sum(parts: &[usize]) -> Result<usize> {
    parts
        .iter()
        .try_fold(0usize, |total, part| total.checked_add(*part))
        .context("activation memory size overflow")
}

fn explicit_prefix_context_capacity(
    ctx: &ActivationMemoryContext<'_>,
    model: &dyn ModelConfigLike,
) -> Result<Option<usize>> {
    if let Some(tokens) = ctx.cache_capacity_tokens {
        anyhow::ensure!(tokens > 0, "prefix KV capacity must be nonzero");
        return Ok(Some(tokens.min(model.max_seq_len())));
    }
    if matches!(
        ctx.paged.memory_goal,
        MemoryGoal::Auto | MemoryGoal::Utilization(_)
    ) {
        return Ok(None);
    }
    let block_size = ctx
        .paged
        .block_size
        .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE);
    let elements = (0..model.num_layers()).try_fold(0usize, |total, layer| {
        total
            .checked_add(model.layer_kv_cache_elements_per_token(layer).unwrap_or(0))
            .context("prefix KV geometry overflow")
    })?;
    let bytes_per_block = product(&[
        elements,
        block_size,
        ctx.paged.cache_type.to_dtype(ctx.dtype).size_in_bytes(),
    ])?;
    anyhow::ensure!(
        bytes_per_block > 0,
        "The model has no paged KV cache storage"
    );
    let blocks = requested_cache_blocks(ctx.paged.memory_goal, block_size, bytes_per_block)?
        .context("explicit KV capacity is missing")?;
    Ok(Some(
        product(&[blocks - 1, block_size])?.min(model.max_seq_len()),
    ))
}

fn additional_metal_mask_bytes(
    batch: usize,
    query_len: usize,
    kv_len: usize,
    dtype: DType,
    variants: usize,
) -> Result<usize> {
    product(&[
        variants.saturating_sub(1),
        batch,
        query_len,
        kv_len,
        DType::F32.size_in_bytes() + dtype.size_in_bytes(),
        METAL_BUFFER_ROUNDING_BOUND,
    ])
}

pub(crate) fn attention_workspace_bytes(shape: AttentionMemoryShape) -> Result<usize> {
    let AttentionMemoryShape {
        batch,
        query_len,
        kv_len,
        query_heads,
        kv_heads,
        k_head_dim,
        v_head_dim,
        dtype,
        backend,
        materialized_mask,
    } = shape;
    let dtype_bytes = dtype.size_in_bytes();
    let output = product(&[batch, query_len, query_heads, v_head_dim, dtype_bytes])?;
    let mask = if materialized_mask {
        product(&[
            batch,
            query_len,
            kv_len,
            DType::F32.size_in_bytes() + dtype_bytes,
        ])?
    } else {
        0
    };
    match backend {
        AttentionWorkspaceKind::Eager => {
            let scores = product(&[
                batch,
                query_heads,
                query_len.min(ATTENTION_CHUNK_SIZE),
                kv_len,
                EAGER_F32_SCORE_BUFFERS * DType::F32.size_in_bytes() + dtype_bytes,
            ])?;
            let kv_heads = if query_heads > kv_heads {
                query_heads
            } else {
                kv_heads
            };
            let repeated_kv = product(&[
                batch,
                kv_len,
                kv_heads,
                sum(&[k_head_dim, v_head_dim])?,
                dtype_bytes,
            ])?;
            sum(&[scores, repeated_kv, mask, output])
        }
        AttentionWorkspaceKind::Fused => {
            let packed_qkv = product(&[
                batch,
                dtype_bytes,
                sum(&[
                    product(&[query_len, query_heads, k_head_dim])?,
                    product(&[kv_len, kv_heads, sum(&[k_head_dim, v_head_dim])?])?,
                ])?,
            ])?;
            let logsumexp = product(&[batch, query_len, query_heads, DType::F32.size_in_bytes()])?;
            sum(&[packed_qkv, output, mask, logsumexp])
        }
        AttentionWorkspaceKind::Paged => {
            let partitions = kv_len.div_ceil(PAGED_PARTITION_TOKENS);
            let scratch = product(&[
                batch,
                query_heads,
                partitions,
                sum(&[v_head_dim, 2])?,
                DType::F32.size_in_bytes(),
            ])?;
            sum(&[scratch, output])
        }
    }
}

struct ModelActivationConfig<'a> {
    model: &'a dyn ModelConfigLike,
    architecture: ActivationModelConfig,
}

impl ModelActivationConfig<'_> {
    fn attention_features(&self, layer_idx: usize) -> Option<PrefixPrefillAttentionFeatures> {
        self.model
            .prefix_prefill_attention_features(layer_idx)
            .or(self.architecture.attention_features)
    }
}

fn prefill_backend(
    ctx: &ActivationMemoryContext<'_>,
    shape: &AttentionMemoryShape,
    features: Option<PrefixPrefillAttentionFeatures>,
    custom_mask: bool,
) -> AttentionWorkspaceKind {
    let Some(features) = features else {
        return AttentionWorkspaceKind::Eager;
    };
    let same_heads = shape.k_head_dim == shape.v_head_dim;
    if same_heads
        && ctx.device.is_cuda()
        && !custom_mask
        && !features.has_alibi
        && !features.has_sinks
        && shape.query_heads / shape.kv_heads <= MAX_NATIVE_FLASH_GQA_GROUPS
        && matches!(ctx.dtype, DType::F16 | DType::BF16)
        && flash_backend_supports_sdpa(
            shape.k_head_dim,
            features.has_softcap,
            features.has_sliding_window,
        )
    {
        return AttentionWorkspaceKind::Fused;
    }
    if same_heads
        && ctx.device.is_metal()
        && !features.has_alibi
        && !features.has_sinks
        && !features.has_softcap
        && matches!(ctx.dtype, DType::F16 | DType::BF16 | DType::F32)
        && matches!(shape.k_head_dim, 32 | 64 | 72 | 80 | 96 | 128 | 256)
        && shape.query_len > 1
    {
        return AttentionWorkspaceKind::Fused;
    }
    AttentionWorkspaceKind::Eager
}

fn prefix_query_lengths(max_query: usize, capacity: usize) -> Vec<usize> {
    let max_query = max_query.min(capacity);
    let mut lengths = vec![1, max_query.min(METAL_D512_SMALL_QUERY_TOKENS), max_query];
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

fn native_metal_prefill_workspace(
    ctx: &ActivationMemoryContext<'_>,
    cfg: &ModelActivationConfig<'_>,
    layer: usize,
    query_lens: &[usize],
    context_lens: &[usize],
) -> Result<Option<usize>> {
    #[cfg(feature = "metal")]
    if ctx.device.is_metal() {
        use crate::paged_attention::{
            configured_metal_cache_layout,
            plan::{metal_paged_prefill_workspace, MetalPagedPrefillWorkspaceInput},
            KvCacheLayout,
        };
        if matches!(
            cfg.model.kv_cache_layout_for_layer(layer),
            KvCacheLayout::Mla { .. }
        ) {
            return Ok(None);
        }
        let hnd = configured_metal_cache_layout()? == KvCacheLayout::MetalHnd;
        return Ok(metal_paged_prefill_workspace(
            MetalPagedPrefillWorkspaceInput {
                query_lens,
                context_lens,
                query_heads: cfg.model.num_attn_heads_for_layer(layer),
                kv_heads: cfg.model.num_kv_heads_for_layer(layer),
                key_head_dim: cfg.model.k_head_dim_for_layer(layer),
                value_head_dim: cfg.model.v_head_dim_for_layer(layer),
                block_size: ctx
                    .paged
                    .block_size
                    .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE),
                query_dtype: ctx.dtype,
                cache_dtype: ctx.paged.cache_type.to_dtype(ctx.dtype),
                hnd,
                has_custom_mask: cfg.architecture.custom_prefill_mask
                    && !cfg.architecture.metal_prefill_mask_is_structured,
                has_multimodal_ranges: cfg.architecture.custom_prefill_mask,
            },
        )?);
    }
    #[cfg(not(feature = "metal"))]
    let _ = (ctx, cfg, layer, query_lens, context_lens);
    Ok(None)
}

fn prefix_prefill_workspace(
    ctx: &ActivationMemoryContext<'_>,
    cfg: &ModelActivationConfig<'_>,
    layer_idx: usize,
    capacity: usize,
    max_query: usize,
) -> Result<usize> {
    use crate::paged_attention::plan::{
        gather_prefill_workspace_for_lengths, GatherPrefillWorkspaceRequest, PrefixPrefillPlanInput,
    };

    let features = cfg.attention_features(layer_idx);
    let known = features.is_some();
    let features = features.unwrap_or_default();
    prefix_query_lengths(max_query, capacity)
        .into_iter()
        .try_fold(0, |peak, query_len| {
            if let Some(bytes) =
                native_metal_prefill_workspace(ctx, cfg, layer_idx, &[query_len], &[capacity])?
            {
                return Ok(peak.max(bytes));
            }
            let bytes = gather_prefill_workspace_for_lengths(GatherPrefillWorkspaceRequest {
                query_lens: &[query_len],
                kv_lens: &[capacity],
                q_heads: cfg.model.num_attn_heads_for_layer(layer_idx),
                kv_heads: cfg.model.num_kv_heads_for_layer(layer_idx),
                k_head_dim: cfg.model.k_head_dim_for_layer(layer_idx),
                v_head_dim: cfg.model.v_head_dim_for_layer(layer_idx),
                dtype: ctx.dtype,
                device_is_metal: ctx.device.is_metal(),
                packed_queries: false,
                metal_mask_is_compatible: known,
                plan_input: PrefixPrefillPlanInput {
                    device_is_cuda: ctx.device.is_cuda(),
                    dtype: ctx.dtype,
                    cache_dtype: ctx.paged.cache_type.to_dtype(ctx.dtype),
                    has_alibi: features.has_alibi,
                    has_sinks: features.has_sinks,
                    has_custom_mask: !known || cfg.architecture.custom_prefill_mask,
                    causality_known: known,
                    head_size: cfg.model.k_head_dim_for_layer(layer_idx),
                    has_softcap: features.has_softcap,
                    has_sliding_window: !known || features.has_sliding_window,
                    query_layout_is_dense: true,
                    query_len,
                    q_heads: cfg.model.num_attn_heads_for_layer(layer_idx),
                    kv_heads: cfg.model.num_kv_heads_for_layer(layer_idx),
                    writes_cache: cfg.model.uses_own_kv_cache_for_layer(layer_idx),
                    is_causal: known && !cfg.architecture.custom_prefill_mask,
                    has_noncausal_mm_context: !known || cfg.architecture.custom_prefill_mask,
                    fa3_supported: false,
                    block_size: ctx
                        .paged
                        .block_size
                        .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE),
                    attention_backend: cfg.model.attention_backend_kind_for_layer(layer_idx),
                },
            })?;
            let bytes = bytes
                .checked_mul(if ctx.device.is_metal() {
                    METAL_BUFFER_ROUNDING_BOUND
                } else {
                    1
                })
                .context("prefix attention buffer rounding overflow")?;
            Ok(peak.max(bytes))
        })
}

fn model_activation_memory(
    ctx: &ActivationMemoryContext<'_>,
    cfg: ModelActivationConfig<'_>,
) -> Result<ActivationMemoryEstimate> {
    let tokens = ctx.query_tokens(cfg.model.max_seq_len())?;
    let max_query = tokens.min(cfg.model.max_seq_len());
    let batch = tokens.div_ceil(max_query);
    let prefix_context_capacity = explicit_prefix_context_capacity(ctx, cfg.model)?;
    let dtype_bytes = ctx.dtype.size_in_bytes();
    let hidden = product(&[
        tokens,
        cfg.model.hidden_size(),
        RESIDUAL_HIDDEN_BUFFERS,
        dtype_bytes,
    ])?;
    let extra_hidden = product(&[
        tokens,
        cfg.architecture.extra_hidden_elements_per_token,
        DType::F32.size_in_bytes(),
    ])?;
    let mlp = product(&[
        tokens,
        cfg.architecture.intermediate_size,
        MLP_INTERMEDIATE_BUFFERS,
        dtype_bytes,
    ])?;
    let mut mapped_non_attention_bytes = sum(&[hidden, extra_hidden, mlp])?;
    let mut prefill_attention_bytes = 0;
    let mut prefix_attention_bytes = 0;
    let mut decode_attention_bytes = 0;
    let mut cold_mask_variants = std::collections::HashSet::new();
    let mut prefix_mask_variants = std::collections::HashSet::new();
    let mut mask_heads = 1;
    for layer_idx in 0..cfg.model.num_layers() {
        if !cfg.model.layer_has_paged_kv_cache(layer_idx) {
            continue;
        }
        let features = cfg.attention_features(layer_idx);
        let mask_key = match features {
            Some(features) if !features.has_sliding_window => (None, None),
            Some(_) => match cfg.model.sliding_window_for_layer(layer_idx) {
                Some(window) => (Some(window), None),
                None => (None, Some(layer_idx)),
            },
            None => (None, Some(layer_idx)),
        };
        let query_heads = cfg.model.num_attn_heads_for_layer(layer_idx);
        let has_native_custom_mask = cfg.architecture.custom_prefill_mask
            && !cfg.architecture.metal_prefill_mask_is_structured;
        if has_native_custom_mask {
            mask_heads = mask_heads.max(query_heads);
        }
        let kv_heads = cfg.model.num_kv_heads_for_layer(layer_idx);
        let k_head_dim = cfg.model.k_head_dim_for_layer(layer_idx);
        let v_head_dim = cfg.model.v_head_dim_for_layer(layer_idx);
        let projections = product(&[
            tokens,
            sum(&[
                product(&[query_heads, k_head_dim])?,
                product(&[kv_heads, sum(&[k_head_dim, v_head_dim])?])?,
            ])?,
            PROJECTION_BUFFERS,
            dtype_bytes,
        ])?;
        mapped_non_attention_bytes =
            mapped_non_attention_bytes.max(sum(&[hidden, extra_hidden, projections])?);
        let mut prefill = AttentionMemoryShape {
            batch,
            query_len: max_query,
            kv_len: max_query,
            query_heads,
            kv_heads,
            k_head_dim,
            v_head_dim,
            dtype: ctx.dtype,
            backend: AttentionWorkspaceKind::Eager,
            materialized_mask: !ctx.device.is_cuda() || cfg.architecture.custom_prefill_mask,
        };
        prefill.backend = prefill_backend(
            ctx,
            &prefill,
            features,
            cfg.architecture.custom_prefill_mask,
        );
        prefill.materialized_mask |= prefill.backend == AttentionWorkspaceKind::Eager;
        let decode = AttentionMemoryShape {
            batch: ctx.limits.max_num_seqs,
            query_len: 1,
            kv_len: cfg.model.max_seq_len(),
            backend: AttentionWorkspaceKind::Paged,
            materialized_mask: false,
            ..prefill
        };
        let decode_bytes = attention_workspace_bytes(decode)?;
        let native_prefill = native_metal_prefill_workspace(
            ctx,
            &cfg,
            layer_idx,
            &vec![max_query; batch],
            &vec![max_query; batch],
        )?;
        let prefill_bytes = if let Some(bytes) = native_prefill {
            bytes
        } else {
            attention_workspace_bytes(prefill)?
                .checked_mul(if ctx.device.is_metal() {
                    METAL_BUFFER_ROUNDING_BOUND
                } else {
                    1
                })
                .context("attention buffer rounding overflow")?
        };
        if native_prefill.is_none() || has_native_custom_mask {
            cold_mask_variants.insert(mask_key);
        }
        prefill_attention_bytes = prefill_attention_bytes.max(prefill_bytes);
        // Reserve one full-prefix row; runtime admission reduces larger cohorts when their gathers cannot fit.
        if let Some(capacity) = prefix_context_capacity {
            let bytes = prefix_prefill_workspace(ctx, &cfg, layer_idx, capacity, max_query)?;
            prefix_attention_bytes = prefix_attention_bytes.max(bytes);
            if has_native_custom_mask
                || native_metal_prefill_workspace(
                    ctx,
                    &cfg,
                    layer_idx,
                    &[max_query.min(capacity)],
                    &[capacity],
                )?
                .is_none()
            {
                prefix_mask_variants.insert(mask_key);
            }
        }
        decode_attention_bytes = decode_attention_bytes.max(decode_bytes);
    }
    if ctx.device.is_metal() {
        prefill_attention_bytes = sum(&[
            prefill_attention_bytes,
            additional_metal_mask_bytes(
                batch,
                max_query,
                max_query,
                ctx.dtype,
                cold_mask_variants.len(),
            )?
            .checked_mul(mask_heads)
            .context("retained mask size overflow")?,
        ])?;
        if let Some(capacity) = prefix_context_capacity {
            prefix_attention_bytes = sum(&[
                prefix_attention_bytes,
                additional_metal_mask_bytes(
                    1,
                    max_query.min(capacity),
                    capacity,
                    ctx.dtype,
                    prefix_mask_variants.len(),
                )?
                .checked_mul(mask_heads)
                .context("retained mask size overflow")?,
            ])?;
        }
    }
    #[cfg(feature = "metal")]
    if ctx.device.is_metal() {
        use crate::paged_attention::{
            configured_metal_cache_layout,
            plan::{metal_decode_workspace, MetalDecodeWorkspaceInput},
            AttentionFeaturesConfig, KvCacheLayout,
        };
        let model = AttentionFeaturesConfig::new(cfg.model, cfg.architecture.attention_features);
        let hnd = configured_metal_cache_layout()? == KvCacheLayout::MetalHnd;
        let hnd_by_layer = (0..model.num_layers())
            .map(|layer| {
                hnd && !matches!(
                    model.kv_cache_layout_for_layer(layer),
                    KvCacheLayout::Mla { .. }
                )
            })
            .collect::<Vec<_>>();
        let context = prefix_context_capacity.unwrap_or(max_query).max(max_query);
        decode_attention_bytes = metal_decode_workspace(
            Some(&model),
            MetalDecodeWorkspaceInput {
                activation_dtype: ctx.dtype,
                cache_dtype: ctx.paged.cache_type.to_dtype(ctx.dtype),
                block_size: ctx
                    .paged
                    .block_size
                    .unwrap_or(DEFAULT_PAGED_ATTENTION_BLOCK_SIZE),
                query_lens: &vec![1; ctx.limits.max_num_seqs],
                context_lens: &vec![context; ctx.limits.max_num_seqs],
                hnd_by_layer: &hnd_by_layer,
                capacity_bound: true,
                has_custom_mask: cfg.architecture.custom_prefill_mask
                    && !cfg.architecture.metal_prefill_mask_is_structured,
                has_multimodal_ranges: cfg.architecture.custom_prefill_mask,
            },
        )?;
    }
    let attention_bytes = prefill_attention_bytes
        .max(prefix_attention_bytes)
        .max(decode_attention_bytes);
    let logits = product(&[
        ctx.limits.max_num_seqs,
        cfg.architecture.vocab_size,
        dtype_bytes + DType::F32.size_in_bytes(),
    ])?;
    let non_mapped_non_attention_bytes = sum(&[hidden, extra_hidden, logits])?;
    let rounding = if ctx.device.is_metal() {
        METAL_BUFFER_ROUNDING_BOUND
    } else {
        1
    };
    let mapped_non_attention_bytes = mapped_non_attention_bytes
        .checked_mul(rounding)
        .context("mapped activation buffer rounding overflow")?;
    let non_mapped_non_attention_bytes = non_mapped_non_attention_bytes
        .checked_mul(rounding)
        .context("non-mapped activation buffer rounding overflow")?;
    Ok(ActivationMemoryEstimate {
        mapped_bytes: sum(&[mapped_non_attention_bytes, attention_bytes])?,
        non_mapped_bytes: non_mapped_non_attention_bytes,
        mapped_non_attention_bytes,
        non_mapped_non_attention_bytes,
        planned_query_tokens: tokens,
        logits_bytes_per_query_token: product(&[
            cfg.architecture.vocab_size,
            dtype_bytes + DType::F32.size_in_bytes(),
            rounding,
        ])?,
        custom_prefill_mask: cfg.architecture.custom_prefill_mask,
        metal_prefill_mask_is_structured: cfg.architecture.metal_prefill_mask_is_structured,
    })
}

pub(crate) fn reserve_model_activation_memory(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    paged: &mut PagedAttentionConfig,
    dtype: DType,
    device: &Device,
    params: &AutoDeviceMapParams,
) -> Result<ActivationMemoryEstimate> {
    let estimate = estimate_model_activation_memory(
        loader,
        config,
        ActivationMemoryContext {
            params,
            limits: paged.execution_limits,
            dtype,
            device,
            paged,
            cache_capacity_tokens: None,
        },
    )?;
    apply_model_activation_memory(paged, estimate);
    Ok(estimate)
}

pub(crate) fn estimate_model_activation_memory(
    loader: &dyn DeviceMappedModelLoader,
    config: &str,
    ctx: ActivationMemoryContext<'_>,
) -> Result<ActivationMemoryEstimate> {
    let model = loader.model_config(config)?;
    let max_query = model.max_seq_len().min(ctx.limits.max_num_batched_tokens);
    let params = match ctx.params {
        AutoDeviceMapParams::Text { .. } => AutoDeviceMapParams::Text {
            max_seq_len: max_query,
            max_batch_size: 1,
        },
        AutoDeviceMapParams::Multimodal {
            max_image_shape,
            max_num_images,
            ..
        } => AutoDeviceMapParams::Multimodal {
            max_seq_len: max_query,
            max_batch_size: 1,
            max_image_shape: *max_image_shape,
            max_num_images: *max_num_images,
        },
    };
    let ctx = ActivationMemoryContext {
        params: &params,
        ..ctx
    };
    let architecture = loader.activation_model_config(config)?;
    let cfg = ModelActivationConfig {
        model: &*model,
        architecture,
    };
    let mut native_prefill = ctx.device.is_metal()
        && (!architecture.custom_prefill_mask || architecture.metal_prefill_mask_is_structured);
    let mut paged_layers = 0;
    for layer in (0..model.num_layers()).filter(|&layer| model.layer_has_paged_kv_cache(layer)) {
        paged_layers += 1;
        native_prefill &=
            native_metal_prefill_workspace(&ctx, &cfg, layer, &[max_query], &[max_query])?
                .is_some();
    }
    let mut estimate = model_activation_memory(&ctx, cfg)?;
    let encoder_params = match &params {
        AutoDeviceMapParams::Text { max_seq_len, .. } => AutoDeviceMapParams::Text {
            max_seq_len: *max_seq_len,
            max_batch_size: ctx.limits.max_num_seqs,
        },
        AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_image_shape,
            max_num_images,
            ..
        } => AutoDeviceMapParams::Multimodal {
            max_seq_len: *max_seq_len,
            max_batch_size: ctx.limits.max_num_seqs,
            max_image_shape: *max_image_shape,
            max_num_images: *max_num_images,
        },
    };
    let encoder = product(&[
        loader.non_mapped_max_act_size_elems(config, &encoder_params)?,
        DType::F32.size_in_bytes(),
    ])?;
    estimate.non_mapped_non_attention_bytes =
        sum(&[estimate.non_mapped_non_attention_bytes, encoder])?;
    estimate.non_mapped_bytes = estimate.non_mapped_non_attention_bytes;
    if !native_prefill || paged_layers == 0 {
        let legacy = product(&[
            loader.mapped_max_act_size_elems(config, &params)?,
            DType::F32.size_in_bytes(),
        ])?;
        estimate.mapped_bytes = estimate.mapped_bytes.max(legacy);
    }
    #[cfg(feature = "metal")]
    let cache_write_workspace = if ctx.device.is_metal() {
        crate::paged_attention::plan::metal_cache_write_workspace(
            Some(model.as_ref()),
            estimate.planned_query_tokens,
            ctx.dtype,
        )?
    } else {
        0
    };
    #[cfg(not(feature = "metal"))]
    let cache_write_workspace = 0;
    estimate.reserve_attention_inventory(cache_write_workspace)
}

pub(crate) fn apply_model_activation_memory(
    paged: &mut PagedAttentionConfig,
    estimate: ActivationMemoryEstimate,
) {
    paged.custom_prefill_mask |= estimate.custom_prefill_mask;
    paged.metal_prefill_mask_is_structured = estimate.metal_prefill_mask_is_structured;
    paged.reserve_activation_memory(
        estimate.non_mapped_bytes.max(estimate.mapped_bytes),
        estimate.mapped_bytes,
    );
    paged.primary_non_attention_memory_reservation_bytes =
        paged.primary_non_attention_memory_reservation_bytes.max(
            estimate
                .non_mapped_non_attention_bytes
                .max(estimate.mapped_non_attention_bytes),
        );
    paged.mapped_non_attention_memory_reservation_bytes = paged
        .mapped_non_attention_memory_reservation_bytes
        .max(estimate.mapped_non_attention_bytes);
    paged.planned_query_tokens = paged
        .planned_query_tokens
        .max(estimate.planned_query_tokens);
    paged.logits_bytes_per_query_token = paged
        .logits_bytes_per_query_token
        .max(estimate.logits_bytes_per_query_token);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primary_encoder_reservation_includes_planned_attention() -> Result<()> {
        let baseline = ActivationMemoryEstimate {
            mapped_bytes: 30,
            non_mapped_bytes: 100,
            mapped_non_attention_bytes: 20,
            non_mapped_non_attention_bytes: 100,
            planned_query_tokens: 512,
            logits_bytes_per_query_token: 7,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        };
        for attention in [10, 80, 200] {
            let estimate = ActivationMemoryEstimate {
                mapped_bytes: baseline.mapped_non_attention_bytes + attention,
                ..baseline
            }
            .reserve_attention_inventory(0)?;
            assert_eq!(estimate.non_mapped_bytes, 100 + attention);
            assert_eq!(estimate.mapped_bytes, 20 + attention);
            let mut paged = PagedAttentionConfig::new(
                Some(32),
                MemoryGoal::Auto,
                crate::paged_attention::PagedCacheType::Auto,
            )?;
            apply_model_activation_memory(&mut paged, estimate);
            let plan = paged.runtime_memory_plan();
            assert_eq!(
                plan.activation_bytes(true, baseline.planned_query_tokens)?,
                plan.non_attention_bytes(true, baseline.planned_query_tokens)? + attention
            );
            assert_eq!(plan.mapped_activation_bytes, 20 + attention);
            assert_eq!(plan.primary_non_attention_bytes, 100);
            assert_eq!(plan.mapped_non_attention_bytes, 20);
            assert_eq!(plan.logits_bytes_per_query_token, 7);
        }
        Ok(())
    }

    #[test]
    fn primary_attention_normalization_preserves_larger_phase_reservations() -> Result<()> {
        let estimate = ActivationMemoryEstimate {
            mapped_bytes: 90,
            non_mapped_bytes: 200,
            mapped_non_attention_bytes: 80,
            non_mapped_non_attention_bytes: 50,
            planned_query_tokens: 512,
            logits_bytes_per_query_token: 0,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        };
        assert_eq!(estimate.reserve_attention_inventory(0)?, estimate);
        assert!(ActivationMemoryEstimate {
            non_mapped_non_attention_bytes: usize::MAX,
            ..estimate
        }
        .reserve_attention_inventory(0)
        .is_err());
        Ok(())
    }

    #[test]
    fn cache_write_inventory_is_reserved_on_primary_and_mapped_devices() -> Result<()> {
        const QUERY_TOKENS: usize = 512;
        const PREFIX_ATTENTION: usize = 80;
        const CACHE_WRITE: usize = 32;
        let estimate = ActivationMemoryEstimate {
            mapped_bytes: 20 + PREFIX_ATTENTION,
            non_mapped_bytes: 100,
            mapped_non_attention_bytes: 20,
            non_mapped_non_attention_bytes: 100,
            planned_query_tokens: QUERY_TOKENS,
            logits_bytes_per_query_token: 0,
            custom_prefill_mask: false,
            metal_prefill_mask_is_structured: false,
        }
        .reserve_attention_inventory(CACHE_WRITE)?;
        let mut paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?;
        apply_model_activation_memory(&mut paged, estimate);
        let plan = paged.runtime_memory_plan();
        for primary in [false, true] {
            assert_eq!(
                plan.activation_bytes(primary, QUERY_TOKENS)?,
                plan.non_attention_bytes(primary, QUERY_TOKENS)? + PREFIX_ATTENTION + CACHE_WRITE
            );
        }
        assert!(ActivationMemoryEstimate {
            mapped_bytes: usize::MAX,
            ..estimate
        }
        .reserve_attention_inventory(CACHE_WRITE)
        .is_err());
        Ok(())
    }

    #[test]
    fn structured_mask_metadata_reaches_runtime_without_changing_non_attention() -> Result<()> {
        let mut paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?;
        apply_model_activation_memory(
            &mut paged,
            ActivationMemoryEstimate {
                mapped_bytes: 120,
                non_mapped_bytes: 220,
                mapped_non_attention_bytes: 100,
                non_mapped_non_attention_bytes: 200,
                planned_query_tokens: 512,
                custom_prefill_mask: true,
                metal_prefill_mask_is_structured: true,
                ..Default::default()
            },
        );
        let plan = paged.runtime_memory_plan();
        assert!(plan.custom_prefill_mask);
        assert!(plan.metal_prefill_mask_is_structured);
        assert_eq!(plan.non_attention_bytes(true, 1)?, 200);
        assert_eq!(plan.non_attention_bytes(false, 1)?, 100);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    #[ignore = "requires a Metal device"]
    fn native_metal_startup_reservation_has_no_full_prefix_gather() -> Result<()> {
        let device = Device::new_metal(0)?;
        let metadata = crate::paged_attention::ModelConfigMetadata {
            max_seq_len: 65_536,
            num_layers: 2,
            hidden_size: 1024,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 512,
            v_head_dim: 512,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let model = crate::paged_attention::AttentionFeaturesConfig::new(
            &metadata,
            Some(PrefixPrefillAttentionFeatures::default()),
        );
        let paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?
        .with_execution_limits(ExecutionMemoryLimits {
            max_num_seqs: 1,
            max_num_batched_tokens: 512,
            max_prefill_chunk_tokens: 512,
        })?;
        let params = AutoDeviceMapParams::default_text();
        let estimate = |capacity, custom, structured| {
            model_activation_memory(
                &ActivationMemoryContext {
                    params: &params,
                    limits: paged.execution_limits,
                    dtype: DType::BF16,
                    device: &device,
                    paged: &paged,
                    cache_capacity_tokens: Some(capacity),
                },
                ModelActivationConfig {
                    model: &model,
                    architecture: ActivationModelConfig {
                        intermediate_size: 4096,
                        vocab_size: 1024,
                        custom_prefill_mask: custom,
                        metal_prefill_mask_is_structured: structured,
                        ..Default::default()
                    },
                },
            )
        };
        let short = estimate(512, false, false)?;
        let long = estimate(65_536, false, false)?;
        assert_eq!(
            long.mapped_non_attention_bytes,
            short.mapped_non_attention_bytes
        );
        assert!(long.mapped_bytes < short.mapped_bytes * 2);
        let structured = estimate(65_536, true, true)?;
        let custom = estimate(65_536, true, false)?;
        assert!(custom.mapped_bytes > structured.mapped_bytes * 8);
        assert_eq!(
            custom.mapped_non_attention_bytes,
            structured.mapped_non_attention_bytes
        );
        Ok(())
    }

    #[test]
    fn prefix_capacity_uses_usable_blocks_and_actual_cache_dtype() -> Result<()> {
        let model = crate::paged_attention::ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers: 2,
            hidden_size: 512,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 96,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let params = AutoDeviceMapParams::default_text();
        let capacity = |goal, cache_type| -> Result<_> {
            let paged = PagedAttentionConfig::new(Some(32), goal, cache_type)?;
            explicit_prefix_context_capacity(
                &ActivationMemoryContext {
                    params: &params,
                    limits: paged.execution_limits,
                    dtype: DType::BF16,
                    device: &Device::Cpu,
                    paged: &paged,
                    cache_capacity_tokens: None,
                },
                &model,
            )
        };
        let auto = crate::paged_attention::PagedCacheType::Auto;
        let fp8 = crate::paged_attention::PagedCacheType::F8E4M3;
        let bytes_per_block = 2 * 2 * (64 + 96) * 32 * 2;
        assert_eq!(capacity(MemoryGoal::KvTokens(33), auto)?, Some(64));
        assert_eq!(capacity(MemoryGoal::KvTokens(9000), auto)?, Some(8192));
        assert_eq!(
            capacity(MemoryGoal::KvBytes(3 * bytes_per_block), auto)?,
            Some(64)
        );
        assert_eq!(
            capacity(MemoryGoal::KvBytes(3 * bytes_per_block), fp8)?,
            Some(160)
        );
        assert_eq!(capacity(MemoryGoal::Auto, auto)?, None);
        assert_eq!(capacity(MemoryGoal::Utilization(0.9), auto)?, None);
        assert!(capacity(MemoryGoal::KvBytes(bytes_per_block), auto).is_err());
        Ok(())
    }

    #[test]
    fn shared_estimator_reserves_prefix_attention_and_logits() -> Result<()> {
        let model = crate::paged_attention::ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers: 2,
            hidden_size: 512,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let params = AutoDeviceMapParams::default_text();
        let paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?;
        let mut context = ActivationMemoryContext {
            params: &params,
            limits: paged.execution_limits,
            dtype: DType::BF16,
            device: &Device::Cpu,
            paged: &paged,
            cache_capacity_tokens: None,
        };
        let architecture = ActivationModelConfig {
            intermediate_size: 1536,
            vocab_size: 1024,
            ..Default::default()
        };
        let baseline = model_activation_memory(
            &context,
            ModelActivationConfig {
                model: &model,
                architecture,
            },
        )?;
        context.cache_capacity_tokens = Some(8192);
        let prefix = model_activation_memory(
            &context,
            ModelActivationConfig {
                model: &model,
                architecture,
            },
        )?;
        assert!(prefix.mapped_bytes > baseline.mapped_bytes);
        assert_eq!(
            prefix.mapped_non_attention_bytes,
            baseline.mapped_non_attention_bytes
        );
        assert_eq!(prefix.non_mapped_bytes, baseline.non_mapped_bytes);
        assert_eq!(prefix.logits_bytes_per_query_token, 1024 * (2 + 4));
        Ok(())
    }

    fn attention_shape() -> AttentionMemoryShape {
        AttentionMemoryShape {
            batch: 1,
            query_len: 1024,
            kv_len: 8192,
            query_heads: 16,
            kv_heads: 8,
            k_head_dim: 128,
            v_head_dim: 128,
            dtype: DType::BF16,
            backend: AttentionWorkspaceKind::Eager,
            materialized_mask: true,
        }
    }

    #[test]
    fn retained_token_state_is_reserved_at_f32_width() -> Result<()> {
        let model = crate::paged_attention::ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers: 2,
            hidden_size: 512,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };
        let params = AutoDeviceMapParams::default_text();
        let paged = PagedAttentionConfig::new(
            Some(32),
            MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?;
        let estimate = |dtype, extra_hidden_elements_per_token| {
            model_activation_memory(
                &ActivationMemoryContext {
                    params: &params,
                    limits: paged.execution_limits,
                    dtype,
                    device: &Device::Cpu,
                    paged: &paged,
                    cache_capacity_tokens: None,
                },
                ModelActivationConfig {
                    model: &model,
                    architecture: ActivationModelConfig {
                        intermediate_size: 1536,
                        vocab_size: 1024,
                        extra_hidden_elements_per_token,
                        ..Default::default()
                    },
                },
            )
        };
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let baseline = estimate(dtype, 0)?;
            let retained = estimate(dtype, 512)?;
            let expected = baseline.planned_query_tokens * 512 * DType::F32.size_in_bytes();
            assert_eq!(
                retained.mapped_non_attention_bytes - baseline.mapped_non_attention_bytes,
                expected
            );
            assert_eq!(
                retained.non_mapped_non_attention_bytes - baseline.non_mapped_non_attention_bytes,
                expected
            );
        }
        assert!(estimate(DType::BF16, usize::MAX).is_err());
        Ok(())
    }

    #[test]
    fn eager_attention_retains_full_keys_and_f32_scores() -> Result<()> {
        let shape = attention_shape();
        let scores = 16 * 1024 * 8192 * (2 * 4 + 2);
        let repeated_kv = 8192 * 16 * (128 + 128) * 2;
        let mask = 1024 * 8192 * (4 + 2);
        let output = 1024 * 16 * 128 * 2;
        assert_eq!(
            attention_workspace_bytes(shape)?,
            scores + repeated_kv + mask + output
        );
        let shorter = AttentionMemoryShape {
            kv_len: 1024,
            ..shape
        };
        assert!(attention_workspace_bytes(shape)? > 7 * attention_workspace_bytes(shorter)?);
        Ok(())
    }

    #[test]
    fn fused_workspace_does_not_reserve_attention_scores() -> Result<()> {
        let eager = attention_shape();
        let fused = AttentionMemoryShape {
            backend: AttentionWorkspaceKind::Fused,
            materialized_mask: false,
            ..eager
        };
        assert!(attention_workspace_bytes(fused)? < attention_workspace_bytes(eager)? / 8);
        Ok(())
    }

    #[test]
    fn attention_workspace_rejects_size_overflow() {
        assert!(attention_workspace_bytes(AttentionMemoryShape {
            batch: usize::MAX,
            ..attention_shape()
        })
        .is_err());
    }

    #[test]
    fn metal_masks_count_distinct_full_and_window_variants() -> Result<()> {
        assert_eq!(additional_metal_mask_bytes(2, 16, 32, DType::BF16, 1)?, 0);
        assert_eq!(
            additional_metal_mask_bytes(2, 16, 32, DType::BF16, 2)?,
            2 * 16 * 32 * (4 + 2) * METAL_BUFFER_ROUNDING_BOUND
        );
        assert!(additional_metal_mask_bytes(usize::MAX, 2, 2, DType::BF16, 2).is_err());
        Ok(())
    }

    #[test]
    fn idle_prefill_uses_batch_budget_not_latency_chunk_target() -> Result<()> {
        let params = AutoDeviceMapParams::default_text();
        let paged = PagedAttentionConfig::new(
            Some(32),
            crate::MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?
        .with_execution_limits(ExecutionMemoryLimits {
            max_num_seqs: 4,
            max_num_batched_tokens: 4096,
            max_prefill_chunk_tokens: 512,
        })?;
        let context = ActivationMemoryContext {
            params: &params,
            limits: paged.execution_limits,
            dtype: DType::BF16,
            device: &Device::Cpu,
            paged: &paged,
            cache_capacity_tokens: None,
        };
        assert_eq!(context.query_tokens(8192)?, 4096);
        assert_eq!(context.query_tokens(128)?, 512);
        Ok(())
    }

    #[test]
    fn qwen_reservations_follow_execution_limits_and_include_logits() -> Result<()> {
        let config = crate::models::qwen3::Config {
            vocab_size: 1024,
            hidden_size: 512,
            intermediate_size: 1536,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            max_position_embeddings: 8192,
            head_dim: Some(128),
            ..Default::default()
        };
        let config = serde_json::to_string(&config)?;
        let metadata = super::super::Qwen3Loader.model_config(&config)?;
        assert_eq!(metadata.k_head_dim(), 128);
        assert_eq!(metadata.v_head_dim(), 128);
        let mut paged = PagedAttentionConfig::new(
            Some(32),
            crate::MemoryGoal::Auto,
            crate::paged_attention::PagedCacheType::Auto,
        )?
        .with_execution_limits(ExecutionMemoryLimits {
            max_num_seqs: 4,
            max_num_batched_tokens: 1024,
            max_prefill_chunk_tokens: 128,
        })?;
        let first = reserve_model_activation_memory(
            &super::super::Qwen3Loader,
            &config,
            &mut paged,
            DType::BF16,
            &Device::Cpu,
            &AutoDeviceMapParams::Text {
                max_seq_len: 4,
                max_batch_size: 999,
            },
        )?;
        let second = reserve_model_activation_memory(
            &super::super::Qwen3Loader,
            &config,
            &mut paged,
            DType::BF16,
            &Device::Cpu,
            &AutoDeviceMapParams::default_text(),
        )?;
        assert_eq!(first, second);
        assert_eq!(first.planned_query_tokens, 1024);
        assert_eq!(paged.planned_query_tokens, 1024);
        assert_eq!(first.logits_bytes_per_query_token, 1024 * (2 + 4));
        assert!(first.mapped_non_attention_bytes > 0);
        assert!(first.non_mapped_non_attention_bytes > 0);
        assert!(first.mapped_bytes > first.mapped_non_attention_bytes);
        assert_eq!(
            paged.mapped_activation_memory_reservation_bytes,
            first.mapped_bytes
        );
        Ok(())
    }

    #[test]
    fn prefix_reservation_follows_capacity_without_inflating_auto_placement() -> Result<()> {
        let config = serde_json::to_string(&crate::models::qwen3::Config {
            vocab_size: 1024,
            hidden_size: 512,
            intermediate_size: 1536,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            max_position_embeddings: 8192,
            head_dim: Some(128),
            ..Default::default()
        })?;
        let params = AutoDeviceMapParams::default_text();
        let estimate = |goal, capacity| -> Result<_> {
            let paged = PagedAttentionConfig::new(
                Some(32),
                goal,
                crate::paged_attention::PagedCacheType::Auto,
            )?
            .with_execution_limits(ExecutionMemoryLimits {
                max_num_seqs: 4,
                max_num_batched_tokens: 512,
                max_prefill_chunk_tokens: 128,
            })?;
            estimate_model_activation_memory(
                &super::super::Qwen3Loader,
                &config,
                ActivationMemoryContext {
                    params: &params,
                    limits: paged.execution_limits,
                    dtype: DType::BF16,
                    device: &Device::Cpu,
                    paged: &paged,
                    cache_capacity_tokens: capacity,
                },
            )
        };
        let baseline = estimate(MemoryGoal::Auto, None)?;
        let explicit = estimate(MemoryGoal::KvTokens(8192), None)?;
        let resolved = estimate(MemoryGoal::Auto, Some(8192))?;
        assert_eq!(explicit, resolved);
        assert_eq!(baseline.planned_query_tokens, resolved.planned_query_tokens);
        assert_eq!(
            baseline.mapped_non_attention_bytes,
            resolved.mapped_non_attention_bytes
        );
        assert!(resolved.mapped_bytes > baseline.mapped_bytes);
        let mut previous = baseline.mapped_bytes;
        for capacity in [32, 64, 512, 1024, 2048, 8192] {
            let next = estimate(MemoryGoal::Auto, Some(capacity))?.mapped_bytes;
            assert!(
                next >= previous,
                "capacity {capacity} reduced the reservation"
            );
            previous = next;
        }
        assert_eq!(estimate(MemoryGoal::Auto, Some(16384))?, resolved);
        assert!(estimate(MemoryGoal::Auto, Some(0)).is_err());
        Ok(())
    }
}
