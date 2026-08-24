use candle_core::{DType, Result};

use super::{
    attention_backend::AttentionBackendKind, config::PrefixPrefillAttentionFeatures,
    ModelConfigLike,
};
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::attention::flash_backend_supports_sdpa;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{self, FlashInferDecodePlan, FlashInferDecodePlanInput};

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct PrefixPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub cache_dtype: DType,
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_custom_mask: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub has_softcap: bool,
    pub has_sliding_window: bool,
    pub query_layout_is_dense: bool,
    pub query_len: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub writes_cache: bool,
    pub is_causal: bool,
    pub has_noncausal_mm_context: bool,
    pub fa3_supported: bool,
    pub block_size: usize,
    pub attention_backend: AttentionBackendKind,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefixPrefillPlan {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    Fa3Fp8Paged,
    #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
    FlashAttentionPaged,
    GatherSdpa,
}

impl PrefixPrefillPlan {
    pub fn choose(input: PrefixPrefillPlanInput) -> Self {
        #[cfg(not(all(feature = "cuda", feature = "flash-attn", target_family = "unix")))]
        let _ = (
            input.device_is_cuda,
            input.dtype,
            input.cache_dtype,
            input.has_alibi,
            input.has_sinks,
            input.has_custom_mask,
            input.causality_known,
            input.head_size,
            input.has_softcap,
            input.has_sliding_window,
            input.query_layout_is_dense,
            input.query_len,
            input.q_heads,
            input.kv_heads,
            input.writes_cache,
            input.is_causal,
            input.has_noncausal_mm_context,
            input.fa3_supported,
            input.block_size,
            input.attention_backend,
        );

        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if mistralrs_paged_attn::USE_FA3_FP8_PAGED && fa3_paged_prefill_supported(input) {
            return Self::Fa3Fp8Paged;
        }

        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if input.device_is_cuda
            && matches!(input.dtype, DType::F16 | DType::BF16)
            && input.cache_dtype == input.dtype
            && !input.has_alibi
            && !input.has_sinks
            && !input.has_custom_mask
            && input.causality_known
            && input.query_layout_is_dense
            && paged_flash_attention_supports(
                input.head_size,
                input.block_size,
                input.has_softcap,
                input.has_sliding_window,
            )
            && matches!(input.attention_backend, AttentionBackendKind::FlashInfer)
        {
            return Self::FlashAttentionPaged;
        }

        Self::GatherSdpa
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub(crate) fn fa3_paged_prefill_supported(input: PrefixPrefillPlanInput) -> bool {
    input.fa3_supported
        && input.device_is_cuda
        && input.dtype == DType::BF16
        && input.cache_dtype == DType::F8E4M3
        && input.writes_cache
        && !input.has_alibi
        && !input.has_sinks
        && !input.has_custom_mask
        && input.causality_known
        && (input.query_len == 1 || input.is_causal)
        && !input.has_noncausal_mm_context
        && input.head_size == 256
        && input.query_layout_is_dense
        && input.query_len > 0
        && input.query_len <= mistralrs_paged_attn::FA3_DECODE_MAX_QUERY_LEN
        && fa3_group_size_is_supported(input.q_heads, input.kv_heads)
        && input.block_size.is_multiple_of(32)
        && !input.has_softcap
        && !input.has_sliding_window
        && matches!(input.attention_backend, AttentionBackendKind::FlashInfer)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PromptPrefillWorkspaceInput<'a> {
    pub activation_dtype: DType,
    pub cache_dtype: DType,
    pub device_is_cuda: bool,
    pub block_size: usize,
    pub query_lens: &'a [usize],
    pub full_context_lens: &'a [usize],
    pub max_pages_per_sequence: usize,
    pub requires_prefix_attention: bool,
    pub is_causal: bool,
    pub causality_known: bool,
    pub has_custom_mask: bool,
    pub has_noncausal_mm_context: bool,
    pub has_sliding_window: bool,
    pub fa3_num_sm_by_layer: &'a [Option<usize>],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct PromptPrefillWorkspace {
    pub bytes: usize,
    pub gather_workspace_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
struct GatherPrefillWorkspaceInput {
    batch: usize,
    total_q: usize,
    max_q: usize,
    total_kv: usize,
    max_kv: usize,
    q_heads: usize,
    kv_heads: usize,
    k_head_dim: usize,
    v_head_dim: usize,
    dtype: DType,
    packed_varlen: bool,
}

fn gather_prefill_workspace_bytes(input: GatherPrefillWorkspaceInput) -> Result<usize> {
    if input.batch == 0
        || input.total_q == 0
        || input.max_q == 0
        || input.total_kv == 0
        || input.max_kv == 0
        || input.q_heads == 0
        || input.kv_heads == 0
        || !input.q_heads.is_multiple_of(input.kv_heads)
        || input.k_head_dim == 0
        || input.v_head_dim == 0
    {
        candle_core::bail!("invalid gather prefill workspace shape");
    }
    let dtype_bytes = input.dtype.size_in_bytes();
    let packed_k = checked_tensor_bytes(
        &[input.total_kv, input.kv_heads, input.k_head_dim],
        dtype_bytes,
        "packed key",
    )?;
    let packed_v = checked_tensor_bytes(
        &[input.total_kv, input.kv_heads, input.v_head_dim],
        dtype_bytes,
        "packed value",
    )?;
    let packed_kv = checked_sum(&[packed_k, packed_v], "packed KV")?;

    let packed_output = checked_tensor_bytes(
        &[input.total_q, input.q_heads, input.v_head_dim],
        dtype_bytes,
        "packed attention output",
    )?;
    let padded_output = checked_tensor_bytes(
        &[input.batch, input.q_heads, input.max_q, input.v_head_dim],
        dtype_bytes,
        "padded attention output",
    )?;
    let output = packed_output.max(padded_output);

    if input.packed_varlen {
        let group_size = input.q_heads / input.kv_heads;
        let (expanded_k, expanded_v) =
            if group_size > crate::attention::FLASH_ATTN_NATIVE_MAX_GQA_GROUP {
                (
                    checked_tensor_bytes(
                        &[input.total_kv, input.q_heads, input.k_head_dim],
                        dtype_bytes,
                        "expanded packed key",
                    )?,
                    checked_tensor_bytes(
                        &[input.total_kv, input.q_heads, input.v_head_dim],
                        dtype_bytes,
                        "expanded packed value",
                    )?,
                )
            } else {
                (0, 0)
            };
        return checked_sum(
            &[packed_kv, expanded_k, expanded_v, output],
            "packed varlen attention peak",
        );
    }

    let padded_k = checked_tensor_bytes(
        &[input.batch, input.kv_heads, input.max_kv, input.k_head_dim],
        dtype_bytes,
        "padded key",
    )?;
    let padded_v = checked_tensor_bytes(
        &[input.batch, input.kv_heads, input.max_kv, input.v_head_dim],
        dtype_bytes,
        "padded value",
    )?;
    let unpack_peak = checked_sum(
        &[
            padded_k,
            padded_v
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::msg("padded value workspace overflow"))?,
        ],
        "KV unpack",
    )?
    .max(
        padded_k
            .checked_mul(2)
            .ok_or_else(|| candle_core::Error::msg("padded key workspace overflow"))?,
    );

    let padded_mask_elements = checked_product(
        &[input.batch, input.max_q, input.max_kv],
        "padded attention mask",
    )?;
    let packed_mask_elements =
        checked_product(&[input.total_q, input.total_kv], "packed attention mask")?;
    let mask_elements = padded_mask_elements.max(packed_mask_elements);
    let mask_f32 = mask_elements
        .checked_mul(DType::F32.size_in_bytes())
        .ok_or_else(|| candle_core::Error::msg("F32 attention mask workspace overflow"))?;
    let mask_dtype = mask_elements
        .checked_mul(dtype_bytes)
        .ok_or_else(|| candle_core::Error::msg("attention mask workspace overflow"))?;
    let mask_peak = checked_sum(&[packed_kv, mask_f32, mask_dtype], "attention mask peak")?;

    let repeated_k = if input.q_heads > input.kv_heads {
        checked_tensor_bytes(
            &[input.batch, input.q_heads, input.max_kv, input.k_head_dim],
            dtype_bytes,
            "padded repeated key",
        )?
        .max(checked_tensor_bytes(
            &[input.total_kv, input.q_heads, input.k_head_dim],
            dtype_bytes,
            "packed repeated key",
        )?)
    } else {
        0
    };
    let repeated_v = if input.q_heads > input.kv_heads {
        checked_tensor_bytes(
            &[input.batch, input.q_heads, input.max_kv, input.v_head_dim],
            dtype_bytes,
            "padded repeated value",
        )?
        .max(checked_tensor_bytes(
            &[input.total_kv, input.q_heads, input.v_head_dim],
            dtype_bytes,
            "packed repeated value",
        )?)
    } else {
        0
    };

    let padded_score_elements = checked_product(
        &[
            input.batch,
            input.q_heads,
            input.max_q.min(crate::attention::ATTENTION_CHUNK_SIZE),
            input.max_kv,
        ],
        "padded attention scores",
    )?;
    let packed_score_elements = checked_product(
        &[
            input.q_heads,
            input.total_q.min(crate::attention::ATTENTION_CHUNK_SIZE),
            input.total_kv,
        ],
        "packed attention scores",
    )?;
    let score_elements = padded_score_elements.max(packed_score_elements);
    let score_peak = score_elements
        .checked_mul(
            DType::F32
                .size_in_bytes()
                .checked_mul(2)
                .and_then(|bytes| bytes.checked_add(dtype_bytes))
                .ok_or_else(|| candle_core::Error::msg("attention score element size overflow"))?,
        )
        .ok_or_else(|| candle_core::Error::msg("attention score workspace overflow"))?;
    let score_and_output_peak = checked_sum(&[score_peak, output], "attention score/output peak")?;
    let score_dtype = score_elements
        .checked_mul(dtype_bytes)
        .ok_or_else(|| candle_core::Error::msg("attention score dtype workspace overflow"))?;
    let retained_bias_and_scores = score_dtype
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::msg("retained attention bias workspace overflow"))?;
    let attended_v = checked_tensor_bytes(
        &[input.batch, input.q_heads, input.max_kv, input.v_head_dim],
        dtype_bytes,
        "padded attended value",
    )?
    .max(checked_tensor_bytes(
        &[input.total_kv, input.q_heads, input.v_head_dim],
        dtype_bytes,
        "packed attended value",
    )?);
    let context_matmul_peak = checked_sum(
        &[retained_bias_and_scores, attended_v, output],
        "attention context matmul peak",
    )?;
    let retained_output_peak = output
        .checked_mul(2)
        .ok_or_else(|| candle_core::Error::msg("retained attention output overflow"))?;
    let unpack_peak = checked_sum(&[packed_kv, mask_dtype, unpack_peak], "gather unpack peak")?;
    let attention_peak = checked_sum(
        &[
            packed_kv,
            mask_dtype,
            padded_k,
            padded_v,
            repeated_k,
            repeated_v,
            score_and_output_peak
                .max(context_matmul_peak)
                .max(retained_output_peak),
        ],
        "gather attention peak",
    )?;
    Ok(mask_peak.max(unpack_peak).max(attention_peak))
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherPrefillWorkspaceRequest<'a> {
    pub query_lens: &'a [usize],
    pub kv_lens: &'a [usize],
    pub q_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub dtype: DType,
    pub plan_input: PrefixPrefillPlanInput,
}

pub(crate) fn gather_prefill_workspace_for_lengths(
    input: GatherPrefillWorkspaceRequest<'_>,
) -> Result<usize> {
    if input.query_lens.is_empty()
        || input.query_lens.len() != input.kv_lens.len()
        || input.query_lens.contains(&0)
        || input.kv_lens.contains(&0)
    {
        candle_core::bail!("invalid gather prefill sequence lengths");
    }
    let total_q = input
        .query_lens
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| candle_core::Error::msg("gather query length sum overflow"))?;
    let total_kv = input
        .kv_lens
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| candle_core::Error::msg("gather context length sum overflow"))?;
    gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
        batch: input.query_lens.len(),
        total_q,
        max_q: input.query_lens.iter().copied().max().unwrap_or_default(),
        total_kv,
        max_kv: input.kv_lens.iter().copied().max().unwrap_or_default(),
        q_heads: input.q_heads,
        kv_heads: input.kv_heads,
        k_head_dim: input.k_head_dim,
        v_head_dim: input.v_head_dim,
        dtype: input.dtype,
        packed_varlen: gather_prefill_uses_packed_varlen(input.plan_input),
    })
}

fn gather_prefill_uses_packed_varlen(input: PrefixPrefillPlanInput) -> bool {
    input.device_is_cuda
        && crate::using_flash_attn()
        && matches!(input.dtype, DType::F16 | DType::BF16)
        && !input.has_alibi
        && !input.has_sinks
        && !input.has_custom_mask
        && input.causality_known
        && input.is_causal
        && !input.has_noncausal_mm_context
        && !input.has_sliding_window
        && input.query_layout_is_dense
        && input.query_len > 1
        && input.kv_heads > 0
        && input.q_heads.is_multiple_of(input.kv_heads)
        && crate::attention::flash_backend_supports_sdpa(
            input.head_size,
            input.has_softcap,
            input.has_sliding_window,
        )
}

fn checked_product(parts: &[usize], name: &str) -> Result<usize> {
    parts.iter().try_fold(1usize, |value, part| {
        value
            .checked_mul(*part)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} size overflow")))
    })
}

fn checked_tensor_bytes(dimensions: &[usize], element_size: usize, name: &str) -> Result<usize> {
    checked_product(dimensions, name)?
        .checked_mul(element_size)
        .ok_or_else(|| candle_core::Error::msg(format!("{name} workspace overflow")))
}

fn checked_sum(values: &[usize], name: &str) -> Result<usize> {
    values.iter().try_fold(0usize, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| candle_core::Error::msg(format!("{name} workspace overflow")))
    })
}

pub(crate) fn model_has_donor_paged_cache_layers(
    model: &(dyn ModelConfigLike + Send + Sync),
) -> bool {
    (0..model.num_layers()).any(|layer_idx| {
        model.layer_has_paged_kv_cache(layer_idx) && !model.uses_own_kv_cache_for_layer(layer_idx)
    })
}

pub(crate) fn prompt_prefill_workspace(
    model: Option<&(dyn ModelConfigLike + Send + Sync)>,
    input: PromptPrefillWorkspaceInput<'_>,
) -> Result<PromptPrefillWorkspace> {
    if !input.requires_prefix_attention {
        return Ok(PromptPrefillWorkspace::default());
    }
    let Some(model) = model else {
        candle_core::bail!("prompt prefix attention requires model metadata");
    };
    if input.query_lens.is_empty()
        || input.query_lens.len() != input.full_context_lens.len()
        || input.query_lens.contains(&0)
    {
        candle_core::bail!("invalid prompt prefill workspace dimensions");
    }
    let query_len = input.query_lens[0];
    let query_layout_is_dense = input.query_lens.iter().all(|&len| len == query_len);
    let total_q_tokens = input
        .query_lens
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| candle_core::Error::msg("prompt query length sum overflow"))?;
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    let mut fa3_pool = crate::flashinfer::Fa3PrefillPoolBytes::default();
    let mut max_layer_transient = 0usize;
    let mut gather_workspace_bytes = 0usize;
    for layer_idx in
        (0..model.num_layers()).filter(|&layer_idx| model.layer_has_paged_kv_cache(layer_idx))
    {
        let fa3_num_sm = input.fa3_num_sm_by_layer.get(layer_idx).copied().flatten();
        let features = model.prefix_prefill_attention_features(layer_idx);
        let plan_input = features.map(|features| {
            prompt_plan_input(
                model,
                layer_idx,
                features,
                input,
                query_len,
                query_layout_is_dense,
                fa3_num_sm,
            )
        });
        let plan = plan_input.map_or(PrefixPrefillPlan::GatherSdpa, PrefixPrefillPlan::choose);
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if matches!(plan, PrefixPrefillPlan::Fa3Fp8Paged) {
            let fa3_workspace = crate::flashinfer::fa3_prefill_workspace_components(
                input.query_lens.len(),
                query_len,
                model.num_attn_heads_for_layer(layer_idx),
                model.num_kv_heads_for_layer(layer_idx),
                model.k_head_dim_for_layer(layer_idx),
                input.max_pages_per_sequence,
                fa3_num_sm.ok_or_else(|| {
                    candle_core::Error::msg("FA3 prefill workspace is missing the SM count")
                })?,
            )?;
            fa3_pool = fa3_pool.component_max(fa3_workspace.pool());
            max_layer_transient = max_layer_transient.max(fa3_workspace.transient_bytes());
            continue;
        }
        if !matches!(plan, PrefixPrefillPlan::GatherSdpa) {
            let output = checked_tensor_bytes(
                &[
                    total_q_tokens,
                    model.num_attn_heads_for_layer(layer_idx),
                    model.v_head_dim_for_layer(layer_idx),
                ],
                input.activation_dtype.size_in_bytes(),
                "paged FlashAttention output",
            )?;
            let output_peak = output.checked_mul(2).ok_or_else(|| {
                candle_core::Error::msg("paged FlashAttention output workspace overflow")
            })?;
            max_layer_transient = max_layer_transient.max(output_peak);
            continue;
        }
        let layer_workspace =
            gather_prefill_workspace_for_lengths(GatherPrefillWorkspaceRequest {
                query_lens: input.query_lens,
                kv_lens: input.full_context_lens,
                q_heads: model.num_attn_heads_for_layer(layer_idx),
                kv_heads: model.num_kv_heads_for_layer(layer_idx),
                k_head_dim: model.k_head_dim_for_layer(layer_idx),
                v_head_dim: model.v_head_dim_for_layer(layer_idx),
                dtype: input.activation_dtype,
                plan_input: plan_input.unwrap_or(PrefixPrefillPlanInput {
                    device_is_cuda: false,
                    dtype: input.activation_dtype,
                    cache_dtype: input.cache_dtype,
                    has_alibi: false,
                    has_sinks: false,
                    has_custom_mask: true,
                    causality_known: false,
                    head_size: model.k_head_dim_for_layer(layer_idx),
                    has_softcap: false,
                    has_sliding_window: input.has_sliding_window,
                    query_layout_is_dense,
                    query_len,
                    q_heads: model.num_attn_heads_for_layer(layer_idx),
                    kv_heads: model.num_kv_heads_for_layer(layer_idx),
                    writes_cache: model.uses_own_kv_cache_for_layer(layer_idx),
                    is_causal: input.is_causal,
                    has_noncausal_mm_context: input.has_noncausal_mm_context,
                    fa3_supported: false,
                    block_size: input.block_size,
                    attention_backend: model.attention_backend_kind_for_layer(layer_idx),
                }),
            })?;
        gather_workspace_bytes = gather_workspace_bytes.max(layer_workspace);
        max_layer_transient = max_layer_transient.max(layer_workspace);
    }
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    let fa3_pool_bytes = fa3_pool.bytes()?;
    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
    let fa3_pool_bytes = 0usize;
    let bytes = fa3_pool_bytes
        .checked_add(max_layer_transient)
        .ok_or_else(|| candle_core::Error::msg("prompt prefill workspace size overflow"))?;
    Ok(PromptPrefillWorkspace {
        bytes,
        gather_workspace_bytes,
    })
}

fn prompt_plan_input(
    model: &(dyn ModelConfigLike + Send + Sync),
    layer_idx: usize,
    features: PrefixPrefillAttentionFeatures,
    input: PromptPrefillWorkspaceInput<'_>,
    query_len: usize,
    query_layout_is_dense: bool,
    fa3_num_sm: Option<usize>,
) -> PrefixPrefillPlanInput {
    PrefixPrefillPlanInput {
        device_is_cuda: input.device_is_cuda || fa3_num_sm.is_some(),
        dtype: input.activation_dtype,
        cache_dtype: input.cache_dtype,
        has_alibi: features.has_alibi,
        has_sinks: features.has_sinks,
        has_custom_mask: input.has_custom_mask,
        causality_known: input.causality_known,
        head_size: model.k_head_dim_for_layer(layer_idx),
        has_softcap: features.has_softcap,
        has_sliding_window: input.has_sliding_window || features.has_sliding_window,
        query_layout_is_dense,
        query_len,
        q_heads: model.num_attn_heads_for_layer(layer_idx),
        kv_heads: model.num_kv_heads_for_layer(layer_idx),
        writes_cache: model.uses_own_kv_cache_for_layer(layer_idx),
        is_causal: input.is_causal,
        has_noncausal_mm_context: input.has_noncausal_mm_context,
        fa3_supported: fa3_num_sm.is_some()
            && model.k_head_dim_for_layer(layer_idx) == model.v_head_dim_for_layer(layer_idx),
        block_size: input.block_size,
        attention_backend: model.attention_backend_kind_for_layer(layer_idx),
    }
}

#[cfg(all(feature = "cuda", target_family = "unix"))]
fn fa3_group_size_is_supported(q_heads: usize, kv_heads: usize) -> bool {
    kv_heads > 0
        && q_heads.is_multiple_of(kv_heads)
        && matches!(q_heads / kv_heads, 1 | 2 | 3 | 4 | 6 | 8 | 16)
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn paged_flash_attention_supports(
    head_size: usize,
    block_size: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    flash_backend_supports_sdpa(head_size, has_softcap, has_sliding_window)
        && block_size.is_multiple_of(32)
}

#[allow(dead_code)]
pub(crate) struct DecodePlanInput {
    pub attention_backend: AttentionBackendKind,
    pub head_size: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_sliding_window: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodePlan {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    FlashInfer(FlashInferDecodePlan),
    GatherSdpa,
    PagedAttention,
}

impl DecodePlan {
    pub(crate) fn requires_host_context_lengths(
        attention_backend: AttentionBackendKind,
        head_size: usize,
    ) -> bool {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            head_size > FlashInferDecodePlan::head_size_limit(attention_backend)
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        {
            let _ = head_size;
            matches!(attention_backend, AttentionBackendKind::FlashInfer)
        }
    }

    pub fn choose(input: DecodePlanInput) -> Result<Self> {
        if Self::requires_host_context_lengths(input.attention_backend, input.head_size) {
            return Ok(Self::GatherSdpa);
        }
        match input.attention_backend {
            #[cfg(all(feature = "cuda", target_family = "unix"))]
            AttentionBackendKind::FlashInfer => {
                flashinfer::decode_plan(FlashInferDecodePlanInput {
                    head_size: input.head_size,
                    has_alibi: input.has_alibi,
                    has_sinks: input.has_sinks,
                })
                .map(Self::FlashInfer)
            }
            #[cfg(not(all(feature = "cuda", target_family = "unix")))]
            AttentionBackendKind::FlashInfer => Ok(Self::GatherSdpa),
            AttentionBackendKind::Standard if input.has_sliding_window => Ok(Self::GatherSdpa),
            AttentionBackendKind::Standard => Ok(Self::PagedAttention),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paged_attention::{
        HybridPagedKvCacheConfig, KvCacheLayout, KvCacheTopology, ModelConfigMetadata,
    };

    struct DonorWorkspaceModel;

    impl ModelConfigLike for DonorWorkspaceModel {
        fn max_seq_len(&self) -> usize {
            131_072
        }

        fn num_layers(&self) -> usize {
            2
        }

        fn hidden_size(&self) -> usize {
            4096
        }

        fn num_kv_heads(&self) -> usize {
            4
        }

        fn num_attn_heads(&self) -> usize {
            16
        }

        fn k_head_dim(&self) -> usize {
            256
        }

        fn v_head_dim(&self) -> usize {
            256
        }

        fn has_kv_cache_sharing(&self) -> bool {
            true
        }

        fn kv_cache_topology(&self) -> KvCacheTopology {
            KvCacheTopology::from_layer_owners(vec![0, 0])
        }

        fn prefix_prefill_attention_features(
            &self,
            _layer_idx: usize,
        ) -> Option<PrefixPrefillAttentionFeatures> {
            Some(PrefixPrefillAttentionFeatures::default())
        }
    }

    fn workspace_model(
        features: Option<PrefixPrefillAttentionFeatures>,
    ) -> HybridPagedKvCacheConfig {
        let model = HybridPagedKvCacheConfig::new(
            ModelConfigMetadata {
                max_seq_len: 131_072,
                num_layers: 4,
                hidden_size: 4096,
                num_kv_heads: 4,
                num_attn_heads: 16,
                sliding_window: None,
                k_head_dim: 256,
                v_head_dim: 256,
                kv_cache_layout: KvCacheLayout::FlashInferHnd,
            },
            vec![true, false, true, false],
        );
        match features {
            Some(features) => model.with_uniform_prefix_prefill_attention_features(features),
            None => model,
        }
    }

    fn workspace_input<'a>(
        query_lens: &'a [usize],
        full_context_lens: &'a [usize],
    ) -> PromptPrefillWorkspaceInput<'a> {
        PromptPrefillWorkspaceInput {
            activation_dtype: DType::BF16,
            cache_dtype: DType::F8E4M3,
            device_is_cuda: true,
            block_size: 32,
            query_lens,
            full_context_lens,
            max_pages_per_sequence: full_context_lens
                .iter()
                .copied()
                .max()
                .unwrap_or_default()
                .div_ceil(32),
            requires_prefix_attention: true,
            is_causal: true,
            causality_known: true,
            has_custom_mask: false,
            has_noncausal_mm_context: false,
            has_sliding_window: false,
            fa3_num_sm_by_layer: &[Some(132), None, Some(132), None],
        }
    }

    fn prefix_plan(
        head_size: usize,
        has_softcap: bool,
        has_sliding_window: bool,
    ) -> PrefixPrefillPlan {
        PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::F16,
            cache_dtype: DType::F16,
            has_alibi: false,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size,
            has_softcap,
            has_sliding_window,
            query_layout_is_dense: true,
            query_len: 64,
            q_heads: 24,
            kv_heads: 4,
            writes_cache: true,
            is_causal: true,
            has_noncausal_mm_context: false,
            fa3_supported: true,
            block_size: 32,
            attention_backend: AttentionBackendKind::FlashInfer,
        })
    }

    #[test]
    fn paged_prefix_rejects_disabled_large_head_features() {
        assert!(matches!(
            prefix_plan(320, true, false),
            PrefixPrefillPlan::GatherSdpa
        ));
        assert!(matches!(
            prefix_plan(320, false, true),
            PrefixPrefillPlan::GatherSdpa
        ));
    }

    #[test]
    fn paged_prefix_gathers_mixed_dtype_cache() {
        let plan = PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::BF16,
            cache_dtype: DType::F8E4M3,
            has_alibi: false,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size: 128,
            has_softcap: false,
            has_sliding_window: false,
            query_layout_is_dense: true,
            query_len: 64,
            q_heads: 24,
            kv_heads: 4,
            writes_cache: true,
            is_causal: true,
            has_noncausal_mm_context: false,
            fa3_supported: false,
            block_size: 32,
            attention_backend: AttentionBackendKind::FlashInfer,
        });
        assert!(matches!(plan, PrefixPrefillPlan::GatherSdpa));
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn fp8_dense_short_prefix_uses_fa3_when_available() {
        let input = PrefixPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::BF16,
            cache_dtype: DType::F8E4M3,
            has_alibi: false,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size: 256,
            has_softcap: false,
            has_sliding_window: false,
            query_layout_is_dense: true,
            query_len: 128,
            q_heads: 24,
            kv_heads: 4,
            writes_cache: true,
            is_causal: true,
            has_noncausal_mm_context: false,
            fa3_supported: true,
            block_size: 32,
            attention_backend: AttentionBackendKind::FlashInfer,
        };
        assert!(fa3_paged_prefill_supported(input));
        assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
            query_len: 129,
            ..input
        }));
        assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
            query_layout_is_dense: false,
            ..input
        }));
        assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
            writes_cache: false,
            ..input
        }));
        assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
            has_noncausal_mm_context: true,
            ..input
        }));
        assert!(!fa3_paged_prefill_supported(PrefixPrefillPlanInput {
            q_heads: 20,
            ..input
        }));
        let plan = PrefixPrefillPlan::choose(input);
        if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
            assert!(matches!(plan, PrefixPrefillPlan::Fa3Fp8Paged));
        } else {
            assert!(matches!(plan, PrefixPrefillPlan::GatherSdpa));
        }
    }

    #[test]
    fn prefix_gather_workspace_is_checked_and_exact() {
        let packed = GatherPrefillWorkspaceInput {
            batch: 16,
            total_q: 2_048,
            max_q: 128,
            total_kv: 1_600_000,
            max_kv: 100_000,
            q_heads: 16,
            kv_heads: 4,
            k_head_dim: 256,
            v_head_dim: 256,
            dtype: DType::BF16,
            packed_varlen: true,
        };
        assert_eq!(
            gather_prefill_workspace_bytes(packed).unwrap(),
            6_570_377_216
        );

        let mixed_fallback = GatherPrefillWorkspaceInput {
            total_kv: 115_000,
            packed_varlen: false,
            ..packed
        };
        assert_eq!(
            gather_prefill_workspace_bytes(mixed_fallback).unwrap(),
            66_494_857_216
        );
        assert!(gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
            total_kv: usize::MAX,
            ..packed
        })
        .is_err());
    }

    #[test]
    fn mixed_short_query_gather_covers_value_transpose_peak() {
        let expected = [
            (1, 46_467_811_072usize),
            (8, 47_318_808_576),
            (32, 50_236_514_304),
            (64, 54_126_788_608),
            (128, 66_494_857_216),
        ];
        for (query_len, expected_bytes) in expected {
            let input = GatherPrefillWorkspaceInput {
                batch: 16,
                total_q: 16 * query_len,
                max_q: query_len,
                total_kv: 115_000,
                max_kv: 100_000,
                q_heads: 16,
                kv_heads: 4,
                k_head_dim: 256,
                v_head_dim: 256,
                dtype: DType::BF16,
                packed_varlen: false,
            };
            assert_eq!(
                gather_prefill_workspace_bytes(input).unwrap(),
                expected_bytes
            );
        }

        assert_eq!(
            gather_prefill_workspace_bytes(GatherPrefillWorkspaceInput {
                batch: 16,
                total_q: 16,
                max_q: 1,
                total_kv: 115_000,
                max_kv: 100_000,
                q_heads: 16,
                kv_heads: 16,
                k_head_dim: 256,
                v_head_dim: 256,
                dtype: DType::BF16,
                packed_varlen: false,
            })
            .unwrap(),
            41_327_331_072
        );
    }

    #[test]
    fn prompt_workspace_is_zero_before_prefix_attention() {
        let model = workspace_model(None);
        let query_lens = [128, 128];
        let context_lens = [128, 128];
        let mut input = workspace_input(&query_lens, &context_lens);
        input.requires_prefix_attention = false;
        assert_eq!(
            prompt_prefill_workspace(Some(&model), input).unwrap(),
            PromptPrefillWorkspace::default()
        );
        assert_eq!(
            prompt_prefill_workspace(None, input).unwrap(),
            PromptPrefillWorkspace::default()
        );
    }

    #[test]
    fn donor_cache_first_prompt_requires_gather_workspace() {
        let model = DonorWorkspaceModel;
        assert!(model_has_donor_paged_cache_layers(&model));
        let query_lens = [128];
        let context_lens = [128];
        let workspace =
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap();
        assert!(workspace.gather_workspace_bytes > 0);
        assert!(workspace.bytes >= workspace.gather_workspace_bytes);
    }

    #[test]
    fn prompt_workspace_rejects_prefix_without_model_metadata() {
        let query_lens = [128, 128];
        let context_lens = [1_000, 8_000];
        assert!(
            prompt_prefill_workspace(None, workspace_input(&query_lens, &context_lens)).is_err()
        );
    }

    #[test]
    fn prompt_workspace_is_conservative_for_unknown_attention_features() {
        let model = workspace_model(None);
        let query_lens = [128, 128];
        let context_lens = [1_000, 8_000];
        assert_eq!(
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap()
                .bytes,
            739_889_152
        );

        let model = workspace_model(Some(PrefixPrefillAttentionFeatures {
            has_sliding_window: true,
            ..Default::default()
        }));
        assert_eq!(
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap()
                .bytes,
            739_889_152
        );
    }

    #[test]
    fn prompt_workspace_falls_back_for_ineligible_known_attention() {
        let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
        let query_lens = [129, 129];
        let context_lens = [1_000, 8_000];
        assert_eq!(
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap()
                .bytes,
            38_977_536
        );

        let model = workspace_model(Some(PrefixPrefillAttentionFeatures {
            has_sinks: true,
            ..Default::default()
        }));
        let query_lens = [128, 128];
        assert_eq!(
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap()
                .bytes,
            739_889_152
        );
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn prompt_workspace_matches_direct_fa3_availability() {
        let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
        let query_lens = [128, 128];
        let context_lens = [1_000, 8_000];
        let workspace =
            prompt_prefill_workspace(Some(&model), workspace_input(&query_lens, &context_lens))
                .unwrap();
        assert_eq!(
            workspace.bytes,
            if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
                crate::flashinfer::fa3_prefill_workspace_bytes(2, 128, 16, 4, 256, 250, 132)
                    .unwrap()
            } else {
                77_922_304
            }
        );
        assert_eq!(workspace.gather_workspace_bytes, 0);
    }

    #[cfg(all(feature = "cuda", target_family = "unix"))]
    #[test]
    fn prompt_workspace_tracks_fa3_support_per_layer() {
        let model = workspace_model(Some(PrefixPrefillAttentionFeatures::default()));
        let query_lens = [128];
        let context_lens = [256];
        let mut input = workspace_input(&query_lens, &context_lens);
        input.device_is_cuda = false;
        input.fa3_num_sm_by_layer = &[Some(132), None, None, None];
        input.max_pages_per_sequence = 3_125;
        let workspace = prompt_prefill_workspace(Some(&model), input).unwrap();
        let expected_bytes = if mistralrs_paged_attn::USE_FA3_FP8_PAGED {
            let fa3 =
                crate::flashinfer::fa3_prefill_workspace_components(1, 128, 16, 4, 256, 3_125, 132)
                    .unwrap();
            fa3.pool().bytes().unwrap() + fa3.transient_bytes().max(12_648_448)
        } else {
            12_648_448
        };
        let gather_bytes = 12_648_448;
        assert_eq!(workspace.bytes, expected_bytes);
        assert_eq!(workspace.gather_workspace_bytes, gather_bytes);
    }

    #[test]
    fn standard_sliding_decode_uses_exact_gather_path() {
        let plan = DecodePlan::choose(DecodePlanInput {
            attention_backend: AttentionBackendKind::Standard,
            head_size: 128,
            has_alibi: false,
            has_sinks: false,
            has_sliding_window: true,
        })
        .unwrap();

        assert!(matches!(plan, DecodePlan::GatherSdpa));
    }

    #[test]
    fn standard_full_decode_keeps_paged_kernel() {
        let plan = DecodePlan::choose(DecodePlanInput {
            attention_backend: AttentionBackendKind::Standard,
            head_size: 128,
            has_alibi: false,
            has_sinks: false,
            has_sliding_window: false,
        })
        .unwrap();

        assert!(matches!(plan, DecodePlan::PagedAttention));
    }
}
