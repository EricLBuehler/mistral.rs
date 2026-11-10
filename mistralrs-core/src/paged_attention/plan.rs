use candle_core::{DType, Result};

use super::{
    attention_backend::AttentionBackendKind, config::PrefixPrefillAttentionFeatures,
    ModelConfigLike,
};
#[cfg(cuda_unix)]
use crate::flashinfer::{self, FlashInferDecodePlan, FlashInferDecodePlanInput};

#[cfg(cuda_unix)]
// Matches the instantiated head sizes in pagedattention.cuh.
const CUDA_PAGED_ATTENTION_HEAD_DIMS: [usize; 8] = [64, 80, 96, 112, 128, 192, 256, 512];
const METAL_SDPA_HEAD_DIMS: [usize; 7] = [32, 64, 72, 80, 96, 128, 256];
const METAL_D512_HEAD_DIM: usize = 512;
const METAL_SDPA_SMALL_QUERY_LIMIT: usize = 8;
const METAL_GATHER_UNPACK_BUFFERS: usize = 3;
const METAL_GATHER_OUTPUT_BUFFERS: usize = 2;

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
    #[cfg(cuda_unix)]
    Fa3Fp8Paged,
    #[cfg(cuda_flash_attn)]
    FlashAttentionPaged,
    GatherSdpa,
}

impl PrefixPrefillPlan {
    pub fn choose(input: PrefixPrefillPlanInput) -> Self {
        #[cfg(not(cuda_flash_attn))]
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

        #[cfg(cuda_unix)]
        if mistralrs_paged_attn::USE_FA3_FP8_PAGED && fa3_paged_prefill_supported(input) {
            return Self::Fa3Fp8Paged;
        }

        #[cfg(cuda_flash_attn)]
        if Self::flash_paged_blocker(input).is_none() {
            return Self::FlashAttentionPaged;
        }

        Self::GatherSdpa
    }

    /// Why the paged flash kernel cannot take this prefill, if anything stops it.
    pub(crate) fn flash_paged_blocker(input: PrefixPrefillPlanInput) -> Option<&'static str> {
        if !cfg!(cuda_flash_attn) {
            return Some("built without flash-attn");
        }
        if !input.device_is_cuda {
            return Some("not on CUDA");
        }
        if !crate::attention::flash_dtype_supported(input.dtype) {
            return Some("activations are not F16/BF16");
        }
        if input.cache_dtype != input.dtype {
            return Some("KV cache dtype differs from the activations");
        }
        if input.has_alibi {
            return Some("ALiBi bias");
        }
        if input.has_sinks {
            return Some("attention sinks");
        }
        if input.has_custom_mask {
            return Some("explicit attention mask");
        }
        if !input.causality_known {
            return Some("causality is not known ahead of the forward");
        }
        if !input.query_layout_is_dense {
            return Some("sparse query layout");
        }
        if !paged_flash_attention_supports(
            input.head_size,
            input.block_size,
            input.has_softcap,
            input.has_sliding_window,
        ) {
            return Some("head dim, softcap, sliding window or block size outside flash-attn");
        }
        if !matches!(input.attention_backend, AttentionBackendKind::FlashInfer) {
            return Some("KV cache is not in the FlashInfer layout");
        }
        None
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            #[cfg(cuda_unix)]
            Self::Fa3Fp8Paged => "FA3 FP8 paged",
            #[cfg(cuda_flash_attn)]
            Self::FlashAttentionPaged => "flash-attn paged",
            Self::GatherSdpa => "gather + SDPA",
        }
    }
}

#[cfg(cuda_unix)]
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
#[allow(dead_code)]
pub(crate) struct PromptPrefillWorkspaceInput<'a> {
    pub activation_dtype: DType,
    pub cache_dtype: DType,
    pub device_is_cuda: bool,
    pub device_is_metal: bool,
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
    pub hnd_by_layer: &'a [bool],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct PromptPrefillWorkspace {
    pub bytes: usize,
    pub gather_workspace_bytes: usize,
    pub attention_is_complete: bool,
}

#[cfg(feature = "metal")]
pub(crate) struct MetalPagedPrefillWorkspaceInput<'a> {
    pub query_lens: &'a [usize],
    pub context_lens: &'a [usize],
    pub query_heads: usize,
    pub kv_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub block_size: usize,
    pub query_dtype: DType,
    pub cache_dtype: DType,
    pub hnd: bool,
    pub has_custom_mask: bool,
    pub has_multimodal_ranges: bool,
}

#[cfg(feature = "metal")]
pub(crate) fn metal_paged_prefill_workspace(
    input: MetalPagedPrefillWorkspaceInput<'_>,
) -> Result<Option<usize>> {
    if input.query_lens.is_empty()
        || input.query_lens.len() != input.context_lens.len()
        || input
            .query_lens
            .iter()
            .zip(input.context_lens)
            .any(|(&query, &context)| query == 0 || query > context)
    {
        candle_core::bail!("invalid Metal paged prefill lengths");
    }
    let total_query_tokens = checked_sum(input.query_lens, "Metal prefill query lengths")?;
    let max_query_len = *input.query_lens.iter().max().unwrap();
    let max_context_len = *input.context_lens.iter().max().unwrap();
    let shape = mistralrs_paged_attn::MetalPagedPrefillShape {
        num_seqs: input.query_lens.len(),
        num_heads: input.query_heads,
        num_kv_heads: input.kv_heads,
        max_query_len,
        total_query_tokens,
        max_context_len,
        key_head_dim: input.key_head_dim,
        value_head_dim: input.value_head_dim,
        block_size: input.block_size,
        num_multimodal_ranges: if input.has_multimodal_ranges {
            checked_sum(input.context_lens, "Metal prefill multimodal ranges")?
        } else {
            0
        },
        query_dtype: input.query_dtype,
        cache_dtype: input.cache_dtype,
        packed_queries: false,
        hnd: input.hnd,
    };
    if !shape.supports() {
        return Ok(None);
    }
    let external_mask = if input.has_custom_mask {
        metal_gather_workspace_bytes(checked_tensor_bytes(
            &[
                shape.num_seqs,
                shape.num_heads,
                max_query_len,
                max_context_len,
            ],
            DType::F32.size_in_bytes() + input.query_dtype.size_in_bytes(),
            "Metal prefill external mask",
        )?)?
    } else {
        0
    };
    Ok(Some(checked_sum(
        &[
            shape.workspace_bytes()?,
            shape.query_copy_workspace_bytes()?,
            shape.output_copy_workspace_bytes()?,
            external_mask,
        ],
        "Metal paged prefill workspace",
    )?))
}

#[cfg(feature = "metal")]
const METAL_BACKING_ROUNDING_BOUND: usize = 2;

#[cfg(feature = "metal")]
pub(crate) fn metal_gather_workspace_bytes(bytes: usize) -> Result<usize> {
    bytes
        .checked_mul(METAL_BACKING_ROUNDING_BOUND)
        .ok_or_else(|| candle_core::Error::msg("Metal gather backing size overflow"))
}

#[cfg(feature = "metal")]
pub(crate) fn metal_workspace_capacity(
    device: &candle_core::Device,
    required: usize,
    allocation_limit: Option<usize>,
) -> Result<(usize, usize)> {
    let capacity = || -> Result<_> {
        let snapshot = crate::MemoryUsage
            .query(device)
            .map_err(candle_core::Error::wrap)?;
        let available = if let Some(limit) = allocation_limit {
            snapshot
                .available()
                .min(limit.saturating_sub(snapshot.allocated()))
        } else {
            super::device_memory_cap(snapshot.available(), device)
        };
        Ok((available, snapshot.total()))
    };
    let available = capacity()?;
    if required <= available.0 {
        return Ok(available);
    }
    device.synchronize()?;
    capacity()
}

#[cfg(feature = "metal")]
pub(crate) fn check_metal_attention_workspace(
    device: &candle_core::Device,
    required: usize,
    budget: Option<&crate::pipeline::text_models_inputs_processor::AttentionMemoryBudget>,
) -> Result<()> {
    let limit = budget.and_then(|budget| budget.metal_allocation_limit(device.location()));
    if let Some(budget) = budget {
        if required > budget.workspace_bytes {
            candle_core::bail!("Metal attention requires {required} bytes, exceeding its admitted workspace of {} bytes", budget.workspace_bytes);
        }
        if let Some(limit) = limit {
            if required
                <= limit.saturating_sub(device.as_metal_device()?.device().current_allocated_size())
            {
                return Ok(());
            }
        }
    }
    let (available, _) = metal_workspace_capacity(device, required, limit)?;
    if required > available {
        candle_core::bail!("Metal attention requires {required} bytes, but only {available} bytes remain within its memory budget");
    }
    Ok(())
}

#[cfg(feature = "metal")]
pub(crate) fn metal_cache_write_workspace(
    model: Option<&dyn ModelConfigLike>,
    tokens: usize,
    dtype: DType,
) -> Result<usize> {
    let Some(model) = model else {
        return Ok(0);
    };
    (0..model.num_layers())
        .filter(|&layer| model.layer_has_paged_kv_cache(layer))
        .try_fold(0usize, |peak, layer| {
            let dimensions = checked_sum(
                &[
                    model.k_head_dim_for_layer(layer),
                    model.v_head_dim_for_layer(layer),
                ],
                "KV head dimensions",
            )?;
            let bytes = checked_tensor_bytes(
                &[tokens, model.num_kv_heads_for_layer(layer), dimensions],
                dtype.size_in_bytes(),
                "KV input packing",
            )?;
            Ok(peak.max(metal_gather_workspace_bytes(bytes)?))
        })
}

#[cfg(feature = "metal")]
pub(crate) struct MetalDecodeWorkspaceInput<'a> {
    pub activation_dtype: DType,
    pub cache_dtype: DType,
    pub block_size: usize,
    pub query_lens: &'a [usize],
    pub context_lens: &'a [usize],
    pub hnd_by_layer: &'a [bool],
    pub capacity_bound: bool,
    pub has_custom_mask: bool,
    pub has_multimodal_ranges: bool,
}

#[cfg(feature = "metal")]
pub(crate) fn metal_decode_workspace(
    model: Option<&dyn ModelConfigLike>,
    input: MetalDecodeWorkspaceInput<'_>,
) -> Result<usize> {
    let model =
        model.ok_or_else(|| candle_core::Error::msg("Metal decode requires model metadata"))?;
    if input.query_lens.len() != input.context_lens.len() || input.query_lens.is_empty() {
        candle_core::bail!("invalid Metal decode workspace dimensions");
    }
    let mut row_contexts = Vec::new();
    for (&query, &context) in input.query_lens.iter().zip(input.context_lens) {
        if query == 0 || query > context {
            candle_core::bail!("invalid Metal decode context length");
        }
        row_contexts.extend((context - query + 1)..=context);
    }
    let query_lens = vec![1; row_contexts.len()];
    let max_context_len = row_contexts.iter().copied().max().unwrap_or_default();
    let mut workspace = 0;
    for layer in (0..model.num_layers()).filter(|&layer| model.layer_has_paged_kv_cache(layer)) {
        let features = model.prefix_prefill_attention_features(layer);
        let hnd = input.hnd_by_layer.get(layer).copied().unwrap_or(false);
        let q_heads = model.num_attn_heads_for_layer(layer);
        let kv_heads = model.num_kv_heads_for_layer(layer);
        let k_head_dim = model.k_head_dim_for_layer(layer);
        let v_head_dim = model.v_head_dim_for_layer(layer);
        let gather = input.has_custom_mask
            || (!hnd && features.is_none_or(|features| features.has_sliding_window))
            || k_head_dim != v_head_dim;
        let native_prefill = if !input.has_custom_mask
            && !matches!(
                model.kv_cache_layout_for_layer(layer),
                super::KvCacheLayout::Mla { .. }
            ) {
            metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
                query_lens: input.query_lens,
                context_lens: input.context_lens,
                query_heads: q_heads,
                kv_heads,
                key_head_dim: k_head_dim,
                value_head_dim: v_head_dim,
                block_size: input.block_size,
                query_dtype: input.activation_dtype,
                cache_dtype: input.cache_dtype,
                hnd,
                has_custom_mask: false,
                has_multimodal_ranges: input.has_multimodal_ranges,
            })?
        } else {
            None
        };
        let attention = if let Some(bytes) =
            native_prefill.filter(|_| input.query_lens.iter().any(|&query| query > 1))
        {
            bytes
        } else if gather {
            metal_gather_workspace_bytes(gather_prefill_workspace_bytes(
                GatherPrefillWorkspaceInput {
                    batch: row_contexts.len(),
                    total_q: row_contexts.len(),
                    max_q: 1,
                    total_kv: checked_sum(&row_contexts, "decode context lengths")?,
                    max_kv: max_context_len,
                    q_heads,
                    kv_heads,
                    k_head_dim,
                    v_head_dim,
                    dtype: input.activation_dtype,
                    backend: GatherAttentionBackend::Conservative,
                },
            )?)?
        } else {
            let known_features = features;
            let features = features.unwrap_or_default();
            let workload = mistralrs_paged_attn::MetalPagedAttentionWorkspace {
                num_seqs: query_lens.len(),
                num_heads: q_heads,
                num_kv_heads: kv_heads,
                head_size: k_head_dim,
                block_size: input.block_size,
                max_context_len,
                sliding_window: None,
                dtype: input.activation_dtype,
                cache_dtype: input.cache_dtype,
                hnd,
                packed_x: if hnd {
                    1
                } else {
                    mistralrs_paged_attn::METAL_PAGED_CACHE_VECTOR_BYTES
                        / input.cache_dtype.size_in_bytes()
                },
                has_alibi: features.has_alibi,
                has_sinks: features.has_sinks,
                has_scales: input.cache_dtype == DType::F8E4M3,
                softcapping: if features.has_softcap { 0.0 } else { 1.0 },
            };
            let workspace_bytes = |workload: mistralrs_paged_attn::MetalPagedAttentionWorkspace| {
                if input.capacity_bound {
                    workload.capacity_bytes()
                } else {
                    workload.bytes()
                }
            };
            let bytes = workspace_bytes(workload)?;
            if known_features.is_none() {
                bytes.max(workspace_bytes(
                    mistralrs_paged_attn::MetalPagedAttentionWorkspace {
                        has_alibi: true,
                        has_sinks: true,
                        softcapping: 0.0,
                        ..workload
                    },
                )?)
            } else {
                bytes
            }
        };
        let attention = attention.max(native_prefill.unwrap_or_default());
        let pack = checked_tensor_bytes(
            &[row_contexts.len(), kv_heads, k_head_dim + v_head_dim],
            input.activation_dtype.size_in_bytes(),
            "decode KV packing",
        )?;
        workspace = workspace.max(checked_sum(
            &[attention, metal_gather_workspace_bytes(pack)?],
            "decode workspace",
        )?);
    }
    Ok(workspace)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GatherAttentionBackend {
    Conservative,
    Sinks,
    FusedSinks,
    PackedVarlen,
    MetalFused,
    #[cfg(feature = "metal")]
    MetalD512,
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
    backend: GatherAttentionBackend,
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

    if input.backend == GatherAttentionBackend::FusedSinks {
        let padded_kv = checked_tensor_bytes(
            &[
                input.batch,
                input.kv_heads,
                input.max_kv,
                checked_sum(
                    &[input.k_head_dim, input.v_head_dim],
                    "fused sinks KV width",
                )?,
            ],
            dtype_bytes,
            "padded fused sinks KV",
        )?;
        let query = checked_tensor_bytes(
            &[input.batch, input.q_heads, input.max_q, input.k_head_dim],
            dtype_bytes,
            "fused sinks query",
        )?
        .max(checked_tensor_bytes(
            &[input.total_q, input.q_heads, input.k_head_dim],
            dtype_bytes,
            "packed fused sinks query",
        )?);
        let sinks = checked_tensor_bytes(
            &[input.q_heads],
            DType::F32.size_in_bytes(),
            "fused attention sinks",
        )?;
        let offsets = checked_tensor_bytes(
            &[2, input.batch + 1],
            DType::U32.size_in_bytes(),
            "fused sinks sequence offsets",
        )?;
        return checked_sum(
            &[
                packed_kv.max(padded_kv),
                query,
                output,
                output,
                sinks,
                offsets,
            ],
            "fused sinks gather peak",
        );
    }

    if input.backend == GatherAttentionBackend::PackedVarlen {
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
        let query = checked_tensor_bytes(
            &[input.total_q, input.q_heads, input.k_head_dim],
            dtype_bytes,
            "packed query",
        )?;
        let logsumexp = checked_tensor_bytes(
            &[input.total_q, input.q_heads],
            DType::F32.size_in_bytes(),
            "packed attention logsumexp",
        )?;
        return checked_sum(
            &[
                packed_kv, expanded_k, expanded_v, query, output, output, logsumexp,
            ],
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
    match input.backend {
        GatherAttentionBackend::MetalFused => {
            return metal_fused_gather_workspace(input, packed_kv, padded_k, padded_v, output);
        }
        #[cfg(feature = "metal")]
        GatherAttentionBackend::MetalD512 => {
            return metal_fused_gather_workspace(input, packed_kv, padded_k, padded_v, output);
        }
        _ => {}
    }
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

    let repeated_k =
        if input.q_heads > input.kv_heads || input.backend == GatherAttentionBackend::Sinks {
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
    let repeated_v =
        if input.q_heads > input.kv_heads || input.backend == GatherAttentionBackend::Sinks {
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

    // The sinks fallback materializes every query row instead of chunking the score matrix.
    let query_limit = if input.backend == GatherAttentionBackend::Sinks {
        usize::MAX
    } else {
        crate::attention::ATTENTION_CHUNK_SIZE
    };
    let padded_score_elements = checked_product(
        &[
            input.batch,
            input.q_heads,
            input.max_q.min(query_limit),
            input.max_kv,
        ],
        "padded attention scores",
    )?;
    let packed_score_elements = checked_product(
        &[
            input.q_heads,
            input.total_q.min(query_limit),
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
    let contiguous_query = if input.backend == GatherAttentionBackend::Sinks {
        checked_tensor_bytes(
            &[input.batch, input.q_heads, input.max_q, input.k_head_dim],
            dtype_bytes,
            "contiguous sinks query",
        )?
        .max(checked_tensor_bytes(
            &[input.total_q, input.q_heads, input.k_head_dim],
            dtype_bytes,
            "packed contiguous sinks query",
        )?)
    } else {
        0
    };
    let attention_peak = checked_sum(
        &[
            packed_kv,
            mask_dtype,
            padded_k,
            padded_v,
            repeated_k,
            repeated_v,
            contiguous_query,
            score_and_output_peak
                .max(context_matmul_peak)
                .max(retained_output_peak),
        ],
        "gather attention peak",
    )?;
    Ok(mask_peak.max(unpack_peak).max(attention_peak))
}

fn metal_fused_gather_workspace(
    input: GatherPrefillWorkspaceInput,
    gathered_kv: usize,
    padded_k: usize,
    padded_v: usize,
    output: usize,
) -> Result<usize> {
    let masks = checked_tensor_bytes(
        &[input.batch, input.max_q, input.max_kv],
        DType::F32.size_in_bytes() + input.dtype.size_in_bytes(),
        "Metal prefix masks",
    )?;
    let q = checked_tensor_bytes(
        &[input.batch, input.q_heads, input.max_q, input.k_head_dim],
        input.dtype.size_in_bytes(),
        "Metal packed query",
    )?;
    // Ragged unpacking retains zero pads, per-row copies, and the final concatenation in the pools.
    let padded_kv = checked_sum(&[padded_k, padded_v], "Metal padded KV")?
        .checked_mul(METAL_GATHER_UNPACK_BUFFERS)
        .ok_or_else(|| candle_core::Error::msg("Metal KV unpack inventory overflow"))?;
    let output = output
        .checked_mul(METAL_GATHER_OUTPUT_BUFFERS)
        .ok_or_else(|| candle_core::Error::msg("Metal attention output inventory overflow"))?;
    let bytes = checked_sum(
        &[gathered_kv, padded_kv, masks, q, output],
        "Metal fused gather",
    )?;
    #[cfg(feature = "metal")]
    let bytes = if input.backend == GatherAttentionBackend::MetalD512 {
        use mistralrs_quant::metal_kernels::{FA_NCPSG, FA_NQPSG};

        let mask_f16 = checked_tensor_bytes(
            &[input.batch, input.max_q, input.max_kv],
            DType::F16.size_in_bytes(),
            "Metal D512 mask cast",
        )?;
        let block_scratch = checked_product(
            &[
                input.max_q.div_ceil(FA_NQPSG),
                input.max_kv.div_ceil(FA_NCPSG),
                input.batch,
            ],
            "Metal D512 block scratch",
        )?;
        let kv_tail = checked_tensor_bytes(
            &[2, input.batch, input.kv_heads, FA_NCPSG, input.k_head_dim],
            DType::BF16.size_in_bytes(),
            "Metal D512 KV tail",
        )?;
        let mask_tail = checked_tensor_bytes(
            &[input.batch, input.max_q, FA_NCPSG],
            DType::F16.size_in_bytes(),
            "Metal D512 mask tail",
        )?;
        // Retain the possible tail at aligned lengths so capacity searches have a monotonic bound.
        let pad_scratch = checked_sum(&[kv_tail, mask_tail], "Metal D512 padding")?;
        checked_sum(
            &[bytes, mask_f16, block_scratch, pad_scratch],
            "Metal D512 gather",
        )?
    } else {
        bytes
    };
    Ok(bytes)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct GatherPrefillWorkspaceRequest<'a> {
    pub device_is_metal: bool,
    pub packed_queries: bool,
    pub metal_mask_is_compatible: bool,
    pub query_lens: &'a [usize],
    pub kv_lens: &'a [usize],
    pub q_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub dtype: DType,
    pub plan_input: PrefixPrefillPlanInput,
}

fn gather_attention_backend(input: GatherPrefillWorkspaceRequest<'_>) -> GatherAttentionBackend {
    if gather_prefill_uses_sinks(input.plan_input) && input.k_head_dim == input.v_head_dim {
        return GatherAttentionBackend::FusedSinks;
    }
    if input.plan_input.has_sinks {
        return GatherAttentionBackend::Sinks;
    }
    if !input.device_is_metal && gather_prefill_uses_packed_varlen(input.plan_input) {
        return GatherAttentionBackend::PackedVarlen;
    }
    let plan = input.plan_input;
    let max_q = input.query_lens.iter().copied().max().unwrap_or_default();
    let max_kv = input.kv_lens.iter().copied().max().unwrap_or_default();
    if !input.device_is_metal
        || input.packed_queries
        || !input.metal_mask_is_compatible
        || max_q <= 1
        || max_q > max_kv
        || input.k_head_dim != input.v_head_dim
        || plan.has_alibi
        || plan.has_sinks
        || plan.has_softcap
        || !matches!(input.dtype, DType::BF16 | DType::F16 | DType::F32)
    {
        return GatherAttentionBackend::Conservative;
    }
    if METAL_SDPA_HEAD_DIMS.contains(&input.k_head_dim)
        || input.k_head_dim == METAL_D512_HEAD_DIM
            && max_q <= METAL_SDPA_SMALL_QUERY_LIMIT
            && input.dtype != DType::F32
    {
        return GatherAttentionBackend::MetalFused;
    }
    #[cfg(feature = "metal")]
    if input.k_head_dim == METAL_D512_HEAD_DIM
        && input.dtype == DType::BF16
        && max_q > METAL_SDPA_SMALL_QUERY_LIMIT
        && (plan.has_custom_mask || plan.is_causal || plan.has_sliding_window)
    {
        return GatherAttentionBackend::MetalD512;
    }
    GatherAttentionBackend::Conservative
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
        backend: gather_attention_backend(input),
    })
}

pub(crate) fn gather_prefill_uses_packed_varlen(input: PrefixPrefillPlanInput) -> bool {
    gather_prefill_uses_packed_varlen_with_backend(
        input,
        crate::using_flash_attn(),
        crate::attention::flash_backend_supports_sdpa(
            input.head_size,
            input.has_softcap,
            input.has_sliding_window,
        ),
    )
}

pub(crate) fn gather_prefill_uses_sinks(input: PrefixPrefillPlanInput) -> bool {
    input.device_is_cuda
        && input.has_sinks
        && !input.has_alibi
        && !input.has_softcap
        && !input.has_custom_mask
        && !input.has_noncausal_mm_context
        && input.causality_known
        && (input.query_len == 1 || input.is_causal)
        && input.query_len > 0
        && input.kv_heads > 0
        && input.q_heads.is_multiple_of(input.kv_heads)
        && crate::attention::sinks_backend_supports(
            input.dtype,
            candle_core::DeviceLocation::Cuda { gpu_id: 0 },
            input.head_size,
        )
}

fn gather_prefill_uses_packed_varlen_with_backend(
    input: PrefixPrefillPlanInput,
    flash_enabled: bool,
    shape_supported: bool,
) -> bool {
    input.device_is_cuda
        && flash_enabled
        && crate::attention::flash_dtype_supported(input.dtype)
        && !input.has_alibi
        && !input.has_sinks
        && !input.has_custom_mask
        && input.causality_known
        && (input.query_len == 1 || input.is_causal)
        && !input.has_noncausal_mm_context
        && !input.has_sliding_window
        && input.query_layout_is_dense
        && input.query_len > 0
        && input.kv_heads > 0
        && input.q_heads.is_multiple_of(input.kv_heads)
        && shape_supported
}

fn cuda_gather_workspace_is_complete(plan: PrefixPrefillPlanInput, value_head_dim: usize) -> bool {
    if !plan.device_is_cuda
        || plan.has_custom_mask
        || plan.has_noncausal_mm_context
        || plan.has_alibi
        || !plan.causality_known
        || plan.head_size != value_head_dim
    {
        return false;
    }
    if plan.has_sinks {
        return !plan.has_softcap && matches!(plan.dtype, DType::F16 | DType::BF16 | DType::F32);
    }
    !plan.has_sliding_window
        && crate::using_flash_attn()
        && crate::attention::flash_backend_supports_sdpa(
            plan.head_size,
            plan.has_softcap,
            plan.has_sliding_window,
        )
        && crate::attention::flash_dtype_supported(plan.dtype)
}

/// Noncausal multimodal spans need range metadata; a packed batch without it cannot fall back.
pub(crate) fn select_paged_mm_prefix_path(
    requires_noncausal: bool,
    is_paged: bool,
    supported_backend: bool,
    packed: bool,
    has_range_metadata: bool,
) -> Result<bool> {
    if requires_noncausal && packed && !has_range_metadata {
        candle_core::bail!("packed multimodal prefill is missing noncausal range metadata");
    }
    Ok(requires_noncausal && is_paged && supported_backend && has_range_metadata)
}

pub(crate) fn paged_flash_prefill_workspace(
    total_query_tokens: usize,
    query_heads: usize,
    head_dim: usize,
    dtype: DType,
) -> Result<usize> {
    let tensor = checked_tensor_bytes(
        &[total_query_tokens, query_heads, head_dim],
        dtype.size_in_bytes(),
        "paged FlashAttention query/output",
    )?;
    let logsumexp = checked_tensor_bytes(
        &[total_query_tokens, query_heads],
        DType::F32.size_in_bytes(),
        "paged FlashAttention logsumexp",
    )?;
    checked_sum(
        &[tensor, tensor, tensor, logsumexp],
        "paged FlashAttention peak",
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

#[allow(dead_code)]
pub(crate) fn model_has_donor_paged_cache_layers(
    model: &(dyn ModelConfigLike + Send + Sync),
) -> bool {
    (0..model.num_layers()).any(|layer_idx| {
        model.layer_has_paged_kv_cache(layer_idx) && !model.uses_own_kv_cache_for_layer(layer_idx)
    })
}

#[allow(dead_code)]
pub(crate) fn prompt_prefill_workspace(
    model: Option<&dyn ModelConfigLike>,
    input: PromptPrefillWorkspaceInput<'_>,
) -> Result<PromptPrefillWorkspace> {
    if !input.requires_prefix_attention
        && ((!input.device_is_metal && !input.device_is_cuda) || model.is_none())
    {
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
    let capped_lens;
    let input = match model.prefix_prefill_kv_len_cap() {
        Some(cap) => {
            capped_lens = input
                .full_context_lens
                .iter()
                .zip(input.query_lens)
                .map(|(full, query)| (*full).min(cap.max(*query)))
                .collect::<Vec<_>>();
            let max_len = capped_lens.iter().copied().max().unwrap_or(0);
            PromptPrefillWorkspaceInput {
                full_context_lens: &capped_lens,
                max_pages_per_sequence: input
                    .max_pages_per_sequence
                    .min(max_len.div_ceil(input.block_size.max(1))),
                ..input
            }
        }
        None => input,
    };
    let query_len = input.query_lens[0];
    let query_layout_is_dense = input.query_lens.iter().all(|&len| len == query_len);
    let total_q_tokens = input
        .query_lens
        .iter()
        .try_fold(0usize, |total, &len| total.checked_add(len))
        .ok_or_else(|| candle_core::Error::msg("prompt query length sum overflow"))?;
    #[cfg(cuda_unix)]
    let mut fa3_pool = crate::flashinfer::Fa3PrefillPoolBytes::default();
    let mut max_layer_transient = 0usize;
    let mut gather_workspace_bytes = 0usize;
    let mut attention_is_complete = input.device_is_metal || input.device_is_cuda;
    #[cfg(feature = "metal")]
    let mut native_workspace_bytes = 0usize;
    #[cfg(feature = "metal")]
    let mut has_native_donor = false;
    #[cfg(feature = "metal")]
    let mut retained_mask_keys = std::collections::HashSet::new();
    #[cfg(feature = "metal")]
    let mut retained_mask_heads = 1usize;
    for layer_idx in
        (0..model.num_layers()).filter(|&layer_idx| model.layer_has_paged_kv_cache(layer_idx))
    {
        let fa3_num_sm = input.fa3_num_sm_by_layer.get(layer_idx).copied().flatten();
        let features = model.prefix_prefill_attention_features(layer_idx);
        #[cfg(feature = "metal")]
        if input.device_is_metal
            && !matches!(
                model.kv_cache_layout_for_layer(layer_idx),
                super::KvCacheLayout::Mla { .. }
            )
        {
            let native = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
                query_lens: input.query_lens,
                context_lens: input.full_context_lens,
                query_heads: model.num_attn_heads_for_layer(layer_idx),
                kv_heads: model.num_kv_heads_for_layer(layer_idx),
                key_head_dim: model.k_head_dim_for_layer(layer_idx),
                value_head_dim: model.v_head_dim_for_layer(layer_idx),
                block_size: input.block_size,
                query_dtype: input.activation_dtype,
                cache_dtype: input.cache_dtype,
                hnd: input.hnd_by_layer.get(layer_idx).copied().unwrap_or(false),
                has_custom_mask: input.has_custom_mask,
                has_multimodal_ranges: input.has_noncausal_mm_context,
            })?;
            if let Some(bytes) = native {
                native_workspace_bytes = native_workspace_bytes.max(bytes);
                has_native_donor |= !model.uses_own_kv_cache_for_layer(layer_idx)
                    && input.hnd_by_layer.get(layer_idx).copied().unwrap_or(false);
                if input.has_custom_mask {
                    retained_mask_keys.insert(metal_mask_key(model, layer_idx, features));
                    retained_mask_heads =
                        retained_mask_heads.max(model.num_attn_heads_for_layer(layer_idx));
                }
                continue;
            }
        }
        if !input.requires_prefix_attention && !input.device_is_cuda {
            attention_is_complete = false;
            continue;
        }
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
        let plan = if input.requires_prefix_attention || query_len == 1 {
            plan_input.map_or(PrefixPrefillPlan::GatherSdpa, PrefixPrefillPlan::choose)
        } else {
            PrefixPrefillPlan::GatherSdpa
        };
        #[cfg(cuda_unix)]
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
            let output_peak = paged_flash_prefill_workspace(
                total_q_tokens,
                model.num_attn_heads_for_layer(layer_idx),
                model.v_head_dim_for_layer(layer_idx),
                input.activation_dtype,
            )?;
            max_layer_transient = max_layer_transient.max(output_peak);
            continue;
        }
        #[cfg(feature = "metal")]
        if input.device_is_metal {
            retained_mask_keys.insert(metal_mask_key(model, layer_idx, features));
        }
        attention_is_complete &= plan_input.is_some_and(|plan| {
            cuda_gather_workspace_is_complete(plan, model.v_head_dim_for_layer(layer_idx))
        });
        let layer_workspace =
            gather_prefill_workspace_for_lengths(GatherPrefillWorkspaceRequest {
                device_is_metal: input.device_is_metal,
                packed_queries: false,
                metal_mask_is_compatible: !input.has_custom_mask && features.is_some(),
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
    #[cfg(feature = "metal")]
    if has_native_donor && !input.has_custom_mask && !input.has_noncausal_mm_context {
        // Shared-cache tails can reduce to one query and select a larger decode workspace.
        let query_lens = vec![1; input.query_lens.len()];
        native_workspace_bytes = native_workspace_bytes.max(metal_decode_workspace(
            Some(model),
            MetalDecodeWorkspaceInput {
                activation_dtype: input.activation_dtype,
                cache_dtype: input.cache_dtype,
                block_size: input.block_size,
                query_lens: &query_lens,
                context_lens: input.full_context_lens,
                hnd_by_layer: input.hnd_by_layer,
                capacity_bound: true,
                has_custom_mask: false,
                has_multimodal_ranges: false,
            },
        )?);
    }
    #[cfg(cuda_unix)]
    let fa3_pool_bytes = fa3_pool.bytes()?;
    #[cfg(not(cuda_unix))]
    let fa3_pool_bytes = 0usize;
    #[cfg(feature = "metal")]
    let retained_mask_bytes = if retained_mask_keys.len() > 1 {
        let max_q = input.query_lens.iter().copied().max().unwrap();
        let max_kv = input.full_context_lens.iter().copied().max().unwrap();
        let total_kv = checked_sum(input.full_context_lens, "prefix mask KV lengths")?;
        let elements = checked_product(
            &[input.query_lens.len(), max_q, max_kv],
            "retained dense prefix masks",
        )?
        .max(checked_product(
            &[total_q_tokens, total_kv],
            "retained packed prefix masks",
        )?);
        // One mask is already included in the per-layer gather peak.
        checked_product(
            &[
                retained_mask_keys.len() - 1,
                retained_mask_heads,
                elements,
                DType::F32.size_in_bytes() + input.activation_dtype.size_in_bytes(),
            ],
            "retained prefix masks",
        )?
    } else {
        0
    };
    #[cfg(not(feature = "metal"))]
    let retained_mask_bytes = 0usize;
    #[cfg(feature = "metal")]
    let (max_layer_transient, retained_mask_bytes) = if input.device_is_metal {
        (
            metal_gather_workspace_bytes(max_layer_transient)?.max(native_workspace_bytes),
            metal_gather_workspace_bytes(retained_mask_bytes)?,
        )
    } else {
        (max_layer_transient, retained_mask_bytes)
    };
    let bytes = fa3_pool_bytes
        .checked_add(max_layer_transient)
        .and_then(|bytes| bytes.checked_add(retained_mask_bytes))
        .ok_or_else(|| candle_core::Error::msg("prompt prefill workspace size overflow"))?;
    Ok(PromptPrefillWorkspace {
        bytes,
        gather_workspace_bytes,
        attention_is_complete,
    })
}

#[cfg(feature = "metal")]
fn metal_mask_key(
    model: &dyn ModelConfigLike,
    layer: usize,
    features: Option<PrefixPrefillAttentionFeatures>,
) -> (Option<usize>, Option<usize>) {
    match features {
        Some(features) if !features.has_sliding_window => (None, None),
        Some(_) => match model.sliding_window_for_layer(layer) {
            Some(window) => (Some(window), None),
            None => (None, Some(layer)),
        },
        None => (None, Some(layer)),
    }
}

#[allow(dead_code)]
fn prompt_plan_input(
    model: &dyn ModelConfigLike,
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

#[cfg(cuda_unix)]
fn fa3_group_size_is_supported(q_heads: usize, kv_heads: usize) -> bool {
    kv_heads > 0
        && q_heads.is_multiple_of(kv_heads)
        && matches!(q_heads / kv_heads, 1 | 2 | 3 | 4 | 6 | 8 | 16)
}

fn paged_flash_attention_supports(
    head_size: usize,
    block_size: usize,
    has_softcap: bool,
    has_sliding_window: bool,
) -> bool {
    crate::attention::flash_backend_supports_sdpa(head_size, has_softcap, has_sliding_window)
        && block_size.is_multiple_of(32)
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
#[allow(dead_code)]
pub(crate) struct DecodePlanInput {
    pub attention_backend: AttentionBackendKind,
    pub head_size: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
    pub has_sliding_window: bool,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodePlan {
    #[cfg(cuda_unix)]
    FlashInfer(FlashInferDecodePlan),
    GatherSdpa,
    PagedAttention,
}

#[cfg(any(feature = "cuda", feature = "metal", test))]
impl DecodePlan {
    pub(crate) fn requires_host_context_lengths(
        attention_backend: AttentionBackendKind,
        head_size: usize,
        has_sliding_window: bool,
    ) -> bool {
        #[cfg(cuda_unix)]
        {
            let _ = has_sliding_window;
            match attention_backend {
                AttentionBackendKind::Standard => {
                    !CUDA_PAGED_ATTENTION_HEAD_DIMS.contains(&head_size)
                }
                AttentionBackendKind::FlashInfer => {
                    head_size > FlashInferDecodePlan::head_size_limit(attention_backend)
                }
            }
        }
        #[cfg(not(cuda_unix))]
        {
            let _ = head_size;
            matches!(attention_backend, AttentionBackendKind::FlashInfer) || has_sliding_window
        }
    }

    pub fn choose(input: DecodePlanInput) -> Result<Self> {
        if Self::requires_host_context_lengths(
            input.attention_backend,
            input.head_size,
            input.has_sliding_window,
        ) {
            return Ok(Self::GatherSdpa);
        }
        match input.attention_backend {
            #[cfg(cuda_unix)]
            AttentionBackendKind::FlashInfer => {
                flashinfer::decode_plan(FlashInferDecodePlanInput {
                    head_size: input.head_size,
                    has_alibi: input.has_alibi,
                    has_sinks: input.has_sinks,
                })
                .map(Self::FlashInfer)
            }
            #[cfg(not(cuda_unix))]
            AttentionBackendKind::FlashInfer => Ok(Self::GatherSdpa),
            AttentionBackendKind::Standard => Ok(Self::PagedAttention),
        }
    }
}

#[cfg(test)]
#[path = "plan_tests.rs"]
mod tests;

#[cfg(all(test, feature = "metal"))]
mod metal_workspace_tests {
    use super::*;
    use crate::paged_attention::{HybridPagedKvCacheConfig, KvCacheLayout, ModelConfigMetadata};

    fn model(features: PrefixPrefillAttentionFeatures) -> HybridPagedKvCacheConfig {
        HybridPagedKvCacheConfig::new(
            ModelConfigMetadata {
                max_seq_len: 8192,
                num_layers: 1,
                hidden_size: 4096,
                num_kv_heads: 2,
                num_attn_heads: 8,
                sliding_window: None,
                k_head_dim: 512,
                v_head_dim: 512,
                kv_cache_layout: KvCacheLayout::MetalHnd,
            },
            vec![true],
        )
        .with_uniform_prefix_prefill_attention_features(features)
    }

    #[test]
    fn decode_reserve_covers_split_and_kv_packing() {
        let model = model(PrefixPrefillAttentionFeatures::default());
        let bytes = metal_decode_workspace(
            Some(&model),
            MetalDecodeWorkspaceInput {
                activation_dtype: DType::BF16,
                cache_dtype: DType::BF16,
                block_size: 32,
                query_lens: &[1],
                context_lens: &[1024],
                hnd_by_layer: &[true],
                capacity_bound: false,
                has_multimodal_ranges: false,
                has_custom_mask: false,
            },
        )
        .unwrap();
        let split_output_and_scratch = 8192 + 2 * 1024 + 524288;
        let packed_kv_backing = 2 * (2 * 512 * 2 * 2);
        assert_eq!(bytes, split_output_and_scratch + packed_kv_backing);
    }

    #[test]
    fn speculative_decode_reserves_native_layout_copies_and_custom_fallback() -> Result<()> {
        let model = model(PrefixPrefillAttentionFeatures::default());
        let query_lens = [4, 2];
        let context_lens = [1024, 513];
        let native = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
            query_lens: &query_lens,
            context_lens: &context_lens,
            query_heads: 8,
            kv_heads: 2,
            key_head_dim: 512,
            value_head_dim: 512,
            block_size: 32,
            query_dtype: DType::BF16,
            cache_dtype: DType::BF16,
            hnd: true,
            has_custom_mask: false,
            has_multimodal_ranges: false,
        })?
        .unwrap();
        let estimate = |has_custom_mask| {
            metal_decode_workspace(
                Some(&model),
                MetalDecodeWorkspaceInput {
                    activation_dtype: DType::BF16,
                    cache_dtype: DType::BF16,
                    block_size: 32,
                    query_lens: &query_lens,
                    context_lens: &context_lens,
                    hnd_by_layer: &[true],
                    capacity_bound: false,
                    has_multimodal_ranges: false,
                    has_custom_mask,
                },
            )
        };
        let packing = metal_cache_write_workspace(Some(&model), 6, DType::BF16)?;
        assert_eq!(estimate(false)?, native + packing);
        assert!(estimate(true)? > estimate(false)?);
        Ok(())
    }

    #[test]
    fn single_query_decode_also_reserves_structured_native_attention() -> Result<()> {
        let model = model(PrefixPrefillAttentionFeatures::default());
        let native = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
            query_lens: &[1],
            context_lens: &[1],
            query_heads: 8,
            kv_heads: 2,
            key_head_dim: 512,
            value_head_dim: 512,
            block_size: 32,
            query_dtype: DType::BF16,
            cache_dtype: DType::BF16,
            hnd: true,
            has_custom_mask: false,
            has_multimodal_ranges: false,
        })?
        .unwrap();
        let decode = metal_decode_workspace(
            Some(&model),
            MetalDecodeWorkspaceInput {
                activation_dtype: DType::BF16,
                cache_dtype: DType::BF16,
                block_size: 32,
                query_lens: &[1],
                context_lens: &[1],
                hnd_by_layer: &[true],
                capacity_bound: false,
                has_multimodal_ranges: false,
                has_custom_mask: false,
            },
        )?;
        assert_eq!(
            decode,
            native + metal_cache_write_workspace(Some(&model), 1, DType::BF16)?
        );
        Ok(())
    }

    #[test]
    fn decode_reserves_retained_multimodal_range_metadata() -> Result<()> {
        let model = ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers: 1,
            hidden_size: 128,
            num_kv_heads: 1,
            num_attn_heads: 2,
            sliding_window: None,
            k_head_dim: 64,
            v_head_dim: 64,
            kv_cache_layout: KvCacheLayout::MetalHnd,
        };
        let estimate = |has_multimodal_ranges| {
            metal_decode_workspace(
                Some(&model),
                MetalDecodeWorkspaceInput {
                    activation_dtype: DType::BF16,
                    cache_dtype: DType::BF16,
                    block_size: 32,
                    query_lens: &[1, 1],
                    context_lens: &[4096, 8192],
                    hnd_by_layer: &[true],
                    capacity_bound: false,
                    has_custom_mask: false,
                    has_multimodal_ranges,
                },
            )
        };
        let native = metal_paged_prefill_workspace(MetalPagedPrefillWorkspaceInput {
            query_lens: &[1, 1],
            context_lens: &[4096, 8192],
            query_heads: 2,
            kv_heads: 1,
            key_head_dim: 64,
            value_head_dim: 64,
            block_size: 32,
            query_dtype: DType::BF16,
            cache_dtype: DType::BF16,
            hnd: true,
            has_custom_mask: false,
            has_multimodal_ranges: true,
        })?
        .unwrap();
        assert!(estimate(true)? > estimate(false)?);
        assert_eq!(
            estimate(true)?,
            native + metal_cache_write_workspace(Some(&model), 2, DType::BF16)?
        );
        Ok(())
    }

    #[test]
    fn legacy_sliding_gather_reserve_grows_with_key_length() {
        let model = model(PrefixPrefillAttentionFeatures {
            has_sliding_window: true,
            ..Default::default()
        });
        let estimate = |context| {
            metal_decode_workspace(
                Some(&model),
                MetalDecodeWorkspaceInput {
                    activation_dtype: DType::BF16,
                    cache_dtype: DType::BF16,
                    block_size: 32,
                    query_lens: &[1, 1],
                    context_lens: &[context, context],
                    hnd_by_layer: &[false],
                    capacity_bound: false,
                    has_multimodal_ranges: false,
                    has_custom_mask: false,
                },
            )
            .unwrap()
        };
        assert!(estimate(4096) > 2 * estimate(1024));
    }

    #[test]
    fn unknown_hnd_features_reserve_direct_fallback_without_gathering_long_kv() {
        let model = ModelConfigMetadata {
            max_seq_len: 8192,
            num_layers: 1,
            hidden_size: 4096,
            num_kv_heads: 2,
            num_attn_heads: 8,
            sliding_window: None,
            k_head_dim: 512,
            v_head_dim: 512,
            kv_cache_layout: KvCacheLayout::MetalHnd,
        };
        let estimate = |hnd| {
            metal_decode_workspace(
                Some(&model),
                MetalDecodeWorkspaceInput {
                    activation_dtype: DType::BF16,
                    cache_dtype: DType::BF16,
                    block_size: 32,
                    query_lens: &[1],
                    context_lens: &[8192],
                    hnd_by_layer: &[hnd],
                    capacity_bound: false,
                    has_multimodal_ranges: false,
                    has_custom_mask: false,
                },
            )
            .unwrap()
        };
        assert!(estimate(true) < estimate(false) / 2);
    }

    #[test]
    fn rejects_before_allocation_when_admitted_workspace_is_exceeded() {
        let error = check_metal_attention_workspace(
            &candle_core::Device::Cpu,
            2,
            Some(
                &crate::pipeline::text_models_inputs_processor::AttentionMemoryBudget {
                    workspace_bytes: 1,
                    metal_allocation_limits: None,
                },
            ),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("exceeding its admitted workspace"));
    }
}
