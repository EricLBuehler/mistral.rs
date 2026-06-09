use candle_core::{DType, Result};

use super::attention_backend::AttentionBackendKind;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{self, FlashInferDecodePlan, FlashInferDecodePlanInput};

pub(crate) struct PrefixPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub has_sinks: bool,
    pub has_custom_mask: bool,
    pub has_noncausal_mm_context: bool,
    pub has_mm_prefix_ranges: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub query_lens_match_seq_len: bool,
    pub block_size: usize,
    pub attention_backend: AttentionBackendKind,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefixPrefillPlan {
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
            input.has_sinks,
            input.has_custom_mask,
            input.has_noncausal_mm_context,
            input.has_mm_prefix_ranges,
            input.causality_known,
            input.head_size,
            input.query_lens_match_seq_len,
            input.block_size,
            input.attention_backend,
        );

        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if input.device_is_cuda
            && matches!(input.dtype, DType::F16 | DType::BF16)
            && !input.has_sinks
            && !input.has_custom_mask
            && (!input.has_noncausal_mm_context || input.has_mm_prefix_ranges)
            && input.causality_known
            && input.query_lens_match_seq_len
            && paged_flash_attention_supports(input.head_size, input.block_size)
            && matches!(input.attention_backend, AttentionBackendKind::FlashInfer)
        {
            return Self::FlashAttentionPaged;
        }

        Self::GatherSdpa
    }
}

#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
fn paged_flash_attention_supports(head_size: usize, block_size: usize) -> bool {
    matches!(head_size, 64 | 128 | 256) && block_size % 32 == 0
}

pub(crate) struct DecodePlanInput {
    pub attention_backend: AttentionBackendKind,
    pub head_size: usize,
    pub has_alibi: bool,
    pub has_sinks: bool,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum DecodePlan {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    FlashInfer(FlashInferDecodePlan),
    GatherSdpa,
    PagedAttention,
}

impl DecodePlan {
    pub fn choose(input: DecodePlanInput) -> Result<Self> {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        if input.head_size > FlashInferDecodePlan::head_size_limit(input.attention_backend) {
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
            AttentionBackendKind::Standard => Ok(Self::PagedAttention),
        }
    }
}
