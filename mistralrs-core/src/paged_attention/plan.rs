use candle_core::{DType, Result};

use super::attention_backend::AttentionBackendKind;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{
    self, FlashInferDecodePlan, FlashInferDecodePlanInput, FlashInferPrefillPlan,
    FlashInferPrefillPlanInput,
};

pub(crate) struct PrefixPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub has_sinks: bool,
    pub causal: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub attention_heads: usize,
    pub key_value_heads: usize,
    pub query_lens_match_seq_len: bool,
    pub attention_backend: AttentionBackendKind,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum PrefixPrefillPlan {
    #[cfg(all(feature = "cuda", target_family = "unix"))]
    FlashInfer(FlashInferPrefillPlan),
    GatherSdpa,
}

impl PrefixPrefillPlan {
    pub fn choose(input: PrefixPrefillPlanInput) -> Self {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            flashinfer::prefill_plan(FlashInferPrefillPlanInput {
                device_is_cuda: input.device_is_cuda,
                dtype: input.dtype,
                has_sinks: input.has_sinks,
                causal: input.causal,
                causality_known: input.causality_known,
                head_size: input.head_size,
                attention_heads: input.attention_heads,
                key_value_heads: input.key_value_heads,
                query_lens_match_seq_len: input.query_lens_match_seq_len,
                attention_backend: input.attention_backend,
            })
            .map_or(Self::GatherSdpa, Self::FlashInfer)
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        {
            let _ = input;
            Self::GatherSdpa
        }
    }
}

pub(crate) struct DecodePlanInput {
    pub attention_backend: AttentionBackendKind,
    pub dtype: DType,
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
                    dtype: input.dtype,
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
