use candle_core::{DType, Result};

use super::attention_backend::AttentionBackendKind;
#[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
use crate::attention::flash_backend_supports_sdpa;
#[cfg(all(feature = "cuda", target_family = "unix"))]
use crate::flashinfer::{self, FlashInferDecodePlan, FlashInferDecodePlanInput};

#[allow(dead_code)]
pub(crate) struct PrefixPrefillPlanInput {
    pub device_is_cuda: bool,
    pub dtype: DType,
    pub cache_dtype: DType,
    pub has_sinks: bool,
    pub has_custom_mask: bool,
    pub causality_known: bool,
    pub head_size: usize,
    pub has_softcap: bool,
    pub has_sliding_window: bool,
    pub query_layout_is_dense: bool,
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
            input.cache_dtype,
            input.has_sinks,
            input.has_custom_mask,
            input.causality_known,
            input.head_size,
            input.has_softcap,
            input.has_sliding_window,
            input.query_layout_is_dense,
            input.block_size,
            input.attention_backend,
        );

        #[cfg(all(feature = "cuda", feature = "flash-attn", target_family = "unix"))]
        if input.device_is_cuda
            && matches!(input.dtype, DType::F16 | DType::BF16)
            && input.cache_dtype == input.dtype
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

    fn prefix_plan(
        head_size: usize,
        has_softcap: bool,
        has_sliding_window: bool,
    ) -> PrefixPrefillPlan {
        PrefixPrefillPlan::choose(PrefixPrefillPlanInput {
            device_is_cuda: true,
            dtype: DType::F16,
            cache_dtype: DType::F16,
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size,
            has_softcap,
            has_sliding_window,
            query_layout_is_dense: true,
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
            has_sinks: false,
            has_custom_mask: false,
            causality_known: true,
            head_size: 128,
            has_softcap: false,
            has_sliding_window: false,
            query_layout_is_dense: true,
            block_size: 32,
            attention_backend: AttentionBackendKind::FlashInfer,
        });
        assert!(matches!(plan, PrefixPrefillPlan::GatherSdpa));
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
