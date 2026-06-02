use candle_core::Tensor;

use crate::pipeline::text_models_inputs_processor::FLASHINFER_PREFILL_MAX_GROUP_SIZE;

#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const STANDARD_PAGED_ATTENTION_MAX_HEAD_SIZE: usize = 256;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_PREFILL_MAX_HEAD_SIZE: usize = 256;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub const FLASHINFER_DECODE_MAX_HEAD_SIZE: usize = 512;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_ENABLED: bool = false;
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const FLASHINFER_TENSOR_CORE_DECODE_MAX_HEAD_SIZE: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionBackendKind {
    Standard,
    FlashInfer,
}

impl AttentionBackendKind {
    pub fn from_cache(key_cache: &Tensor, value_cache: &Tensor) -> Self {
        #[cfg(all(feature = "cuda", target_family = "unix"))]
        {
            if mistralrs_paged_attn::is_flashinfer_cache(key_cache, value_cache) {
                return Self::FlashInfer;
            }
        }
        #[cfg(not(all(feature = "cuda", target_family = "unix")))]
        let _ = (key_cache, value_cache);

        Self::Standard
    }
}

#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerSpec {
    pub q_heads: usize,
    pub kv_heads: usize,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
}

pub trait AttentionBackend {
    fn kind(&self) -> AttentionBackendKind;
    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool;
}

pub struct FlashInferAttentionBackend;

impl AttentionBackend for FlashInferAttentionBackend {
    fn kind(&self) -> AttentionBackendKind {
        AttentionBackendKind::FlashInfer
    }

    fn supports_layer(&self, spec: AttentionLayerSpec) -> bool {
        if !cfg!(feature = "cuda") || !crate::perf_flags::flashinfer_decode_enabled() {
            return false;
        }
        if spec.kv_heads == 0 || !spec.q_heads.is_multiple_of(spec.kv_heads) {
            return false;
        }
        let q_group = spec.q_heads / spec.kv_heads;
        spec.k_head_dim == spec.v_head_dim
            && matches!(spec.k_head_dim, 64 | 128 | 256 | 512)
            && q_group <= FLASHINFER_PREFILL_MAX_GROUP_SIZE
    }
}
