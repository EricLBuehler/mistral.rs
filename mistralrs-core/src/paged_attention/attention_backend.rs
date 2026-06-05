use candle_core::Tensor;

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
