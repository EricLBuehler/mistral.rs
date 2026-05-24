const FLASHINFER_DECODE_ENV: &str = "MISTRALRS_FLASHINFER_DECODE";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    Standard,
    StandardNoFlashInfer,
    FlashInferHnd,
    Mla {
        kv_lora_rank: usize,
        kpe_head_dim: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FlashInferDecodePolicy {
    Auto,
    Off,
}

impl FlashInferDecodePolicy {
    fn from_env() -> Self {
        std::env::var(FLASHINFER_DECODE_ENV)
            .map(|value| {
                if matches!(value.as_str(), "0" | "false" | "FALSE" | "no" | "off") {
                    Self::Off
                } else {
                    Self::Auto
                }
            })
            .unwrap_or(Self::Auto)
    }
}

fn select_kv_cache_layout(
    requested_layout: KvCacheLayout,
    k_head_dim: usize,
    v_head_dim: usize,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            if cfg!(feature = "cuda")
                && FlashInferDecodePolicy::from_env() == FlashInferDecodePolicy::Auto
                && k_head_dim == v_head_dim
            {
                KvCacheLayout::FlashInferHnd
            } else {
                KvCacheLayout::Standard
            }
        }
    }
}

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads()
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim()
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim()
    }
    fn uses_own_kv_cache_for_layer(&self, _layer_idx: usize) -> bool {
        true
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(
            KvCacheLayout::Standard,
            self.k_head_dim(),
            self.v_head_dim(),
        )
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        2 * self.num_kv_heads() * self.k_head_dim().max(self.v_head_dim())
    }
}

#[derive(Clone)]
pub struct ModelConfigMetadata {
    pub max_seq_len: usize,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub num_kv_heads: usize,
    pub num_attn_heads: usize,
    pub sliding_window: Option<usize>,
    pub k_head_dim: usize,
    pub v_head_dim: usize,
    pub kv_cache_layout: KvCacheLayout,
}

impl ModelConfigLike for ModelConfigMetadata {
    fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attn_heads(&self) -> usize {
        self.num_attn_heads
    }
    fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    fn num_layers(&self) -> usize {
        self.num_layers
    }
    fn k_head_dim(&self) -> usize {
        self.k_head_dim
    }
    fn v_head_dim(&self) -> usize {
        self.v_head_dim
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(self.kv_cache_layout, self.k_head_dim, self.v_head_dim)
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        match self.kv_cache_layout() {
            KvCacheLayout::Standard
            | KvCacheLayout::StandardNoFlashInfer
            | KvCacheLayout::FlashInferHnd => {
                2 * self.num_kv_heads * self.k_head_dim.max(self.v_head_dim)
            }
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
        }
    }
}
