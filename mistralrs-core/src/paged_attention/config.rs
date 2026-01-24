#[derive(Clone, Copy, Debug, PartialEq)]
pub enum KvCacheLayout {
    Standard,
    Mla {
        kv_lora_rank: usize,
        kpe_head_dim: usize,
    },
}

pub trait ModelConfigLike {
    fn max_seq_len(&self) -> usize;
    fn num_layers(&self) -> usize;
    fn hidden_size(&self) -> usize;
    fn num_kv_heads(&self) -> usize;
    fn num_attn_heads(&self) -> usize;
    fn k_head_dim(&self) -> usize;
    fn v_head_dim(&self) -> usize;
    fn kv_cache_layout(&self) -> KvCacheLayout {
        KvCacheLayout::Standard
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
        self.kv_cache_layout
    }
    fn kv_cache_elements_per_token(&self) -> usize {
        match self.kv_cache_layout {
            KvCacheLayout::Standard => 2 * self.num_kv_heads * self.k_head_dim.max(self.v_head_dim),
            KvCacheLayout::Mla {
                kv_lora_rank,
                kpe_head_dim,
            } => kv_lora_rank + kpe_head_dim,
        }
    }
}
