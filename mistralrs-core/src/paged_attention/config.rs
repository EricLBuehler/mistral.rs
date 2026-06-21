use super::attention_backend::{AttentionBackend, AttentionBackendKind, AttentionLayerSpec};
use crate::flashinfer::FlashInferAttentionBackend;

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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KvCacheTopology {
    layer_to_owner: Vec<usize>,
}

impl KvCacheTopology {
    pub fn all_own(num_layers: usize) -> Self {
        Self {
            layer_to_owner: (0..num_layers).collect(),
        }
    }

    pub fn from_layer_owners(layer_to_owner: Vec<usize>) -> Self {
        debug_assert!(layer_to_owner
            .iter()
            .all(|owner| *owner < layer_to_owner.len()));
        Self { layer_to_owner }
    }

    pub fn owner_for_layer(&self, layer_idx: usize) -> usize {
        self.layer_to_owner
            .get(layer_idx)
            .copied()
            .unwrap_or(layer_idx)
    }

    pub fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        self.owner_for_layer(layer_idx) == layer_idx
    }

    pub fn has_shared_layers(&self) -> bool {
        self.layer_to_owner
            .iter()
            .enumerate()
            .any(|(layer_idx, owner)| layer_idx != *owner)
    }

    pub fn group_ids(&self) -> Vec<u32> {
        if !self.has_shared_layers() {
            return vec![0];
        }

        let mut group_ids = self
            .layer_to_owner
            .iter()
            .map(|owner| u32::try_from(*owner).expect("KV cache owner index exceeds u32"))
            .collect::<Vec<_>>();
        group_ids.sort_unstable();
        group_ids.dedup();
        if group_ids.is_empty() {
            vec![0]
        } else {
            group_ids
        }
    }
}

fn flashinfer_supported_for_model<M: ModelConfigLike + ?Sized>(config: &M) -> bool {
    let backend = FlashInferAttentionBackend;
    (0..config.num_layers())
        .all(|layer_idx| backend.supports_layer(config.attention_layer_spec(layer_idx)))
}

fn select_attention_backend<M: ModelConfigLike + ?Sized>(config: &M) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if flashinfer_supported_for_model(config) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_attention_backend_for_layer<M: ModelConfigLike + ?Sized>(
    config: &M,
    layer_idx: usize,
) -> AttentionBackendKind {
    let backend = FlashInferAttentionBackend;
    if backend.supports_layer(config.attention_layer_spec(layer_idx)) {
        backend.kind()
    } else {
        AttentionBackendKind::Standard
    }
}

fn select_kv_cache_layout<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind() {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
            }
        }
    }
}

fn select_kv_cache_layout_for_layer<M: ModelConfigLike + ?Sized>(
    requested_layout: KvCacheLayout,
    config: &M,
    layer_idx: usize,
) -> KvCacheLayout {
    match requested_layout {
        KvCacheLayout::Mla { .. } => requested_layout,
        KvCacheLayout::StandardNoFlashInfer => KvCacheLayout::Standard,
        KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
            match config.attention_backend_kind_for_layer(layer_idx) {
                AttentionBackendKind::FlashInfer => KvCacheLayout::FlashInferHnd,
                AttentionBackendKind::Standard => KvCacheLayout::Standard,
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
    fn num_attn_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_attn_heads()
    }
    fn num_kv_heads_for_layer(&self, _layer_idx: usize) -> usize {
        self.num_kv_heads()
    }
    fn k_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.k_head_dim()
    }
    fn v_head_dim_for_layer(&self, _layer_idx: usize) -> usize {
        self.v_head_dim()
    }
    fn has_kv_cache_sharing(&self) -> bool {
        false
    }
    fn kv_cache_topology(&self) -> KvCacheTopology {
        KvCacheTopology::all_own(self.num_layers())
    }
    fn uses_own_kv_cache_for_layer(&self, layer_idx: usize) -> bool {
        !self.has_kv_cache_sharing()
            || self
                .kv_cache_topology()
                .uses_own_kv_cache_for_layer(layer_idx)
    }
    fn kv_cache_group_ids(&self) -> Vec<u32> {
        if self.has_kv_cache_sharing() {
            self.kv_cache_topology().group_ids()
        } else {
            vec![0]
        }
    }
    fn attention_layer_spec(&self, layer_idx: usize) -> AttentionLayerSpec {
        AttentionLayerSpec {
            q_heads: self.num_attn_heads_for_layer(layer_idx),
            kv_heads: self.num_kv_heads_for_layer(layer_idx),
            k_head_dim: self.k_head_dim_for_layer(layer_idx),
            v_head_dim: self.v_head_dim_for_layer(layer_idx),
        }
    }
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        select_attention_backend(self)
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        select_attention_backend_for_layer(self, layer_idx)
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(KvCacheLayout::Standard, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(KvCacheLayout::Standard, self, layer_idx)
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
    fn attention_backend_kind(&self) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend(self)
            }
        }
    }
    fn attention_backend_kind_for_layer(&self, layer_idx: usize) -> AttentionBackendKind {
        match self.kv_cache_layout {
            KvCacheLayout::Mla { .. } | KvCacheLayout::StandardNoFlashInfer => {
                AttentionBackendKind::Standard
            }
            KvCacheLayout::FlashInferHnd | KvCacheLayout::Standard => {
                select_attention_backend_for_layer(self, layer_idx)
            }
        }
    }
    fn kv_cache_layout(&self) -> KvCacheLayout {
        select_kv_cache_layout(self.kv_cache_layout, self)
    }
    fn kv_cache_layout_for_layer(&self, layer_idx: usize) -> KvCacheLayout {
        select_kv_cache_layout_for_layer(self.kv_cache_layout, self, layer_idx)
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

#[cfg(test)]
mod tests {
    use super::KvCacheTopology;

    #[test]
    fn all_own_topology_uses_single_prefix_cache_group() {
        let topology = KvCacheTopology::all_own(4);
        assert_eq!(topology.group_ids(), vec![0]);
        assert!(!topology.has_shared_layers());
        assert!(topology.uses_own_kv_cache_for_layer(3));
    }

    #[test]
    fn shared_topology_groups_by_owner_layers() {
        let topology = KvCacheTopology::from_layer_owners(vec![0, 1, 0, 1, 4]);
        assert_eq!(topology.group_ids(), vec![0, 1, 4]);
        assert!(topology.has_shared_layers());
        assert!(topology.uses_own_kv_cache_for_layer(1));
        assert!(!topology.uses_own_kv_cache_for_layer(2));
        assert_eq!(topology.owner_for_layer(3), 1);
    }
}
