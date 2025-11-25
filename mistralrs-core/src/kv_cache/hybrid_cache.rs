//! Hybrid cache for models that mix attention and Mamba layers (e.g., GraniteMoeHybrid)
//!
//! This implements vLLM-style continuous batching for hybrid models:
//! - Attention layers use standard KV cache batching
//! - Mamba layers use a pre-allocated state pool with indexed access
//!
//! The key insight is that Mamba state is accessed via `state_indices` which map
//! each sequence in the current batch to its slot in the pre-allocated pool.

use candle_core::{Device, IndexOp, Result, Tensor};

use super::KvCache;

/// Pool-based Mamba state cache for continuous batching.
///
/// Instead of dynamically sized state tensors, we pre-allocate a pool of
/// `max_num_seqs` state slots. Each sequence is assigned a slot index, and
/// the forward pass uses `index_select` (gather) and index assignment (scatter)
/// to access the correct states.
#[derive(Debug)]
pub struct MambaStatePool {
    /// Convolution state pool: (max_num_seqs, conv_dim, d_conv)
    pub conv_state: Tensor,
    /// SSM state pool: (max_num_seqs, n_heads, head_dim, d_state)
    pub ssm_state: Tensor,
    /// Per-slot sequence length offsets (for tracking generation position)
    seqlen_offsets: Vec<usize>,
    /// Stack of free slot indices (for allocation)
    free_slots: Vec<usize>,
    /// Configuration
    max_num_seqs: usize,
    conv_dim: usize,
    d_conv: usize,
    n_heads: usize,
    head_dim: usize,
    d_state: usize,
    dtype: candle_core::DType,
    device: Device,
}

impl MambaStatePool {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_num_seqs: usize,
        conv_dim: usize,
        d_conv: usize,
        n_heads: usize,
        head_dim: usize,
        d_state: usize,
        dtype: candle_core::DType,
        device: &Device,
    ) -> Result<Self> {
        let conv_state = Tensor::zeros((max_num_seqs, conv_dim, d_conv), dtype, device)?;
        let ssm_state = Tensor::zeros((max_num_seqs, n_heads, head_dim, d_state), dtype, device)?;

        // All slots start as free
        let free_slots: Vec<usize> = (0..max_num_seqs).rev().collect();
        let seqlen_offsets = vec![0; max_num_seqs];

        Ok(Self {
            conv_state,
            ssm_state,
            seqlen_offsets,
            free_slots,
            max_num_seqs,
            conv_dim,
            d_conv,
            n_heads,
            head_dim,
            d_state,
            dtype,
            device: device.clone(),
        })
    }

    /// Allocate a state slot for a new sequence. Returns the slot index.
    /// The slot's state is reset to zeros to prevent state bleeding from previous sequences.
    pub fn allocate(&mut self) -> Option<usize> {
        let slot_idx = self.free_slots.pop()?;
        // Reset the state for the newly allocated slot to prevent state bleeding
        if self.reset_slot(slot_idx).is_err() {
            // If reset fails, still return the slot but log warning
            tracing::warn!("Failed to reset Mamba state slot {slot_idx}, state may be stale");
        }
        Some(slot_idx)
    }

    /// Free a state slot when a sequence completes.
    pub fn free(&mut self, slot_idx: usize) {
        debug_assert!(slot_idx < self.max_num_seqs);
        // Reset the state for this slot
        self.seqlen_offsets[slot_idx] = 0;
        self.free_slots.push(slot_idx);
    }

    /// Get the seqlen offset for a slot
    pub fn get_seqlen_offset(&self, slot_idx: usize) -> usize {
        self.seqlen_offsets[slot_idx]
    }

    /// Set the seqlen offset for a slot
    pub fn set_seqlen_offset(&mut self, slot_idx: usize, offset: usize) {
        self.seqlen_offsets[slot_idx] = offset;
    }

    /// Increment seqlen offset for a slot
    pub fn increment_seqlen_offset(&mut self, slot_idx: usize, delta: usize) {
        self.seqlen_offsets[slot_idx] += delta;
    }

    /// Gather conv states for the given slot indices
    /// Returns tensor of shape (batch_size, conv_dim, d_conv)
    pub fn gather_conv_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.conv_state.index_select(state_indices, 0)
    }

    /// Gather SSM states for the given slot indices
    /// Returns tensor of shape (batch_size, n_heads, head_dim, d_state)
    pub fn gather_ssm_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.ssm_state.index_select(state_indices, 0)
    }

    /// Scatter conv states back to the pool for the given slot indices
    pub fn scatter_conv_state(&mut self, state_indices: &Tensor, values: &Tensor) -> Result<()> {
        let indices: Vec<u32> = state_indices.to_vec1()?;
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values.i(batch_idx)?.unsqueeze(0)?;
            // Use slice_set for in-place update (faster than slice_assign)
            self.conv_state.slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    /// Scatter SSM states back to the pool for the given slot indices
    pub fn scatter_ssm_state(&mut self, state_indices: &Tensor, values: &Tensor) -> Result<()> {
        let indices: Vec<u32> = state_indices.to_vec1()?;
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values.i(batch_idx)?.unsqueeze(0)?;
            // Use slice_set for in-place update (faster than slice_assign)
            self.ssm_state.slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    /// Reset a specific slot's state to zeros
    pub fn reset_slot(&mut self, slot_idx: usize) -> Result<()> {
        let zero_conv = Tensor::zeros((1, self.conv_dim, self.d_conv), self.dtype, &self.device)?;
        let zero_ssm = Tensor::zeros(
            (1, self.n_heads, self.head_dim, self.d_state),
            self.dtype,
            &self.device,
        )?;

        // Use slice_set for in-place update
        self.conv_state.slice_set(&zero_conv, 0, slot_idx)?;
        self.ssm_state.slice_set(&zero_ssm, 0, slot_idx)?;
        self.seqlen_offsets[slot_idx] = 0;
        Ok(())
    }

    /// Reset all slots
    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.ssm_state = self.ssm_state.zeros_like()?;
        self.seqlen_offsets.fill(0);
        self.free_slots = (0..self.max_num_seqs).rev().collect();
        Ok(())
    }

    pub fn max_num_seqs(&self) -> usize {
        self.max_num_seqs
    }

    pub fn num_free_slots(&self) -> usize {
        self.free_slots.len()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> candle_core::DType {
        self.dtype
    }
}

impl Clone for MambaStatePool {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            ssm_state: self.ssm_state.clone(),
            seqlen_offsets: self.seqlen_offsets.clone(),
            free_slots: self.free_slots.clone(),
            max_num_seqs: self.max_num_seqs,
            conv_dim: self.conv_dim,
            d_conv: self.d_conv,
            n_heads: self.n_heads,
            head_dim: self.head_dim,
            d_state: self.d_state,
            dtype: self.dtype,
            device: self.device.clone(),
        }
    }
}

/// Per-layer cache that can be either attention (KV) or Mamba (state pool)
#[derive(Clone, Debug)]
pub enum HybridLayerCache {
    Attention(KvCache),
    Mamba(MambaStatePool),
}

impl HybridLayerCache {
    pub fn reset(&mut self) {
        match self {
            Self::Attention(kv) => kv.reset(),
            Self::Mamba(pool) => {
                let _ = pool.reset();
            }
        }
    }

    pub fn as_kv_cache(&self) -> Option<&KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Mamba(_) => None,
        }
    }

    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Mamba(_) => None,
        }
    }

    pub fn as_mamba_pool(&self) -> Option<&MambaStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Mamba(pool) => Some(pool),
        }
    }

    pub fn as_mamba_pool_mut(&mut self) -> Option<&mut MambaStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Mamba(pool) => Some(pool),
        }
    }
}

/// Layer type indicator for hybrid models
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridLayerType {
    Attention,
    Mamba,
}

/// Configuration for creating a hybrid cache
#[derive(Clone, Debug)]
pub struct HybridCacheConfig {
    pub layer_types: Vec<HybridLayerType>,
    pub max_seq_len: usize,
    pub max_num_seqs: usize,
    // Mamba-specific config
    pub mamba_conv_dim: usize,
    pub mamba_d_conv: usize,
    pub mamba_n_heads: usize,
    pub mamba_head_dim: usize,
    pub mamba_d_state: usize,
}

/// Hybrid cache that stores per-layer caches for mixed attention/Mamba models
///
/// For continuous batching:
/// - Attention layers use standard KV cache with batching support
/// - Mamba layers use MambaStatePool with indexed access via state_indices
#[derive(Clone, Debug)]
pub struct HybridCache {
    pub caches: Vec<HybridLayerCache>,
    config: HybridCacheConfig,
    /// Current batch's state indices for Mamba pool access.
    /// Set by clone_in_cache before forward, used by model during forward.
    /// Shape: (batch_size,) containing pool slot indices.
    state_indices: Option<Tensor>,
}

impl HybridCache {
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(
        config: HybridCacheConfig,
        dtype: candle_core::DType,
        device: &Device,
    ) -> Result<Self> {
        let mut caches = Vec::with_capacity(config.layer_types.len());

        for layer_type in &config.layer_types {
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Mamba => HybridLayerCache::Mamba(MambaStatePool::new(
                    config.max_num_seqs,
                    config.mamba_conv_dim,
                    config.mamba_d_conv,
                    config.mamba_n_heads,
                    config.mamba_head_dim,
                    config.mamba_d_state,
                    dtype,
                    device,
                )?),
            };
            caches.push(cache);
        }

        Ok(Self {
            caches,
            config,
            state_indices: None,
        })
    }

    /// Allocate state slots for a new sequence across all Mamba layers.
    /// Returns the slot index (same for all layers).
    pub fn allocate_seq(&mut self) -> Option<usize> {
        // All Mamba layers share the same slot index for a sequence
        let mut slot_idx = None;
        for cache in &mut self.caches {
            if let HybridLayerCache::Mamba(pool) = cache {
                if slot_idx.is_none() {
                    slot_idx = pool.allocate();
                    slot_idx?;
                } else {
                    // Allocate same slot in other layers (they should be in sync)
                    let _ = pool.allocate();
                }
            }
        }
        slot_idx
    }

    /// Free state slots for a sequence across all Mamba layers.
    pub fn free_seq(&mut self, slot_idx: usize) {
        for cache in &mut self.caches {
            if let HybridLayerCache::Mamba(pool) = cache {
                pool.free(slot_idx);
            }
        }
    }

    /// Reset a specific sequence's state in all Mamba layers.
    pub fn reset_seq(&mut self, slot_idx: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Mamba(pool) = cache {
                pool.reset_slot(slot_idx)?;
            }
        }
        Ok(())
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }

    pub fn num_layers(&self) -> usize {
        self.caches.len()
    }

    pub fn layer_types(&self) -> &[HybridLayerType] {
        &self.config.layer_types
    }

    pub fn config(&self) -> &HybridCacheConfig {
        &self.config
    }

    pub fn max_num_seqs(&self) -> usize {
        self.config.max_num_seqs
    }

    /// Get a mutable reference to a specific layer's cache
    pub fn get_mut(&mut self, layer: usize) -> Option<&mut HybridLayerCache> {
        self.caches.get_mut(layer)
    }

    /// Get a reference to a specific layer's cache
    pub fn get(&self, layer: usize) -> Option<&HybridLayerCache> {
        self.caches.get(layer)
    }

    /// Set the state indices for the current batch.
    /// Called by HybridCacheManager::clone_in_cache before forward.
    pub fn set_state_indices(&mut self, indices: Option<Tensor>) {
        self.state_indices = indices;
    }

    /// Get the state indices for the current batch.
    /// Used by the model during forward to access Mamba state pool.
    pub fn state_indices(&self) -> Option<&Tensor> {
        self.state_indices.as_ref()
    }
}
