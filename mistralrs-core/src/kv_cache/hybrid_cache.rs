//! Hybrid cache for models that mix attention and recurrent layers (e.g., GraniteMoeHybrid, Qwen3 Next)
//!
//! This implements vLLM-style continuous batching for hybrid models:
//! - Attention layers use standard KV cache batching
//! - Recurrent layers (Mamba SSM or GDN) use a pool-based state with indexed access
//!
//! The key insight is that recurrent state is accessed via `state_indices` which map
//! each sequence in the current batch to its slot in the pool.

use candle_core::{DType, Device, DeviceLocation, IndexOp, Result, Tensor};
use std::collections::{HashMap, HashSet};

use super::KvCache;
use crate::layers_masker::PastKvLenCache;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecurrentStateSpec {
    Opaque {
        dims: Vec<usize>,
    },
    Gdn {
        heads: usize,
        key_dim: usize,
        value_dim: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecurrentStateLayout {
    Opaque,
    GdnKeyMajor,
    GdnValueMajor,
}

impl RecurrentStateSpec {
    fn physical_layout(
        &self,
        input_dtype: DType,
        device: &Device,
    ) -> Result<(Vec<usize>, RecurrentStateLayout)> {
        match self {
            Self::Opaque { dims } => Ok((dims.clone(), RecurrentStateLayout::Opaque)),
            Self::Gdn {
                heads,
                key_dim,
                value_dim,
            } => {
                if crate::cuda::gdn::v_major_state_supported(
                    device,
                    input_dtype,
                    *key_dim,
                    *value_dim,
                )? {
                    Ok((
                        vec![*heads, *value_dim, *key_dim],
                        RecurrentStateLayout::GdnValueMajor,
                    ))
                } else {
                    Ok((
                        vec![*heads, *key_dim, *value_dim],
                        RecurrentStateLayout::GdnKeyMajor,
                    ))
                }
            }
        }
    }
}

/// Pool-based recurrent state cache for continuous batching.
///
/// Works for both Mamba SSM and GDN (Gated Delta Net) recurrent layers.
/// Instead of dynamically sized state tensors, we maintain a pool of
/// state slots that grows dynamically. Each sequence is assigned a slot index; CUDA GDN kernels
/// address the pool rows in place through the batch's slot table, other backends gather the rows
/// with `index_select` and scatter them back after the layer.
#[derive(Debug)]
pub struct RecurrentStatePool {
    /// Convolution state pool: (capacity * checkpoint_lanes, conv_dim, conv_width)
    pub conv_state: Tensor,
    /// Recurrent state pool: (capacity * checkpoint_lanes, ...state_dims)
    pub recurrent_state: Tensor,
    allocated_slots: Vec<bool>,
    allocated_count: usize,
    /// Current capacity (grows dynamically)
    capacity: usize,
    checkpoint_lanes: usize,
    /// Shape parameters for growing
    conv_dim: usize,
    conv_width: usize,
    state_dims: Vec<usize>,
    state_layout: RecurrentStateLayout,
    conv_dtype: DType,
    recurrent_dtype: DType,
    device: Device,
}

/// Initial pool capacity before dynamic growth: the pre-captured CUDA graph batch range plus the
/// graph pad slot, so growth (which invalidates captured graphs) only happens past that.
const INITIAL_POOL_CAPACITY: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecurrentSlotOwner {
    Sequence(usize),
    GraphPad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecurrentSlotInitialization {
    Deferred,
    Zeroed,
}

fn logical_slot_from_physical_slot(physical_slot: u32, checkpoint_lanes: usize) -> u32 {
    if physical_slot == u32::MAX {
        return u32::MAX;
    }
    let checkpoint_lanes =
        u32::try_from(checkpoint_lanes).expect("recurrent checkpoint lane count exceeds u32");
    physical_slot / checkpoint_lanes
}

struct RecurrentStatePoolConfig<'a> {
    conv_dim: usize,
    conv_width: usize,
    state_dims: Vec<usize>,
    state_layout: RecurrentStateLayout,
    conv_dtype: DType,
    recurrent_dtype: DType,
    device: &'a Device,
}

impl RecurrentStatePool {
    /// Create a new recurrent state pool.
    ///
    /// - `conv_dim`: dimension of the convolution state
    /// - `conv_width`: kernel size / d_conv for causal conv1d
    fn new(config: RecurrentStatePoolConfig<'_>) -> Result<Self> {
        let RecurrentStatePoolConfig {
            conv_dim,
            conv_width,
            state_dims,
            state_layout,
            conv_dtype,
            recurrent_dtype,
            device,
        } = config;
        let capacity = INITIAL_POOL_CAPACITY;
        let checkpoint_lanes = 1;

        let conv_state = Tensor::zeros((capacity, conv_dim, conv_width), conv_dtype, device)?;

        let mut recurrent_shape = vec![capacity];
        recurrent_shape.extend_from_slice(&state_dims);
        let recurrent_state = Tensor::zeros(recurrent_shape, recurrent_dtype, device)?;

        Ok(Self {
            conv_state,
            recurrent_state,
            allocated_slots: vec![false; capacity],
            allocated_count: 0,
            capacity,
            checkpoint_lanes,
            conv_dim,
            conv_width,
            state_dims,
            state_layout,
            conv_dtype,
            recurrent_dtype,
            device: device.clone(),
        })
    }

    fn resized_storage(&self, new_capacity: usize) -> Result<(Tensor, Tensor)> {
        let physical_capacity = new_capacity
            .checked_mul(self.checkpoint_lanes)
            .ok_or_else(|| candle_core::Error::msg("recurrent physical capacity overflow"))?;
        let new_conv = Tensor::zeros(
            (physical_capacity, self.conv_dim, self.conv_width),
            self.conv_dtype,
            &self.device,
        )?;
        new_conv.slice_set(&self.conv_state, 0, 0)?;

        let mut recurrent_shape = vec![physical_capacity];
        recurrent_shape.extend_from_slice(&self.state_dims);
        let new_recurrent = Tensor::zeros(recurrent_shape, self.recurrent_dtype, &self.device)?;
        new_recurrent.slice_set(&self.recurrent_state, 0, 0)?;
        Ok((new_conv, new_recurrent))
    }

    fn install_resized_storage(
        &mut self,
        new_capacity: usize,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) {
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
        self.allocated_slots.resize(new_capacity, false);
        self.capacity = new_capacity;
    }

    fn checkpoint_storage(&self, checkpoint_lanes: usize) -> Result<(Tensor, Tensor)> {
        let physical_capacity = self
            .capacity
            .checked_mul(checkpoint_lanes)
            .ok_or_else(|| candle_core::Error::msg("recurrent physical capacity overflow"))?;
        let conv_state = Tensor::zeros(
            (physical_capacity, self.conv_dim, self.conv_width),
            self.conv_dtype,
            &self.device,
        )?;
        let mut recurrent_shape = vec![physical_capacity];
        recurrent_shape.extend_from_slice(&self.state_dims);
        let recurrent_state = Tensor::zeros(recurrent_shape, self.recurrent_dtype, &self.device)?;
        Ok((conv_state, recurrent_state))
    }

    fn install_checkpoint_storage(
        &mut self,
        checkpoint_lanes: usize,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) {
        self.checkpoint_lanes = checkpoint_lanes;
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
    }

    fn allocate_at(&mut self, slot_idx: usize) -> Result<()> {
        self.reserve_at(slot_idx)?;
        if let Err(err) = self.reset_slot(slot_idx) {
            let released = self.free(slot_idx);
            debug_assert!(released);
            return Err(candle_core::Error::msg(format!(
                "failed to reset recurrent state slot {slot_idx}: {err}"
            )));
        }
        Ok(())
    }

    fn reserve_at(&mut self, slot_idx: usize) -> Result<()> {
        let allocated = self.allocated_slots.get(slot_idx).copied().ok_or_else(|| {
            candle_core::Error::msg(format!(
                "recurrent state slot {slot_idx} exceeds capacity {}",
                self.capacity
            ))
        })?;
        if allocated {
            candle_core::bail!("recurrent state slot {slot_idx} is already allocated");
        }
        self.allocated_slots[slot_idx] = true;
        self.allocated_count += 1;
        Ok(())
    }

    fn free(&mut self, slot_idx: usize) -> bool {
        let Some(allocated) = self.allocated_slots.get_mut(slot_idx) else {
            return false;
        };
        if !*allocated {
            return false;
        }
        *allocated = false;
        self.allocated_count -= 1;
        true
    }

    /// Gather conv states for the given slot indices
    pub fn gather_conv_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.conv_state.index_select(state_indices, 0)
    }

    /// Gather recurrent states for the given slot indices
    pub fn gather_recurrent_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.recurrent_state.index_select(state_indices, 0)
    }

    /// Scatter conv states back to the pool for the given slot indices
    pub fn scatter_conv_state(&mut self, state_indices: &Tensor, values: &Tensor) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return crate::cuda::indexed_copy::copy_rows(values, &self.conv_state, state_indices);
        }
        let indices: Vec<u32> = state_indices.to_vec1()?;
        self.scatter_conv_state_for_indices(&indices, values)
    }

    pub fn scatter_conv_state_for_indices(
        &mut self,
        indices: &[u32],
        values: &Tensor,
    ) -> Result<()> {
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values.i(batch_idx)?.unsqueeze(0)?.contiguous()?;
            self.conv_state.slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    pub fn scatter_conv_state_with_host_indices(
        &mut self,
        state_indices: &Tensor,
        host_indices: Option<&[u32]>,
        values: &Tensor,
    ) -> Result<()> {
        if let Some(indices) = host_indices {
            self.scatter_conv_state_for_indices(indices, values)
        } else {
            self.scatter_conv_state(state_indices, values)
        }
    }

    /// Scatter recurrent states back to the pool for the given slot indices
    pub fn scatter_recurrent_state(
        &mut self,
        state_indices: &Tensor,
        values: &Tensor,
    ) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return crate::cuda::indexed_copy::copy_rows(
                values,
                &self.recurrent_state,
                state_indices,
            );
        }
        let indices: Vec<u32> = state_indices.to_vec1()?;
        self.scatter_recurrent_state_for_indices(&indices, values)
    }

    pub fn scatter_recurrent_state_for_indices(
        &mut self,
        indices: &[u32],
        values: &Tensor,
    ) -> Result<()> {
        for (batch_idx, &slot_idx) in indices.iter().enumerate() {
            let value = values.i(batch_idx)?.unsqueeze(0)?.contiguous()?;
            self.recurrent_state
                .slice_set(&value, 0, slot_idx as usize)?;
        }
        Ok(())
    }

    pub fn scatter_recurrent_state_with_host_indices(
        &mut self,
        state_indices: &Tensor,
        host_indices: Option<&[u32]>,
        values: &Tensor,
    ) -> Result<()> {
        if let Some(indices) = host_indices {
            self.scatter_recurrent_state_for_indices(indices, values)
        } else {
            self.scatter_recurrent_state(state_indices, values)
        }
    }

    /// Reset a specific slot's state to zeros
    pub fn reset_slot(&mut self, slot_idx: usize) -> Result<()> {
        let zero_conv = Tensor::zeros(
            (self.checkpoint_lanes, self.conv_dim, self.conv_width),
            self.conv_dtype,
            &self.device,
        )?;

        let mut recurrent_shape = vec![self.checkpoint_lanes];
        recurrent_shape.extend_from_slice(&self.state_dims);
        let zero_recurrent = Tensor::zeros(recurrent_shape, self.recurrent_dtype, &self.device)?;

        let physical_slot = self.physical_slot(slot_idx, 0)?;
        self.conv_state.slice_set(&zero_conv, 0, physical_slot)?;
        self.recurrent_state
            .slice_set(&zero_recurrent, 0, physical_slot)?;
        Ok(())
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn physical_capacity(&self) -> usize {
        self.capacity * self.checkpoint_lanes
    }

    pub fn checkpoint_lanes(&self) -> usize {
        self.checkpoint_lanes
    }

    pub fn physical_slot(&self, logical_slot: usize, lane: usize) -> Result<usize> {
        if logical_slot >= self.capacity {
            candle_core::bail!(
                "recurrent logical slot {logical_slot} exceeds capacity {}",
                self.capacity
            );
        }
        if lane >= self.checkpoint_lanes {
            candle_core::bail!(
                "recurrent checkpoint lane {lane} exceeds lane count {}",
                self.checkpoint_lanes
            );
        }
        logical_slot
            .checked_mul(self.checkpoint_lanes)
            .and_then(|base| base.checked_add(lane))
            .ok_or_else(|| candle_core::Error::msg("recurrent physical slot overflow"))
    }

    pub fn num_free_slots(&self) -> usize {
        self.capacity - self.allocated_count
    }

    fn is_allocated(&self, slot_idx: usize) -> bool {
        self.allocated_slots.get(slot_idx).copied().unwrap_or(false)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn conv_dtype(&self) -> DType {
        self.conv_dtype
    }

    pub fn recurrent_dtype(&self) -> DType {
        self.recurrent_dtype
    }

    fn snapshot_bytes(&self) -> Result<usize> {
        let conv_elements = self.conv_dim.checked_mul(self.conv_width).ok_or_else(|| {
            candle_core::Error::msg("recurrent convolution snapshot size overflow")
        })?;
        let recurrent_elements = self.state_dims.iter().try_fold(1usize, |elements, dim| {
            elements
                .checked_mul(*dim)
                .ok_or_else(|| candle_core::Error::msg("recurrent state snapshot size overflow"))
        })?;
        let conv_bytes = conv_elements
            .checked_mul(self.conv_dtype.size_in_bytes())
            .ok_or_else(|| {
                candle_core::Error::msg("recurrent convolution snapshot size overflow")
            })?;
        let recurrent_bytes = recurrent_elements
            .checked_mul(self.recurrent_dtype.size_in_bytes())
            .ok_or_else(|| candle_core::Error::msg("recurrent state snapshot size overflow"))?;
        conv_bytes
            .checked_add(recurrent_bytes)
            .ok_or_else(|| candle_core::Error::msg("recurrent snapshot size overflow"))
    }

    pub fn state_layout(&self) -> RecurrentStateLayout {
        self.state_layout
    }
}

impl Clone for RecurrentStatePool {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            allocated_slots: self.allocated_slots.clone(),
            allocated_count: self.allocated_count,
            capacity: self.capacity,
            checkpoint_lanes: self.checkpoint_lanes,
            conv_dim: self.conv_dim,
            conv_width: self.conv_width,
            state_dims: self.state_dims.clone(),
            state_layout: self.state_layout,
            conv_dtype: self.conv_dtype,
            recurrent_dtype: self.recurrent_dtype,
            device: self.device.clone(),
        }
    }
}

/// Per-layer cache that can be either attention (KV) or recurrent (state pool)
#[derive(Clone, Debug)]
pub enum HybridLayerCache {
    Attention(KvCache),
    Recurrent(RecurrentStatePool),
}

impl HybridLayerCache {
    pub fn as_kv_cache(&self) -> Option<&KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Recurrent(_) => None,
        }
    }

    pub fn as_kv_cache_mut(&mut self) -> Option<&mut KvCache> {
        match self {
            Self::Attention(kv) => Some(kv),
            Self::Recurrent(_) => None,
        }
    }

    pub fn as_recurrent_pool(&self) -> Option<&RecurrentStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Recurrent(pool) => Some(pool),
        }
    }

    pub fn as_recurrent_pool_mut(&mut self) -> Option<&mut RecurrentStatePool> {
        match self {
            Self::Attention(_) => None,
            Self::Recurrent(pool) => Some(pool),
        }
    }
}

/// Layer type indicator for hybrid models
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HybridLayerType {
    Attention,
    Recurrent,
}

/// Configuration for the recurrent layer state dimensions
#[derive(Clone, Debug)]
pub struct RecurrentLayerConfig {
    /// Dimension of the convolution state
    pub conv_dim: usize,
    /// Kernel size for causal conv1d
    pub conv_width: usize,
    pub state: RecurrentStateSpec,
    pub recurrent_dtype: Option<DType>,
}

/// Configuration for creating a hybrid cache
#[derive(Clone, Debug)]
pub struct HybridCacheConfig {
    pub layer_types: Vec<HybridLayerType>,
    pub max_seq_len: usize,
    pub recurrent: RecurrentLayerConfig,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecurrentBatchMapping {
    pub logical_slots: Vec<u32>,
    pub physical_slots: Vec<u32>,
}

/// Hybrid cache that stores per-layer caches for mixed attention/recurrent models
///
/// For continuous batching:
/// - Attention layers use standard KV cache with batching support
/// - Recurrent layers use RecurrentStatePool with indexed access via state_indices
#[derive(Clone, Debug)]
pub struct HybridCache {
    pub caches: Vec<HybridLayerCache>,
    config: HybridCacheConfig,
    /// Current batch's state indices for recurrent pool access.
    /// Set by clone_in_cache before forward, used by model during forward.
    /// Shape: (batch_size,) containing pool slot indices.
    state_indices: Option<Tensor>,
    state_indices_host: Option<Vec<u32>>,
    logical_state_indices_host: Option<Vec<u32>>,
    device_state_indices: Vec<(Device, Tensor)>,
    checkpoint_lanes: usize,
    committed_lanes: Vec<usize>,
    slot_owners: Vec<Option<RecurrentSlotOwner>>,
    initialized_slots: Vec<bool>,
    last_released_sequence_owners: Vec<Option<usize>>,
    recurrent_storage_generation: u64,
    recurrent_storage_locked: bool,
    // Scratch slot CUDA graph pad rows write into; allocated on first use, dropped on reset
    graph_pad_slot: Option<usize>,
}

impl HybridCache {
    pub const CACHE_GROW_SIZE: usize = 512;

    pub fn new(
        config: HybridCacheConfig,
        dtype: candle_core::DType,
        layer_devices: &[Device],
    ) -> Result<Self> {
        if layer_devices.len() != config.layer_types.len() {
            candle_core::bail!(
                "Hybrid cache has {} layers but {} layer devices",
                config.layer_types.len(),
                layer_devices.len()
            );
        }
        let mut caches = Vec::with_capacity(config.layer_types.len());

        for (layer_type, device) in config.layer_types.iter().zip(layer_devices) {
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Recurrent => {
                    let (state_dims, state_layout) =
                        config.recurrent.state.physical_layout(dtype, device)?;
                    HybridLayerCache::Recurrent(RecurrentStatePool::new(
                        RecurrentStatePoolConfig {
                            conv_dim: config.recurrent.conv_dim,
                            conv_width: config.recurrent.conv_width,
                            state_dims,
                            state_layout,
                            conv_dtype: dtype,
                            recurrent_dtype: config.recurrent.recurrent_dtype.unwrap_or(dtype),
                            device,
                        },
                    )?)
                }
            };
            caches.push(cache);
        }

        let cache = Self {
            caches,
            config,
            state_indices: None,
            state_indices_host: None,
            logical_state_indices_host: None,
            device_state_indices: Vec::new(),
            checkpoint_lanes: 1,
            committed_lanes: vec![0; INITIAL_POOL_CAPACITY],
            slot_owners: vec![None; INITIAL_POOL_CAPACITY],
            initialized_slots: vec![false; INITIAL_POOL_CAPACITY],
            last_released_sequence_owners: vec![None; INITIAL_POOL_CAPACITY],
            recurrent_storage_generation: 0,
            recurrent_storage_locked: false,
            graph_pad_slot: None,
        };
        cache.publish_recurrent_slot_metrics();
        Ok(cache)
    }

    /// Slot reserved for CUDA graph pad rows; never handed to a sequence while it lives.
    pub fn graph_pad_slot(&mut self) -> Result<Option<usize>> {
        if self.graph_pad_slot.is_none() {
            self.graph_pad_slot = Some(self.allocate_slot(
                RecurrentSlotOwner::GraphPad,
                RecurrentSlotInitialization::Zeroed,
            )?);
        }
        if let Some(slot_idx) = self.graph_pad_slot {
            self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::GraphPad)?;
        }
        Ok(self.graph_pad_slot)
    }

    pub fn release_graph_pad_slot(&mut self) -> Result<bool> {
        let Some(slot_idx) = self.graph_pad_slot else {
            return Ok(false);
        };
        self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::GraphPad)?;
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let released = pool.free(slot_idx);
                debug_assert!(released);
            }
        }
        self.slot_owners[slot_idx] = None;
        self.initialized_slots[slot_idx] = false;
        self.last_released_sequence_owners[slot_idx] = None;
        self.committed_lanes[slot_idx] = 0;
        self.graph_pad_slot = None;
        if self
            .logical_state_indices_host
            .as_ref()
            .is_some_and(|slots| {
                slots
                    .iter()
                    .any(|&logical_slot| logical_slot as usize == slot_idx)
            })
        {
            self.clear_state_indices();
        }
        self.publish_recurrent_slot_metrics();
        Ok(true)
    }

    pub fn configure_checkpoint_lanes(&mut self, checkpoint_lanes: usize) -> Result<bool> {
        if checkpoint_lanes == 0 {
            candle_core::bail!("recurrent checkpoint lane count must be nonzero");
        }
        if checkpoint_lanes == self.checkpoint_lanes {
            return Ok(false);
        }
        if self.recurrent_storage_locked {
            candle_core::bail!(
                "recurrent checkpoint lanes must be configured before reservation or allocation"
            );
        }
        if self.slot_owners.iter().any(Option::is_some) {
            candle_core::bail!(
                "recurrent checkpoint lanes cannot change while sequence slots are allocated"
            );
        }

        let storage = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.checkpoint_storage(checkpoint_lanes))
            .collect::<Result<Vec<_>>>()?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let (conv_state, recurrent_state) = storage
                    .next()
                    .expect("one checkpoint allocation per recurrent pool");
                pool.install_checkpoint_storage(checkpoint_lanes, conv_state, recurrent_state);
            }
        }
        self.checkpoint_lanes = checkpoint_lanes;
        self.committed_lanes.fill(0);
        self.initialized_slots.fill(false);
        self.clear_state_indices();
        self.advance_recurrent_storage_generation();
        Ok(true)
    }

    pub fn recurrent_storage_generation(&self) -> u64 {
        self.recurrent_storage_generation
    }

    fn advance_recurrent_storage_generation(&mut self) {
        self.recurrent_storage_generation = self
            .recurrent_storage_generation
            .checked_add(1)
            .expect("recurrent storage generation overflow");
    }

    pub fn checkpoint_lanes(&self) -> usize {
        self.checkpoint_lanes
    }

    pub fn committed_lane(&self, logical_slot: usize) -> Result<usize> {
        self.committed_lanes
            .get(logical_slot)
            .copied()
            .ok_or_else(|| {
                candle_core::Error::msg(format!(
                    "recurrent logical slot {logical_slot} exceeds capacity {}",
                    self.committed_lanes.len()
                ))
            })
    }

    pub fn physical_slot(&self, logical_slot: usize, lane: usize) -> Result<usize> {
        if logical_slot >= self.recurrent_capacity() {
            candle_core::bail!(
                "recurrent logical slot {logical_slot} exceeds capacity {}",
                self.recurrent_capacity()
            );
        }
        if lane >= self.checkpoint_lanes {
            candle_core::bail!(
                "recurrent checkpoint lane {lane} exceeds lane count {}",
                self.checkpoint_lanes
            );
        }
        logical_slot
            .checked_mul(self.checkpoint_lanes)
            .and_then(|base| base.checked_add(lane))
            .ok_or_else(|| candle_core::Error::msg("recurrent physical slot overflow"))
    }

    pub fn active_physical_slot(&self, logical_slot: usize) -> Result<usize> {
        self.physical_slot(logical_slot, self.committed_lane(logical_slot)?)
    }

    pub fn map_recurrent_batch(&self, logical_slots: &[u32]) -> Result<RecurrentBatchMapping> {
        self.map_recurrent_batch_with_lanes(logical_slots, &self.committed_lanes)
    }

    fn map_recurrent_batch_with_lanes(
        &self,
        logical_slots: &[u32],
        committed_lanes: &[usize],
    ) -> Result<RecurrentBatchMapping> {
        let physical_slots = logical_slots
            .iter()
            .map(|&logical_slot| {
                let logical_slot = logical_slot as usize;
                let lane = committed_lanes.get(logical_slot).copied().ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "recurrent logical slot {logical_slot} exceeds capacity {}",
                        committed_lanes.len()
                    ))
                })?;
                let physical_slot = self.physical_slot(logical_slot, lane)?;
                u32::try_from(physical_slot).map_err(|_| {
                    candle_core::Error::msg(format!(
                        "recurrent physical slot {physical_slot} exceeds u32"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RecurrentBatchMapping {
            logical_slots: logical_slots.to_vec(),
            physical_slots,
        })
    }

    pub fn current_recurrent_batch_mapping(&self) -> Option<RecurrentBatchMapping> {
        Some(RecurrentBatchMapping {
            logical_slots: self.logical_state_indices_host.clone()?,
            physical_slots: self.state_indices_host.clone()?,
        })
    }

    pub fn recurrent_devices(&self) -> Vec<Device> {
        self.caches
            .iter()
            .filter_map(|cache| cache.as_recurrent_pool().map(|pool| pool.device().clone()))
            .fold(Vec::new(), |mut devices, device| {
                if !devices.iter().any(|d: &Device| d.same_device(&device)) {
                    devices.push(device);
                }
                devices
            })
    }

    /// Install caller-owned per-device index tensors (e.g. CUDA graph buffers) as the batch's state indices.
    pub fn set_state_indices_tensors(
        &mut self,
        host: Vec<u32>,
        mut tensors: Vec<(Device, Tensor)>,
    ) {
        self.logical_state_indices_host = Some(
            host.iter()
                .map(|&slot| logical_slot_from_physical_slot(slot, self.checkpoint_lanes))
                .collect(),
        );
        self.state_indices = (!tensors.is_empty()).then(|| tensors.remove(0).1);
        self.device_state_indices = tensors;
        self.state_indices_host = Some(host);
    }

    pub fn set_physical_state_indices_with_host(
        &mut self,
        indices: Option<Tensor>,
        physical_host: Option<Vec<u32>>,
    ) {
        self.logical_state_indices_host = physical_host.as_ref().map(|slots| {
            slots
                .iter()
                .map(|&slot| logical_slot_from_physical_slot(slot, self.checkpoint_lanes))
                .collect()
        });
        self.state_indices = indices;
        self.state_indices_host = physical_host;
        self.cache_device_state_indices();
    }

    /// Slot capacity of the recurrent pools; changes whenever the pool storage is reallocated.
    pub fn recurrent_capacity(&self) -> usize {
        if self
            .caches
            .iter()
            .any(|cache| cache.as_recurrent_pool().is_some())
        {
            self.slot_owners.len()
        } else {
            0
        }
    }

    pub(crate) fn recurrent_snapshot_bytes_by_device(
        &self,
    ) -> Result<HashMap<DeviceLocation, usize>> {
        let mut bytes_by_device: HashMap<DeviceLocation, usize> = HashMap::new();
        for pool in self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
        {
            let bytes = pool.snapshot_bytes()?;
            let entry = bytes_by_device.entry(pool.device().location()).or_default();
            *entry = (*entry).checked_add(bytes).ok_or_else(|| {
                candle_core::Error::msg("recurrent snapshot device size overflow")
            })?;
        }
        Ok(bytes_by_device)
    }

    pub fn recurrent_slots_used(&self) -> usize {
        let used = self
            .slot_owners
            .iter()
            .filter(|owner| owner.is_some())
            .count();
        debug_assert!(self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .all(|pool| pool.capacity() - pool.num_free_slots() == used));
        used
    }

    fn ensure_recurrent_slot_allocated(&self, slot_idx: usize) -> Result<()> {
        let mut pools = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool);
        let Some(first) = pools.next() else {
            candle_core::bail!("hybrid cache has no recurrent state pool");
        };
        let allocated = first.is_allocated(slot_idx);
        if pools.any(|pool| pool.is_allocated(slot_idx) != allocated) {
            candle_core::bail!(
                "hybrid recurrent pool allocation state diverged for slot {slot_idx}"
            );
        }
        if !allocated {
            candle_core::bail!("recurrent state slot {slot_idx} is not allocated");
        }
        Ok(())
    }

    fn ensure_recurrent_slot_owned(
        &self,
        slot_idx: usize,
        expected_owner: RecurrentSlotOwner,
    ) -> Result<()> {
        self.ensure_recurrent_slot_allocated(slot_idx)?;
        let owner = self.slot_owners.get(slot_idx).copied().flatten();
        if owner != Some(expected_owner) {
            candle_core::bail!(
                "recurrent state slot {slot_idx} is owned by {owner:?}, expected {expected_owner:?}"
            );
        }
        Ok(())
    }

    fn ensure_recurrent_slot_initialized(&self, slot_idx: usize) -> Result<()> {
        self.ensure_recurrent_slot_allocated(slot_idx)?;
        if !self.initialized_slots[slot_idx] {
            candle_core::bail!("recurrent state slot {slot_idx} is not initialized");
        }
        Ok(())
    }

    pub fn validate_sequence_slots(&self, sequence_slots: &[(usize, usize)]) -> Result<()> {
        let mut unique_slots = HashSet::with_capacity(sequence_slots.len());
        for &(sequence_id, slot_idx) in sequence_slots {
            if !unique_slots.insert(slot_idx) {
                candle_core::bail!(
                    "recurrent state slot {slot_idx} is assigned to multiple sequences in one batch"
                );
            }
            self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::Sequence(sequence_id))?;
            self.ensure_recurrent_slot_initialized(slot_idx)?;
        }
        Ok(())
    }

    fn publish_recurrent_slot_metrics(&self) {
        if self.recurrent_capacity() == 0 {
            return;
        }
        let slots_used = u32::try_from(self.recurrent_slots_used())
            .expect("recurrent state slot usage exceeds u32");
        let slots_total = u32::try_from(self.recurrent_capacity())
            .expect("recurrent state slot capacity exceeds u32");
        metrics::gauge!("mistralrs_recurrent_state_slots_used").set(f64::from(slots_used));
        metrics::gauge!("mistralrs_recurrent_state_slots_total").set(f64::from(slots_total));
    }

    fn resize_recurrent_storage(&mut self, min_capacity: usize) -> Result<bool> {
        let current_capacity = self.recurrent_capacity();
        if current_capacity == 0 {
            candle_core::bail!("hybrid cache has no recurrent state pool");
        }
        if min_capacity <= current_capacity {
            return Ok(false);
        }
        if self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .any(|pool| pool.capacity() != current_capacity)
        {
            candle_core::bail!("hybrid recurrent pool capacities diverged before resize");
        }

        let storage = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.resized_storage(min_capacity))
            .collect::<Result<Vec<_>>>()?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let (conv_state, recurrent_state) = storage
                    .next()
                    .expect("one resized allocation per recurrent pool");
                pool.install_resized_storage(min_capacity, conv_state, recurrent_state);
            }
        }
        self.slot_owners.resize(min_capacity, None);
        self.initialized_slots.resize(min_capacity, false);
        self.last_released_sequence_owners
            .resize(min_capacity, None);
        self.committed_lanes.resize(min_capacity, 0);
        self.advance_recurrent_storage_generation();
        tracing::info!("Recurrent state pool grew to capacity {min_capacity}");
        Ok(true)
    }

    pub(crate) fn reserve_recurrent_capacity(&mut self, min_capacity: usize) -> Result<bool> {
        let grew = self.resize_recurrent_storage(min_capacity)?;
        self.recurrent_storage_locked = true;
        self.publish_recurrent_slot_metrics();
        Ok(grew)
    }

    pub fn allocate_seq(&mut self, sequence_id: usize) -> Result<usize> {
        let slot = self.allocate_slot(
            RecurrentSlotOwner::Sequence(sequence_id),
            RecurrentSlotInitialization::Zeroed,
        );
        self.publish_recurrent_slot_metrics();
        slot
    }

    pub(crate) fn reserve_seq_uninitialized(&mut self, sequence_id: usize) -> Result<usize> {
        let slot = self.allocate_slot(
            RecurrentSlotOwner::Sequence(sequence_id),
            RecurrentSlotInitialization::Deferred,
        );
        self.publish_recurrent_slot_metrics();
        slot
    }

    fn allocate_slot(
        &mut self,
        owner: RecurrentSlotOwner,
        initialization: RecurrentSlotInitialization,
    ) -> Result<usize> {
        self.recurrent_storage_locked = true;
        if self.slot_owners.contains(&Some(owner)) {
            candle_core::bail!("recurrent slot owner {owner:?} already has an allocation");
        }
        if self.recurrent_capacity() == 0 {
            candle_core::bail!("hybrid cache has no recurrent state pool");
        }
        if !self.slot_owners.iter().any(Option::is_none) {
            let new_capacity = self
                .recurrent_capacity()
                .checked_mul(2)
                .ok_or_else(|| candle_core::Error::msg("recurrent state capacity overflow"))?;
            self.resize_recurrent_storage(new_capacity)?;
        }
        let slot_idx = self
            .slot_owners
            .iter()
            .position(Option::is_none)
            .expect("capacity growth must create a free recurrent slot");
        let recurrent_layers: Vec<usize> = self
            .caches
            .iter()
            .enumerate()
            .filter_map(|(idx, cache)| match cache {
                HybridLayerCache::Recurrent(_) => Some(idx),
                HybridLayerCache::Attention(_) => None,
            })
            .collect();
        let mut allocated_layers = Vec::with_capacity(recurrent_layers.len());
        for &layer_idx in &recurrent_layers {
            let HybridLayerCache::Recurrent(pool) = &mut self.caches[layer_idx] else {
                unreachable!("recurrent_layers only contains recurrent entries");
            };
            let allocation = match initialization {
                RecurrentSlotInitialization::Deferred => pool.reserve_at(slot_idx),
                RecurrentSlotInitialization::Zeroed => pool.allocate_at(slot_idx),
            };
            if let Err(err) = allocation {
                for rollback_layer_idx in allocated_layers {
                    let HybridLayerCache::Recurrent(pool) = &mut self.caches[rollback_layer_idx]
                    else {
                        unreachable!("recurrent_layers only contains recurrent entries");
                    };
                    let released = pool.free(slot_idx);
                    debug_assert!(released);
                }
                return Err(err);
            }
            allocated_layers.push(layer_idx);
        }
        self.slot_owners[slot_idx] = Some(owner);
        self.initialized_slots[slot_idx] = initialization == RecurrentSlotInitialization::Zeroed;
        self.last_released_sequence_owners[slot_idx] = None;
        self.committed_lanes[slot_idx] = 0;
        Ok(slot_idx)
    }

    pub fn release_seq(&mut self, sequence_id: usize, slot_idx: usize) -> Result<bool> {
        let owner = self.slot_owners.get(slot_idx).copied().flatten();
        match owner {
            Some(RecurrentSlotOwner::Sequence(owner_id)) if owner_id == sequence_id => {}
            Some(owner) => candle_core::bail!(
                "cannot release recurrent state slot {slot_idx} for sequence {sequence_id}: owned by {owner:?}"
            ),
            None
                if self
                    .last_released_sequence_owners
                    .get(slot_idx)
                    .copied()
                    .flatten()
                    == Some(sequence_id) =>
            {
                return Ok(false);
            }
            None => candle_core::bail!(
                "cannot release unowned recurrent state slot {slot_idx} for sequence {sequence_id}"
            ),
        }

        self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::Sequence(sequence_id))?;
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let released = pool.free(slot_idx);
                debug_assert!(released);
            }
        }
        self.slot_owners[slot_idx] = None;
        self.initialized_slots[slot_idx] = false;
        self.last_released_sequence_owners[slot_idx] = Some(sequence_id);
        self.committed_lanes[slot_idx] = 0;
        if self
            .logical_state_indices_host
            .as_ref()
            .is_some_and(|slots| {
                slots
                    .iter()
                    .any(|&logical_slot| logical_slot as usize == slot_idx)
            })
        {
            self.clear_state_indices();
        }
        self.publish_recurrent_slot_metrics();
        Ok(true)
    }

    /// Reset a specific sequence's state in all recurrent layers.
    pub fn reset_seq(&mut self, sequence_id: usize, slot_idx: usize) -> Result<()> {
        self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::Sequence(sequence_id))?;
        self.initialized_slots[slot_idx] = false;
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.reset_slot(slot_idx)?;
            }
        }
        let capacity = self.committed_lanes.len();
        *self.committed_lanes.get_mut(slot_idx).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "recurrent logical slot {slot_idx} exceeds capacity {capacity}"
            ))
        })? = 0;
        self.initialized_slots[slot_idx] = true;
        self.refresh_current_batch_mapping()
    }

    pub fn reset(&mut self) -> Result<()> {
        if self.graph_pad_slot.is_some() {
            candle_core::bail!(
                "cannot reset recurrent storage while CUDA graph storage is registered"
            );
        }
        if self
            .slot_owners
            .iter()
            .any(|owner| matches!(owner, Some(RecurrentSlotOwner::Sequence(_))))
        {
            candle_core::bail!("cannot reset recurrent storage while sequence slots are allocated");
        }
        let storage = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.checkpoint_storage(self.checkpoint_lanes))
            .collect::<Result<Vec<_>>>()?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            match cache {
                HybridLayerCache::Attention(kv) => kv.reset(),
                HybridLayerCache::Recurrent(pool) => {
                    let (conv_state, recurrent_state) = storage
                        .next()
                        .expect("one reset allocation per recurrent pool");
                    pool.install_checkpoint_storage(
                        self.checkpoint_lanes,
                        conv_state,
                        recurrent_state,
                    );
                    pool.allocated_slots.fill(false);
                    pool.allocated_count = 0;
                }
            }
        }
        self.committed_lanes.fill(0);
        self.slot_owners.fill(None);
        self.initialized_slots.fill(false);
        self.last_released_sequence_owners.fill(None);
        self.clear_state_indices();
        self.graph_pad_slot = None;
        if self.recurrent_capacity() > 0 {
            self.advance_recurrent_storage_generation();
        }
        self.publish_recurrent_slot_metrics();
        Ok(())
    }

    /// Reset the attention caches and only the given recurrent slots; other sequences keep their state.
    pub fn reset_attention_and_slots(&mut self, slots: &[usize]) -> Result<()> {
        for &slot in slots {
            self.ensure_recurrent_slot_allocated(slot)?;
        }
        for &slot in slots {
            self.initialized_slots[slot] = false;
        }
        let capacity = self.committed_lanes.len();
        for cache in &mut self.caches {
            match cache {
                HybridLayerCache::Attention(kv) => kv.reset(),
                HybridLayerCache::Recurrent(pool) => {
                    for slot in slots {
                        pool.reset_slot(*slot)?;
                        *self.committed_lanes.get_mut(*slot).ok_or_else(|| {
                            candle_core::Error::msg(format!(
                                "recurrent logical slot {slot} exceeds capacity {capacity}"
                            ))
                        })? = 0;
                    }
                }
            }
        }
        for &slot in slots {
            self.initialized_slots[slot] = true;
        }
        self.clear_state_indices();
        Ok(())
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
    pub fn set_state_indices(&mut self, indices: Option<Tensor>) -> Result<()> {
        if self.checkpoint_lanes == 1 || indices.is_none() {
            self.state_indices = indices;
            self.state_indices_host = None;
            self.logical_state_indices_host = None;
            self.cache_device_state_indices();
            return Ok(());
        }
        let indices = indices.expect("checked above");
        let device = indices.device().clone();
        indices.to_vec1::<u32>().and_then(|logical| {
            let mapping = self.map_recurrent_batch(&logical)?;
            self.install_batch_mapping(mapping, Some(device))
        })
    }

    pub fn set_state_indices_with_host(
        &mut self,
        indices: Option<Tensor>,
        host_indices: Option<Vec<u32>>,
    ) -> Result<()> {
        let Some(logical_slots) = host_indices else {
            return self.set_state_indices(indices);
        };
        if self.checkpoint_lanes == 1 {
            self.state_indices = indices;
            self.state_indices_host = Some(logical_slots.clone());
            self.logical_state_indices_host = Some(logical_slots);
            self.cache_device_state_indices();
            return Ok(());
        }
        let preferred_device = indices.as_ref().map(|indices| indices.device().clone());
        self.map_recurrent_batch(&logical_slots)
            .and_then(|mapping| self.install_batch_mapping(mapping, preferred_device))
    }

    pub fn install_sequence_state_indices(
        &mut self,
        sequence_slots: &[(usize, usize)],
    ) -> Result<()> {
        self.validate_sequence_slots(sequence_slots)?;
        let logical_slots = sequence_slots
            .iter()
            .map(|&(_, slot_idx)| {
                u32::try_from(slot_idx).map_err(|_| {
                    candle_core::Error::msg(format!(
                        "recurrent logical slot {slot_idx} exceeds u32"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mapping = self.map_recurrent_batch(&logical_slots)?;
        let device = self.recurrent_devices().into_iter().next();
        self.install_batch_mapping(mapping, device)
    }

    pub fn commit_speculative_rows(&mut self, rows: &[(usize, usize)]) -> Result<bool> {
        if self.checkpoint_lanes == 1 {
            return Ok(false);
        }
        let logical_slots = self.logical_state_indices_host.clone().ok_or_else(|| {
            candle_core::Error::msg("recurrent batch has no logical slot mapping")
        })?;
        let updates = rows
            .iter()
            .map(|&(batch_idx, keep_rows)| {
                if keep_rows == 0 || keep_rows > self.checkpoint_lanes {
                    candle_core::bail!(
                        "recurrent keep row count {keep_rows} is outside 1..={}",
                        self.checkpoint_lanes
                    );
                }
                let logical_slot = *logical_slots.get(batch_idx).ok_or_else(|| {
                    candle_core::Error::msg(format!(
                        "recurrent batch row {batch_idx} exceeds batch size {}",
                        logical_slots.len()
                    ))
                })? as usize;
                if logical_slot >= self.committed_lanes.len() {
                    candle_core::bail!(
                        "recurrent logical slot {logical_slot} exceeds capacity {}",
                        self.committed_lanes.len()
                    );
                }
                Ok((logical_slot, keep_rows - 1))
            })
            .collect::<Result<Vec<_>>>()?;
        let mut committed_lanes = self.committed_lanes.clone();
        for (logical_slot, lane) in updates {
            committed_lanes[logical_slot] = lane;
        }
        let preferred_device = self
            .state_indices
            .as_ref()
            .map(|indices| indices.device().clone());
        let mapping = self.map_recurrent_batch_with_lanes(&logical_slots, &committed_lanes)?;
        self.install_batch_mapping(mapping, preferred_device)?;
        self.committed_lanes = committed_lanes;
        Ok(true)
    }

    fn install_batch_mapping(
        &mut self,
        mapping: RecurrentBatchMapping,
        preferred_device: Option<Device>,
    ) -> Result<()> {
        let device = preferred_device.or_else(|| self.recurrent_devices().into_iter().next());
        self.state_indices = match device {
            Some(device) => Some(Tensor::from_vec(
                mapping.physical_slots.clone(),
                (mapping.physical_slots.len(),),
                &device,
            )?),
            None => None,
        };
        self.logical_state_indices_host = Some(mapping.logical_slots);
        self.state_indices_host = Some(mapping.physical_slots);
        self.cache_device_state_indices();
        Ok(())
    }

    fn refresh_current_batch_mapping(&mut self) -> Result<()> {
        let Some(logical_slots) = self.logical_state_indices_host.clone() else {
            return Ok(());
        };
        let preferred_device = self
            .state_indices
            .as_ref()
            .map(|indices| indices.device().clone());
        let mapping = self.map_recurrent_batch(&logical_slots)?;
        self.install_batch_mapping(mapping, preferred_device)
    }

    fn clear_state_indices(&mut self) {
        self.state_indices = None;
        self.state_indices_host = None;
        self.logical_state_indices_host = None;
        self.device_state_indices.clear();
    }

    fn cache_device_state_indices(&mut self) {
        self.device_state_indices.clear();
        let devices = self
            .caches
            .iter()
            .filter_map(|cache| match cache {
                HybridLayerCache::Recurrent(pool) => Some(pool.device().clone()),
                HybridLayerCache::Attention(_) => None,
            })
            .fold(Vec::<Device>::new(), |mut devices, device| {
                if !devices
                    .iter()
                    .any(|cached_device| cached_device.same_device(&device))
                {
                    devices.push(device);
                }
                devices
            });
        for device in devices {
            if self
                .state_indices
                .as_ref()
                .is_some_and(|indices| indices.device().same_device(&device))
            {
                continue;
            }
            let indices = if let Some(host_indices) = &self.state_indices_host {
                Tensor::from_vec(host_indices.clone(), (host_indices.len(),), &device)
            } else if let Some(indices) = &self.state_indices {
                indices.to_device(&device)
            } else {
                continue;
            };
            if let Ok(indices) = indices {
                self.device_state_indices.push((device, indices));
            }
        }
    }

    /// Get the state indices for the current batch.
    /// Used by the model during forward to access recurrent state pool.
    pub fn state_indices(&self) -> Option<&Tensor> {
        self.state_indices.as_ref()
    }

    pub fn state_indices_host(&self) -> Option<&[u32]> {
        self.state_indices_host.as_deref()
    }

    pub fn logical_state_indices_host(&self) -> Option<&[u32]> {
        self.logical_state_indices_host.as_deref()
    }

    pub fn state_indices_for_layer(&mut self, layer: usize) -> Result<Option<Tensor>> {
        let device = match self.caches.get(layer) {
            Some(HybridLayerCache::Recurrent(pool)) => pool.device().clone(),
            _ => return Ok(None),
        };
        if let Some(indices) = &self.state_indices {
            if indices.device().same_device(&device) {
                return Ok(Some(indices.clone()));
            }
        }
        if let Some((_, indices)) = self
            .device_state_indices
            .iter()
            .find(|(cached_device, _)| cached_device.same_device(&device))
        {
            return Ok(Some(indices.clone()));
        }
        let indices = if let Some(host_indices) = &self.state_indices_host {
            Tensor::from_vec(host_indices.clone(), (host_indices.len(),), &device)?
        } else if let Some(indices) = &self.state_indices {
            indices.to_device(&device)?
        } else {
            return Ok(None);
        };
        self.device_state_indices.push((device, indices.clone()));
        Ok(Some(indices))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    fn config(layer_types: Vec<HybridLayerType>) -> HybridCacheConfig {
        HybridCacheConfig {
            layer_types,
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 2,
                conv_width: 3,
                state: RecurrentStateSpec::Opaque { dims: vec![2, 2] },
                recurrent_dtype: None,
            },
        }
    }

    fn gdn_config(layer_types: Vec<HybridLayerType>) -> HybridCacheConfig {
        HybridCacheConfig {
            layer_types,
            max_seq_len: 32,
            recurrent: RecurrentLayerConfig {
                conv_dim: 2,
                conv_width: 3,
                state: RecurrentStateSpec::Gdn {
                    heads: 3,
                    key_dim: 4,
                    value_dim: 5,
                },
                recurrent_dtype: Some(DType::F32),
            },
        }
    }

    #[test]
    fn requires_one_device_per_layer() {
        let error = HybridCache::new(
            config(vec![HybridLayerType::Attention, HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )
        .unwrap_err();
        assert!(error.to_string().contains("1 layer devices"));
    }

    #[test]
    fn recurrent_snapshot_bytes_use_one_physical_row_per_layer() -> Result<()> {
        let cache = HybridCache::new(
            config(vec![
                HybridLayerType::Recurrent,
                HybridLayerType::Attention,
                HybridLayerType::Recurrent,
            ]),
            DType::F32,
            &[Device::Cpu, Device::Cpu, Device::Cpu],
        )?;

        let bytes = cache.recurrent_snapshot_bytes_by_device()?;
        assert_eq!(bytes.len(), 1);
        assert_eq!(bytes[&DeviceLocation::Cpu], 80);
        Ok(())
    }

    #[test]
    fn recurrent_indices_are_local_to_the_layer_pool() -> Result<()> {
        let devices = vec![Device::Cpu, Device::Cpu, Device::Cpu];
        let mut cache = HybridCache::new(
            config(vec![
                HybridLayerType::Recurrent,
                HybridLayerType::Attention,
                HybridLayerType::Recurrent,
            ]),
            DType::F32,
            &devices,
        )?;
        let state_indices = Tensor::from_vec(vec![1u32, 3], (2,), &Device::Cpu)?;
        cache.set_state_indices_with_host(Some(state_indices), Some(vec![1, 3]))?;

        assert!(cache.state_indices_for_layer(1)?.is_none());
        for layer in [0, 2] {
            let indices = cache.state_indices_for_layer(layer)?.unwrap();
            let HybridLayerCache::Recurrent(pool) = cache.get(layer).unwrap() else {
                unreachable!()
            };
            assert!(indices.device().same_device(pool.device()));
            assert_eq!(indices.to_vec1::<u32>()?, vec![1, 3]);
        }
        Ok(())
    }

    #[test]
    fn recurrent_capacity_can_be_reserved_before_admission() -> Result<()> {
        let devices = vec![Device::Cpu, Device::Cpu];
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &devices,
        )?;

        assert!(cache.reserve_recurrent_capacity(17)?);
        assert_eq!(cache.recurrent_capacity(), 17);
        assert!(!cache.reserve_recurrent_capacity(16)?);

        let slots = (0..17)
            .map(|sequence_id| cache.allocate_seq(sequence_id).unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(slots.len(), 17);
        assert_eq!(cache.recurrent_capacity(), 17);
        Ok(())
    }

    #[test]
    fn recurrent_slot_accounting_tracks_all_lifecycle_transitions() -> Result<()> {
        let devices = vec![Device::Cpu, Device::Cpu];
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &devices,
        )?;

        assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
        assert_eq!(cache.recurrent_slots_used(), 0);

        let first = cache.allocate_seq(10)?;
        let second = cache.allocate_seq(20)?;
        assert_eq!(cache.recurrent_slots_used(), 2);

        assert!(cache.release_seq(10, first)?);
        assert_eq!(cache.recurrent_slots_used(), 1);
        assert!(cache.reserve_recurrent_capacity(17)?);
        assert_eq!(cache.recurrent_capacity(), 17);
        assert_eq!(cache.recurrent_slots_used(), 1);

        assert!(cache.graph_pad_slot()?.is_some());
        assert_eq!(cache.recurrent_slots_used(), 2);
        assert!(cache.release_seq(20, second)?);
        assert_eq!(cache.recurrent_slots_used(), 1);

        assert!(cache.release_graph_pad_slot()?);
        cache.reset()?;
        assert_eq!(cache.recurrent_capacity(), 17);
        assert_eq!(cache.recurrent_slots_used(), 0);
        Ok(())
    }

    #[test]
    fn failed_slot_reset_rolls_back_every_recurrent_layer() -> Result<()> {
        let devices = vec![Device::Cpu, Device::Cpu];
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &devices,
        )?;
        let original_conv_state = {
            let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
                unreachable!()
            };
            let original = pool.conv_state.clone();
            pool.conv_state =
                Tensor::zeros((INITIAL_POOL_CAPACITY, 1, 1), DType::F32, &Device::Cpu)?;
            original
        };

        let error = cache.allocate_seq(10).unwrap_err();
        assert!(error
            .to_string()
            .contains("failed to reset recurrent state slot"));
        assert_eq!(cache.recurrent_slots_used(), 0);
        for layer in [0, 1] {
            let HybridLayerCache::Recurrent(pool) = cache.get(layer).unwrap() else {
                unreachable!()
            };
            assert_eq!(pool.num_free_slots(), pool.capacity());
            assert!(!pool.is_allocated(0));
        }

        let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
            unreachable!()
        };
        pool.conv_state = original_conv_state;
        assert_eq!(cache.allocate_seq(10)?, 0);
        assert_eq!(cache.recurrent_slots_used(), 1);
        Ok(())
    }

    #[test]
    fn duplicate_free_cannot_duplicate_or_alias_a_slot() -> Result<()> {
        let devices = vec![Device::Cpu, Device::Cpu];
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &devices,
        )?;

        let slot = cache.allocate_seq(10)?;
        assert!(cache.release_seq(10, slot)?);
        assert!(!cache.release_seq(10, slot)?);
        assert_eq!(cache.recurrent_slots_used(), 0);

        let allocated = (0..cache.recurrent_capacity())
            .map(|sequence_id| cache.allocate_seq(sequence_id + 100).unwrap())
            .collect::<HashSet<_>>();
        assert_eq!(allocated.len(), cache.recurrent_capacity());
        assert!(allocated.contains(&slot));
        Ok(())
    }

    #[test]
    fn checkpoint_lanes_expand_physical_storage_before_reservation() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;

        assert!(cache.configure_checkpoint_lanes(3)?);
        assert!(!cache.configure_checkpoint_lanes(3)?);
        assert_eq!(cache.checkpoint_lanes(), 3);
        assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        assert_eq!(pool.checkpoint_lanes(), 3);
        assert_eq!(pool.physical_capacity(), INITIAL_POOL_CAPACITY * 3);
        assert_eq!(pool.conv_state.dims(), &[27, 2, 3]);
        assert_eq!(pool.recurrent_state.dims(), &[27, 2, 2]);

        assert!(cache.reserve_recurrent_capacity(17)?);
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        assert_eq!(pool.conv_state.dims(), &[51, 2, 3]);
        assert_eq!(pool.recurrent_state.dims(), &[51, 2, 2]);
        assert!(cache.configure_checkpoint_lanes(4).is_err());
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn checkpoint_batch_maps_logical_slots_and_commits_keep_rows() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        cache.configure_checkpoint_lanes(4)?;
        let first = cache.allocate_seq(10)?;
        let second = cache.allocate_seq(20)?;
        assert_eq!((first, second), (0, 1));

        let logical = vec![second as u32, first as u32];
        let indices = Tensor::from_vec(logical.clone(), (2,), &Device::Cpu)?;
        cache.set_state_indices_with_host(Some(indices), Some(logical.clone()))?;
        assert_eq!(cache.logical_state_indices_host(), Some(logical.as_slice()));
        assert_eq!(cache.state_indices_host(), Some([4u32, 0].as_slice()));
        assert_eq!(cache.state_indices().unwrap().to_vec1::<u32>()?, vec![4, 0]);

        assert!(cache.commit_speculative_rows(&[(0, 3), (1, 2)])?);
        assert_eq!(cache.committed_lane(second)?, 2);
        assert_eq!(cache.committed_lane(first)?, 1);
        assert_eq!(cache.state_indices_host(), Some([6u32, 1].as_slice()));
        assert_eq!(
            cache.current_recurrent_batch_mapping(),
            Some(RecurrentBatchMapping {
                logical_slots: logical,
                physical_slots: vec![6, 1],
            })
        );

        assert!(cache.commit_speculative_rows(&[(0, 0)]).is_err());
        assert!(cache.commit_speculative_rows(&[(2, 1)]).is_err());
        assert_eq!(cache.state_indices_host(), Some([6u32, 1].as_slice()));
        Ok(())
    }

    #[test]
    fn lane_one_declines_speculative_commit() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        let indices = Tensor::from_vec(vec![0u32], (1,), &Device::Cpu)?;
        cache.set_state_indices_with_host(Some(indices), Some(vec![0]))?;
        assert!(!cache.commit_speculative_rows(&[(0, 1)])?);
        assert_eq!(cache.state_indices_host(), Some([0u32].as_slice()));
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn checkpoint_snapshot_uses_active_lane_and_restore_canonicalizes() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        cache.configure_checkpoint_lanes(3)?;
        let source = cache.allocate_seq(10)?;
        let destination = cache.allocate_seq(20)?;
        let indices = Tensor::from_vec(vec![source as u32], (1,), &Device::Cpu)?;
        cache.set_state_indices_with_host(Some(indices), Some(vec![source as u32]))?;
        cache.commit_speculative_rows(&[(0, 3)])?;

        let source_physical = cache.active_physical_slot(source)?;
        let destination_base = cache.physical_slot(destination, 0)?;
        let conv = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], (1, 2, 3), &Device::Cpu)?;
        let recurrent = Tensor::from_vec(vec![7f32, 8., 9., 10.], (1, 2, 2), &Device::Cpu)?;
        {
            let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
                unreachable!()
            };
            let source_idx = Tensor::from_vec(vec![source_physical as u32], (1,), &Device::Cpu)?;
            pool.scatter_conv_state(&source_idx, &conv)?;
            pool.scatter_recurrent_state(&source_idx, &recurrent)?;

            let destination_indices = Tensor::from_vec(
                (destination_base..destination_base + 3)
                    .map(|idx| idx as u32)
                    .collect::<Vec<_>>(),
                (3,),
                &Device::Cpu,
            )?;
            pool.scatter_conv_state(
                &destination_indices,
                &Tensor::ones((3, 2, 3), DType::F32, &Device::Cpu)?,
            )?;
            pool.scatter_recurrent_state(
                &destination_indices,
                &Tensor::ones((3, 2, 2), DType::F32, &Device::Cpu)?,
            )?;
        }

        let snapshots = cache.snapshot_recurrent_state(10, source)?;
        assert_eq!(
            snapshots[0].conv_state.to_vec3::<f32>()?,
            conv.to_vec3::<f32>()?
        );
        assert_eq!(
            snapshots[0].recurrent_state.to_vec3::<f32>()?,
            recurrent.to_vec3::<f32>()?
        );

        assert!(cache.release_seq(20, destination)?);
        assert_eq!(cache.reserve_seq_uninitialized(30)?, destination);
        cache.restore_recurrent_state(30, destination, &snapshots)?;
        cache.validate_sequence_slots(&[(30, destination)])?;
        assert_eq!(cache.committed_lane(destination)?, 0);
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        let restored_indices = Tensor::from_vec(
            (destination_base..destination_base + 3)
                .map(|idx| idx as u32)
                .collect::<Vec<_>>(),
            (3,),
            &Device::Cpu,
        )?;
        let restored_conv = pool.gather_conv_state(&restored_indices)?;
        let restored_recurrent = pool.gather_recurrent_state(&restored_indices)?;
        assert_eq!(
            restored_conv.i(0)?.to_vec2::<f32>()?,
            conv.i(0)?.to_vec2::<f32>()?
        );
        assert_eq!(
            restored_recurrent.i(0)?.to_vec2::<f32>()?,
            recurrent.i(0)?.to_vec2::<f32>()?
        );
        assert_eq!(
            restored_conv
                .narrow(0, 1, 2)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        assert_eq!(
            restored_recurrent
                .narrow(0, 1, 2)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn deferred_recurrent_reservation_requires_initialization_without_resetting() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        let slot = cache.allocate_seq(1)?;
        let physical_slot = cache.active_physical_slot(slot)?;
        let indices = Tensor::from_vec(vec![physical_slot as u32], (1,), &Device::Cpu)?;
        {
            let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
                unreachable!()
            };
            pool.scatter_conv_state(
                &indices,
                &Tensor::ones((1, 2, 3), DType::F32, &Device::Cpu)?,
            )?;
            pool.scatter_recurrent_state(
                &indices,
                &Tensor::ones((1, 2, 2), DType::F32, &Device::Cpu)?,
            )?;
        }
        assert!(cache.release_seq(1, slot)?);
        assert_eq!(cache.reserve_seq_uninitialized(10)?, slot);

        let error = cache.validate_sequence_slots(&[(10, slot)]).unwrap_err();
        assert!(error.to_string().contains("is not initialized"));
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        assert_eq!(
            pool.gather_conv_state(&indices)?
                .sum_all()?
                .to_scalar::<f32>()?,
            6.0
        );
        assert_eq!(
            pool.gather_recurrent_state(&indices)?
                .sum_all()?
                .to_scalar::<f32>()?,
            4.0
        );

        cache.reset_seq(10, slot)?;
        cache.validate_sequence_slots(&[(10, slot)])?;
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        assert_eq!(
            pool.gather_conv_state(&indices)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        assert_eq!(
            pool.gather_recurrent_state(&indices)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }

    #[test]
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn graph_snapshot_preserves_active_checkpoint_lane() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        cache.configure_checkpoint_lanes(3)?;
        let slot = cache.allocate_seq(10)?;
        let indices = Tensor::from_vec(vec![slot as u32], (1,), &Device::Cpu)?;
        cache.set_state_indices_with_host(Some(indices), Some(vec![slot as u32]))?;
        cache.commit_speculative_rows(&[(0, 3)])?;

        let base = cache.physical_slot(slot, 0)?;
        let physical_indices = Tensor::from_vec(
            (base..base + 3).map(|idx| idx as u32).collect::<Vec<_>>(),
            (3,),
            &Device::Cpu,
        )?;
        let conv = Tensor::from_vec(
            (1..=18).map(|value| value as f32).collect::<Vec<_>>(),
            (3, 2, 3),
            &Device::Cpu,
        )?;
        let recurrent = Tensor::from_vec(
            (21..=32).map(|value| value as f32).collect::<Vec<_>>(),
            (3, 2, 2),
            &Device::Cpu,
        )?;
        {
            let HybridLayerCache::Recurrent(pool) = cache.get_mut(0).unwrap() else {
                unreachable!()
            };
            pool.scatter_conv_state(&physical_indices, &conv)?;
            pool.scatter_recurrent_state(&physical_indices, &recurrent)?;
        }

        let snapshot = cache.snapshot_recurrent_checkpoint_state(slot)?;
        cache.reset_seq(10, slot)?;
        cache.restore_recurrent_checkpoint_state(slot, &snapshot)?;

        assert_eq!(cache.committed_lane(slot)?, 2);
        assert_eq!(cache.state_indices_host(), Some([2u32].as_slice()));
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        let restored_conv = pool.gather_conv_state(&physical_indices)?;
        let restored_recurrent = pool.gather_recurrent_state(&physical_indices)?;
        assert_eq!(
            restored_conv.i(2)?.to_vec2::<f32>()?,
            conv.i(2)?.to_vec2::<f32>()?
        );
        assert_eq!(
            restored_recurrent.i(2)?.to_vec2::<f32>()?,
            recurrent.i(2)?.to_vec2::<f32>()?
        );
        assert_eq!(
            restored_conv
                .narrow(0, 0, 2)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        assert_eq!(
            restored_recurrent
                .narrow(0, 0, 2)?
                .sum_all()?
                .to_scalar::<f32>()?,
            0.0
        );
        Ok(())
    }

    #[test]
    fn graph_state_indices_preserve_pad_sentinel() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        cache.configure_checkpoint_lanes(3)?;
        let physical = vec![2u32, u32::MAX, 2];
        let indices = Tensor::from_vec(physical.clone(), (3,), &Device::Cpu)?;
        cache.set_state_indices_tensors(physical.clone(), vec![(Device::Cpu, indices)]);

        assert_eq!(cache.state_indices_host(), Some(physical.as_slice()));
        assert_eq!(
            cache.logical_state_indices_host(),
            Some([0u32, u32::MAX, 0].as_slice())
        );
        Ok(())
    }

    #[test]
    fn cpu_gdn_pool_is_explicitly_key_major() -> Result<()> {
        let cache = HybridCache::new(
            gdn_config(vec![HybridLayerType::Recurrent]),
            DType::BF16,
            &[Device::Cpu],
        )?;
        let HybridLayerCache::Recurrent(pool) = cache.get(0).unwrap() else {
            unreachable!()
        };
        assert_eq!(pool.state_layout(), RecurrentStateLayout::GdnKeyMajor);
        assert_eq!(pool.recurrent_state.dims(), &[9, 3, 4, 5]);
        Ok(())
    }

    #[test]
    fn recurrent_snapshot_restore_validates_layout_and_count() -> Result<()> {
        let mut cache = HybridCache::new(
            gdn_config(vec![HybridLayerType::Recurrent]),
            DType::BF16,
            &[Device::Cpu],
        )?;
        let source = cache.allocate_seq(10)?;
        let destination = cache.allocate_seq(20)?;
        let snapshots = cache.snapshot_recurrent_state(10, source)?;
        let valid_snapshots = snapshots.clone();
        assert_eq!(snapshots[0].state_layout, RecurrentStateLayout::GdnKeyMajor);
        assert!(cache.restore_recurrent_state(20, destination, &[]).is_err());

        let mut wrong_layout = snapshots;
        wrong_layout[0].state_layout = RecurrentStateLayout::GdnValueMajor;
        let error = cache
            .restore_recurrent_state(20, destination, &wrong_layout)
            .unwrap_err();
        assert!(error.to_string().contains("layout mismatch"));

        assert!(cache.release_seq(20, destination)?);
        let error = cache
            .restore_recurrent_state(20, destination, &valid_snapshots)
            .unwrap_err();
        assert!(error.to_string().contains("is not allocated"));
        Ok(())
    }

    #[test]
    fn sequence_slot_validation_rejects_missing_stale_and_duplicate_slots() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu, Device::Cpu],
        )?;
        let slot = cache.allocate_seq(10)?;

        cache.validate_sequence_slots(&[(10, slot)])?;
        let missing = cache
            .validate_sequence_slots(&[(20, slot + 1)])
            .unwrap_err();
        assert!(missing.to_string().contains("is not allocated"));
        let stale = cache.validate_sequence_slots(&[(20, slot)]).unwrap_err();
        assert!(stale.to_string().contains("expected Sequence(20)"));
        let duplicate = cache
            .validate_sequence_slots(&[(10, slot), (10, slot)])
            .unwrap_err();
        assert!(duplicate.to_string().contains("multiple sequences"));
        Ok(())
    }

    #[test]
    fn stale_aba_release_cannot_free_a_reused_slot() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu, Device::Cpu],
        )?;
        let slot = cache.allocate_seq(10)?;
        assert!(cache.release_seq(10, slot)?);
        assert!(!cache.release_seq(10, slot)?);

        let reused = cache.allocate_seq(20)?;
        assert_eq!(reused, slot);
        let stale = cache.release_seq(10, slot).unwrap_err();
        assert!(stale.to_string().contains("owned by Sequence(20)"));
        cache.validate_sequence_slots(&[(20, slot)])?;
        assert!(cache.release_seq(20, slot)?);
        Ok(())
    }

    #[test]
    fn graph_pad_has_distinct_ownership_and_blocks_storage_reset() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        let pad = cache.graph_pad_slot()?.unwrap();

        assert!(cache.release_seq(10, pad).is_err());
        assert!(cache.reset().is_err());
        assert!(cache.release_graph_pad_slot()?);
        assert!(!cache.release_graph_pad_slot()?);
        cache.reset()?;
        Ok(())
    }

    #[test]
    fn recurrent_storage_generation_tracks_committed_replacements() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu],
        )?;
        assert_eq!(cache.recurrent_storage_generation(), 0);
        assert!(cache.configure_checkpoint_lanes(2)?);
        assert_eq!(cache.recurrent_storage_generation(), 1);
        assert!(cache.reserve_recurrent_capacity(17)?);
        assert_eq!(cache.recurrent_storage_generation(), 2);
        assert!(!cache.reserve_recurrent_capacity(16)?);
        assert_eq!(cache.recurrent_storage_generation(), 2);
        cache.reset()?;
        assert_eq!(cache.recurrent_storage_generation(), 3);
        Ok(())
    }

    #[test]
    fn failed_growth_does_not_partially_replace_recurrent_storage() -> Result<()> {
        let mut cache = HybridCache::new(
            config(vec![HybridLayerType::Recurrent, HybridLayerType::Recurrent]),
            DType::F32,
            &[Device::Cpu, Device::Cpu],
        )?;
        let first_storage_id = match cache.get(0).unwrap() {
            HybridLayerCache::Recurrent(pool) => pool.conv_state.id(),
            HybridLayerCache::Attention(_) => unreachable!(),
        };
        let original_second = {
            let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
                unreachable!()
            };
            std::mem::replace(
                &mut pool.conv_state,
                Tensor::zeros((INITIAL_POOL_CAPACITY, 1, 1), DType::F32, &Device::Cpu)?,
            )
        };
        let generation = cache.recurrent_storage_generation();

        assert!(cache.reserve_recurrent_capacity(17).is_err());
        assert_eq!(cache.recurrent_storage_generation(), generation);
        assert_eq!(cache.recurrent_capacity(), INITIAL_POOL_CAPACITY);
        let first = match cache.get(0).unwrap() {
            HybridLayerCache::Recurrent(pool) => pool,
            HybridLayerCache::Attention(_) => unreachable!(),
        };
        assert_eq!(first.capacity(), INITIAL_POOL_CAPACITY);
        assert_eq!(first.conv_state.id(), first_storage_id);

        let HybridLayerCache::Recurrent(pool) = cache.get_mut(1).unwrap() else {
            unreachable!()
        };
        pool.conv_state = original_second;
        Ok(())
    }
}

impl PastKvLenCache for HybridCache {
    fn get_past_kv_len(&self) -> Result<usize> {
        for cache in &self.caches {
            if let HybridLayerCache::Attention(kv) = cache {
                return Ok(kv.current_seq_len());
            }
        }
        Ok(0)
    }
}

impl HybridCache {
    /// Truncate all attention layer KV caches to the given sequence length.
    /// Recurrent layers are unchanged, use snapshot/restore for recurrent rollback.
    pub fn truncate_attention_to(&mut self, len: usize) -> Result<()> {
        for cache in &mut self.caches {
            if let HybridLayerCache::Attention(kv) = cache {
                kv.set_len(len)?;
            }
        }
        Ok(())
    }
}

/// Snapshot of a single recurrent layer's state for prefix caching.
#[derive(Clone, Debug)]
pub struct RecurrentStateSnapshot {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub state_layout: RecurrentStateLayout,
}

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Debug)]
pub(crate) struct RecurrentCheckpointStateSnapshot {
    states: Vec<RecurrentStateSnapshot>,
    checkpoint_lanes: usize,
    committed_lane: usize,
}

impl HybridCache {
    /// Snapshot the recurrent state for a sequence at the given slot index.
    /// Returns one snapshot per recurrent layer, in layer order.
    pub fn snapshot_recurrent_state(
        &self,
        sequence_id: usize,
        slot_idx: usize,
    ) -> Result<Vec<RecurrentStateSnapshot>> {
        self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::Sequence(sequence_id))?;
        self.ensure_recurrent_slot_initialized(slot_idx)?;
        let physical_slot = self.active_physical_slot(slot_idx)?;
        let physical_slot = u32::try_from(physical_slot).map_err(|_| {
            candle_core::Error::msg(format!(
                "recurrent physical slot {physical_slot} exceeds u32"
            ))
        })?;
        let mut snapshots = Vec::new();
        for cache in &self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let idx_tensor = Tensor::from_vec(vec![physical_slot], (1,), pool.device())?;
                let conv = pool.gather_conv_state(&idx_tensor)?;
                let recurrent = pool.gather_recurrent_state(&idx_tensor)?;
                snapshots.push(RecurrentStateSnapshot {
                    conv_state: conv,
                    recurrent_state: recurrent,
                    state_layout: pool.state_layout(),
                });
            }
        }
        Ok(snapshots)
    }

    /// Restore recurrent state snapshots into the pool at the given slot index.
    /// Snapshots must be in the same layer order as returned by `snapshot_recurrent_state`.
    pub fn restore_recurrent_state(
        &mut self,
        sequence_id: usize,
        slot_idx: usize,
        snapshots: &[RecurrentStateSnapshot],
    ) -> Result<()> {
        self.ensure_recurrent_slot_owned(slot_idx, RecurrentSlotOwner::Sequence(sequence_id))?;
        let expected = self
            .caches
            .iter()
            .filter(|cache| matches!(cache, HybridLayerCache::Recurrent(_)))
            .count();
        if snapshots.len() != expected {
            candle_core::bail!(
                "recurrent snapshot count mismatch: got {}, expected {expected}",
                snapshots.len()
            );
        }
        for (cache, snap) in self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .zip(snapshots)
        {
            if snap.state_layout != cache.state_layout() {
                candle_core::bail!(
                    "recurrent state layout mismatch: snapshot {:?}, pool {:?}",
                    snap.state_layout,
                    cache.state_layout()
                );
            }
        }
        self.initialized_slots[slot_idx] = false;
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.reset_slot(slot_idx)?;
            }
        }
        self.committed_lanes[slot_idx] = 0;
        let physical_slot = self.physical_slot(slot_idx, 0)?;
        let physical_slot = u32::try_from(physical_slot).map_err(|_| {
            candle_core::Error::msg(format!(
                "recurrent physical slot {physical_slot} exceeds u32"
            ))
        })?;
        let mut snap_iter = snapshots.iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let snap = snap_iter.next().expect("snapshot count checked above");
                let conv = snap.conv_state.to_device(pool.device())?;
                let recurrent = snap.recurrent_state.to_device(pool.device())?;
                let idx_tensor = Tensor::from_vec(vec![physical_slot], (1,), pool.device())?;
                pool.scatter_conv_state(&idx_tensor, &conv)?;
                pool.scatter_recurrent_state(&idx_tensor, &recurrent)?;
            }
        }
        self.initialized_slots[slot_idx] = true;
        self.refresh_current_batch_mapping()
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn snapshot_recurrent_checkpoint_state(
        &self,
        slot_idx: usize,
    ) -> Result<RecurrentCheckpointStateSnapshot> {
        self.ensure_recurrent_slot_initialized(slot_idx)?;
        let committed_lane = self.committed_lane(slot_idx)?;
        let physical_slot = self.physical_slot(slot_idx, committed_lane)?;
        let physical_slot = u32::try_from(physical_slot).map_err(|_| {
            candle_core::Error::msg(format!(
                "recurrent physical slot {physical_slot} exceeds u32"
            ))
        })?;
        let mut states = Vec::new();
        for cache in &self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let idx_tensor = Tensor::from_vec(vec![physical_slot], (1,), pool.device())?;
                states.push(RecurrentStateSnapshot {
                    conv_state: pool
                        .gather_conv_state(&idx_tensor)?
                        .to_device(&Device::Cpu)?,
                    recurrent_state: pool
                        .gather_recurrent_state(&idx_tensor)?
                        .to_device(&Device::Cpu)?,
                    state_layout: pool.state_layout(),
                });
            }
        }
        Ok(RecurrentCheckpointStateSnapshot {
            states,
            checkpoint_lanes: self.checkpoint_lanes,
            committed_lane,
        })
    }

    #[cfg(any(feature = "cuda", test))]
    pub(crate) fn restore_recurrent_checkpoint_state(
        &mut self,
        slot_idx: usize,
        snapshot: &RecurrentCheckpointStateSnapshot,
    ) -> Result<()> {
        self.ensure_recurrent_slot_initialized(slot_idx)?;
        if snapshot.checkpoint_lanes != self.checkpoint_lanes {
            candle_core::bail!(
                "recurrent checkpoint lane mismatch: snapshot {}, pool {}",
                snapshot.checkpoint_lanes,
                self.checkpoint_lanes
            );
        }
        if snapshot.committed_lane >= self.checkpoint_lanes {
            candle_core::bail!(
                "recurrent checkpoint committed lane {} exceeds lane count {}",
                snapshot.committed_lane,
                self.checkpoint_lanes
            );
        }
        let pools = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .collect::<Vec<_>>();
        if snapshot.states.len() != pools.len() {
            candle_core::bail!(
                "recurrent checkpoint snapshot count mismatch: got {}, expected {}",
                snapshot.states.len(),
                pools.len()
            );
        }
        for (pool, state) in pools.into_iter().zip(&snapshot.states) {
            if state.state_layout != pool.state_layout() {
                candle_core::bail!(
                    "recurrent state layout mismatch: snapshot {:?}, pool {:?}",
                    state.state_layout,
                    pool.state_layout()
                );
            }
            if state.conv_state.dim(0)? != 1 || state.recurrent_state.dim(0)? != 1 {
                candle_core::bail!("recurrent checkpoint snapshot must contain its active lane");
            }
        }

        let physical_slot = self.physical_slot(slot_idx, snapshot.committed_lane)?;
        let physical_slot = u32::try_from(physical_slot).map_err(|_| {
            candle_core::Error::msg(format!(
                "recurrent physical slot {physical_slot} exceeds u32"
            ))
        })?;
        let logical_slot = u32::try_from(slot_idx).map_err(|_| {
            candle_core::Error::msg(format!("recurrent logical slot {slot_idx} exceeds u32"))
        })?;
        self.initialized_slots[slot_idx] = false;
        let mut states = snapshot.states.iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let state = states.next().expect("snapshot count checked above");
                let idx_tensor = Tensor::from_vec(vec![physical_slot], (1,), pool.device())?;
                let conv_state = state.conv_state.to_device(pool.device())?;
                let recurrent_state = state.recurrent_state.to_device(pool.device())?;
                pool.scatter_conv_state(&idx_tensor, &conv_state)?;
                pool.scatter_recurrent_state(&idx_tensor, &recurrent_state)?;
            }
        }

        let previous_lane = self.committed_lane(slot_idx)?;
        self.committed_lanes[slot_idx] = snapshot.committed_lane;
        self.initialized_slots[slot_idx] = true;
        if previous_lane != snapshot.committed_lane
            && self
                .logical_state_indices_host
                .as_ref()
                .is_some_and(|slots| slots.contains(&logical_slot))
        {
            self.refresh_current_batch_mapping()?;
        }
        Ok(())
    }
}
