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

mod gdn_pools;
mod recurrent_pool;

pub(crate) use gdn_pools::*;
pub use recurrent_pool::*;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecurrentSpeculativeStorage {
    FullCheckpoints,
    TransitionLog,
}

impl RecurrentSpeculativeStorage {
    fn physical_lanes(self, checkpoint_lanes: usize) -> usize {
        match self {
            Self::FullCheckpoints => checkpoint_lanes,
            Self::TransitionLog => 1,
        }
    }
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

/// Per-layer cache that can be either attention (KV) or recurrent (state pool)
#[allow(clippy::large_enum_variant)]
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
    speculative_storage: RecurrentSpeculativeStorage,
    committed_lanes: Vec<usize>,
    slot_owners: Vec<Option<RecurrentSlotOwner>>,
    initialized_slots: Vec<bool>,
    // True when every recurrent layer's slot row and transition metadata are known zero.
    pristine_zero_slots: Vec<bool>,
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
        Self::new_with_recurrent_overrides(config, dtype, layer_devices, &HashMap::new())
    }

    /// Like `new`, but recurrent layers listed in `overrides` get their own state shapes instead of
    /// `config.recurrent`. Their pools still share slot allocation, snapshots and resets.
    pub fn new_with_recurrent_overrides(
        config: HybridCacheConfig,
        dtype: candle_core::DType,
        layer_devices: &[Device],
        overrides: &HashMap<usize, RecurrentLayerConfig>,
    ) -> Result<Self> {
        if layer_devices.len() != config.layer_types.len() {
            candle_core::bail!(
                "Hybrid cache has {} layers but {} layer devices",
                config.layer_types.len(),
                layer_devices.len()
            );
        }
        let mut caches = Vec::with_capacity(config.layer_types.len());

        for (layer_idx, (layer_type, device)) in
            config.layer_types.iter().zip(layer_devices).enumerate()
        {
            let recurrent = overrides.get(&layer_idx).unwrap_or(&config.recurrent);
            let cache = match layer_type {
                HybridLayerType::Attention => HybridLayerCache::Attention(KvCache::new_normal(
                    2,
                    config.max_seq_len,
                    Self::CACHE_GROW_SIZE,
                )),
                HybridLayerType::Recurrent => {
                    let (state_dims, state_layout) =
                        recurrent.state.physical_layout(dtype, device)?;
                    HybridLayerCache::Recurrent(RecurrentStatePool::new(
                        RecurrentStatePoolConfig {
                            conv_dim: recurrent.conv_dim,
                            conv_width: recurrent.conv_width,
                            state_dims,
                            state_layout,
                            conv_dtype: dtype,
                            recurrent_dtype: recurrent.recurrent_dtype.unwrap_or(dtype),
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
            speculative_storage: RecurrentSpeculativeStorage::FullCheckpoints,
            committed_lanes: vec![0; INITIAL_POOL_CAPACITY],
            slot_owners: vec![None; INITIAL_POOL_CAPACITY],
            initialized_slots: vec![false; INITIAL_POOL_CAPACITY],
            pristine_zero_slots: vec![true; INITIAL_POOL_CAPACITY],
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
                pool.clear_pending_transition_slot(slot_idx)?;
                pool.clear_deferred_state_slot(slot_idx)?;
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
        self.configure_checkpoint_storage(
            checkpoint_lanes,
            RecurrentSpeculativeStorage::FullCheckpoints,
        )
    }

    pub(crate) fn configure_transition_lanes(&mut self, checkpoint_lanes: usize) -> Result<bool> {
        self.configure_checkpoint_storage(
            checkpoint_lanes,
            RecurrentSpeculativeStorage::TransitionLog,
        )
    }

    fn configure_checkpoint_storage(
        &mut self,
        checkpoint_lanes: usize,
        speculative_storage: RecurrentSpeculativeStorage,
    ) -> Result<bool> {
        if checkpoint_lanes == 0 {
            candle_core::bail!("recurrent checkpoint lane count must be nonzero");
        }
        if checkpoint_lanes == self.checkpoint_lanes
            && speculative_storage == self.speculative_storage
        {
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

        let physical_lanes = speculative_storage.physical_lanes(checkpoint_lanes);
        let storage = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.checkpoint_storage(physical_lanes))
            .collect::<Result<Vec<_>>>()?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let (conv_state, recurrent_state) = storage
                    .next()
                    .expect("one checkpoint allocation per recurrent pool");
                pool.install_checkpoint_storage(physical_lanes, conv_state, recurrent_state);
                pool.pending_transitions = None;
                pool.deferred_state = None;
            }
        }
        self.checkpoint_lanes = checkpoint_lanes;
        self.speculative_storage = speculative_storage;
        self.committed_lanes.fill(0);
        self.initialized_slots.fill(false);
        self.pristine_zero_slots.fill(true);
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

    pub(crate) fn recurrent_speculative_storage(&self) -> RecurrentSpeculativeStorage {
        self.speculative_storage
    }

    pub(crate) fn uses_recurrent_transition_log(&self) -> bool {
        self.recurrent_speculative_storage() == RecurrentSpeculativeStorage::TransitionLog
    }

    pub(crate) fn physical_checkpoint_lanes(&self) -> usize {
        self.speculative_storage
            .physical_lanes(self.checkpoint_lanes)
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
        let physical_lanes = self.physical_checkpoint_lanes();
        if lane >= physical_lanes {
            candle_core::bail!(
                "recurrent physical lane {lane} exceeds lane count {physical_lanes}"
            );
        }
        logical_slot
            .checked_mul(physical_lanes)
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
        let physical_lanes = self.physical_checkpoint_lanes();
        self.logical_state_indices_host = Some(
            host.iter()
                .map(|&slot| logical_slot_from_physical_slot(slot, physical_lanes))
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
        let physical_lanes = self.physical_checkpoint_lanes();
        self.logical_state_indices_host = physical_host.as_ref().map(|slots| {
            slots
                .iter()
                .map(|&slot| logical_slot_from_physical_slot(slot, physical_lanes))
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

    pub(crate) fn recurrent_slots_for_sequences(&self, sequence_ids: &[usize]) -> Vec<u32> {
        let sequence_ids = sequence_ids.iter().copied().collect::<HashSet<_>>();
        self.slot_owners
            .iter()
            .enumerate()
            .filter_map(|(slot, owner)| match owner {
                Some(RecurrentSlotOwner::Sequence(sequence_id))
                    if sequence_ids.contains(sequence_id) =>
                {
                    u32::try_from(slot).ok()
                }
                _ => None,
            })
            .collect()
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
                let (conv_state, recurrent_state, pending_transitions, deferred_state) = storage
                    .next()
                    .expect("one resized allocation per recurrent pool");
                pool.install_resized_storage(
                    min_capacity,
                    conv_state,
                    recurrent_state,
                    pending_transitions,
                    deferred_state,
                );
            }
        }
        self.slot_owners.resize(min_capacity, None);
        self.initialized_slots.resize(min_capacity, false);
        self.pristine_zero_slots.resize(min_capacity, true);
        self.last_released_sequence_owners
            .resize(min_capacity, None);
        self.committed_lanes.resize(min_capacity, 0);
        self.advance_recurrent_storage_generation();
        tracing::info!("Recurrent state pool grew to capacity {min_capacity}");
        Ok(true)
    }

    pub(crate) fn reserve_recurrent_layout(
        &mut self,
        min_capacity: usize,
        checkpoint_lanes: usize,
    ) -> Result<bool> {
        self.reserve_recurrent_layout_with_storage(
            min_capacity,
            checkpoint_lanes,
            RecurrentSpeculativeStorage::FullCheckpoints,
        )
    }

    pub(crate) fn reserve_recurrent_transition_layout(
        &mut self,
        min_capacity: usize,
        checkpoint_lanes: usize,
    ) -> Result<bool> {
        self.reserve_recurrent_layout_with_storage(
            min_capacity,
            checkpoint_lanes,
            RecurrentSpeculativeStorage::TransitionLog,
        )
    }

    pub(crate) fn reserve_gdn_pending_transitions(
        &mut self,
        spec: GdnPendingTransitionSpec,
    ) -> Result<bool> {
        if !self.uses_recurrent_transition_log() {
            candle_core::bail!("GDN pending transitions require transition-log storage");
        }
        if spec.max_rows != self.checkpoint_lanes {
            candle_core::bail!(
                "GDN pending transition depth {} does not match logical checkpoint depth {}",
                spec.max_rows,
                self.checkpoint_lanes
            );
        }
        if self.slot_owners.iter().any(Option::is_some) {
            candle_core::bail!(
                "GDN pending transitions must be reserved before recurrent slot allocation"
            );
        }
        let recurrent_pools = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .filter(|pool| pool.state_layout() != RecurrentStateLayout::Opaque)
            .collect::<Vec<_>>();
        if recurrent_pools.is_empty() {
            candle_core::bail!("hybrid cache has no GDN recurrent state pool");
        }
        let existing = recurrent_pools
            .iter()
            .map(|pool| {
                pool.pending_transitions()
                    .map(GdnPendingTransitionPool::spec)
            })
            .collect::<Vec<_>>();
        if existing.iter().all(|configured| *configured == Some(spec)) {
            return Ok(false);
        }
        if existing.iter().any(Option::is_some) {
            candle_core::bail!("GDN pending transition pool configuration diverged");
        }
        let storage = recurrent_pools
            .into_iter()
            .map(|pool| pool.pending_transition_storage(spec))
            .collect::<Result<Vec<_>>>()?;
        let storage_bytes = storage.iter().try_fold(0usize, |bytes, pending| {
            bytes.checked_add(pending.storage_bytes()?).ok_or_else(|| {
                candle_core::Error::msg("GDN pending transition storage size overflow")
            })
        })?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                if pool.state_layout() == RecurrentStateLayout::Opaque {
                    continue;
                }
                pool.install_pending_transition_storage(
                    storage
                        .next()
                        .expect("one pending allocation per GDN recurrent pool"),
                );
            }
        }
        self.advance_recurrent_storage_generation();
        tracing::info!(storage_bytes, "Reserved GDN speculative transition storage");
        Ok(true)
    }

    pub(crate) fn reserve_gdn_deferred_state(
        &mut self,
        specs: &[(usize, GdnDeferredStateSpec)],
    ) -> Result<bool> {
        if specs.is_empty() {
            return Ok(false);
        }
        if self.slot_owners.iter().any(Option::is_some) {
            candle_core::bail!(
                "GDN deferred state must be reserved before recurrent slot allocation"
            );
        }
        let mut selected = HashSet::new();
        let mut storage = Vec::new();
        for &(layer_idx, spec) in specs {
            if !selected.insert(layer_idx) {
                candle_core::bail!("GDN deferred state repeats layer {layer_idx}");
            }
            let Some(HybridLayerCache::Recurrent(pool)) = self.get(layer_idx) else {
                candle_core::bail!("GDN layer {layer_idx} has no recurrent state pool");
            };
            if let Some(deferred) = pool.deferred_state() {
                if deferred.spec() != spec {
                    candle_core::bail!("GDN deferred state pool configuration diverged");
                }
            } else {
                storage.push((layer_idx, pool.deferred_state_storage(spec)?));
            }
        }
        if storage.is_empty() {
            return Ok(false);
        }
        let storage_bytes = storage.iter().try_fold(0usize, |bytes, (_, deferred)| {
            bytes
                .checked_add(deferred.storage_bytes()?)
                .ok_or_else(|| candle_core::Error::msg("GDN deferred state storage size overflow"))
        })?;
        for (layer_idx, deferred) in storage {
            let HybridLayerCache::Recurrent(pool) = &mut self.caches[layer_idx] else {
                unreachable!("deferred-state pool was validated above")
            };
            pool.install_deferred_state_storage(deferred);
        }
        self.advance_recurrent_storage_generation();
        tracing::info!(storage_bytes, "Reserved GDN deferred-state storage");
        Ok(true)
    }

    pub(crate) fn uses_gdn_deferred_state(&self) -> bool {
        self.caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .any(|pool| pool.deferred_state().is_some())
    }

    pub(crate) fn materialize_deferred_state_for_current_batch(&self) -> Result<()> {
        if !self
            .state_indices()
            .is_some_and(|slots| slots.elem_count() != 0)
        {
            return Ok(());
        }
        for pool in self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
        {
            if pool.deferred_state().is_some() {
                let slots = self
                    .state_indices_for_device(pool.device())
                    .ok_or_else(|| {
                        candle_core::Error::msg(
                            "GDN deferred-state flush has no device-local state slots",
                        )
                    })?;
                pool.materialize_deferred_state(&slots)?;
            }
        }
        Ok(())
    }

    pub(crate) fn materialize_deferred_state_for_slots(&self, slots: &[u32]) -> Result<()> {
        let mut slots = slots
            .iter()
            .copied()
            .filter(|slot| *slot != crate::cuda::gdn::GDN_PAD_SLOT)
            .collect::<Vec<_>>();
        slots.sort_unstable();
        slots.dedup();
        if slots.is_empty() {
            return Ok(());
        }
        for pool in self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
        {
            if pool.deferred_state().is_some() {
                let active_slots = Tensor::from_vec(slots.clone(), (slots.len(),), pool.device())?;
                pool.materialize_deferred_state(&active_slots)?;
            }
        }
        Ok(())
    }

    pub(crate) fn disable_gdn_deferred_state(&mut self) -> Result<bool> {
        if self.slot_owners.iter().any(Option::is_some) {
            candle_core::bail!(
                "GDN deferred state must be disabled before recurrent slot allocation"
            );
        }
        if !self.uses_gdn_deferred_state() {
            return Ok(false);
        }
        let storage_bytes = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .filter_map(RecurrentStatePool::deferred_state)
            .try_fold(0usize, |bytes, deferred| {
                bytes.checked_add(deferred.storage_bytes()?).ok_or_else(|| {
                    candle_core::Error::msg("GDN deferred state storage size overflow")
                })
            })?;
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                pool.deferred_state = None;
            }
        }
        self.advance_recurrent_storage_generation();
        tracing::info!(storage_bytes, "Disabled GDN deferred-state storage");
        Ok(true)
    }

    fn reserve_recurrent_layout_with_storage(
        &mut self,
        min_capacity: usize,
        checkpoint_lanes: usize,
        speculative_storage: RecurrentSpeculativeStorage,
    ) -> Result<bool> {
        if checkpoint_lanes == 0 {
            candle_core::bail!("recurrent checkpoint lane count must be nonzero");
        }
        let current_capacity = self.recurrent_capacity();
        if current_capacity == 0 {
            candle_core::bail!("hybrid cache has no recurrent state pool");
        }
        if self.recurrent_storage_locked {
            candle_core::bail!("recurrent storage is already reserved or allocated");
        }
        if self.slot_owners.iter().any(Option::is_some) {
            candle_core::bail!(
                "recurrent layout cannot change while recurrent slots are allocated"
            );
        }
        if self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .any(|pool| pool.capacity() != current_capacity)
        {
            candle_core::bail!("hybrid recurrent pool capacities diverged before reservation");
        }
        let current_physical_lanes = self.physical_checkpoint_lanes();
        if self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .any(|pool| pool.checkpoint_lanes() != current_physical_lanes)
        {
            candle_core::bail!(
                "hybrid recurrent pool checkpoint lanes diverged before reservation"
            );
        }

        let capacity = current_capacity.max(min_capacity);
        let physical_lanes = speculative_storage.physical_lanes(checkpoint_lanes);
        let changed = capacity != current_capacity
            || checkpoint_lanes != self.checkpoint_lanes
            || speculative_storage != self.speculative_storage;
        if changed {
            let storage = self
                .caches
                .iter()
                .filter_map(HybridLayerCache::as_recurrent_pool)
                .map(|pool| pool.layout_storage(capacity, physical_lanes))
                .collect::<Result<Vec<_>>>()?;
            let mut storage = storage.into_iter();
            for cache in &mut self.caches {
                if let HybridLayerCache::Recurrent(pool) = cache {
                    let (conv_state, recurrent_state) = storage
                        .next()
                        .expect("one layout allocation per recurrent pool");
                    pool.install_layout_storage(
                        capacity,
                        physical_lanes,
                        conv_state,
                        recurrent_state,
                    );
                }
            }
            self.checkpoint_lanes = checkpoint_lanes;
            self.speculative_storage = speculative_storage;
            self.slot_owners.resize(capacity, None);
            self.slot_owners.fill(None);
            self.initialized_slots.resize(capacity, false);
            self.initialized_slots.fill(false);
            self.pristine_zero_slots.resize(capacity, true);
            self.pristine_zero_slots.fill(true);
            self.last_released_sequence_owners.resize(capacity, None);
            self.last_released_sequence_owners.fill(None);
            self.committed_lanes.resize(capacity, 0);
            self.committed_lanes.fill(0);
            self.graph_pad_slot = None;
            self.advance_recurrent_storage_generation();
            tracing::info!(
                capacity,
                checkpoint_lanes,
                physical_lanes,
                ?speculative_storage,
                "Recurrent state pool reserved"
            );
        }
        self.clear_state_indices();
        self.recurrent_storage_locked = true;
        self.publish_recurrent_slot_metrics();
        Ok(changed)
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
        let slot_is_pristine = self.pristine_zero_slots[slot_idx];
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
                RecurrentSlotInitialization::Zeroed if slot_is_pristine => {
                    pool.reserve_at(slot_idx)
                }
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
        if initialization == RecurrentSlotInitialization::Zeroed {
            self.pristine_zero_slots[slot_idx] = false;
        }
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
                pool.clear_pending_transition_slot(slot_idx)?;
                pool.clear_deferred_state_slot(slot_idx)?;
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
        let slot_is_pristine = self.pristine_zero_slots[slot_idx];
        self.pristine_zero_slots[slot_idx] = false;
        if !slot_is_pristine {
            for cache in &mut self.caches {
                if let HybridLayerCache::Recurrent(pool) = cache {
                    pool.reset_slot(slot_idx)?;
                }
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
        let physical_lanes = self.physical_checkpoint_lanes();
        let storage = self
            .caches
            .iter()
            .filter_map(HybridLayerCache::as_recurrent_pool)
            .map(|pool| pool.checkpoint_storage(physical_lanes))
            .collect::<Result<Vec<_>>>()?;
        let mut storage = storage.into_iter();
        for cache in &mut self.caches {
            match cache {
                HybridLayerCache::Attention(kv) => kv.reset(),
                HybridLayerCache::Recurrent(pool) => {
                    let (conv_state, recurrent_state) = storage
                        .next()
                        .expect("one reset allocation per recurrent pool");
                    pool.install_checkpoint_storage(physical_lanes, conv_state, recurrent_state);
                    pool.clear_pending_transition_metadata()?;
                    pool.clear_deferred_state_metadata()?;
                    pool.allocated_slots.fill(false);
                    pool.allocated_count = 0;
                }
            }
        }
        self.committed_lanes.fill(0);
        self.slot_owners.fill(None);
        self.initialized_slots.fill(false);
        self.pristine_zero_slots.fill(true);
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
            self.pristine_zero_slots[slot] = false;
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
        if self.checkpoint_lanes == 1 || self.uses_recurrent_transition_log() {
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

    pub(crate) fn state_indices_for_device(&self, device: &Device) -> Option<Tensor> {
        if let Some(indices) = &self.state_indices {
            if indices.device().same_device(device) {
                return Some(indices.clone());
            }
        }
        self.device_state_indices
            .iter()
            .find(|(cached_device, _)| cached_device.same_device(device))
            .map(|(_, indices)| indices.clone())
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
#[allow(clippy::cast_possible_truncation)]
mod tests;

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
    speculative_storage: RecurrentSpeculativeStorage,
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
        let slot_is_pristine = self.pristine_zero_slots[slot_idx];
        self.pristine_zero_slots[slot_idx] = false;
        if !slot_is_pristine {
            for cache in &mut self.caches {
                if let HybridLayerCache::Recurrent(pool) = cache {
                    pool.reset_slot(slot_idx)?;
                }
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
                // Device-resident so the copy returns to the allocator pool instead of fragmenting the host heap
                states.push(RecurrentStateSnapshot {
                    conv_state: pool.gather_conv_state(&idx_tensor)?,
                    recurrent_state: pool.gather_recurrent_state(&idx_tensor)?,
                    state_layout: pool.state_layout(),
                });
            }
        }
        Ok(RecurrentCheckpointStateSnapshot {
            states,
            checkpoint_lanes: self.checkpoint_lanes,
            speculative_storage: self.speculative_storage,
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
        if snapshot.speculative_storage != self.speculative_storage {
            candle_core::bail!(
                "recurrent checkpoint storage mismatch: snapshot {:?}, pool {:?}",
                snapshot.speculative_storage,
                self.speculative_storage
            );
        }
        let physical_lanes = self.physical_checkpoint_lanes();
        if snapshot.committed_lane >= physical_lanes {
            candle_core::bail!(
                "recurrent checkpoint committed lane {} exceeds physical lane count {}",
                snapshot.committed_lane,
                physical_lanes
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
        self.pristine_zero_slots[slot_idx] = false;
        let mut states = snapshot.states.iter();
        for cache in &mut self.caches {
            if let HybridLayerCache::Recurrent(pool) = cache {
                let state = states.next().expect("snapshot count checked above");
                pool.clear_pending_transition_slot(slot_idx)?;
                pool.clear_deferred_state_slot(slot_idx)?;
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

#[cfg(test)]
#[path = "gdn_deferred_lifecycle_tests.rs"]
mod gdn_deferred_lifecycle_tests;
