use super::*;

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
    pub(super) allocated_slots: Vec<bool>,
    pub(super) allocated_count: usize,
    /// Current capacity (grows dynamically)
    pub(super) capacity: usize,
    pub(super) checkpoint_lanes: usize,
    /// Shape parameters for growing
    pub(super) conv_dim: usize,
    pub(super) conv_width: usize,
    pub(super) state_dims: Vec<usize>,
    pub(super) state_layout: RecurrentStateLayout,
    pub(super) conv_dtype: DType,
    pub(super) recurrent_dtype: DType,
    pub(super) device: Device,
    pub(super) pending_transitions: Option<GdnPendingTransitionPool>,
    pub(super) deferred_state: Option<GdnDeferredStatePool>,
}

/// Initial pool capacity before dynamic growth: the pre-captured CUDA graph batch range plus the
/// graph pad slot, so growth (which invalidates captured graphs) only happens past that.
pub(super) const INITIAL_POOL_CAPACITY: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RecurrentSlotOwner {
    Sequence(usize),
    GraphPad,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum RecurrentSlotInitialization {
    Deferred,
    Zeroed,
}

pub(super) fn logical_slot_from_physical_slot(physical_slot: u32, physical_lanes: usize) -> u32 {
    if physical_slot == u32::MAX {
        return u32::MAX;
    }
    let physical_lanes =
        u32::try_from(physical_lanes).expect("recurrent physical lane count exceeds u32");
    physical_slot / physical_lanes
}

pub(super) struct RecurrentStatePoolConfig<'a> {
    pub(super) conv_dim: usize,
    pub(super) conv_width: usize,
    pub(super) state_dims: Vec<usize>,
    pub(super) state_layout: RecurrentStateLayout,
    pub(super) conv_dtype: DType,
    pub(super) recurrent_dtype: DType,
    pub(super) device: &'a Device,
}

impl RecurrentStatePool {
    /// Create a new recurrent state pool.
    ///
    /// - `conv_dim`: dimension of the convolution state
    /// - `conv_width`: kernel size / d_conv for causal conv1d
    pub(super) fn new(config: RecurrentStatePoolConfig<'_>) -> Result<Self> {
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
            pending_transitions: None,
            deferred_state: None,
        })
    }

    pub(super) fn resized_storage(
        &self,
        new_capacity: usize,
    ) -> Result<(
        Tensor,
        Tensor,
        Option<GdnPendingTransitionPool>,
        Option<GdnDeferredStatePool>,
    )> {
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
        let pending_transitions = self
            .pending_transitions
            .as_ref()
            .map(|pending| pending.resized(new_capacity))
            .transpose()?;
        let deferred_state = self
            .deferred_state
            .as_ref()
            .map(|deferred| deferred.resized(new_capacity))
            .transpose()?;
        Ok((new_conv, new_recurrent, pending_transitions, deferred_state))
    }

    pub(super) fn install_resized_storage(
        &mut self,
        new_capacity: usize,
        conv_state: Tensor,
        recurrent_state: Tensor,
        pending_transitions: Option<GdnPendingTransitionPool>,
        deferred_state: Option<GdnDeferredStatePool>,
    ) {
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
        self.pending_transitions = pending_transitions;
        self.deferred_state = deferred_state;
        self.allocated_slots.resize(new_capacity, false);
        self.capacity = new_capacity;
    }

    pub(super) fn checkpoint_storage(&self, checkpoint_lanes: usize) -> Result<(Tensor, Tensor)> {
        self.layout_storage(self.capacity, checkpoint_lanes)
    }

    pub(super) fn layout_storage(
        &self,
        capacity: usize,
        checkpoint_lanes: usize,
    ) -> Result<(Tensor, Tensor)> {
        let physical_capacity = capacity
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

    pub(super) fn install_layout_storage(
        &mut self,
        capacity: usize,
        checkpoint_lanes: usize,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) {
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
        self.allocated_slots.resize(capacity, false);
        self.allocated_slots.fill(false);
        self.allocated_count = 0;
        self.capacity = capacity;
        self.checkpoint_lanes = checkpoint_lanes;
        self.pending_transitions = None;
        self.deferred_state = None;
    }

    pub(super) fn install_checkpoint_storage(
        &mut self,
        checkpoint_lanes: usize,
        conv_state: Tensor,
        recurrent_state: Tensor,
    ) {
        self.checkpoint_lanes = checkpoint_lanes;
        self.conv_state = conv_state;
        self.recurrent_state = recurrent_state;
    }

    pub(super) fn pending_transition_storage(
        &self,
        spec: GdnPendingTransitionSpec,
    ) -> Result<GdnPendingTransitionPool> {
        GdnPendingTransitionPool::new(
            self.capacity,
            self.conv_dim,
            &self.state_dims,
            self.state_layout,
            self.conv_dtype,
            &self.device,
            spec,
        )
    }

    pub(super) fn install_pending_transition_storage(&mut self, pending: GdnPendingTransitionPool) {
        self.pending_transitions = Some(pending);
    }

    pub(super) fn deferred_state_storage(
        &self,
        spec: GdnDeferredStateSpec,
    ) -> Result<GdnDeferredStatePool> {
        GdnDeferredStatePool::new(
            self.capacity,
            &self.state_dims,
            self.state_layout,
            &self.device,
            spec,
        )
    }

    pub(super) fn install_deferred_state_storage(&mut self, deferred: GdnDeferredStatePool) {
        self.deferred_state = Some(deferred);
    }

    pub(super) fn clear_pending_transition_slot(&mut self, slot_idx: usize) -> Result<()> {
        if let Some(pending) = &mut self.pending_transitions {
            pending.clear_slot(slot_idx)?;
        }
        Ok(())
    }

    pub(super) fn clear_pending_transition_metadata(&mut self) -> Result<()> {
        if let Some(pending) = &mut self.pending_transitions {
            pending.clear_metadata()?;
        }
        Ok(())
    }

    pub(super) fn clear_deferred_state_slot(&self, slot_idx: usize) -> Result<()> {
        if let Some(deferred) = &self.deferred_state {
            deferred.clear_slot(slot_idx)?;
        }
        Ok(())
    }

    pub(super) fn clear_deferred_state_metadata(&self) -> Result<()> {
        if let Some(deferred) = &self.deferred_state {
            deferred.clear_metadata()?;
        }
        Ok(())
    }

    pub(super) fn allocate_at(&mut self, slot_idx: usize) -> Result<()> {
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

    pub(super) fn reserve_at(&mut self, slot_idx: usize) -> Result<()> {
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

    pub(super) fn free(&mut self, slot_idx: usize) -> bool {
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

    pub(crate) fn materialize_deferred_state(&self, slots: &Tensor) -> Result<()> {
        if let Some(deferred) = &self.deferred_state {
            deferred.materialize(&self.recurrent_state, slots)?;
        }
        Ok(())
    }

    /// Gather recurrent states for the given slot indices
    pub fn gather_recurrent_state(&self, state_indices: &Tensor) -> Result<Tensor> {
        self.materialize_deferred_state(state_indices)?;
        self.recurrent_state.index_select(state_indices, 0)
    }

    /// Scatter conv states back to the pool for the given slot indices
    pub fn scatter_conv_state(&mut self, state_indices: &Tensor, values: &Tensor) -> Result<()> {
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return crate::cuda::indexed_copy::copy_rows(
                &values.contiguous()?,
                &self.conv_state,
                state_indices,
            );
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
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return self.scatter_conv_state(state_indices, values);
        }
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
                &values.contiguous()?,
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
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return self.scatter_recurrent_state(state_indices, values);
        }
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
        if let Some(pending) = &mut self.pending_transitions {
            pending.clear_slot(slot_idx)?;
        }
        self.clear_deferred_state_slot(slot_idx)?;
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

    pub(super) fn is_allocated(&self, slot_idx: usize) -> bool {
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

    pub(super) fn snapshot_bytes(&self) -> Result<usize> {
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

    pub(crate) fn pending_transitions(&self) -> Option<&GdnPendingTransitionPool> {
        self.pending_transitions.as_ref()
    }

    pub(crate) fn deferred_state(&self) -> Option<&GdnDeferredStatePool> {
        self.deferred_state.as_ref()
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
            pending_transitions: self.pending_transitions.clone(),
            deferred_state: self.deferred_state.clone(),
        }
    }
}
