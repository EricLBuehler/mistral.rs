use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnPendingTransitionSpec {
    pub num_k_heads: usize,
    pub max_rows: usize,
}

pub(super) const GDN_PENDING_KEEP_ROWS: usize = 0;
pub(super) const GDN_PENDING_EPOCHS: usize = 1;
pub(super) const GDN_PENDING_KEY_BANK: usize = 2;
pub(super) const GDN_PENDING_METADATA_FIELDS: usize = 3;
pub(crate) const GDN_PENDING_KEY_BANK_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GdnDeferredStateSpec {
    pub num_k_heads: usize,
    pub tiled_v_heads: bool,
    pub depth: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct GdnPendingTransitionPool {
    pub conv_input: Tensor,
    pub key_banks: Tensor,
    pub delta: Tensor,
    pub decay: Tensor,
    pub keep_rows: Tensor,
    pub pending_epochs: Tensor,
    pub key_bank: Tensor,
    pub conv_applied_epochs: Tensor,
    pub recurrent_applied_epochs: Tensor,
    pub(super) metadata: Tensor,
    pub(super) metadata_zero_column: Tensor,
    pub(super) conv_applied_zero_row: Tensor,
    pub(super) recurrent_applied_zero_row: Tensor,
    pub(super) spec: GdnPendingTransitionSpec,
    pub(super) capacity: usize,
    pub(super) num_v_heads: usize,
    pub(super) head_k_dim: usize,
    pub(super) head_v_dim: usize,
    pub(super) conv_dim: usize,
    pub(super) conv_dtype: DType,
    pub(super) state_layout: RecurrentStateLayout,
    pub(super) device: Device,
}

impl GdnPendingTransitionPool {
    pub(super) fn new(
        capacity: usize,
        conv_dim: usize,
        state_dims: &[usize],
        state_layout: RecurrentStateLayout,
        conv_dtype: DType,
        device: &Device,
        spec: GdnPendingTransitionSpec,
    ) -> Result<Self> {
        if spec.num_k_heads == 0
            || spec.max_rows == 0
            || spec.max_rows > crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS
        {
            candle_core::bail!("invalid GDN pending transition dimensions");
        }
        let (num_v_heads, head_k_dim, head_v_dim) = match (state_layout, state_dims) {
            (RecurrentStateLayout::GdnKeyMajor, [heads, key_dim, value_dim]) => {
                (*heads, *key_dim, *value_dim)
            }
            (RecurrentStateLayout::GdnValueMajor, [heads, value_dim, key_dim]) => {
                (*heads, *key_dim, *value_dim)
            }
            _ => candle_core::bail!("pending transitions require a GDN recurrent state layout"),
        };
        if num_v_heads == 0
            || head_k_dim == 0
            || head_v_dim == 0
            || !num_v_heads.is_multiple_of(spec.num_k_heads)
        {
            candle_core::bail!("GDN pending transition head dimensions are incompatible");
        }

        let conv_blocks = conv_dim.div_ceil(crate::cuda::gdn::GDN_CHANNEL_BLOCK_SIZE);
        let metadata = Tensor::zeros((GDN_PENDING_METADATA_FIELDS, capacity), DType::U32, device)?;
        let keep_rows = metadata.i(GDN_PENDING_KEEP_ROWS)?;
        let pending_epochs = metadata.i(GDN_PENDING_EPOCHS)?;
        let key_bank = metadata.i(GDN_PENDING_KEY_BANK)?;
        let conv_applied_epochs = Tensor::zeros((capacity, conv_blocks), DType::U32, device)?;
        let recurrent_applied_epochs = Tensor::zeros((capacity, num_v_heads), DType::U32, device)?;
        Ok(Self {
            conv_input: Tensor::zeros((capacity, spec.max_rows, conv_dim), conv_dtype, device)?,
            key_banks: Tensor::zeros(
                (
                    GDN_PENDING_KEY_BANK_COUNT,
                    capacity,
                    spec.max_rows,
                    spec.num_k_heads,
                    head_k_dim,
                ),
                DType::F32,
                device,
            )?,
            delta: Tensor::zeros(
                (capacity, spec.max_rows, num_v_heads, head_v_dim),
                DType::F32,
                device,
            )?,
            decay: Tensor::zeros((capacity, spec.max_rows, num_v_heads), DType::F32, device)?,
            keep_rows,
            pending_epochs,
            key_bank,
            conv_applied_epochs,
            recurrent_applied_epochs,
            metadata,
            metadata_zero_column: Tensor::zeros(
                (GDN_PENDING_METADATA_FIELDS, 1),
                DType::U32,
                device,
            )?,
            conv_applied_zero_row: Tensor::zeros((1, conv_blocks), DType::U32, device)?,
            recurrent_applied_zero_row: Tensor::zeros((1, num_v_heads), DType::U32, device)?,
            spec,
            capacity,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            conv_dim,
            conv_dtype,
            state_layout,
            device: device.clone(),
        })
    }

    pub(super) fn resized(&self, new_capacity: usize) -> Result<Self> {
        let state_dims = match self.state_layout {
            RecurrentStateLayout::GdnKeyMajor => {
                [self.num_v_heads, self.head_k_dim, self.head_v_dim]
            }
            RecurrentStateLayout::GdnValueMajor => {
                [self.num_v_heads, self.head_v_dim, self.head_k_dim]
            }
            RecurrentStateLayout::Opaque => unreachable!("pending transitions require GDN state"),
        };
        let resized = Self::new(
            new_capacity,
            self.conv_dim,
            &state_dims,
            self.state_layout,
            self.conv_dtype,
            &self.device,
            self.spec,
        )?;
        resized.conv_input.slice_set(&self.conv_input, 0, 0)?;
        resized.key_banks.slice_set(&self.key_banks, 1, 0)?;
        resized.delta.slice_set(&self.delta, 0, 0)?;
        resized.decay.slice_set(&self.decay, 0, 0)?;
        resized.metadata.slice_set(&self.metadata, 1, 0)?;
        resized
            .conv_applied_epochs
            .slice_set(&self.conv_applied_epochs, 0, 0)?;
        resized
            .recurrent_applied_epochs
            .slice_set(&self.recurrent_applied_epochs, 0, 0)?;
        Ok(resized)
    }

    pub(super) fn clear_slot(&mut self, slot_idx: usize) -> Result<()> {
        if slot_idx >= self.capacity {
            candle_core::bail!(
                "GDN pending transition slot {slot_idx} exceeds capacity {}",
                self.capacity
            );
        }
        self.metadata
            .slice_set(&self.metadata_zero_column, 1, slot_idx)?;
        self.conv_applied_epochs
            .slice_set(&self.conv_applied_zero_row, 0, slot_idx)?;
        self.recurrent_applied_epochs
            .slice_set(&self.recurrent_applied_zero_row, 0, slot_idx)?;
        Ok(())
    }

    pub(super) fn clear_metadata(&mut self) -> Result<()> {
        let zeros = Tensor::zeros(
            (GDN_PENDING_METADATA_FIELDS, self.capacity),
            DType::U32,
            &self.device,
        )?;
        self.metadata.slice_set(&zeros, 0, 0)?;
        self.conv_applied_epochs = self.conv_applied_epochs.zeros_like()?;
        self.recurrent_applied_epochs = self.recurrent_applied_epochs.zeros_like()?;
        Ok(())
    }

    pub fn spec(&self) -> GdnPendingTransitionSpec {
        self.spec
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub(crate) fn storage_bytes(&self) -> Result<usize> {
        [
            &self.conv_input,
            &self.key_banks,
            &self.delta,
            &self.decay,
            &self.metadata,
            &self.conv_applied_epochs,
            &self.recurrent_applied_epochs,
            &self.metadata_zero_column,
            &self.conv_applied_zero_row,
            &self.recurrent_applied_zero_row,
        ]
        .into_iter()
        .try_fold(0usize, |bytes, tensor| {
            let tensor_bytes = tensor
                .elem_count()
                .checked_mul(tensor.dtype().size_in_bytes())
                .ok_or_else(|| {
                    candle_core::Error::msg("GDN pending transition storage size overflow")
                })?;
            bytes.checked_add(tensor_bytes).ok_or_else(|| {
                candle_core::Error::msg("GDN pending transition storage size overflow")
            })
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GdnDeferredStatePool {
    pub key: Tensor,
    pub delta: Tensor,
    pub decay: Tensor,
    pub pending_rows: Tensor,
    pub(super) pending_rows_zero: Tensor,
    pub(super) spec: GdnDeferredStateSpec,
    pub(super) capacity: usize,
    pub(super) num_v_heads: usize,
    pub(super) head_k_dim: usize,
    pub(super) head_v_dim: usize,
    pub(super) state_layout: RecurrentStateLayout,
    pub(super) device: Device,
}

impl GdnDeferredStatePool {
    pub(super) fn new(
        capacity: usize,
        state_dims: &[usize],
        state_layout: RecurrentStateLayout,
        device: &Device,
        spec: GdnDeferredStateSpec,
    ) -> Result<Self> {
        let (num_v_heads, head_k_dim, head_v_dim) = match (state_layout, state_dims) {
            (RecurrentStateLayout::GdnKeyMajor, [heads, key_dim, value_dim]) => {
                (*heads, *key_dim, *value_dim)
            }
            (RecurrentStateLayout::GdnValueMajor, [heads, value_dim, key_dim]) => {
                (*heads, *key_dim, *value_dim)
            }
            _ => candle_core::bail!("deferred state requires a GDN recurrent state layout"),
        };
        if spec.num_k_heads == 0
            || spec.depth != crate::cuda::gdn::GDN_DEFERRED_STATE_DEPTH
            || num_v_heads == 0
            || head_k_dim == 0
            || head_v_dim == 0
            || !num_v_heads.is_multiple_of(spec.num_k_heads)
        {
            candle_core::bail!("GDN deferred state dimensions are incompatible");
        }

        Ok(Self {
            key: Tensor::zeros(
                (capacity, spec.depth, spec.num_k_heads, head_k_dim),
                DType::F32,
                device,
            )?,
            delta: Tensor::zeros(
                (capacity, spec.depth, num_v_heads, head_v_dim),
                DType::F32,
                device,
            )?,
            decay: Tensor::zeros((capacity, spec.depth, num_v_heads), DType::F32, device)?,
            pending_rows: Tensor::zeros((capacity,), DType::U32, device)?,
            pending_rows_zero: Tensor::zeros((1,), DType::U32, device)?,
            spec,
            capacity,
            num_v_heads,
            head_k_dim,
            head_v_dim,
            state_layout,
            device: device.clone(),
        })
    }

    pub(super) fn resized(&self, new_capacity: usize) -> Result<Self> {
        let state_dims = match self.state_layout {
            RecurrentStateLayout::GdnKeyMajor => {
                [self.num_v_heads, self.head_k_dim, self.head_v_dim]
            }
            RecurrentStateLayout::GdnValueMajor => {
                [self.num_v_heads, self.head_v_dim, self.head_k_dim]
            }
            RecurrentStateLayout::Opaque => unreachable!("deferred state requires GDN state"),
        };
        let resized = Self::new(
            new_capacity,
            &state_dims,
            self.state_layout,
            &self.device,
            self.spec,
        )?;
        resized.key.slice_set(&self.key, 0, 0)?;
        resized.delta.slice_set(&self.delta, 0, 0)?;
        resized.decay.slice_set(&self.decay, 0, 0)?;
        resized.pending_rows.slice_set(&self.pending_rows, 0, 0)?;
        Ok(resized)
    }

    pub(super) fn clear_slot(&self, slot_idx: usize) -> Result<()> {
        if slot_idx >= self.capacity {
            candle_core::bail!(
                "GDN deferred state slot {slot_idx} exceeds capacity {}",
                self.capacity
            );
        }
        self.pending_rows
            .slice_set(&self.pending_rows_zero, 0, slot_idx)
    }

    pub(super) fn clear_metadata(&self) -> Result<()> {
        let zeros = Tensor::zeros((self.capacity,), DType::U32, &self.device)?;
        self.pending_rows.slice_set(&zeros, 0, 0)
    }

    pub(crate) fn materialize(&self, state: &Tensor, slots: &Tensor) -> Result<()> {
        if slots.elem_count() == 0 {
            return Ok(());
        }
        crate::cuda::gdn::flush_deferred_state_cuda(crate::cuda::gdn::GdnDeferredStateFlush {
            state_pool: state,
            active_slots: slots,
            deferred_key: &self.key,
            deferred_delta: &self.delta,
            deferred_decay: &self.decay,
            deferred_cursor: &self.pending_rows,
            num_k_heads: self.spec.num_k_heads,
            num_v_heads: self.num_v_heads,
            head_k_dim: self.head_k_dim,
            head_v_dim: self.head_v_dim,
            tiled_v_heads: self.spec.tiled_v_heads,
            state_layout: self.state_layout,
        })
    }

    pub fn spec(&self) -> GdnDeferredStateSpec {
        self.spec
    }

    pub(crate) fn storage_bytes(&self) -> Result<usize> {
        [&self.key, &self.delta, &self.decay, &self.pending_rows]
            .into_iter()
            .try_fold(0usize, |bytes, tensor| {
                let tensor_bytes = tensor
                    .elem_count()
                    .checked_mul(tensor.dtype().size_in_bytes())
                    .ok_or_else(|| {
                        candle_core::Error::msg("GDN deferred state storage size overflow")
                    })?;
                bytes.checked_add(tensor_bytes).ok_or_else(|| {
                    candle_core::Error::msg("GDN deferred state storage size overflow")
                })
            })
    }
}
