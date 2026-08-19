use candle_core::{DType, Device, Result, Tensor};

use crate::kv_cache::RecurrentStatePool;

use super::config::{GdnConfig, GdnDims};

/// Per-layer GDN state handed to the kernels. Gathered: `conv_state` `[B, conv_dim, k]` and
/// `recurrent_state` `[B, H, K, V]` hold this batch's rows. Pooled (`slots` set): they are the whole
/// recurrent state pool and `slots` is the `[B]` u32 row table, so CUDA kernels update it in place.
#[derive(Debug)]
pub struct GdnLayerCache {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
    pub slots: Option<Tensor>,
}

#[allow(dead_code)]
impl GdnLayerCache {
    pub fn new(cfg: &dyn GdnConfig, dtype: DType, device: &Device) -> Result<Self> {
        let dims = GdnDims::new(cfg);
        let conv_state = Tensor::zeros((1, dims.conv_dim, dims.conv_kernel_size), dtype, device)?;
        let recurrent_state = Tensor::zeros(
            (1, dims.num_v_heads, dims.head_k_dim, dims.head_v_dim),
            DType::F32,
            device,
        )?;
        Ok(Self::gathered(conv_state, recurrent_state))
    }

    pub fn gathered(conv_state: Tensor, recurrent_state: Tensor) -> Self {
        Self {
            conv_state,
            recurrent_state,
            slots: None,
        }
    }

    pub fn pooled(conv_state: Tensor, recurrent_state: Tensor, slots: Tensor) -> Self {
        Self {
            conv_state,
            recurrent_state,
            slots: Some(slots),
        }
    }

    /// Check out the rows `indices` of `pool` for one layer forward. On CUDA the kernels update the
    /// pool in place through the slot table; elsewhere this is a gathered copy that `commit` scatters back.
    pub fn checkout(pool: &RecurrentStatePool, indices: &Tensor) -> Result<Self> {
        if pool.device().is_cuda() {
            return Ok(Self::pooled(
                pool.conv_state.clone(),
                pool.recurrent_state.clone(),
                indices.clone(),
            ));
        }
        Ok(Self::gathered(
            pool.gather_conv_state(indices)?,
            pool.gather_recurrent_state(indices)?,
        ))
    }

    pub fn commit(
        self,
        pool: &mut RecurrentStatePool,
        indices: &Tensor,
        host_indices: Option<&[u32]>,
    ) -> Result<()> {
        if self.slots.is_some() {
            return Ok(());
        }
        pool.scatter_conv_state_with_host_indices(indices, host_indices, &self.conv_state)?;
        pool.scatter_recurrent_state_with_host_indices(indices, host_indices, &self.recurrent_state)
    }

    pub fn reset(&mut self) -> Result<()> {
        self.conv_state = self.conv_state.zeros_like()?;
        self.recurrent_state = self.recurrent_state.zeros_like()?;
        Ok(())
    }
}

impl Clone for GdnLayerCache {
    fn clone(&self) -> Self {
        Self {
            conv_state: self.conv_state.clone(),
            recurrent_state: self.recurrent_state.clone(),
            slots: self.slots.clone(),
        }
    }
}
