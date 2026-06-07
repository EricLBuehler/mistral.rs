use candle_core::{DType, Device, Result, Tensor};

use super::config::{GdnConfig, GdnDims};

#[derive(Debug)]
pub struct GdnLayerCache {
    pub conv_state: Tensor,
    pub recurrent_state: Tensor,
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
        Ok(Self {
            conv_state,
            recurrent_state,
        })
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
        }
    }
}
