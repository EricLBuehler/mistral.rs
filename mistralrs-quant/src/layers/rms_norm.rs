use candle_core::{CustomOp2, InplaceOp2, Result, Tensor};

pub const CAN_USE_FAST_RMSNORM: bool = cfg!(feature = "metal");

pub struct RmsNorm {
    eps: f32,
}

impl CustomOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "mistralrs-quant-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::CpuStorage, candle_core::Shape)> {
        candle_core::bail!("No CPU implementation of mistralrs-quant-rms-norm")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x_s: &candle_core::MetalStorage,
        x_l: &candle_core::Layout,
        w_s: &candle_core::MetalStorage,
        w_l: &candle_core::Layout,
    ) -> candle_core::Result<(candle_core::MetalStorage, candle_core::Shape)> {
        use candle_core::{backend::BackendStorage, D};

        use crate::metal_kernels;

        if x_l.dims().len() == 0 {
            candle_core::bail!("Expected xs to have at least 1 dim.");
        }
        if w_l.dims().len() != 1 {
            candle_core::bail!("Expected ws to have at 1 dim.");
        }
        if w_l.dim(D::Minus1)? != x_l.dim(D::Minus1)? {
            candle_core::bail!("Weight size must match xs last dim.");
        }
        if w_s.dtype() != x_s.dtype() {
            candle_core::bail!("Weight and xs dtypes must match.");
        }

        let command_buffer = x_s.device().command_buffer()?;
        command_buffer.set_label("mistralrs-quant-rms-norm");

        let device = x_s.device();

        let output = device.new_buffer(
            x_l.shape().elem_count(),
            x_s.dtype(),
            "mistralrs-quant-rms-norm",
        )?;

        metal_kernels::call_rms_norm(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            x_s.dtype(),
            x_s.buffer(),
            x_l.start_offset() * x_s.dtype().size_in_bytes(),
            x_l.dims(),
            w_s.buffer(),
            w_l.start_offset() * w_s.dtype().size_in_bytes(),
            w_l.stride(),
            &output,
            0,
            self.eps,
        )
        .map_err(candle_core::Error::wrap)?;

        let newstorage = candle_core::MetalStorage::new(
            output,
            device.clone(),
            x_l.shape().elem_count(),
            x_s.dtype(),
        );
        Ok((newstorage, x_l.shape().clone()))
    }
}

impl InplaceOp2 for RmsNorm {
    fn name(&self) -> &'static str {
        "mistralrs-quant-rms-norm"
    }

    fn cpu_fwd(
        &self,
        _s1: &mut candle_core::CpuStorage,
        _l1: &candle_core::Layout,
        _s2: &candle_core::CpuStorage,
        _l2: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        candle_core::bail!("No CPU implementation of mistralrs-quant-rms-norm")
    }

    #[cfg(feature = "metal")]
    fn metal_fwd(
        &self,
        x_s: &mut candle_core::MetalStorage,
        x_l: &candle_core::Layout,
        w_s: &candle_core::MetalStorage,
        w_l: &candle_core::Layout,
    ) -> candle_core::Result<()> {
        use candle_core::{backend::BackendStorage, D};

        use crate::metal_kernels;

        if x_l.dims().len() == 0 {
            candle_core::bail!("Expected xs to have at least 1 dim.");
        }
        if w_l.dims().len() != 1 {
            candle_core::bail!("Expected ws to have at 1 dim.");
        }
        if w_l.dim(D::Minus1)? != x_l.dim(D::Minus1)? {
            candle_core::bail!("Weight size must match xs last dim.");
        }

        let command_buffer = x_s.device().command_buffer()?;
        command_buffer.set_label("mistralrs-quant-rms-norm");

        let device = x_s.device();

        metal_kernels::call_rms_norm(
            device.device(),
            &command_buffer,
            &crate::metal_kernels::Kernels::new(),
            x_s.dtype(),
            x_s.buffer(),
            x_l.start_offset() * x_s.dtype().size_in_bytes(),
            x_l.dims(),
            w_s.buffer(),
            w_l.start_offset() * w_s.dtype().size_in_bytes(),
            w_l.stride(),
            &x_s.buffer(),
            x_l.start_offset() * x_s.dtype().size_in_bytes(),
            self.eps,
        )
        .map_err(candle_core::Error::wrap)?;

        Ok(())
    }
}

/// RmsNorm operation. Last dim of xs must match weight length.
pub fn rms_norm(xs: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    xs.apply_op2_no_bwd(weight, &RmsNorm { eps })
}

/// Inplace RmsNorm operation. Last dim of xs must match weight length.
pub fn rms_norm_inplace(xs: &mut Tensor, weight: &Tensor, eps: f32) -> Result<()> {
    xs.inplace_op2(weight, &RmsNorm { eps })
}
