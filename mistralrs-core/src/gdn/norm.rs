use candle_core::{DType, Device, Result, Tensor, D};
#[cfg(feature = "cuda")]
use mistralrs_quant::QuantizedActivation;
use mistralrs_quant::ShardedVarBuilder;

#[cfg(feature = "cuda")]
use crate::cuda::gdn::GdnFp8OutputSpec;

pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
    #[cfg(test)]
    pub(crate) fn from_parts(weight: Tensor, eps: f64) -> Self {
        Self { weight, eps }
    }

    pub fn new(
        size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self { weight, eps })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn eps(&self) -> f64 {
        self.eps
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        #[cfg(any(feature = "cuda", feature = "metal"))]
        if self.fused_kernel_applies(x, gate)? {
            let weight = self.weight.to_dtype(x.dtype())?;
            #[cfg(feature = "cuda")]
            if x.device().is_cuda() {
                return crate::cuda::gdn::rmsnorm_gated_cuda(x, gate, &weight, self.eps);
            }
            #[cfg(feature = "metal")]
            if x.device().is_metal() {
                return crate::metal::gdn::rmsnorm_gated_metal(x, gate, &weight, self.eps);
            }
        }

        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.reshape(x.shape().clone())?.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
    }

    #[cfg(any(feature = "cuda", feature = "metal"))]
    fn fused_kernel_applies(&self, x: &Tensor, gate: &Tensor) -> Result<bool> {
        Ok((x.device().is_cuda() || x.device().is_metal())
            && (2..=4).contains(&x.rank())
            && (2..=4).contains(&gate.rank())
            && gate.elem_count() == x.elem_count()
            && x.dim(D::Minus1)? == self.weight.elem_count()
            && gate.dtype() == x.dtype()
            && matches!(x.dtype(), DType::F16 | DType::BF16))
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_quantized(
        &self,
        x: &Tensor,
        gate: &Tensor,
        spec: &GdnFp8OutputSpec,
        num_v_heads: usize,
        head_v_dim: usize,
    ) -> Result<QuantizedActivation> {
        crate::cuda::gdn::rmsnorm_gated_quantized_cuda(
            x,
            gate,
            &self.weight,
            self.eps,
            spec,
            num_v_heads,
            head_v_dim,
        )
    }
}

#[cfg(all(test, feature = "metal"))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::RmsNormGated;

    #[test]
    fn metal_gated_rmsnorm_matches_reference() -> Result<()> {
        let metal = Device::new_metal(0)?;
        let (rows, dim) = (48, 128);
        for (dtype, tol) in [(DType::F16, 5e-3f32), (DType::BF16, 2e-2)] {
            let x = Tensor::randn(0f32, 1., (1, rows, dim), &Device::Cpu)?.to_dtype(dtype)?;
            let gate = Tensor::randn(0f32, 1., (1, rows, dim), &Device::Cpu)?.to_dtype(dtype)?;
            let weight = Tensor::randn(1f32, 0.1, dim, &Device::Cpu)?.to_dtype(dtype)?;
            let reference = RmsNormGated::from_parts(weight.clone(), 1e-6).forward(&x, &gate)?;
            let actual = RmsNormGated::from_parts(weight.to_device(&metal)?, 1e-6)
                .forward(&x.to_device(&metal)?, &gate.to_device(&metal)?)?;
            let diff = (actual.to_device(&Device::Cpu)?.to_dtype(DType::F32)?
                - reference.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
            assert!(diff < tol, "{dtype:?}: max abs diff {diff}");
        }
        Ok(())
    }
}
