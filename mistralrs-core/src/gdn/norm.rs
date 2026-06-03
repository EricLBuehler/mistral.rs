use candle_core::{DType, Device, Result, Tensor, D};
use mistralrs_quant::ShardedVarBuilder;

pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
}

impl RmsNormGated {
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

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if x.device().is_cuda()
            && x.rank() == 2
            && gate.shape() == x.shape()
            && self.weight.dtype() == x.dtype()
            && matches!(x.dtype(), DType::F16 | DType::BF16)
        {
            return crate::cuda::gdn::rmsnorm_gated_cuda(x, gate, &self.weight, self.eps);
        }

        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = candle_nn::ops::silu(&gate.to_dtype(DType::F32)?)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
    }
}
