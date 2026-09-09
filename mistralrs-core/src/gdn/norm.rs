use candle_core::{DType, Device, Result, Tensor, D};
#[cfg(feature = "cuda")]
use mistralrs_quant::QuantizedActivation;
use mistralrs_quant::ShardedVarBuilder;

#[cfg(feature = "cuda")]
use crate::cuda::gdn::GdnFp8OutputSpec;

use super::GdnOutputGate;

pub struct RmsNormGated {
    pub weight: Tensor,
    eps: f64,
    output_gate: GdnOutputGate,
}

impl RmsNormGated {
    #[cfg(test)]
    pub(crate) fn from_parts(weight: Tensor, eps: f64) -> Self {
        Self::from_parts_with_gate(weight, eps, GdnOutputGate::Silu)
    }

    #[cfg(test)]
    pub(crate) fn from_parts_with_gate(
        weight: Tensor,
        eps: f64,
        output_gate: GdnOutputGate,
    ) -> Self {
        Self {
            weight,
            eps,
            output_gate,
        }
    }

    pub fn new(
        size: usize,
        eps: f64,
        vb: ShardedVarBuilder,
        isq_target_device: Option<&Device>,
        output_gate: GdnOutputGate,
    ) -> Result<Self> {
        let mut weight = vb.get(size, "weight")?;
        if let Some(target_dev) = isq_target_device {
            weight = weight.to_device(target_dev)?;
        }
        Ok(Self {
            weight,
            eps,
            output_gate,
        })
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn eps(&self) -> f64 {
        self.eps
    }

    pub(crate) fn output_gate(&self) -> GdnOutputGate {
        self.output_gate
    }

    pub fn forward(&self, x: &Tensor, gate: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if self.output_gate == GdnOutputGate::Silu
            && x.device().is_cuda()
            && (2..=4).contains(&x.rank())
            && (2..=4).contains(&gate.rank())
            && gate.elem_count() == x.elem_count()
            && x.dim(D::Minus1)? == self.weight.elem_count()
            && gate.dtype() == x.dtype()
            && self.weight.dtype() == x.dtype()
            && matches!(x.dtype(), DType::F16 | DType::BF16)
        {
            return crate::cuda::gdn::rmsnorm_gated_cuda(x, gate, &self.weight, self.eps);
        }

        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let gate = gate.reshape(x.shape().clone())?.to_dtype(DType::F32)?;
        let gate = match self.output_gate {
            GdnOutputGate::Silu => candle_nn::ops::silu(&gate)?,
            GdnOutputGate::Sigmoid => candle_nn::ops::sigmoid(&gate)?,
        };
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        let out = normed
            .broadcast_mul(&self.weight.to_dtype(DType::F32)?)?
            .broadcast_mul(&gate)?;
        out.to_dtype(dtype)
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
        debug_assert_eq!(self.output_gate, GdnOutputGate::Silu);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_sigmoid_gate_matches_reference(dtype: DType, tolerance: f32) -> Result<()> {
        let device = Device::Cpu;
        let x_values = [-1.5f32, -0.5, 0.25, 2.0, 1.25, -2.5, 0.75, 0.5];
        let gate_values = [-3.0f32, -1.0, 0.0, 2.0, 4.0, -2.0, 1.5, 0.5];
        let weight_values = [0.5f32, 1.0, 1.5, 2.0];
        let eps = 1e-6;
        let x = Tensor::from_slice(&x_values, (1, 2, 4), &device)?.to_dtype(dtype)?;
        let gate = Tensor::from_slice(&gate_values, (1, 2, 4), &device)?.to_dtype(dtype)?;
        let weight = Tensor::from_slice(&weight_values, 4, &device)?.to_dtype(dtype)?;
        let norm = RmsNormGated::from_parts_with_gate(weight, eps, GdnOutputGate::Sigmoid);

        let actual = norm
            .forward(&x, &gate)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let quantized_x = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let quantized_gate = gate.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let quantized_weight = norm.weight.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let mut expected = Vec::with_capacity(x_values.len());
        for row in 0..2 {
            let start = row * 4;
            let variance = quantized_x[start..start + 4]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / 4.0;
            let inv_rms = (variance + eps as f32).sqrt().recip();
            for (column, &weight) in quantized_weight.iter().enumerate() {
                let index = start + column;
                let sigmoid = 1.0 / (1.0 + (-quantized_gate[index]).exp());
                expected.push(quantized_x[index] * inv_rms * weight * sigmoid);
            }
        }

        for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
            );
        }
        Ok(())
    }

    #[test]
    fn sigmoid_gated_rms_norm_matches_f32_reference() -> Result<()> {
        assert_sigmoid_gate_matches_reference(DType::F32, 1e-6)
    }

    #[test]
    fn sigmoid_gated_rms_norm_matches_f16_reference() -> Result<()> {
        assert_sigmoid_gate_matches_reference(DType::F16, 2e-3)
    }

    #[test]
    fn sigmoid_gated_rms_norm_matches_bf16_reference() -> Result<()> {
        assert_sigmoid_gate_matches_reference(DType::BF16, 2e-2)
    }
}
