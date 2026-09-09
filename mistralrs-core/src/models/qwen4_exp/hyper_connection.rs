#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::config::Config;

#[allow(dead_code)]
pub(crate) struct GatedResidual {
    norm_weight: Tensor,
    norm_eps: f64,
    down: Arc<dyn QuantMethod>,
    up: Arc<dyn QuantMethod>,
    inject: Option<Arc<dyn QuantMethod>>,
    hidden_size: usize,
    hc_count: usize,
}

#[allow(dead_code)]
impl GatedResidual {
    pub(crate) fn new(
        config: &Config,
        vb: ShardedVarBuilder,
        with_injection: bool,
    ) -> Result<Self> {
        let wide_size = config
            .hidden_size
            .checked_mul(config.hc_count)
            .ok_or_else(|| candle_core::Error::msg("Qwen4Exp hyper-connection width overflow"))?;
        let norm_weight = vb.pp("norm").get(wide_size, "weight")?;
        let down = ReplicatedLayer::new(
            wide_size,
            config.hc_lowrank,
            &config.quantization_config,
            false,
            vb.pp("down"),
        )?;
        let up = ReplicatedLayer::new(
            config.hc_lowrank,
            wide_size,
            &config.quantization_config,
            false,
            vb.pp("up"),
        )?;
        let inject = with_injection
            .then(|| {
                ReplicatedLayer::new(
                    wide_size,
                    config.hc_count,
                    &config.quantization_config,
                    false,
                    vb.pp("inject"),
                )
            })
            .transpose()?;
        Ok(Self {
            norm_weight,
            norm_eps: config.rms_norm_eps,
            down,
            up,
            inject,
            hidden_size: config.hidden_size,
            hc_count: config.hc_count,
        })
    }

    pub(crate) fn mix(&self, residual: &Tensor) -> Result<(Tensor, Option<Tensor>)> {
        let (batch, tokens, streams, hidden) = residual.dims4()?;
        if streams != self.hc_count || hidden != self.hidden_size {
            candle_core::bail!(
                "Qwen4Exp hyper-connection expected residual shape [batch, tokens, {}, {}], got {:?}",
                self.hc_count,
                self.hidden_size,
                residual.dims()
            );
        }
        let wide_size = self.hc_count * self.hidden_size;
        let dtype = residual.dtype();
        let flattened = self.normalized_flattened(residual, batch, tokens, wide_size, dtype)?;

        let low_rank =
            candle_nn::ops::silu(&(self.down.forward(&flattened)? / self.hc_count as f64)?)?;
        let gate = candle_nn::ops::sigmoid(&self.up.forward(&low_rank)?)?;
        let gated = (flattened.clone() * gate)?.reshape((
            batch,
            tokens,
            self.hc_count,
            self.hidden_size,
        ))?;
        let mixed = gated.sum(D::Minus2)? / self.hc_count as f64;
        let injection = self
            .inject
            .as_ref()
            .map(|projection| projection.forward(&flattened))
            .transpose()?;
        Ok((mixed?, injection))
    }

    /// Per-stream RMS normalization plus the flattened gamma multiply, emitting the
    /// projection input in the flattened `[batch, tokens, hc_count * hidden]` layout
    /// with the activation dtype preserved.
    ///
    /// Uses the fused Metal kernel when the inputs qualify (one kernel instead of the
    /// composed cast/sqr/mean/sqrt/div/mul chain); the grouped and flattened layouts are
    /// row-major identical, so the kernel writes the same flat offsets. Everything else
    /// keeps the composed Candle path.
    fn normalized_flattened(
        &self,
        residual: &Tensor,
        batch: usize,
        tokens: usize,
        wide_size: usize,
        dtype: DType,
    ) -> Result<Tensor> {
        #[cfg(feature = "metal")]
        if residual.device().is_metal()
            && residual.is_contiguous()
            && self.norm_weight.is_contiguous()
            && self.norm_weight.dtype() == dtype
            && matches!(dtype, DType::F32 | DType::F16 | DType::BF16)
        {
            return crate::metal::qwen4exp::hc_rmsnorm_flatten_metal(
                residual,
                &self.norm_weight,
                self.norm_eps,
            );
        }
        let residual_f32 = residual.to_dtype(DType::F32)?;
        let variance = residual_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = residual_f32.broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?;
        normalized
            .reshape((batch, tokens, wide_size))?
            .broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)
    }

    /// Flattened-stream RMS gamma, exposed for ISQ residual handling.
    pub(crate) fn norm_weight(&self) -> &Tensor {
        &self.norm_weight
    }

    pub(crate) fn combine(
        &self,
        residual: &Tensor,
        branch_output: &Tensor,
        injection: &Tensor,
    ) -> Result<Tensor> {
        let (batch, tokens, streams, hidden) = residual.dims4()?;
        if streams != self.hc_count
            || hidden != self.hidden_size
            || branch_output.dims() != [batch, tokens, hidden]
            || injection.dims() != [batch, tokens, streams]
        {
            candle_core::bail!("Qwen4Exp hyper-connection combine received incompatible shapes");
        }
        let weights = (candle_nn::ops::sigmoid(&(injection / self.hc_count as f64)?)? * 2.0)?;
        residual
            + branch_output
                .unsqueeze(2)?
                .broadcast_mul(&weights.unsqueeze(D::Minus1)?)?
    }
}

#[cfg(test)]
mod tests {
    use candle_core::Device;
    use candle_nn::Linear;
    use mistralrs_quant::{QuantMethodConfig, UnquantLinear};

    use super::*;

    fn projection(weight: Tensor) -> Result<Arc<dyn QuantMethod>> {
        Ok(Arc::new(UnquantLinear::new(
            QuantMethodConfig::Unquantized(Linear::new(weight, None)),
        )?))
    }

    fn gated_residual() -> Result<GatedResidual> {
        gated_residual_on(&Device::Cpu)
    }

    fn gated_residual_on(device: &Device) -> Result<GatedResidual> {
        Ok(GatedResidual {
            norm_weight: Tensor::ones(4, DType::F32, device)?,
            norm_eps: 1e-6,
            down: projection(Tensor::from_slice(
                &[1f32, 0., 0., 0., 0., 1., 0., 0.],
                (2, 4),
                device,
            )?)?,
            up: projection(Tensor::from_slice(
                &[1f32, 0., 0., 1., 0., 0., 0., 0.],
                (4, 2),
                device,
            )?)?,
            inject: Some(projection(Tensor::zeros((2, 4), DType::F32, device)?)?),
            hidden_size: 2,
            hc_count: 2,
        })
    }

    #[test]
    fn mix_matches_small_reference() -> Result<()> {
        let component = gated_residual()?;
        let residual = Tensor::from_slice(&[1f32, 2., 3., 4.], (1, 1, 2, 2), &Device::Cpu)?;
        let (mixed, injection) = component.mix(&residual)?;
        let actual = mixed.flatten_all()?.to_vec1::<f32>()?;

        let inv0 = (2.5f32 + 1e-6).sqrt().recip();
        let inv1 = (12.5f32 + 1e-6).sqrt().recip();
        let normalized = [inv0, 2. * inv0, 3. * inv1, 4. * inv1];
        let silu = |value: f32| value / (1.0 + (-value).exp());
        let low = [silu(normalized[0] / 2.), silu(normalized[1] / 2.)];
        let expected = [
            (normalized[0] / (1. + (-low[0]).exp()) + normalized[2] * 0.5) / 2.,
            (normalized[1] / (1. + (-low[1]).exp()) + normalized[3] * 0.5) / 2.,
        ];
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() < 1e-5,
                "actual={actual}, expected={expected}"
            );
        }
        assert_eq!(
            injection.unwrap().to_vec3::<f32>()?,
            vec![vec![vec![0., 0.]]]
        );
        Ok(())
    }

    #[test]
    fn zero_injection_is_plain_residual_add() -> Result<()> {
        let component = gated_residual()?;
        let residual = Tensor::from_slice(&[1f32, 2., 3., 4.], (1, 1, 2, 2), &Device::Cpu)?;
        let branch = Tensor::from_slice(&[0.5f32, -1.], (1, 1, 2), &Device::Cpu)?;
        let injection = Tensor::zeros((1, 1, 2), DType::F32, &Device::Cpu)?;
        let actual = component
            .combine(&residual, &branch, &injection)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        assert_eq!(actual, vec![1.5, 1.0, 3.5, 3.0]);
        Ok(())
    }

    /// Completion Phase 7: the fused Metal hyper-connection norm/flatten prefix must
    /// track the composed CPU reference through the full mixer. Skips cleanly when no
    /// Metal device is present.
    #[cfg(feature = "metal")]
    #[test]
    fn metal_mix_matches_cpu_reference() -> Result<()> {
        let Ok(device) = Device::new_metal(0) else {
            return Ok(());
        };
        let cpu_component = gated_residual()?;
        let metal_component = gated_residual_on(&device)?;

        for (batch, tokens) in [(1usize, 1usize), (2usize, 3usize)] {
            let residual = Tensor::from_vec(
                (0..batch * tokens * 4)
                    .map(|index| ((index % 13) as f32 - 6.0) / 3.0)
                    .collect::<Vec<f32>>(),
                (batch, tokens, 2, 2),
                &Device::Cpu,
            )?;
            let (cpu_mixed, cpu_injection) = cpu_component.mix(&residual)?;
            let (metal_mixed, metal_injection) =
                metal_component.mix(&residual.to_device(&device)?)?;

            let cpu_vec = cpu_mixed.flatten_all()?.to_vec1::<f32>()?;
            let metal_vec = metal_mixed
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(cpu_vec.len(), metal_vec.len());
            for (cpu, metal) in cpu_vec.iter().zip(metal_vec.iter()) {
                assert!(
                    (cpu - metal).abs() < 1e-5,
                    "mixed: cpu={cpu}, metal={metal}"
                );
            }

            let cpu_inj = cpu_injection.unwrap().flatten_all()?.to_vec1::<f32>()?;
            let metal_inj = metal_injection
                .unwrap()
                .to_device(&Device::Cpu)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(cpu_inj.len(), metal_inj.len());
            for (cpu, metal) in cpu_inj.iter().zip(metal_inj.iter()) {
                assert!(
                    (cpu - metal).abs() < 1e-5,
                    "injection: cpu={cpu}, metal={metal}"
                );
            }
        }
        Ok(())
    }

    /// Direct-kernel coverage mirroring the GDN gated-RMS-norm metal tests: the fused
    /// norm/flatten kernel must track an F32 reference computed over identical quantized
    /// values, and the flattened gamma must stay `streams * hidden` wide (regression for
    /// the weight-inventory guard). Skips cleanly when no Metal device is present.
    #[cfg(feature = "metal")]
    mod metal_tests {
        use super::*;
        use crate::metal::qwen4exp::hc_rmsnorm_flatten_metal;

        const ROWS: usize = 2;
        const STREAMS: usize = 2;
        const HIDDEN: usize = 4;
        const WIDE: usize = STREAMS * HIDDEN;

        /// Per-(row, stream) F32 RMS normalization plus the flattened gamma, matching
        /// the composed path exactly.
        fn reference_hc_rmsnorm_flatten(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
            let mut expected = Vec::with_capacity(x.len());
            for row in 0..ROWS {
                for stream in 0..STREAMS {
                    let start = (row * STREAMS + stream) * HIDDEN;
                    let variance =
                        x[start..start + HIDDEN].iter().map(|v| v * v).sum::<f32>() / HIDDEN as f32;
                    let inv_rms = (variance + eps).sqrt().recip();
                    for column in 0..HIDDEN {
                        expected
                            .push(x[start + column] * inv_rms * weight[stream * HIDDEN + column]);
                    }
                }
            }
            expected
        }

        fn assert_metal_kernel_matches_reference(dtype: DType, tolerance: f32) -> Result<()> {
            let Ok(device) = Device::new_metal(0) else {
                return Ok(());
            };
            let eps = 1e-6;
            let x_values: Vec<f32> = (0..ROWS * WIDE)
                .map(|index| ((index % 13) as f32 - 6.0) / 3.0)
                .collect();
            let weight_values: Vec<f32> = (0..WIDE)
                .map(|index| ((index % 5) as f32 - 2.0) / 2.0 + 0.25)
                .collect();

            // Quantize to the activation dtype, then read the same quantized values back
            // in F32 so the reference uses identical inputs.
            let x_quant = Tensor::from_vec(x_values, (ROWS, STREAMS, HIDDEN), &Device::Cpu)?
                .to_dtype(dtype)?;
            let weight_quant =
                Tensor::from_vec(weight_values, WIDE, &Device::Cpu)?.to_dtype(dtype)?;
            let x_reference = x_quant
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let weight_reference = weight_quant
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let output = hc_rmsnorm_flatten_metal(
                &x_quant.to_device(&device)?,
                &weight_quant.to_device(&device)?,
                eps,
            )?;
            // The kernel emits the projection input in the flattened layout.
            assert_eq!(output.dims(), [ROWS, WIDE]);
            let actual = output
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;

            let expected =
                reference_hc_rmsnorm_flatten(&x_reference, &weight_reference, eps as f32);
            for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "element {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
                );
            }
            Ok(())
        }

        #[test]
        fn metal_hc_rmsnorm_flatten_matches_f16_reference() -> Result<()> {
            assert_metal_kernel_matches_reference(DType::F16, 2e-3)
        }

        #[test]
        fn metal_hc_rmsnorm_flatten_matches_bf16_reference() -> Result<()> {
            assert_metal_kernel_matches_reference(DType::BF16, 2e-2)
        }

        /// The flattened gamma is `streams * hidden` wide and indexed by stream, not the
        /// full residual element count; a mismatched gamma must fail closed.
        #[test]
        fn metal_hc_rmsnorm_flatten_rejects_undersized_gamma() -> Result<()> {
            let Ok(device) = Device::new_metal(0) else {
                return Ok(());
            };
            let residual = Tensor::zeros((ROWS, STREAMS, HIDDEN), DType::F32, &device)?;
            let undersized = Tensor::zeros(HIDDEN, DType::F32, &device)?;
            let error = hc_rmsnorm_flatten_metal(&residual, &undersized, 1e-6)
                .expect_err("undersized flattened gamma must be rejected");
            assert!(
                error
                    .to_string()
                    .contains("incompatible residual/weight shapes"),
                "unexpected error: {error}"
            );
            Ok(())
        }
    }
}
