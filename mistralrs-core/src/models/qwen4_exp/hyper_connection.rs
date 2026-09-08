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
        let residual_f32 = residual.to_dtype(DType::F32)?;
        let variance = residual_f32.sqr()?.mean_keepdim(D::Minus1)?;
        let normalized = residual_f32.broadcast_div(&(variance + self.norm_eps)?.sqrt()?)?;
        let flattened = normalized
            .reshape((batch, tokens, wide_size))?
            .broadcast_mul(&self.norm_weight.to_dtype(DType::F32)?)?
            .to_dtype(dtype)?;
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
        let device = &Device::Cpu;
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
}
