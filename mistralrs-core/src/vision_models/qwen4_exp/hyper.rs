use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::{QuantMethod, ReplicatedLayer, ShardedVarBuilder};

use super::config::TextConfig;

/// Low-rank gated mixer over the `hc` residual streams; it stands in for every layernorm.
pub(super) struct GatedResidual {
    // (1 + w) in f32, [hc * hidden]
    pub(super) norm_weight: Tensor,
    pub(super) down: Arc<dyn QuantMethod>,
    pub(super) up: Arc<dyn QuantMethod>,
    pub(super) inject: Option<Arc<dyn QuantMethod>>,
    hc: usize,
    hidden: usize,
    eps: f64,
}

impl GatedResidual {
    pub(super) fn load(cfg: &TextConfig, vb: ShardedVarBuilder, with_inject: bool) -> Result<Self> {
        let hc_hidden = cfg.hc_hidden_size();
        let norm_weight = (vb
            .pp("hc_norm")
            .get(hc_hidden, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
        let down = ReplicatedLayer::new(
            hc_hidden,
            cfg.hc_lowrank,
            &cfg.quantization_config,
            false,
            vb.pp("input_mix_weight_down"),
        )?;
        let up = ReplicatedLayer::new(
            cfg.hc_lowrank,
            hc_hidden,
            &cfg.quantization_config,
            false,
            vb.pp("input_mix_weight_up"),
        )?;
        let inject = with_inject
            .then(|| {
                ReplicatedLayer::new(
                    hc_hidden,
                    cfg.hc_count,
                    &cfg.quantization_config,
                    false,
                    vb.pp("block_inject_weight"),
                )
            })
            .transpose()?;
        Ok(Self {
            norm_weight,
            down,
            up,
            inject,
            hc: cfg.hc_count,
            hidden: cfg.hidden_size,
            eps: cfg.rms_norm_eps,
        })
    }

    pub(super) fn norm(&self, res: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_norm(res, &self.norm_weight, self.hc, self.eps);
        }
        grouped_rms_norm(res, &self.norm_weight, self.hc, self.eps)
    }

    /// The block input: mean over streams of `xn * sigmoid(up(silu(down(xn) / hc)))`.
    pub(super) fn mix(&self, xn: &Tensor) -> Result<Tensor> {
        let lo = self.down.forward(xn)?;
        let lo = candle_nn::ops::silu(&(lo / self.hc as f64)?)?;
        let gate = self.up.forward(&lo)?;
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_mix(xn, &gate, self.hc);
        }
        let mut shape = xn.dims().to_vec();
        let last = shape.len() - 1;
        shape[last] = self.hc;
        shape.push(self.hidden);
        let gated = (xn * candle_nn::ops::sigmoid(&gate)?)?.reshape(shape)?;
        gated
            .to_dtype(DType::F32)?
            .mean(D::Minus2)?
            .to_dtype(xn.dtype())
    }

    /// Raw per-stream injection logits `[.., hc]`.
    pub(super) fn inject(&self, xn: &Tensor) -> Result<Tensor> {
        self.inject
            .as_ref()
            .expect("final hyper-connection mixer has no injection")
            .forward(xn)
    }

    /// `res + out * 2 * sigmoid(inject / hc)` per stream, plus the grouped norm `next` needs.
    pub(super) fn combine(
        &self,
        res: &Tensor,
        out: &Tensor,
        inject: &Tensor,
        next: Option<&GatedResidual>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let next = next.filter(|next| next.norm_weight.device().same_device(res.device()));
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_combine(
                res,
                out,
                inject,
                next.map(|next| &next.norm_weight),
                self.hc,
                self.eps,
            );
        }
        let dims = res.dims().to_vec();
        let mut stream_shape = dims.clone();
        let last = stream_shape.len() - 1;
        stream_shape[last] = self.hc;
        stream_shape.push(self.hidden);
        let weights =
            (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / self.hc as f64)?)? * 2.0)?;
        let injection = out
            .to_dtype(DType::F32)?
            .unsqueeze(D::Minus2)?
            .broadcast_mul(&weights.unsqueeze(D::Minus1)?)?;
        let res = (res.to_dtype(DType::F32)?.reshape(stream_shape)? + injection)?
            .reshape(dims)?
            .to_dtype(res.dtype())?;
        let xn = next.map(|next| next.norm(&res)).transpose()?;
        Ok((res, xn))
    }
}

/// RMSNorm of each `hidden`-wide group of the last dim, times a per-group weight.
pub(super) fn grouped_rms_norm(
    x: &Tensor,
    weight: &Tensor,
    groups: usize,
    eps: f64,
) -> Result<Tensor> {
    let dims = x.dims().to_vec();
    let width = dims[dims.len() - 1];
    let hidden = width / groups;
    let mut grouped = dims.clone();
    let last = grouped.len() - 1;
    grouped[last] = groups;
    grouped.push(hidden);
    let xf = x.to_dtype(DType::F32)?.reshape(grouped)?;
    let inv = (xf.sqr()?.mean_keepdim(D::Minus1)? + eps)?
        .sqrt()?
        .recip()?;
    let normed = xf.broadcast_mul(&inv)?.reshape(dims)?;
    normed
        .broadcast_mul(&weight.to_device(x.device())?)?
        .to_dtype(x.dtype())
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use candle_core::{DType, Device, Result, Tensor};

    use super::grouped_rms_norm;

    fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
        (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()
    }

    #[test]
    fn fused_hyper_connection_kernels_match_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let (tokens, hc, hidden) = (5, 4, 96);
        let res =
            Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(DType::BF16)?;
        let weight = Tensor::randn(1f32, 0.1, hc * hidden, &device)?;
        let eps = 1e-6;

        let fused = crate::cuda::qwen4_exp::hc_norm(&res, &weight, hc, eps)?;
        let reference = grouped_rms_norm(&res, &weight, hc, eps)?;
        assert!(max_diff(&fused, &reference)? < 0.05);

        let gate =
            Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(DType::BF16)?;
        let mixed = crate::cuda::qwen4_exp::hc_mix(&fused, &gate, hc)?;
        let reference = (fused.to_dtype(DType::F32)?
            * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
        .reshape((tokens, hc, hidden))?
        .mean(1)?;
        assert!(max_diff(&mixed, &reference)? < 0.02);

        let out = Tensor::randn(0f32, 1.0, (tokens, hidden), &device)?.to_dtype(DType::BF16)?;
        let inject = Tensor::randn(0f32, 1.0, (tokens, hc), &device)?.to_dtype(DType::BF16)?;
        let (combined, next) =
            crate::cuda::qwen4_exp::hc_combine(&res, &out, &inject, Some(&weight), hc, eps)?;
        let w = (candle_nn::ops::sigmoid(&(inject.to_dtype(DType::F32)? / hc as f64)?)? * 2.0)?;
        let reference = (res.to_dtype(DType::F32)?.reshape((tokens, hc, hidden))?
            + out
                .to_dtype(DType::F32)?
                .unsqueeze(1)?
                .broadcast_mul(&w.unsqueeze(2)?)?)?
        .reshape((tokens, hc * hidden))?;
        assert!(max_diff(&combined, &reference)? < 0.05);
        let next_reference = grouped_rms_norm(&combined, &weight, hc, eps)?;
        assert!(max_diff(&next.unwrap(), &next_reference)? < 0.05);
        Ok(())
    }
}
