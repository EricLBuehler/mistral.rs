use std::sync::Arc;

use candle_core::{DType, Result, Tensor, D};
use mistralrs_quant::{
    LoraLinearSpec, QuantMethod, QuantizedConfig, ReplicatedLayer, ShardedVarBuilder,
};

pub(crate) struct GatedResidualConfig {
    pub streams: usize,
    pub hidden: usize,
    pub rank: usize,
    pub eps: f64,
}

/// Low-rank gated mixer over the `hc` residual streams; it stands in for every layernorm.
pub(crate) struct GatedResidual {
    // (1 + w) in f32, [hc * hidden]
    pub(crate) norm_weight: Tensor,
    pub(crate) down: Arc<dyn QuantMethod>,
    pub(crate) up: Arc<dyn QuantMethod>,
    pub(crate) inject: Option<Arc<dyn QuantMethod>>,
    merged_down_inject: Option<crate::ops::MergedDenseProjection>,
    hc: usize,
    hidden: usize,
    eps: f64,
}

impl GatedResidual {
    pub(crate) fn load(
        cfg: &GatedResidualConfig,
        quantization: &Option<QuantizedConfig>,
        vb: ShardedVarBuilder,
        with_inject: bool,
    ) -> Result<Self> {
        let hc_hidden = cfg.streams * cfg.hidden;
        let norm_weight = (vb
            .pp("hc_norm")
            .get(hc_hidden, "weight")?
            .to_dtype(DType::F32)?
            + 1.0)?;
        let packed = if with_inject {
            ReplicatedLayer::new_packed(
                &[
                    LoraLinearSpec::replicated(hc_hidden, cfg.rank),
                    LoraLinearSpec::replicated(hc_hidden, cfg.streams),
                ],
                &["input_mix_weight_down", "block_inject_weight"],
                quantization,
                false,
                vb.clone(),
            )?
        } else {
            None
        };
        let (down, inject, merged_down_inject) = if let Some(group) = packed {
            (
                group.constituents[0].clone(),
                Some(group.constituents[1].clone()),
                Some(crate::ops::MergedDenseProjection::from_packed(&group)),
            )
        } else {
            (
                ReplicatedLayer::new(
                    hc_hidden,
                    cfg.rank,
                    quantization,
                    false,
                    vb.pp("input_mix_weight_down"),
                )?,
                with_inject
                    .then(|| {
                        ReplicatedLayer::new(
                            hc_hidden,
                            cfg.streams,
                            quantization,
                            false,
                            vb.pp("block_inject_weight"),
                        )
                    })
                    .transpose()?,
                None,
            )
        };
        let up = ReplicatedLayer::new(
            cfg.rank,
            hc_hidden,
            quantization,
            false,
            vb.pp("input_mix_weight_up"),
        )?;
        #[cfg(feature = "cutile")]
        if let Some(weight) = up.unquant_forward_weight() {
            mistralrs_quant::cutile::register_hc_up_mix(&weight, cfg.streams);
        }
        Ok(Self {
            norm_weight,
            down,
            up,
            inject,
            merged_down_inject,
            hc: cfg.streams,
            hidden: cfg.hidden,
            eps: cfg.eps,
        })
    }

    pub(crate) fn norm(&self, res: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::hyper_connection::hc_norm(
                res,
                &self.norm_weight,
                self.hc,
                self.eps,
            );
        }
        grouped_rms_norm(res, &self.norm_weight, self.hc, self.eps)
    }

    /// The block input: mean over streams of `xn * sigmoid(up(silu(down(xn) / hc)))`.
    pub(crate) fn mix(&self, xn: &Tensor) -> Result<Tensor> {
        self.mix_down(xn, &self.down.forward(xn)?)
    }

    pub(crate) fn mix_and_inject(&self, xn: &Tensor) -> Result<(Tensor, Tensor)> {
        if let Some(merged) = &self.merged_down_inject {
            if self.down.stats_snapshot().is_none()
                && self
                    .inject
                    .as_ref()
                    .is_some_and(|inject| inject.stats_snapshot().is_none())
            {
                let mut parts = merged.forward(xn)?;
                let inject = parts.pop().expect("injection projection");
                let down = parts.pop().expect("low-rank projection");
                return Ok((self.mix_down(xn, &down)?, inject));
            }
        }
        let inject = self.inject.as_deref().expect("block mixer has injection");
        if let Some(outputs) = mistralrs_quant::forward_group(
            xn,
            &[&*self.down, inject],
            mistralrs_quant::GroupKind::Dense,
        )? {
            let [down, inject]: [Tensor; 2] = outputs.try_into().map_err(|_| {
                candle_core::Error::msg("block mixer pair returned the wrong output count")
            })?;
            return Ok((self.mix_down(xn, &down)?, inject));
        }
        Ok((self.mix(xn)?, self.inject(xn)?))
    }

    fn mix_down(&self, xn: &Tensor, down: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(mixed) = mistralrs_quant::try_fused_quantized_gated_mix(
            down,
            &*self.up,
            xn,
            self.hc,
            mistralrs_quant::QuantizedInputTransform::ScaledSilu {
                divisor: self.hc as f32,
            },
        )? {
            return Ok(mixed);
        }
        self.mix_lowrank(xn, &self.scaled_silu(down)?)
    }

    fn scaled_silu(&self, lo: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if lo.device().is_cuda() {
            return crate::cuda::hyper_connection::hc_silu(lo, self.hc);
        }
        candle_nn::ops::silu(&(lo / self.hc as f64)?)
    }

    fn mix_lowrank(&self, xn: &Tensor, lo: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if let Some(mixed) = mistralrs_quant::try_fused_quantized_gated_mix(
            lo,
            &*self.up,
            xn,
            self.hc,
            mistralrs_quant::QuantizedInputTransform::Identity,
        )? {
            return Ok(mixed);
        }
        #[cfg(feature = "cuda")]
        if let Some(mixed) =
            crate::cuda::hyper_connection::try_hc_up_mix(xn, lo, &*self.up, self.hc)?
        {
            return Ok(mixed);
        }
        let gate = self.up.forward(lo)?;
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() {
            return crate::cuda::hyper_connection::hc_mix(xn, &gate, self.hc);
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
    pub(crate) fn inject(&self, xn: &Tensor) -> Result<Tensor> {
        self.inject
            .as_ref()
            .expect("final hyper-connection mixer has no injection")
            .forward(xn)
    }

    /// `res + out * 2 * sigmoid(inject / hc)` per stream, plus the grouped norm `next` needs.
    pub(crate) fn combine(
        &self,
        res: &Tensor,
        out: &Tensor,
        inject: &Tensor,
        next: Option<&GatedResidual>,
    ) -> Result<(Tensor, Option<Tensor>)> {
        let next = next.filter(|next| next.norm_weight.device().same_device(res.device()));
        #[cfg(feature = "cuda")]
        if res.device().is_cuda() {
            return crate::cuda::hyper_connection::hc_combine(
                res,
                out,
                inject,
                next.map(|next| &next.norm_weight),
                self.hc,
                next.map_or(self.eps, |next| next.eps),
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
pub(crate) fn grouped_rms_norm(
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
#[path = "hyper_connection_tests.rs"]
mod tests;
