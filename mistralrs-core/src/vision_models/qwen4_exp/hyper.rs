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
        self.mix_lowrank(xn, &self.down_silu(xn)?)
    }

    pub(super) fn mix_and_inject(&self, xn: &Tensor) -> Result<(Tensor, Tensor)> {
        #[cfg(feature = "cuda")]
        if let Some((down, inject)) = mistralrs_quant::try_fused_quantized_pair(
            xn,
            &*self.down,
            self.inject.as_deref().expect("block mixer has injection"),
        )? {
            let lo = crate::cuda::qwen4_exp::hc_silu(&down, self.hc)?;
            return Ok((self.mix_lowrank(xn, &lo)?, inject));
        }
        Ok((self.mix(xn)?, self.inject(xn)?))
    }

    fn down_silu(&self, xn: &Tensor) -> Result<Tensor> {
        let lo = self.down.forward(xn)?;
        #[cfg(feature = "cuda")]
        if xn.device().is_cuda() {
            return crate::cuda::qwen4_exp::hc_silu(&lo, self.hc);
        }
        candle_nn::ops::silu(&(lo / self.hc as f64)?)
    }

    fn mix_lowrank(&self, xn: &Tensor, lo: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "cuda")]
        if crate::cuda::qwen4_exp::hc_up_mix_supported(xn, lo, self.hc)? {
            if let Some(up) = crate::cuda::qwen4_exp::HcWeight::from_projection(&*self.up) {
                return crate::cuda::qwen4_exp::hc_up_mix(xn, lo, &up, self.hc);
            }
        }
        let gate = self.up.forward(lo)?;
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
    use std::sync::Arc;

    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{DType, Device, Result, Tensor, Var};
    use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, UnquantLinear};

    use super::{grouped_rms_norm, GatedResidual};

    const TEST_HC: usize = 4;
    const TEST_HIDDEN: usize = 2560;
    const TEST_RANK: usize = 320;
    const TEST_TOLERANCE: f32 = 0.016;

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

    #[test]
    fn fused_hyper_connection_projections_match_reference() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let width = TEST_HC * TEST_HIDDEN;
        for dtype in [DType::BF16, DType::F16] {
            for quant in [None, Some(GgmlDType::Q8_0), Some(GgmlDType::Q6K)] {
                let projection = |rows: usize,
                                  cols: usize,
                                  salt: f32,
                                  quant: Option<GgmlDType>|
                 -> Result<Arc<dyn QuantMethod>> {
                    let values = (0..rows * cols)
                        .map(|i| ((i as f32 * 0.17 + salt).sin()) / (cols as f32).sqrt())
                        .collect::<Vec<_>>();
                    let weight = Tensor::from_vec(values, (rows, cols), &Device::Cpu)?;
                    match quant {
                        Some(quant) => Ok(Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(QTensor::quantize_onto(&weight, quant, &device)?),
                            b: None,
                        })?)),
                        None => Ok(Arc::new(UnquantLinear::new(
                            QuantMethodConfig::Unquantized(candle_nn::Linear::new(
                                weight.to_dtype(dtype)?.to_device(&device)?,
                                None,
                            )),
                        )?)),
                    }
                };
                let up_quant = quant.map(|quant| {
                    if quant == GgmlDType::Q6K {
                        GgmlDType::Q5_1
                    } else {
                        quant
                    }
                });
                let hc = GatedResidual {
                    norm_weight: Tensor::ones(width, DType::F32, &device)?,
                    down: projection(TEST_RANK, width, 0.2, quant)?,
                    up: projection(width, TEST_RANK, 0.8, up_quant)?,
                    inject: Some(projection(TEST_HC, width, 1.3, quant)?),
                    hc: TEST_HC,
                    hidden: TEST_HIDDEN,
                    eps: 1e-6,
                };
                for tokens in [1, 2, 9] {
                    let values = (0..tokens * width)
                        .map(|i| (i as f32 * 0.03).cos())
                        .collect::<Vec<_>>();
                    let input =
                        Tensor::from_vec(values, (1, tokens, width), &device)?.to_dtype(dtype)?;
                    let down = hc.down.forward(&input)?;
                    let lo = candle_nn::ops::silu(&(down / TEST_HC as f64)?)?;
                    let gate = hc.up.forward(&lo)?;
                    let expected = crate::cuda::qwen4_exp::hc_mix(&input, &gate, TEST_HC)?;
                    let (actual, injection) = hc.mix_and_inject(&input)?;
                    assert!(
                        max_diff(&actual, &expected)? < TEST_TOLERANCE,
                        "{dtype:?} {quant:?} tokens={tokens}"
                    );
                    assert!(max_diff(&injection, &hc.inject(&input)?)? < TEST_TOLERANCE);

                    if tokens == 1 {
                        let weight = hc.up.dequantize_w()?.to_dtype(DType::F32)?;
                        let exact_gate = lo
                            .to_dtype(DType::F32)?
                            .reshape((tokens, TEST_RANK))?
                            .matmul(&weight.t()?)?
                            .reshape((1, tokens, width))?
                            .to_dtype(dtype)?;
                        let exact = crate::cuda::qwen4_exp::hc_mix(&input, &exact_gate, TEST_HC)?;
                        let fused = hc.mix_lowrank(&input, &lo)?;
                        assert!(max_diff(&fused, &exact)? < TEST_TOLERANCE);
                        if quant.is_some() {
                            assert_eq!(max_diff(&fused, &expected)?, 0.0);
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn quantized_hyper_connection_graph_replay_matches_reference() -> Result<()> {
        use candle_core::cuda::cudarc::driver::sys;

        use crate::cuda::qwen4_exp::{hc_mix, hc_up_mix, hc_up_mix_supported, HcWeight};

        let device = Device::new_cuda_with_stream(0)?;
        let cuda = device.as_cuda_device()?;
        let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
        let width = TEST_HC * TEST_HIDDEN;
        for dtype in [DType::BF16, DType::F16] {
            let input = Var::zeros((1, 1, width), dtype, &device)?;
            let lowrank = Var::zeros((1, 1, TEST_RANK), dtype, &device)?;
            if !hc_up_mix_supported(&input, &lowrank, TEST_HC)? {
                return Ok(());
            }
            for quant in [GgmlDType::Q8_0, GgmlDType::Q5_1] {
                let values = (0..width * TEST_RANK)
                    .map(|i| (i as f32 * 0.17 + 0.8).sin() / (TEST_RANK as f32).sqrt())
                    .collect::<Vec<_>>();
                let weight = Tensor::from_vec(values, (width, TEST_RANK), &Device::Cpu)?;
                let projection = GgufMatMul::new(QuantMethodConfig::Gguf {
                    q_weight: Arc::new(QTensor::quantize_onto(&weight, quant, &device)?),
                    b: None,
                })?;
                let weight = HcWeight::from_projection(&projection).unwrap();
                let _ = hc_up_mix(&input, &lowrank, &weight, TEST_HC)?;
                device.synchronize()?;
                let stream = cuda.cuda_stream();
                let tracking = stream.context().is_event_tracking();
                if tracking {
                    unsafe { stream.context().disable_event_tracking() };
                }
                if let Err(error) =
                    stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                {
                    if tracking {
                        unsafe { stream.context().enable_event_tracking() };
                    }
                    return Err(candle_core::Error::msg(error));
                }
                let output = hc_up_mix(&input, &lowrank, &weight, TEST_HC);
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                let output = output?;
                let graph = graph
                    .map_err(candle_core::Error::msg)?
                    .ok_or_else(|| candle_core::Error::msg("HC capture returned no graph"))?;
                for scale in [0.0f32, 0.15, 1.25, -3.0, 0.15] {
                    let values = (0..width)
                        .map(|i| (i as f32 * 0.03 + scale).cos())
                        .collect::<Vec<_>>();
                    input
                        .set(&Tensor::from_vec(values, (1, 1, width), &device)?.to_dtype(dtype)?)?;
                    let values = (0..TEST_RANK)
                        .map(|i| (i as f32 * 0.13).sin() * scale)
                        .collect::<Vec<_>>();
                    lowrank.set(
                        &Tensor::from_vec(values, (1, 1, TEST_RANK), &device)?.to_dtype(dtype)?,
                    )?;
                    graph.launch().map_err(candle_core::Error::msg)?;
                    device.synchronize()?;
                    let expected = hc_mix(&input, &projection.forward(&lowrank)?, TEST_HC)?;
                    assert_eq!(
                        max_diff(&output, &expected)?,
                        0.0,
                        "{dtype:?} {quant:?} {scale}"
                    );
                }
            }
        }
        Ok(())
    }
}
