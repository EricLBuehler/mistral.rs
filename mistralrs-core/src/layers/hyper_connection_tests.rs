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
    for (tokens, hc, hidden) in [(5, 4, 96), (1, 2, 193), (17, 3, 511)] {
        for dtype in [DType::F16, DType::BF16, DType::F32] {
            let res = Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(dtype)?;
            let weight = Tensor::randn(1f32, 0.1, hc * hidden, &device)?;
            let eps = 1e-6;

            let fused = crate::cuda::hyper_connection::hc_norm(&res, &weight, hc, eps)?;
            let reference = grouped_rms_norm(&res, &weight, hc, eps)?;
            assert!(max_diff(&fused, &reference)? < 0.05);

            let gate = Tensor::randn(0f32, 1.0, (tokens, hc * hidden), &device)?.to_dtype(dtype)?;
            let mixed = crate::cuda::hyper_connection::hc_mix(&fused, &gate, hc)?;
            let reference = (fused.to_dtype(DType::F32)?
                * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
            .reshape((tokens, hc, hidden))?
            .mean(1)?;
            assert!(max_diff(&mixed, &reference)? < 0.02);

            let out = Tensor::randn(0f32, 1.0, (tokens, hidden), &device)?.to_dtype(dtype)?;
            let inject = Tensor::randn(0f32, 1.0, (tokens, hc), &device)?.to_dtype(dtype)?;
            let (combined, next) = crate::cuda::hyper_connection::hc_combine(
                &res,
                &out,
                &inject,
                Some(&weight),
                hc,
                eps,
            )?;
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
        }
    }
    Ok(())
}

#[test]
fn hyper_connection_orders_cross_stream_inputs_and_outputs() -> Result<()> {
    use crate::cuda::hyper_connection::{hc_combine, hc_mix, hc_norm, hc_silu};

    const TOKENS: usize = 129;
    const STREAMS: usize = 3;
    const HIDDEN: usize = 768;
    const WIDTH: usize = STREAMS * HIDDEN;
    const UPDATES: usize = 32;
    const EPS: f64 = 1e-6;
    const TOLERANCE: f32 = 1e-5;
    let first = Device::new_cuda_with_stream(0)?;
    let second = Device::new_cuda_with_stream(0)?;
    let third = Device::new_cuda_with_stream(0)?;
    let matrix = |rows, cols, salt: f32, device: &Device| {
        Tensor::from_vec(
            (0..rows * cols)
                .map(|i| (i as f32 * 0.17 + salt).sin())
                .collect::<Vec<_>>(),
            (rows, cols),
            device,
        )
    };
    let residual = matrix(TOKENS + 1, WIDTH, 0.2, &first)?.narrow(0, 1, TOKENS)?;
    let mut gate = matrix(TOKENS + 1, WIDTH, 0.8, &second)?;
    let mut weight = matrix(1, WIDTH, 1.3, &third)?.flatten_all()?;
    let injection = matrix(TOKENS, STREAMS, 1.7, &third)?;
    for _ in 0..UPDATES {
        gate = gate.affine(0.99, 0.01)?;
        weight = weight.affine(0.99, 0.01)?;
    }
    let gate = gate.narrow(0, 1, TOKENS)?;
    let normalized = hc_norm(&residual, &weight, STREAMS, EPS)?;
    let activated = hc_silu(&gate, STREAMS)?;
    let mixed = hc_mix(&activated, &normalized, STREAMS)?;
    let (combined, next) = hc_combine(&residual, &mixed, &injection, Some(&weight), STREAMS, EPS)?;
    let consumed = hc_mix(&gate, &next.unwrap(), STREAMS)?.to_device(&Device::Cpu)?;

    let residual = residual.to_device(&Device::Cpu)?;
    let gate = gate.to_device(&Device::Cpu)?;
    let weight = weight.to_device(&Device::Cpu)?;
    let injection = injection.to_device(&Device::Cpu)?;
    let normalized = grouped_rms_norm(&residual, &weight, STREAMS, EPS)?;
    let activated = candle_nn::ops::silu(&(&gate / STREAMS as f64)?)?;
    let mixed = (activated * candle_nn::ops::sigmoid(&normalized)?)?
        .reshape((TOKENS, STREAMS, HIDDEN))?
        .mean(1)?;
    let injection = (candle_nn::ops::sigmoid(&(injection / STREAMS as f64)?)? * 2.0)?;
    let expected = (residual.reshape((TOKENS, STREAMS, HIDDEN))?
        + mixed
            .unsqueeze(1)?
            .broadcast_mul(&injection.unsqueeze(2)?)?)?
    .reshape((TOKENS, WIDTH))?;
    assert!(max_diff(&combined.to_device(&Device::Cpu)?, &expected)? < TOLERANCE);
    let expected = (gate
        * candle_nn::ops::sigmoid(&grouped_rms_norm(&expected, &weight, STREAMS, EPS)?)?)?
    .reshape((TOKENS, STREAMS, HIDDEN))?
    .mean(1)?;
    assert!(max_diff(&consumed, &expected)? < TOLERANCE);
    Ok(())
}

#[test]
fn hyper_connection_consumes_offset_and_strided_projection_views() -> Result<()> {
    use crate::cuda::hyper_connection::{hc_combine, hc_silu};

    const TOKENS: usize = 17;
    const STREAMS: usize = 3;
    const HIDDEN: usize = 193;
    const RANK: usize = 32;
    const PAD: usize = 7;
    const EPS: f64 = 1e-6;
    let device = Device::new_cuda(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for batch in [1, 2] {
            let combined = Tensor::randn(
                0f32,
                1.0,
                (batch, TOKENS + 1, RANK + STREAMS + PAD),
                &device,
            )?
            .to_dtype(dtype)?
            .narrow(1, 1, TOKENS)?;
            let down = combined.narrow(2, 1, RANK)?;
            let inject = combined.narrow(2, RANK + 1, STREAMS)?;
            assert!(!down.is_contiguous());
            assert!(down.layout().start_offset() > 0);
            assert_eq!(
                max_diff(
                    &hc_silu(&down, STREAMS)?,
                    &hc_silu(&down.contiguous()?, STREAMS)?
                )?,
                0.0,
            );
            let residual = Tensor::randn(0f32, 1.0, (batch, TOKENS, STREAMS * HIDDEN), &device)?
                .to_dtype(dtype)?;
            let output =
                Tensor::randn(0f32, 1.0, (batch, TOKENS, HIDDEN), &device)?.to_dtype(dtype)?;
            let norm = Tensor::ones(STREAMS * HIDDEN, DType::F32, &device)?;
            let actual = hc_combine(&residual, &output, &inject, Some(&norm), STREAMS, EPS)?;
            let expected = hc_combine(
                &residual,
                &output,
                &inject.contiguous()?,
                Some(&norm),
                STREAMS,
                EPS,
            )?;
            assert_eq!(max_diff(&actual.0, &expected.0)?, 0.0);
            assert_eq!(max_diff(&actual.1.unwrap(), &expected.1.unwrap())?, 0.0);
        }
    }
    Ok(())
}

#[test]
fn hyper_connection_vector_combine_replays_strided_views_and_fallbacks() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    const EPS: f64 = 1e-6;
    const INJECT_PAD: usize = 5;
    let device = Device::new_cuda_with_stream(0)?;
    let stream = device.as_cuda_device()?.cuda_stream();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let tolerance = match dtype {
            DType::BF16 => 0.02,
            DType::F16 => 0.002,
            _ => 2e-6,
        };
        for (tokens, streams, hidden) in [
            (1, 4, 2560),
            (3, 3, 768),
            (17, 2, 1536),
            (1, 1, 16384),
            (1, 3, 16388),
        ] {
            for offset in [0, 1] {
                let width = streams * hidden;
                let residual = Var::zeros(tokens * width + 1, dtype, &device)?;
                let output = Var::zeros(tokens * hidden + 1, dtype, &device)?;
                let injection = Var::zeros((tokens, streams + INJECT_PAD), dtype, &device)?;
                let weight = Tensor::ones(width, DType::F32, &device)?;
                let residual_view = residual
                    .narrow(0, offset, tokens * width)?
                    .reshape((tokens, width))?;
                let output_view = output
                    .narrow(0, offset, tokens * hidden)?
                    .reshape((tokens, hidden))?;
                let injection_view = injection.narrow(1, 1, streams)?;
                let run = || {
                    crate::cuda::hyper_connection::hc_combine(
                        &residual_view,
                        &output_view,
                        &injection_view,
                        Some(&weight),
                        streams,
                        EPS,
                    )
                };
                run()?;
                device.synchronize()?;
                unsafe { stream.context().disable_event_tracking() };
                stream
                    .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
                    .map_err(candle_core::Error::wrap)?;
                let captured = run();
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                unsafe { stream.context().enable_event_tracking() };
                let (combined, normalized) = captured?;
                let normalized = normalized.unwrap();
                let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
                for scale in [0.0f32, 0.25, 1.0, -0.5] {
                    for (variable, salt) in
                        [(&residual, 0.1f32), (&output, -0.2), (&injection, 0.3)]
                    {
                        variable.set(
                            &Tensor::from_vec(
                                (0..variable.elem_count())
                                    .map(|i| ((i % 127) as f32 - 63.0) / 64.0 * scale + salt)
                                    .collect::<Vec<_>>(),
                                variable.shape(),
                                &device,
                            )?
                            .to_dtype(dtype)?,
                        )?;
                    }
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    let actual = combined.to_device(&Device::Cpu)?;
                    let residual = residual_view
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?;
                    let output = output_view.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
                    let injection = injection_view
                        .to_device(&Device::Cpu)?
                        .to_dtype(DType::F32)?;
                    let gate = (candle_nn::ops::sigmoid(&(injection / streams as f64)?)? * 2.0)?;
                    let reference = (residual.reshape((tokens, streams, hidden))?
                        + output.unsqueeze(1)?.broadcast_mul(&gate.unsqueeze(2)?)?)?
                    .reshape((tokens, width))?
                    .to_dtype(dtype)?;
                    assert!(max_diff(&actual, &reference)? <= tolerance);
                    let values = actual
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let reference = values
                        .chunks_exact(hidden)
                        .flat_map(|row| {
                            let variance = row
                                .iter()
                                .map(|value| f64::from(*value).powi(2))
                                .sum::<f64>()
                                / hidden as f64;
                            let inverse = (variance + EPS).sqrt().recip();
                            row.iter()
                                .map(move |value| (f64::from(*value) * inverse) as f32)
                        })
                        .collect::<Vec<_>>();
                    let normalized_reference =
                        Tensor::from_vec(reference, actual.shape(), &Device::Cpu)?
                            .to_dtype(dtype)?;
                    assert!(
                        max_diff(&normalized.to_device(&Device::Cpu)?, &normalized_reference)?
                            <= tolerance
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn packed_hyper_connection_projections_match_reference() -> Result<()> {
    use mistralrs_quant::ShardedSafeTensors;
    use std::collections::HashMap;

    let device = Device::new_cuda(0)?;
    let cfg = super::GatedResidualConfig {
        streams: 3,
        hidden: 96,
        rank: 32,
        eps: 1e-6,
    };
    let width = cfg.streams * cfg.hidden;
    for dtype in [DType::BF16, DType::F16] {
        let matrix = |rows, cols, salt: f32| {
            Tensor::from_vec(
                (0..rows * cols)
                    .map(|i| (i as f32 * 0.17 + salt).sin() / (cols as f32).sqrt())
                    .collect::<Vec<_>>(),
                (rows, cols),
                &device,
            )?
            .to_dtype(dtype)
        };
        let tensors = HashMap::from([
            (
                "hc_norm.weight".to_string(),
                Tensor::zeros(width, dtype, &device)?,
            ),
            (
                "input_mix_weight_down.weight".to_string(),
                matrix(cfg.rank, width, 0.2)?,
            ),
            (
                "input_mix_weight_up.weight".to_string(),
                matrix(width, cfg.rank, 0.8)?,
            ),
            (
                "block_inject_weight.weight".to_string(),
                matrix(cfg.streams, width, 1.3)?,
            ),
        ]);
        let mixer = GatedResidual::load(
            &cfg,
            &None,
            ShardedSafeTensors::wrap(tensors, dtype, device.clone()),
            true,
        )?;
        assert!(mixer.merged_down_inject.is_some());
        for tokens in [1, 17, 512] {
            let input = matrix(tokens, width, 0.3)?;
            let expected = mixer.mix(&input)?;
            let (actual, injection) = mixer.mix_and_inject(&input)?;
            assert!(max_diff(&actual, &expected)? < TEST_TOLERANCE);
            assert!(max_diff(&injection, &mixer.inject(&input)?)? < TEST_TOLERANCE);
        }
    }
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
                merged_down_inject: None,
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
                let expected = crate::cuda::hyper_connection::hc_mix(&input, &gate, TEST_HC)?;
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
                    let exact =
                        crate::cuda::hyper_connection::hc_mix(&input, &exact_gate, TEST_HC)?;
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

    use crate::cuda::hyper_connection::hc_mix;
    use mistralrs_quant::{try_fused_quantized_gated_mix, QuantizedInputTransform};

    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    let width = TEST_HC * TEST_HIDDEN;
    for dtype in [DType::BF16, DType::F16] {
        let input = Var::zeros((1, 1, width), dtype, &device)?;
        let lowrank = Var::zeros((1, 1, TEST_RANK), dtype, &device)?;
        for quant in [GgmlDType::Q8_0, GgmlDType::Q5_1] {
            let values = (0..width * TEST_RANK)
                .map(|i| (i as f32 * 0.17 + 0.8).sin() / (TEST_RANK as f32).sqrt())
                .collect::<Vec<_>>();
            let weight = Tensor::from_vec(values, (width, TEST_RANK), &Device::Cpu)?;
            let projection = GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(QTensor::quantize_onto(&weight, quant, &device)?),
                b: None,
            })?;
            let _ = try_fused_quantized_gated_mix(
                &lowrank,
                &projection,
                &input,
                TEST_HC,
                QuantizedInputTransform::Identity,
            )?
            .unwrap();
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
            let output = try_fused_quantized_gated_mix(
                &lowrank,
                &projection,
                &input,
                TEST_HC,
                QuantizedInputTransform::Identity,
            );
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = output?.unwrap();
            let graph = graph
                .map_err(candle_core::Error::msg)?
                .ok_or_else(|| candle_core::Error::msg("HC capture returned no graph"))?;
            for scale in [0.0f32, 0.15, 1.25, -3.0, 0.15] {
                let values = (0..width)
                    .map(|i| (i as f32 * 0.03 + scale).cos())
                    .collect::<Vec<_>>();
                input.set(&Tensor::from_vec(values, (1, 1, width), &device)?.to_dtype(dtype)?)?;
                let values = (0..TEST_RANK)
                    .map(|i| (i as f32 * 0.13).sin() * scale)
                    .collect::<Vec<_>>();
                lowrank
                    .set(&Tensor::from_vec(values, (1, 1, TEST_RANK), &device)?.to_dtype(dtype)?)?;
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
