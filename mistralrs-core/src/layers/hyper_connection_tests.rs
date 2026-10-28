use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Result, Tensor, Var};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig, UnquantLinear};

use super::{grouped_rms_norm, GatedResidual};

const TEST_HC: usize = 4;
const TEST_HIDDEN: usize = 2560;
const TEST_RANK: usize = 320;
const TEST_TOLERANCE: f32 = 0.016;
const UP_MIX_ROWS_PER_MULTIPROCESSOR: usize = 64;
const UP_MIX_TOLERANCE: f32 = 0.004;
const UP_MIX_ROUNDING_RANK: usize = 64;
const UP_MIX_ROUNDING_CASES: [(DType, f32, f32, f32, f32); 2] = [
    (
        DType::BF16,
        1.0 / 256.0,
        37.0 / 128.0,
        27.0 / 128.0,
        217.0 / 1024.0,
    ),
    (
        DType::F16,
        1.0 / 2048.0,
        9.0 / 32.0,
        1684.0 / 8192.0,
        1685.0 / 8192.0,
    ),
];

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

fn up_mix_test_hidden(device: &Device) -> Result<usize> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
    let multiprocessors = device
        .as_cuda_device()?
        .cuda_stream()
        .context()
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .map_err(candle_core::Error::wrap)?;
    Ok(multiprocessors as usize * UP_MIX_ROWS_PER_MULTIPROCESSOR + 3)
}

#[test]
fn vector_hyper_connection_upmix_matches_rounded_projection() -> Result<()> {
    use crate::cuda::hyper_connection::{hc_mix, try_hc_up_mix};

    let device = Device::new_cuda(0)?;
    let hidden = up_mix_test_hidden(&device)?;
    for (rank, streams, offset) in [
        (64, 1, 0),
        (320, 3, 0),
        (127, 2, 0),
        (320, 4, 1),
        (32, 5, 0),
    ] {
        let width = hidden * streams;
        for dtype in [DType::F16, DType::BF16] {
            for weight_dtype in [DType::F16, DType::BF16] {
                let values = (0..width * rank + offset)
                    .map(|i| (i as f32 * 0.17 + 0.8).sin() / (rank as f32).sqrt())
                    .collect::<Vec<_>>();
                let weight = Tensor::from_vec(values, width * rank + offset, &device)?
                    .to_dtype(weight_dtype)?
                    .narrow(0, offset, width * rank)?
                    .reshape((width, rank))?;
                let values = (0..rank + offset)
                    .map(|i| (i as f32 * 0.13).cos())
                    .collect::<Vec<_>>();
                let lo = Tensor::from_vec(values, rank + offset, &device)?
                    .to_dtype(dtype)?
                    .narrow(0, offset, rank)?
                    .reshape((1, rank))?;
                let values = (0..width)
                    .map(|i| (i as f32 * 0.03 + 0.2).sin())
                    .collect::<Vec<_>>();
                let residual = Tensor::from_vec(values, (1, width), &device)?.to_dtype(dtype)?;
                let gate = lo
                    .to_dtype(DType::F32)?
                    .matmul(&weight.to_dtype(DType::F32)?.t()?)?
                    .to_dtype(dtype)?;
                let expected = hc_mix(&residual, &gate, streams)?;
                let projection = UnquantLinear::new(QuantMethodConfig::Unquantized(
                    candle_nn::Linear::new(weight, None),
                ))?;
                let actual = try_hc_up_mix(&residual, &lo, &projection, streams)?.unwrap();
                assert!(
                    max_diff(&actual, &expected)? < UP_MIX_TOLERANCE,
                    "{dtype:?} {weight_dtype:?} rank={rank} streams={streams} offset={offset}",
                );
            }
        }
    }
    Ok(())
}

#[test]
fn vector_hyper_connection_upmix_rounds_gates_before_sigmoid() -> Result<()> {
    use crate::cuda::hyper_connection::try_hc_up_mix;

    let device = Device::new_cuda(0)?;
    let hidden = up_mix_test_hidden(&device)?;
    for (dtype, delta, residual, rounded, unrounded) in UP_MIX_ROUNDING_CASES {
        assert_ne!(rounded, unrounded);
        let mut row = vec![0f32; UP_MIX_ROUNDING_RANK];
        row[0] = 1.0;
        row[1] = delta;
        let weight = Tensor::from_vec(row.repeat(hidden), (hidden, UP_MIX_ROUNDING_RANK), &device)?
            .to_dtype(dtype)?;
        row[1] = 1.0;
        let lo = Tensor::from_vec(row, (1, UP_MIX_ROUNDING_RANK), &device)?.to_dtype(dtype)?;
        let input = Tensor::full(residual, (1, hidden), &device)?.to_dtype(dtype)?;
        let expected = Tensor::full(rounded, (1, hidden), &device)?.to_dtype(dtype)?;
        let projection = UnquantLinear::new(QuantMethodConfig::Unquantized(
            candle_nn::Linear::new(weight, None),
        ))?;
        let actual = try_hc_up_mix(&input, &lo, &projection, 1)?.unwrap();
        assert_eq!(max_diff(&actual, &expected)?, 0.0, "{dtype:?}");
    }
    Ok(())
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

#[cfg(feature = "cutile")]
mod hc_prefill_integration {
    use candle_core::cuda::cudarc::driver::{sys, CudaGraph};
    use candle_core::{DType, Device, Result, Tensor, Var};
    use mistralrs_quant::cutile::{hc_prefill_supported, register_hc_up_mix, warmup_moe_kernels};
    use mistralrs_quant::{
        maybe_wrap_dynamic_lora, with_lora_execution, LoraExecution, LoraLayerRegistry,
        LoraLinearSpec, LoraWeights, QuantMethod, QuantMethodConfig, ShardedSafeTensors,
        UnquantLinear,
    };
    use std::{
        collections::HashMap,
        sync::{Arc, Mutex},
    };

    use crate::cuda::hyper_connection::{hc_mix, try_hc_up_mix};

    const ROW_OFFSETS: [usize; 5] = [0, 1, 2, 4, 8];
    const ROW_TILE: usize = 64;
    const COLUMN_TILE: usize = 128;
    const TEST_MIN_ROWS: usize = 512;
    const CACHE_PARTS: usize = 3;
    const BLOCKS_PER_SM: usize = 2;
    const HALF_BYTES: usize = 2;
    const ROW_ALIGNMENT: usize = 16;
    const UNSEEN_ROW_OFFSET: usize = 32;

    fn row_classes(device: &Device, streams: usize, hidden: usize) -> Result<[usize; 5]> {
        use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;
        let stream = device.as_cuda_device()?.cuda_stream();
        let context = stream.context();
        let sms = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(candle_core::Error::msg)? as usize;
        let cache = context
            .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE)
            .map_err(candle_core::Error::msg)? as usize;
        let minimum = TEST_MIN_ROWS
            .max((cache / CACHE_PARTS).div_ceil(streams * hidden * HALF_BYTES))
            .max((sms * BLOCKS_PER_SM).div_ceil(hidden / COLUMN_TILE) * ROW_TILE);
        let base = (minimum + UNSEEN_ROW_OFFSET).next_multiple_of(ROW_ALIGNMENT);
        Ok(ROW_OFFSETS.map(|offset| base + offset))
    }
    const SHAPES: [(usize, usize, usize); 3] = [(4, 2048, 256), (3, 3072, 192), (4, 2560, 320)];
    const REPLAY_SCALES: [(f64, f64, f64); 3] =
        [(0.0, 1.0, 1.0), (-0.5, 0.75, 1.0), (2.0, -0.5, -0.5)];
    const INPUT_SEED: u32 = 71;
    const RESIDUAL_SEED: u32 = 97;
    const WEIGHT_SEED: u32 = 23;
    const BF16_ATOL: f32 = 0.004;
    const BF16_RTOL: f32 = 0.016;
    const F16_ATOL: f32 = 0.0005;
    const F16_RTOL: f32 = 0.002;
    const LORA_GATE_DELTA: f32 = 0.5;
    const LORA_MIX_MIN_CHANGE: f32 = 0.05;
    const MISALIGNED_OFFSET: usize = 1;
    const UNREGISTERED_SHAPE: (usize, usize, usize) = (4, 3072, 192);
    const EXCLUDED_SHAPES: [(usize, usize, usize); 2] = [(3, 1537, 130), (3, 257, 96)];
    static CUDA_TEST_LOCK: Mutex<()> = Mutex::new(());

    fn matrix(
        rows: usize,
        columns: usize,
        seed: u32,
        scale: f32,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let values = (0..rows * columns)
            .map(|index| {
                let mut bits = (index as u32).wrapping_add(seed);
                bits ^= bits >> 16;
                bits = bits.wrapping_mul(0x7feb352d);
                bits ^= bits >> 15;
                bits = bits.wrapping_mul(0x846ca68b);
                bits ^= bits >> 16;
                ((bits & 65535) as f32 / 32768.0 - 1.0) * scale
            })
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (rows, columns), device)?.to_dtype(dtype)
    }

    fn projection(weight: Tensor, bias: Option<Tensor>) -> Result<UnquantLinear> {
        UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
            weight, bias,
        )))
    }

    fn assert_close(actual: &Tensor, expected: &Tensor) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        let (atol, rtol) = match actual.dtype() {
            DType::BF16 => (BF16_ATOL, BF16_RTOL),
            DType::F16 => (F16_ATOL, F16_RTOL),
            _ => unreachable!(),
        };
        let actual = actual
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = expected
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                actual.is_finite() && expected.is_finite(),
                "nonfinite at {index}"
            );
            assert!(
                (actual - expected).abs() <= atol + rtol * expected.abs(),
                "at {index}: actual={actual}, expected={expected}"
            );
        }
        Ok(())
    }

    fn capture(
        device: &Device,
        run: impl FnOnce() -> Result<Option<Tensor>>,
    ) -> Result<(CudaGraph, Tensor)> {
        device.synchronize()?;
        let stream = device.as_cuda_device()?.cuda_stream();
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
            return Err(candle_core::Error::wrap(error));
        }
        let output = run();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?.expect("registered prefill shape must select fusion");
        let graph = graph
            .map_err(candle_core::Error::wrap)?
            .expect("captured graph");
        Ok((graph, output))
    }

    #[test]
    fn hc_prefill_warmup_covers_unseen_rows_and_mutated_graph_inputs() -> Result<()> {
        let _lock = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda_with_stream(0)?;
        let dev = device.as_cuda_device()?;
        if !hc_prefill_supported(dev) {
            return Ok(());
        }
        let mut cases = Vec::new();
        for dtype in [DType::BF16, DType::F16] {
            for (streams, hidden, rank) in SHAPES {
                let width = streams * hidden;
                let weight0 = matrix(
                    width,
                    rank,
                    WEIGHT_SEED,
                    1.0 / (rank as f32).sqrt(),
                    dtype,
                    &device,
                )?;
                let weight = Var::from_tensor(&weight0)?;
                register_hc_up_mix(&weight, streams);
                cases.push((dtype, streams, hidden, rank, weight0, weight));
            }
        }
        warmup_moe_kernels(&device, true)?;
        for (dtype, streams, hidden, rank, weight0, weight) in cases {
            let width = streams * hidden;
            let layer = projection(weight.as_tensor().clone(), None)?;
            for rows in row_classes(&device, streams, hidden)? {
                let lo0 = matrix(rows, rank, INPUT_SEED, 1.0, dtype, &device)?;
                let xn0 = matrix(rows, width, RESIDUAL_SEED, 1.0, dtype, &device)?;
                let lo = Var::from_tensor(&lo0)?;
                let xn = Var::from_tensor(&xn0)?;
                let lo_view = lo.reshape((1, rows, rank))?;
                let xn_view = xn.reshape((1, rows, width))?;
                let (graph, output) = capture(&device, || {
                    try_hc_up_mix(&xn_view, &lo_view, &layer, streams)
                })?;
                assert_eq!(output.dims(), [1, rows, hidden]);
                for (lo_scale, xn_scale, weight_scale) in REPLAY_SCALES {
                    lo.set(&lo0.affine(lo_scale, 0.0)?)?;
                    xn.set(&xn0.affine(xn_scale, 0.0)?)?;
                    weight.set(&weight0.affine(weight_scale, 0.0)?)?;
                    graph.launch().map_err(candle_core::Error::wrap)?;
                    let expected = hc_mix(&xn_view, &layer.forward(&lo_view)?, streams)?;
                    assert_close(&output, &expected)?;
                }
                weight.set(&weight0)?;
            }
        }
        Ok(())
    }

    #[test]
    fn hc_prefill_hook_preserves_bias_and_stats_fallbacks() -> Result<()> {
        let _lock = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda_with_stream(0)?;
        let dev = device.as_cuda_device()?;
        if !hc_prefill_supported(dev) {
            return Ok(());
        }
        let (streams, hidden, rank) = SHAPES[0];
        let rows = row_classes(&device, streams, hidden)?[0];
        let width = streams * hidden;
        for dtype in [DType::BF16, DType::F16] {
            let weight = matrix(
                width,
                rank,
                WEIGHT_SEED,
                1.0 / (rank as f32).sqrt(),
                dtype,
                &device,
            )?;
            register_hc_up_mix(&weight, streams);
            warmup_moe_kernels(&device, true)?;
            let lo = matrix(rows, rank, INPUT_SEED, 1.0, dtype, &device)?;
            let xn = matrix(rows, width, RESIDUAL_SEED, 1.0, dtype, &device)?;
            let bias = Tensor::full(0.25f32, width, &device)?.to_dtype(dtype)?;
            let biased = projection(weight.clone(), Some(bias))?;
            assert!(try_hc_up_mix(&xn, &lo, &biased, streams)?.is_none());
            let tracked = projection(weight, None)?;
            assert!(try_hc_up_mix(&xn, &lo, &tracked, streams)?.is_some());
            tracked.begin_track_stats()?;
            assert_eq!(tracked.stats_snapshot(), Some((0, 0)));
            assert!(try_hc_up_mix(&xn, &lo, &tracked, streams)?.is_none());
            assert_eq!(tracked.stats_snapshot(), Some((0, 0)));
            let gate = tracked.forward(&lo)?;
            assert_eq!(tracked.stats_snapshot(), Some((1, rows)));
            let mixed = hc_mix(&xn, &gate, streams)?;
            assert_eq!(mixed.dims(), [rows, hidden]);
            tracked.end_track_stats()?;
            assert!(try_hc_up_mix(&xn, &lo, &tracked, streams)?.is_some());
        }
        Ok(())
    }

    #[test]
    fn hc_prefill_hook_preserves_dynamic_lora_execution() -> Result<()> {
        let _lock = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda_with_stream(0)?;
        if !hc_prefill_supported(device.as_cuda_device()?) {
            return Ok(());
        }
        let (streams, hidden, rank) = SHAPES[0];
        let rows = row_classes(&device, streams, hidden)?[0];
        let width = streams * hidden;
        for dtype in [DType::BF16, DType::F16] {
            let registry = Arc::new(LoraLayerRegistry::new());
            let vb =
                ShardedSafeTensors::wrap(HashMap::<String, Tensor>::new(), dtype, device.clone())
                    .with_lora_registry(registry.clone());
            let weight = Tensor::zeros((width, rank), dtype, &device)?;
            register_hc_up_mix(&weight, streams);
            warmup_moe_kernels(&device, true)?;
            let base = Arc::new(projection(weight, None)?) as Arc<dyn QuantMethod>;
            let wrapped = maybe_wrap_dynamic_lora(
                &vb.pp("hc_up"),
                base,
                LoraLinearSpec::replicated(rank, width),
            )?;
            registry.finalize()?;
            let site = registry
                .sites()
                .into_iter()
                .find(|site| site.key().path() == "hc_up")
                .expect("HC up LoRA site");
            let lo = Tensor::ones((rows, rank), dtype, &device)?;
            let xn = Tensor::ones((rows, width), dtype, &device)?;
            let inactive = try_hc_up_mix(&xn, &lo, &*wrapped, streams)?
                .expect("inactive wrapper must preserve dense fusion");
            let expected_inactive = hc_mix(&xn, &wrapped.forward(&lo)?, streams)?;
            assert_close(&inactive, &expected_inactive)?;
            let mut a_values = vec![0f32; rank];
            a_values[0] = 1.0;
            let a = Tensor::from_vec(a_values, (1, rank), &device)?.to_dtype(dtype)?;
            let b = Tensor::full(LORA_GATE_DELTA, (width, 1), &device)?.to_dtype(dtype)?;
            for active in [false, true] {
                let mut execution =
                    LoraExecution::new(registry.runtime_id(), vec![active.then_some(0); rows]);
                execution.insert(&site, 0, LoraWeights::new(a.clone(), b.clone(), 1.0)?)?;
                with_lora_execution(Some(Arc::new(execution)), || -> Result<()> {
                    assert_eq!(wrapped.is_dynamic_lora_active(), active);
                    let fused = try_hc_up_mix(&xn, &lo, &*wrapped, streams)?;
                    if active {
                        assert!(fused.is_none());
                        let gate = wrapped.forward(&lo)?;
                        let expected_gate = Tensor::full(LORA_GATE_DELTA, (rows, width), &device)?
                            .to_dtype(dtype)?;
                        assert_close(&gate, &expected_gate)?;
                        let mixed = hc_mix(&xn, &gate, streams)?;
                        assert_close(&mixed, &hc_mix(&xn, &expected_gate, streams)?)?;
                        let change = (&mixed.to_dtype(DType::F32)?
                            - &inactive.to_dtype(DType::F32)?)?
                            .abs()?
                            .flatten_all()?
                            .max(0)?
                            .to_scalar::<f32>()?;
                        assert!(change > LORA_MIX_MIN_CHANGE);
                    } else {
                        assert_close(
                            &fused.expect("unused adapter must preserve fusion"),
                            &inactive,
                        )?;
                    }
                    Ok(())
                })?;
            }
            assert!(!wrapped.is_dynamic_lora_active());
            assert_close(
                &try_hc_up_mix(&xn, &lo, &*wrapped, streams)?
                    .expect("fusion resumes after adapter scope"),
                &inactive,
            )?;
        }
        Ok(())
    }

    fn misaligned_copy(tensor: &Tensor) -> Result<Tensor> {
        let flat = tensor.flatten_all()?;
        let prefix = Tensor::zeros(MISALIGNED_OFFSET, tensor.dtype(), tensor.device())?;
        Tensor::cat(&[&prefix, &flat], 0)?
            .narrow(0, MISALIGNED_OFFSET, tensor.elem_count())?
            .reshape(tensor.shape())
    }

    #[test]
    fn hc_prefill_hook_declines_unsupported_inputs() -> Result<()> {
        let _lock = CUDA_TEST_LOCK.lock().unwrap();
        let device = Device::new_cuda_with_stream(0)?;
        if !hc_prefill_supported(device.as_cuda_device()?) {
            return Ok(());
        }
        let (streams, hidden, rank) = SHAPES[0];
        let rows = row_classes(&device, streams, hidden)?[0];
        let width = streams * hidden;
        for dtype in [DType::BF16, DType::F16] {
            let other_dtype = if dtype == DType::BF16 {
                DType::F16
            } else {
                DType::BF16
            };
            let weight = matrix(
                width,
                rank,
                WEIGHT_SEED,
                1.0 / (rank as f32).sqrt(),
                dtype,
                &device,
            )?;
            let other_weight = weight.to_dtype(other_dtype)?;
            register_hc_up_mix(&weight, streams);
            register_hc_up_mix(&other_weight, streams);
            warmup_moe_kernels(&device, true)?;
            let layer = projection(weight.clone(), None)?;
            let other_layer = projection(other_weight, None)?;
            let lo = matrix(rows, rank, INPUT_SEED, 1.0, dtype, &device)?;
            let xn = matrix(rows, width, RESIDUAL_SEED, 1.0, dtype, &device)?;
            assert!(try_hc_up_mix(&xn, &lo, &layer, streams)?.is_some());
            assert!(try_hc_up_mix(&xn, &lo.to_dtype(other_dtype)?, &layer, streams)?.is_none());
            assert!(try_hc_up_mix(&xn.to_dtype(other_dtype)?, &lo, &layer, streams)?.is_none());
            assert!(try_hc_up_mix(&xn, &lo, &other_layer, streams)?.is_none());

            let strided_lo =
                matrix(rows, rank + 1, INPUT_SEED, 1.0, dtype, &device)?.narrow(1, 0, rank)?;
            let strided_xn =
                matrix(rows, width + 1, RESIDUAL_SEED, 1.0, dtype, &device)?.narrow(1, 0, width)?;
            let strided_weight =
                matrix(width, rank + 1, WEIGHT_SEED, 1.0, dtype, &device)?.narrow(1, 0, rank)?;
            assert!(
                !strided_lo.is_contiguous()
                    && !strided_xn.is_contiguous()
                    && !strided_weight.is_contiguous()
            );
            assert!(try_hc_up_mix(&xn, &strided_lo, &layer, streams)?.is_none());
            assert!(try_hc_up_mix(&strided_xn, &lo, &layer, streams)?.is_none());
            assert!(
                try_hc_up_mix(&xn, &lo, &projection(strided_weight, None)?, streams)?.is_none()
            );

            let offset_lo = misaligned_copy(&lo)?;
            let offset_xn = misaligned_copy(&xn)?;
            let offset_weight = misaligned_copy(&weight)?;
            for tensor in [&offset_lo, &offset_xn, &offset_weight] {
                assert!(tensor.is_contiguous());
                assert_eq!(tensor.layout().start_offset(), MISALIGNED_OFFSET);
            }
            assert!(try_hc_up_mix(&xn, &offset_lo, &layer, streams)?.is_none());
            assert!(try_hc_up_mix(&offset_xn, &lo, &layer, streams)?.is_none());
            assert!(try_hc_up_mix(&xn, &lo, &projection(offset_weight, None)?, streams)?.is_none());

            for (excluded_streams, excluded_hidden, excluded_rank) in EXCLUDED_SHAPES {
                let excluded_width = excluded_streams * excluded_hidden;
                let excluded_weight =
                    Tensor::zeros((excluded_width, excluded_rank), dtype, &device)?;
                register_hc_up_mix(&excluded_weight, excluded_streams);
                warmup_moe_kernels(&device, true)?;
                let excluded_lo = Tensor::zeros((rows, excluded_rank), dtype, &device)?;
                let excluded_xn = Tensor::zeros((rows, excluded_width), dtype, &device)?;
                assert!(try_hc_up_mix(
                    &excluded_xn,
                    &excluded_lo,
                    &projection(excluded_weight, None)?,
                    excluded_streams,
                )?
                .is_none());
            }

            let (unregistered_streams, unregistered_hidden, unregistered_rank) = UNREGISTERED_SHAPE;
            let unregistered_width = unregistered_streams * unregistered_hidden;
            let unregistered_weight = matrix(
                unregistered_width,
                unregistered_rank,
                WEIGHT_SEED,
                1.0,
                dtype,
                &device,
            )?;
            let unregistered_lo = matrix(rows, unregistered_rank, INPUT_SEED, 1.0, dtype, &device)?;
            let unregistered_xn =
                matrix(rows, unregistered_width, RESIDUAL_SEED, 1.0, dtype, &device)?;
            assert!(try_hc_up_mix(
                &unregistered_xn,
                &unregistered_lo,
                &projection(unregistered_weight, None)?,
                unregistered_streams,
            )?
            .is_none());
        }
        Ok(())
    }
}
