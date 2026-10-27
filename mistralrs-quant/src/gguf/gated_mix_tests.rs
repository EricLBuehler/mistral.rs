use std::sync::Arc;

use candle_core::{
    quantized::{GgmlDType, QTensor},
    Device, Var,
};

use super::*;
use crate::{GgufMatMul, QuantMethodConfig};

const FORMATS: [GgmlDType; 11] = [
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
];
const SHAPES: [(usize, usize, usize); 3] = [(2, 193, 256), (3, 511, 512), (4, 2560, 320)];
const TOLERANCE: f32 = 0.008;

fn projection(format: GgmlDType, rows: usize, rank: usize, device: &Device) -> Result<GgufMatMul> {
    let values = (0..rows * rank)
        .map(|i| (i as f32 * 0.17 + 0.8).sin() / (rank as f32).sqrt())
        .collect::<Vec<_>>();
    let weight = Tensor::from_vec(values, (rows, rank), &Device::Cpu)?;
    GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(QTensor::quantize_onto(&weight, format, device)?),
        b: None,
    })
}

fn reference(
    input: &Tensor,
    projection: &GgufMatMul,
    residual: &Tensor,
    streams: usize,
) -> Result<Tensor> {
    let hidden = residual.dim(candle_core::D::Minus1)? / streams;
    let gate = if projection.get_qtensor().unwrap().dtype() == GgmlDType::Q8_1 {
        input
            .to_dtype(DType::F32)?
            .matmul(&projection.dequantize_w()?.to_dtype(DType::F32)?.t()?)?
            .to_dtype(input.dtype())?
    } else {
        projection.forward(input)?
    };
    (residual.to_dtype(DType::F32)? * candle_nn::ops::sigmoid(&gate.to_dtype(DType::F32)?)?)?
        .reshape((streams, hidden))?
        .mean(0)?
        .to_dtype(residual.dtype())?
        .reshape((1, hidden))
}

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar()
}

#[test]
fn quantized_gated_mix_formats_and_shapes_match_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (streams, hidden, rank) in SHAPES {
        for format in FORMATS {
            if !rank.is_multiple_of(format.block_size()) {
                continue;
            }
            let projection = projection(format, streams * hidden, rank, &device)?;
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let input = Tensor::from_vec(
                    (0..rank)
                        .map(|i| (i as f32 * 0.13).sin())
                        .collect::<Vec<_>>(),
                    (1, rank),
                    &device,
                )?
                .to_dtype(dtype)?;
                let residual = Tensor::from_vec(
                    (0..streams * hidden)
                        .map(|i| (i as f32 * 0.03).cos())
                        .collect::<Vec<_>>(),
                    (1, streams * hidden),
                    &device,
                )?
                .to_dtype(dtype)?;
                let actual = try_fused_quantized_gated_mix(
                    &input,
                    &projection,
                    &residual,
                    streams,
                    QuantizedInputTransform::Identity,
                )?
                .unwrap();
                let expected = reference(&input, &projection, &residual, streams)?;
                let error = max_diff(&actual, &expected)?;
                assert!(
                    error <= TOLERANCE,
                    "{format:?} {dtype:?} {streams}x{hidden}x{rank}: {error}"
                );
                let actual = try_fused_quantized_gated_mix(
                    &input,
                    &projection,
                    &residual,
                    streams,
                    QuantizedInputTransform::ScaledSilu {
                        divisor: streams as f32,
                    },
                )?
                .unwrap();
                let transformed = candle_nn::ops::silu(&(&input / streams as f64)?)?;
                let expected = reference(&transformed, &projection, &residual, streams)?;
                let error = max_diff(&actual, &expected)?;
                assert!(
                    error <= TOLERANCE,
                    "{format:?} {dtype:?} {streams}x{hidden}x{rank} scaled silu: {error}"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn shared_quantized_pair_prefill_matches_independent_projections() -> Result<()> {
    const RANK: usize = 512;
    const LEFT_ROWS: usize = 193;
    const RIGHT_ROWS: usize = 3;
    let device = Device::new_cuda(0)?;
    for format in FORMATS {
        if !super::super::fast_mmq::supports(format) {
            continue;
        }
        let left = projection(format, LEFT_ROWS, RANK, &device)?;
        let right = projection(format, RIGHT_ROWS, RANK, &device)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for tokens in [9, 37, 129] {
                let input = Tensor::from_vec(
                    (0..tokens * RANK)
                        .map(|i| (i as f32 * 0.13).sin())
                        .collect::<Vec<_>>(),
                    (tokens, RANK),
                    &device,
                )?
                .to_dtype(dtype)?;
                let left_weight = left.get_qtensor().unwrap();
                let right_weight = right.get_qtensor().unwrap();
                let (actual_left, actual_right) =
                    super::super::fast_mmq::fused_pair(&left_weight, &right_weight, &input)?;
                assert!(
                    max_diff(
                        &actual_left,
                        &super::super::fast_mmq::plain(&left_weight, &input)?
                    )? <= TOLERANCE,
                    "{format:?} {dtype:?} tokens={tokens} left"
                );
                assert!(
                    max_diff(
                        &actual_right,
                        &super::super::fast_mmq::plain(&right_weight, &input)?
                    )? <= TOLERANCE,
                    "{format:?} {dtype:?} tokens={tokens} right"
                );
            }
        }
    }
    Ok(())
}

#[test]
fn quantized_gated_mix_graph_replay_matches_reference() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
    let (streams, hidden, rank) = SHAPES[0];
    let transform = QuantizedInputTransform::ScaledSilu {
        divisor: streams as f32,
    };
    for format in FORMATS {
        let projection = projection(format, streams * hidden, rank, &device)?;
        let input = Var::zeros((1, rank), DType::BF16, &device)?;
        let residual = Tensor::ones((1, streams * hidden), DType::BF16, &device)?;
        let _ = try_fused_quantized_gated_mix(&input, &projection, &residual, streams, transform)?
            .unwrap();
        device.synchronize()?;
        let stream = cuda.cuda_stream();
        let tracking = stream.context().is_event_tracking();
        if tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        let capture =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
        if let Err(error) = capture {
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(error));
        }
        let output =
            try_fused_quantized_gated_mix(&input, &projection, &residual, streams, transform);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?.unwrap();
        let graph = graph.map_err(candle_core::Error::msg)?.unwrap();
        for scale in [0.0f32, 0.15, 1.25, -3.0, 0.15] {
            input.set(
                &Tensor::from_vec(
                    (0..rank)
                        .map(|i| (i as f32 * 0.13).sin() * scale)
                        .collect::<Vec<_>>(),
                    (1, rank),
                    &device,
                )?
                .to_dtype(DType::BF16)?,
            )?;
            graph.launch().map_err(candle_core::Error::msg)?;
            device.synchronize()?;
            let transformed = candle_nn::ops::silu(&(input.as_tensor() / streams as f64)?)?;
            let expected = reference(&transformed, &projection, &residual, streams)?;
            assert!(
                max_diff(&output, &expected)? <= TOLERANCE,
                "{format:?} scale={scale}"
            );
        }
    }
    Ok(())
}
