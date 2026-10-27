use super::*;
use crate::layers::router_linear;
use candle_core::{
    cuda::cudarc::driver::{sys, CudaGraph},
    DType, Device, Var,
};
use candle_nn::{Linear, Module};

const GATE_PATTERNS: usize = 65_536;
const EXHAUSTIVE_COLUMNS: usize = 3;
const REPLAY_FACTORS: [f32; 3] = [0.0, 0.05, -0.09];

fn data(count: usize, mut state: u64) -> Vec<f32> {
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 4.0
        })
        .collect()
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

fn reference(residual: &Tensor, value: &Tensor, gate: &Tensor) -> Result<Tensor> {
    residual + value.broadcast_mul(&candle_nn::ops::sigmoid(gate)?)?
}

fn assert_exact(actual: &Tensor, expected: &Tensor) -> Result<()> {
    assert_eq!(actual.dims(), expected.dims());
    assert_eq!(actual.dtype(), expected.dtype());
    for (actual, expected) in values(actual)?.into_iter().zip(values(expected)?) {
        assert!(
            actual.to_bits() == expected.to_bits() || (actual.is_nan() && expected.is_nan()),
            "{actual:?} != {expected:?}"
        );
    }
    Ok(())
}

fn capture(device: &Device, call: impl FnOnce() -> Result<Tensor>) -> Result<(CudaGraph, Tensor)> {
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
        return Err(candle_core::Error::msg(error));
    }
    let output = call();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    Ok((
        graph
            .map_err(candle_core::Error::msg)?
            .expect("captured graph"),
        output?,
    ))
}

#[test]
#[ignore = "requires a CUDA device"]
fn sigmoid_gated_add_preserves_dtype_rounding_and_broadcast_fallbacks() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let gate = match dtype {
            DType::BF16 => Tensor::from_vec(
                (0..GATE_PATTERNS)
                    .map(|bits| half::bf16::from_bits(bits as u16))
                    .collect::<Vec<_>>(),
                (GATE_PATTERNS, 1),
                &device,
            )?,
            DType::F16 => Tensor::from_vec(
                (0..GATE_PATTERNS)
                    .map(|bits| half::f16::from_bits(bits as u16))
                    .collect::<Vec<_>>(),
                (GATE_PATTERNS, 1),
                &device,
            )?,
            _ => {
                let mut random = 0x7654321u64;
                let values = (0..GATE_PATTERNS)
                    .map(|index| {
                        random ^= random << 13;
                        random ^= random >> 7;
                        random ^= random << 17;
                        if index % 2 == 0 {
                            f32::from_bits((random >> 32) as u32)
                        } else {
                            index as f32 / GATE_PATTERNS as f32 * 220.0 - 110.0
                        }
                    })
                    .collect::<Vec<_>>();
                Tensor::from_vec(values, (GATE_PATTERNS, 1), &device)?
            }
        };
        for pattern in 0..3 {
            let value = Tensor::from_vec(
                data(GATE_PATTERNS * EXHAUSTIVE_COLUMNS, 0x9845 + pattern),
                (GATE_PATTERNS, EXHAUSTIVE_COLUMNS),
                &device,
            )?
            .to_dtype(dtype)?;
            let residual = match pattern {
                0 => Tensor::zeros(value.shape(), dtype, &device)?,
                1 => (&value * -0.5)?,
                _ => Tensor::from_vec(
                    (0..value.elem_count())
                        .map(|index| f32::from_bits((index as u32 * 7919) & 0x807fffff))
                        .collect::<Vec<_>>(),
                    value.shape(),
                    &device,
                )?
                .to_dtype(dtype)?,
            };
            assert_exact(
                &sigmoid_gated_add(&residual, &value, &gate)?,
                &reference(&residual, &value, &gate)?,
            )?;
        }
        let residual = Tensor::ones((2, 2), dtype, &device)?;
        let value = Tensor::full(2.0f32, (2, 2), &device)?.to_dtype(dtype)?;
        let gate = Tensor::new(&[0.0f32, 2.0], &device)?.to_dtype(dtype)?;
        assert!(crate::cuda::gated_add::try_sigmoid_gated_add(&residual, &value, &gate)?.is_none());
        assert_exact(
            &sigmoid_gated_add(&residual, &value, &gate)?,
            &reference(&residual, &value, &gate)?,
        )?;
        let scalar = Tensor::ones((), dtype, &device)?;
        assert!(
            crate::cuda::gated_add::try_sigmoid_gated_add(&scalar, &scalar, &scalar)?.is_none()
        );
        assert_exact(
            &sigmoid_gated_add(&scalar, &scalar, &scalar)?,
            &reference(&scalar, &scalar, &scalar)?,
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn sigmoid_gated_add_replays_offsets_and_strided_fallbacks() -> Result<()> {
    for _ in 0..2 {
        let device = Device::new_cuda_with_stream(0)?;
        let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for (rows, columns) in [(1, 127), (1, 2560), (8, 4096), (17, 513)] {
                let backing = Var::zeros((rows, columns + 1), dtype, &device)?;
                let residual = backing.narrow(1, 1, columns)?;
                let value =
                    Tensor::from_vec(data(rows * columns, 0x76543), (rows, columns), &device)?
                        .to_dtype(dtype)?;
                let gate = Var::zeros((rows, 1), dtype, &device)?;
                let _ = sigmoid_gated_add(&residual, &value, &gate)?;
                let (graph, output) =
                    capture(&device, || sigmoid_gated_add(&residual, &value, &gate))?;
                for factor in REPLAY_FACTORS {
                    backing.set(
                        &Tensor::from_vec(
                            data(rows * (columns + 1), 0x2398)
                                .into_iter()
                                .map(|x| x * factor)
                                .collect::<Vec<_>>(),
                            backing.shape(),
                            &device,
                        )?
                        .to_dtype(dtype)?,
                    )?;
                    gate.set(&Tensor::full(factor, (rows, 1), &device)?.to_dtype(dtype)?)?;
                    graph.launch().map_err(candle_core::Error::msg)?;
                    assert_exact(&output, &reference(&residual, &value, &gate)?)?;
                }
            }
        }
    }
    Ok(())
}

fn round(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}

#[test]
#[ignore = "requires a CUDA device"]
fn shared_expert_gate_matches_f64_reference_on_graph_replay() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let epsilon = match dtype {
            DType::BF16 => 1.0 / 128.0,
            DType::F16 => 1.0 / 1024.0,
            _ => 1e-5,
        };
        for columns in [256, 1024, 2560, 4096, 8192] {
            let weight = Tensor::from_vec(
                data(columns, 0x5931)
                    .into_iter()
                    .map(|x| x / (columns as f32).sqrt())
                    .collect::<Vec<_>>(),
                (1, columns),
                &device,
            )?
            .to_dtype(dtype)?;
            let weight_values = values(&weight)?;
            let linear = Linear::new(weight, None);
            let input = Var::zeros((1, columns), dtype, &device)?;
            let _ = router_linear(&linear, &input)?;
            let (graph, output) = capture(&device, || router_linear(&linear, &input))?;
            for turn in 0..5 {
                let input_values = data(columns, 0x8437 + turn)
                    .into_iter()
                    .map(|x| round(x, dtype))
                    .collect::<Vec<_>>();
                let expected = round(
                    input_values
                        .iter()
                        .zip(&weight_values)
                        .map(|(a, b)| f64::from(*a) * f64::from(*b))
                        .sum::<f64>() as f32,
                    dtype,
                );
                input.set(
                    &Tensor::from_vec(input_values, (1, columns), &device)?.to_dtype(dtype)?,
                )?;
                graph.launch().map_err(candle_core::Error::msg)?;
                let actual = values(&output)?[0];
                assert!(
                    (actual - expected).abs() <= epsilon * expected.abs() + 2e-5,
                    "{dtype:?} width{columns}: {actual} != {expected}"
                );
                let baseline = linear.forward(&input)?;
                assert_eq!(output.dims(), baseline.dims());
                assert_eq!(output.dtype(), baseline.dtype());
            }
        }
    }
    Ok(())
}
