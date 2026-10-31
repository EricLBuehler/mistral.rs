use super::*;
use candle_core::{
    cuda::cudarc::driver::{sys, CudaGraph},
    DType, Device, Var,
};

const GATE_PATTERNS: usize = 17;
const TEST_COLUMNS: usize = 3;
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
        let gate = Tensor::from_vec(
            vec![
                f32::NEG_INFINITY,
                -100.0,
                -16.0,
                -2.0,
                -1.0,
                -0.01,
                -0.0,
                0.0,
                0.01,
                0.5,
                1.0,
                2.0,
                8.0,
                16.0,
                100.0,
                f32::INFINITY,
                f32::NAN,
            ],
            (GATE_PATTERNS, 1),
            &device,
        )?
        .to_dtype(dtype)?;
        for pattern in 0..3 {
            let value = Tensor::from_vec(
                data(GATE_PATTERNS * TEST_COLUMNS, 0x9845 + pattern),
                (GATE_PATTERNS, TEST_COLUMNS),
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
    let device = Device::new_cuda_with_stream(0)?;
    let _htod_cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for (rows, columns) in [(1, 127), (17, 513)] {
            let backing = Var::zeros((rows, columns + 1), dtype, &device)?;
            let residual = backing.narrow(1, 1, columns)?;
            let value = Tensor::from_vec(data(rows * columns, 0x76543), (rows, columns), &device)?
                .to_dtype(dtype)?;
            let gate = Var::zeros((rows, 1), dtype, &device)?;
            let _ = sigmoid_gated_add(&residual, &value, &gate)?;
            let (graph, output) = capture(&device, || sigmoid_gated_add(&residual, &value, &gate))?;
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

    Ok(())
}
