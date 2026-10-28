use super::*;
use candle_core::{cuda::cudarc::driver::sys, Device};

const CASES: [(usize, usize, usize); 5] = [
    (1, 1, 1),
    (1, 10, 2560),
    (7, 3, 257),
    (2, 33, 65),
    (65536, 1, 1),
];
const OFFSET: usize = 3;

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

fn offset(tensor: &Tensor) -> Result<Tensor> {
    let padding = Tensor::zeros(OFFSET, tensor.dtype(), tensor.device())?;
    Tensor::cat(&[&padding, &tensor.flatten_all()?], 0)?
        .narrow(0, OFFSET, tensor.elem_count())?
        .reshape(tensor.dims())
}

fn reference(input: &Tensor, weights: &Tensor, dtype: DType) -> Result<Tensor> {
    let (tokens, topk, hidden) = input.dims3()?;
    let source = values(input)?;
    let weights = values(weights)?;
    let mut output = vec![0f32; tokens * hidden];
    for token in 0..tokens {
        for column in 0..hidden {
            for route in 0..topk {
                let row = token * topk + route;
                output[token * hidden + column] += source[row * hidden + column] * weights[row];
            }
        }
    }
    Tensor::from_vec(output, (tokens, hidden), &Device::Cpu)?.to_dtype(dtype)
}

#[test]
#[ignore = "requires CUDA"]
fn weighted_reduce_matches_independent_reference_for_dtypes_and_views() -> Result<()> {
    let device = Device::new_cuda(0)?;
    for (tokens, topk, hidden) in CASES {
        let weights = Tensor::from_vec(
            (0..tokens * topk)
                .map(|i| (i % 11 + 1) as f32 / 13.0)
                .collect::<Vec<_>>(),
            (tokens, topk),
            &Device::Cpu,
        )?;
        let source = Tensor::from_vec(
            (0..tokens * topk * hidden)
                .map(|i| ((i * 1301 % 32003) as f32 - 16001.0) / 4096.0)
                .collect::<Vec<_>>(),
            (tokens, topk, hidden),
            &Device::Cpu,
        )?;
        for input_dtype in [DType::F16, DType::BF16, DType::F32] {
            let source = source.to_dtype(input_dtype)?;
            let input = offset(&source.to_device(&device)?)?;
            let weights_gpu = offset(&weights.to_device(&device)?)?;
            let strided = input.transpose(1, 2)?.contiguous()?.transpose(1, 2)?;
            for input in [&input, &strided] {
                for output_dtype in [DType::F16, DType::BF16, DType::F32] {
                    let actual = moe_weighted_reduce(input, &weights_gpu, output_dtype)?;
                    let expected = reference(&source, &weights, output_dtype)?;
                    assert_eq!(actual.dims(), expected.dims());
                    assert_eq!(actual.dtype(), output_dtype);
                    assert_eq!(
                        values(&actual)?,
                        values(&expected)?,
                        "{input_dtype:?} to {output_dtype:?} {tokens}/{topk}/{hidden}"
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires CUDA"]
fn weighted_reduce_graph_replay_reads_updated_routes() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let source = Tensor::from_vec(
        vec![1f32, -2., 4., 8., -1., 3., 0.5, 2., -4., 6., 7., -8.],
        (2, 2, 3),
        &device,
    )?
    .to_dtype(DType::BF16)?;
    let weights = Tensor::from_vec(vec![0.25f32, 0.75, 0.5, 0.5], (2, 2), &device)?;
    let alternate = (&weights * 0.5)?;
    let live = weights.zeros_like()?;
    live.slice_set(&weights, 0, 0)?;
    let _ = moe_weighted_reduce(&source, &live, DType::BF16)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    device.synchronize()?;
    let tracking = stream.context().is_event_tracking();
    if tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(candle_core::Error::msg)?;
    let output = moe_weighted_reduce(&source, &live, DType::BF16);
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let output = output?;
    let graph = graph
        .map_err(candle_core::Error::msg)?
        .ok_or_else(|| candle_core::Error::msg("weighted reduction capture returned no graph"))?;
    for weights in [&weights, &alternate, &weights] {
        live.slice_set(weights, 0, 0)?;
        graph.launch().map_err(candle_core::Error::msg)?;
        device.synchronize()?;
        assert_eq!(
            values(&output)?,
            values(&reference(&source, weights, DType::BF16)?)?
        );
    }
    Ok(())
}
