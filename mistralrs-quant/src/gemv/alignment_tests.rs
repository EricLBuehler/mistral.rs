use super::*;
use candle_core::{cuda::cudarc::driver::sys, Device};
use candle_nn::Linear;

use crate::{QuantMethod, QuantMethodConfig, UnquantLinear};

const ROWS: usize = 8;
const COLUMNS: usize = 64;
const STREAM_DELAY_BYTES: usize = 32 * 1024 * 1024;

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

#[test]
#[ignore = "requires a CUDA device"]
fn gemv_alignment_checks_use_compute_stream_during_graph_capture() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let producer = stream
        .context()
        .new_stream()
        .map_err(candle_core::Error::wrap)?;
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let delay = producer
            .alloc_zeros::<u8>(STREAM_DELAY_BYTES)
            .map_err(candle_core::Error::wrap)?;
        let input_values = Tensor::ones(COLUMNS, dtype, &Device::Cpu)?;
        let weight_values = Tensor::ones(ROWS * COLUMNS, dtype, &Device::Cpu)?;
        macro_rules! upload {
            ($type:ty) => {{
                let input = producer
                    .clone_htod(&input_values.to_vec1::<$type>()?)
                    .map_err(candle_core::Error::wrap)?;
                let weight = producer
                    .clone_htod(&weight_values.to_vec1::<$type>()?)
                    .map_err(candle_core::Error::wrap)?;
                (
                    Tensor::from((
                        Storage::Cuda(CudaStorage::wrap_cuda_slice(input, cuda.clone())),
                        (1, COLUMNS),
                    )),
                    Tensor::from((
                        Storage::Cuda(CudaStorage::wrap_cuda_slice(weight, cuda.clone())),
                        (ROWS, COLUMNS),
                    )),
                )
            }};
        }
        let (input, weight) = match dtype {
            DType::BF16 => upload!(bf16),
            DType::F16 => upload!(f16),
            DType::F32 => upload!(f32),
            _ => unreachable!(),
        };
        assert!(should_use_gemv(&input, &weight));
        let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(weight, None)))?;
        assert_eq!(values(&layer.forward(&input)?)?, vec![COLUMNS as f32; ROWS]);
        device.synchronize()?;
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
        let captured = layer.forward(&input);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let captured = captured?;
        let graph = graph
            .map_err(candle_core::Error::wrap)?
            .expect("captured GEMV graph");
        for value in [0f32, -0.5, 0.25, 1.0] {
            input.slice_set(
                &Tensor::full(value, (1, COLUMNS), &device)?.to_dtype(dtype)?,
                0,
                0,
            )?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            assert_eq!(values(&captured)?, vec![COLUMNS as f32 * value; ROWS]);
        }
        device.synchronize()?;
        std::hint::black_box(delay);
    }
    Ok(())
}
