use super::*;
use crate::{mxfp8::Mxfp8Linear, QuantMethod};
use candle_core::Device;

const ROWS: usize = 17;
const OUTPUTS: usize = 129;
const INPUTS: usize = 1056;
const TILE_OUTPUTS: usize = 64;
const SPLITS: usize = 4;

#[test]
fn split_k_graph_replay_reads_new_inputs_with_temporary_workspace() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;
    let device = Device::new_cuda_with_stream(0)?;
    if !super::super::block_scaled_mma_supported(device.as_cuda_device()?) {
        return Ok(());
    }
    let multiprocessors = device_multiprocessor_count(device.as_cuda_device()?);
    let (rows, n, k) = (ROWS, OUTPUTS, INPUTS);
    {
        let dtype = DType::BF16;
        let weight = Tensor::from_vec(vec![F8E4M3::from_f32(1.0); n * k], (n, k), &device)?;
        let scales = Tensor::from_vec(
            vec![127u8; n * k / MXFP8_BLOCK_SIZE],
            (n, k / MXFP8_BLOCK_SIZE),
            &device,
        )?;
        let layer = Mxfp8Linear::from_parts(weight.clone(), scales.clone(), None, dtype)?;
        let input = Tensor::ones((rows, k), dtype, &device)?;
        let cfg = Config {
            bm: MIN_TILE_M,
            bn: TILE_OUTPUTS,
            splits: SPLITS,
            ..config(rows, n, k, multiprocessors)
        };
        let launch = || {
            forward_config(
                &layer.quantize_activation(&input)?,
                &weight,
                &scales,
                Some(cfg),
            )
        };
        let _ = launch()?;
        device.synchronize()?;
        let compiled = cutile::tile_kernel::jit_compile_count();
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
        let output = launch();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph.map_err(candle_core::Error::wrap)?.unwrap();
        assert_eq!(cutile::tile_kernel::jit_compile_count(), compiled);
        for multiplier in [1.0f32, -0.5] {
            let values = (0..rows * k)
                .map(|i| [1.0, -2.0, 0.5][(i / k) % 3] * multiplier)
                .collect::<Vec<_>>();
            input.slice_set(
                &Tensor::from_vec(values, (rows, k), &device)?.to_dtype(dtype)?,
                0,
                0,
            )?;
            let scratch = Tensor::full(
                -1.0f32,
                (rows.div_ceil(cfg.bm) * cfg.bm * cfg.splits, n),
                &device,
            )?;
            graph.launch().map_err(candle_core::Error::wrap)?;
            let actual = output.to_dtype(DType::F32)?.to_vec2::<f32>()?;
            for (row, values) in actual.iter().enumerate() {
                let expected = [1.0, -2.0, 0.5][row % 3] * multiplier * k as f32;
                assert!(
                    values.iter().all(|&value| value == expected),
                    "{dtype:?} row={row}"
                );
            }
            std::hint::black_box(&scratch);
        }
    }
    Ok(())
}
