use super::*;
use candle_core::{cuda::cudarc::driver::sys, quantized::QStorage, Var};
use std::borrow::Cow;

const INPUT_COLS: usize = 2560;
const OUTPUT_ROWS: usize = 6144;
const BACKING_COLS: usize = INPUT_COLS + 1;
const BLOCK_BYTES: usize = 34;

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn sm121_q8_dispatch_graph_replay_matches_legacy_on_strided_inputs() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if !is_sm121(cuda)? {
        return Ok(());
    }
    let blocks = OUTPUT_ROWS * INPUT_COLS / Q8_1_BLOCK_SIZE;
    let mut bytes = vec![0u8; blocks * BLOCK_BYTES];
    let scale = half::f16::from_f32(0.0125).to_bits().to_le_bytes();
    for (block, data) in bytes.chunks_exact_mut(BLOCK_BYTES).enumerate() {
        data[..2].copy_from_slice(&scale);
        for (column, value) in data[2..].iter_mut().enumerate() {
            *value = (((block * 17 + column * 13) % 251) as i32 - 125) as i8 as u8;
        }
    }
    let weight = QTensor::new(
        QStorage::from_data(Cow::Owned(bytes), &device, GgmlDType::Q8_0)?,
        (OUTPUT_ROWS, INPUT_COLS),
    )?;
    let stream = cuda.cuda_stream();
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for strided in [false, true] {
            let backing = Var::zeros((1, BACKING_COLS, 2), dtype, &device)?;
            let input = if strided {
                backing
                    .narrow(1, 1, INPUT_COLS)?
                    .narrow(2, 1, 1)?
                    .transpose(1, 2)?
            } else {
                backing
                    .reshape((1, 1, BACKING_COLS * 2))?
                    .narrow(2, 2, INPUT_COLS)?
            };
            assert!(input.layout().start_offset() > 0);
            assert_eq!(input.is_contiguous(), !strided);
            let _ = plain(&weight, &input)?;
            let _ = plain_impl(&weight, &input, false)?;
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
                return Err(candle_core::Error::msg(error));
            }
            let output = plain(&weight, &input);
            let graph = stream.end_capture(
                sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            let output = output?;
            let graph = graph
                .map_err(candle_core::Error::msg)?
                .ok_or_else(|| candle_core::Error::msg("Q8 MMVQ capture returned no graph"))?;
            for factor in [0.0f32, 0.05, -0.09] {
                let values = (0..BACKING_COLS * 2)
                    .map(|i| ((i * 7 % 19) as f32 - 9.0) * factor)
                    .collect();
                backing.set(
                    &Tensor::from_vec(values, (1, BACKING_COLS, 2), &device)?.to_dtype(dtype)?,
                )?;
                graph.launch().map_err(candle_core::Error::msg)?;
                let expected = plain_impl(&weight, &input, false)?;
                let actual = output
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(
                    actual, expected,
                    "{dtype:?} strided={strided} factor={factor}"
                );
            }
            let batched = Tensor::cat(&[&input, &input], 1)?;
            let actual = plain(&weight, &batched)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let expected = plain_impl(&weight, &batched, false)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            assert_eq!(actual, expected);
        }
    }
    Ok(())
}
