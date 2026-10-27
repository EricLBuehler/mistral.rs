use super::*;
use candle_core::{cuda::cudarc::driver::sys, quantized::QStorage, Var};
use std::borrow::Cow;

const INPUT_COLS: usize = 4096;
const KEY_ROWS: usize = 65;
const VALUE_ROWS: usize = 129;
const BACKING_COLS: usize = INPUT_COLS + 1;

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn prefetch_dispatch_graph_replay_matches_legacy_for_all_formats() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    if !is_sm121(cuda)? {
        return Ok(());
    }
    for quant in [
        GgmlDType::Q4_0,
        GgmlDType::Q4_1,
        GgmlDType::Q5_0,
        GgmlDType::Q5_1,
        GgmlDType::Q8_0,
        GgmlDType::Q2K,
        GgmlDType::Q3K,
        GgmlDType::Q4K,
        GgmlDType::Q5K,
        GgmlDType::Q6K,
    ] {
        let row_bytes = INPUT_COLS / quant.block_size() * quant.type_size();
        let output_rows = pad(PREFETCH_MIN_WEIGHT_BYTES.div_ceil(row_bytes), 2);
        let blocks = output_rows * INPUT_COLS / quant.block_size();
        let mut bytes = vec![0u8; blocks * quant.type_size()];
        let scale = half::f16::from_f32(0.0001).to_bits().to_le_bytes();
        for (block, data) in bytes.chunks_exact_mut(quant.type_size()).enumerate() {
            for (column, value) in data.iter_mut().enumerate() {
                *value = (((block * 17 + column * 13) % 251) as i32 - 125) as i8 as u8;
            }
            let scale_offset = match quant {
                GgmlDType::Q2K => data.len() - 4,
                GgmlDType::Q3K | GgmlDType::Q6K => data.len() - 2,
                _ => 0,
            };
            data[scale_offset..scale_offset + 2].copy_from_slice(&scale);
            if matches!(
                quant,
                GgmlDType::Q4_1
                    | GgmlDType::Q5_1
                    | GgmlDType::Q2K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
            ) {
                data[scale_offset + 2..scale_offset + 4].copy_from_slice(&scale);
            }
        }
        let weight = QTensor::new(
            QStorage::from_data(Cow::Owned(bytes), &device, quant)?,
            (output_rows, INPUT_COLS),
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
                    .ok_or_else(|| candle_core::Error::msg("MMVQ capture returned no graph"))?;
                for factor in [0.0f32, 0.05, -0.09] {
                    let values = (0..BACKING_COLS * 2)
                        .map(|i| ((i * 7 % 19) as f32 - 9.0) * factor)
                        .collect();
                    backing.set(
                        &Tensor::from_vec(values, (1, BACKING_COLS, 2), &device)?
                            .to_dtype(dtype)?,
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
                    for (actual, expected) in actual.iter().zip(&expected) {
                        assert!((actual - expected).abs() <= 1e-5 * (1.0 + expected.abs()),
                        "{quant:?} {dtype:?} strided={strided} factor={factor}: {actual} != {expected}");
                    }
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
        let bytes = weight.data()?;
        let make_projection = |rows| {
            QTensor::new(
                QStorage::from_data(
                    Cow::Owned(bytes[..rows * row_bytes].to_vec()),
                    &device,
                    quant,
                )?,
                (rows, INPUT_COLS),
            )
        };
        let key = make_projection(KEY_ROWS)?;
        let value = make_projection(VALUE_ROWS)?;
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let input = Tensor::ones((1, INPUT_COLS), dtype, &device)?;
            let expected = plain_impl(&weight, &input, false)?;
            let (query_out, key_out, value_out) = fused_qkv(&weight, &key, &value, &input)?;
            let (left, right) = fused_pair(&weight, &key, &input)?;
            for (actual, rows) in [
                (query_out, output_rows),
                (key_out, KEY_ROWS),
                (value_out, VALUE_ROWS),
                (left, output_rows),
                (right, KEY_ROWS),
            ] {
                let actual = actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                let expected = expected
                    .narrow(1, 0, rows)?
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?;
                assert_eq!(actual, expected, "{quant:?} {dtype:?} rows={rows}");
            }
        }
    }
    Ok(())
}
