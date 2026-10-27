use super::*;
use crate::{try_fused_quantized_pair, GgufMatMul, QuantMethod, QuantMethodConfig, UnquantLinear};
use candle_core::{cuda::cudarc::driver::sys, Var};
use std::sync::Arc;

const FORMATS: [GgmlDType; 10] = [
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
];
const INPUT_OFFSET: usize = 2;
const TEST_COLUMNS: usize = 512;
const TEST_QUANT_ROWS: usize = 64;
const TEST_DENSE_ROWS: usize = 3;
const STREAM_DELAY_BYTES: usize = 32 * 1024 * 1024;

fn data(elements: usize, salt: usize) -> Vec<f32> {
    (0..elements)
        .map(|index| ((index * 37 + salt * 19) % 211) as f32 / 512.0 - 0.2)
        .collect()
}

fn quant(format: GgmlDType, rows: usize, columns: usize, device: &Device) -> Result<GgufMatMul> {
    let source = Tensor::from_vec(data(rows * columns, rows), (rows, columns), &Device::Cpu)?;
    GgufMatMul::new(QuantMethodConfig::Gguf {
        q_weight: Arc::new(QTensor::quantize_onto(&source, format, device)?),
        b: None,
    })
}

fn dense(weight: Tensor) -> Result<UnquantLinear> {
    UnquantLinear::new(QuantMethodConfig::Unquantized(candle_nn::Linear::new(
        weight, None,
    )))
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
}

#[test]
#[ignore = "requires a CUDA device"]
fn mixed_pair_formats_graph_preserve_inputs_and_output_order() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let _htod_cache = cuda.enable_cuda_graph_htod_cache();
    for format in FORMATS {
        let mut shapes = vec![(64, 3, 512), (96, 7, 2048)];
        if is_sm121(cuda)? {
            let rows = dense_pair_prefetch_rows(cuda)?.next_multiple_of(2);
            let columns = (DENSE_PAIR_PREFETCH_BYTES_PER_THREAD * DENSE_PAIR_THREADS)
                .div_ceil(format.type_size())
                * format.block_size();
            let columns = columns.next_multiple_of(MATRIX_ROW_PADDING);
            shapes.push((rows, TEST_DENSE_ROWS, columns));
        }
        for (rows, dense_rows, columns) in shapes {
            let quant = quant(format, rows, columns, &device)?;
            for dtype in [DType::BF16, DType::F16, DType::F32] {
                let dense = dense(
                    Tensor::from_vec(
                        data(dense_rows * columns, dense_rows),
                        (dense_rows, columns),
                        &device,
                    )?
                    .to_dtype(dtype)?,
                )?;
                let backing = Var::zeros((1, 1, columns + INPUT_OFFSET), dtype, &device)?;
                let input = backing.narrow(2, INPUT_OFFSET, columns)?;
                let other = Tensor::ones(input.shape(), dtype, &device)?;
                let run = || -> Result<Vec<Tensor>> {
                    let pair =
                        try_fused_quantized_pair(&input, &quant, &dense)?.expect("mixed pair");
                    let reverse = try_fused_quantized_pair(&input, &dense, &quant)?
                        .expect("reverse mixed pair");
                    let overwrite = quant.forward(&other)?;
                    let repeat = try_fused_quantized_pair(&input, &quant, &dense)?
                        .expect("repeated mixed pair");
                    Ok(vec![
                        pair.0, pair.1, reverse.0, reverse.1, overwrite, repeat.0, repeat.1,
                    ])
                };
                let _ = run()?;
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
                let outputs = run();
                let graph = stream.end_capture(
                    sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
                );
                if tracking {
                    unsafe { stream.context().enable_event_tracking() };
                }
                let outputs = outputs?;
                let graph = graph
                    .map_err(candle_core::Error::msg)?
                    .expect("captured graph");
                for salt in [0, 7, 29] {
                    backing.set(
                        &Tensor::from_vec(
                            data(columns + INPUT_OFFSET, salt),
                            (1, 1, columns + INPUT_OFFSET),
                            &device,
                        )?
                        .to_dtype(dtype)?,
                    )?;
                    graph.launch().map_err(candle_core::Error::msg)?;
                    let quant_expected = quant.forward(&input)?;
                    let dense_expected = dense.forward(&input)?;
                    for index in [0, 3, 5] {
                        assert_eq!(outputs[index].dims(), quant_expected.dims());
                        assert_eq!(
                            values(&outputs[index])?,
                            values(&quant_expected)?,
                            "{format:?} {dtype:?} quant"
                        );
                    }
                    for index in [1, 2, 6] {
                        assert_eq!(outputs[index].dims(), dense_expected.dims());
                        assert_eq!(
                            values(&outputs[index])?,
                            values(&dense_expected)?,
                            "{format:?} {dtype:?} dense"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn mixed_pair_declines_unsupported_tails_and_layouts() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let dtype = DType::BF16;
    let input = Tensor::ones((1, TEST_COLUMNS), dtype, &device)?;
    let dense_weight = Tensor::ones((TEST_DENSE_ROWS, TEST_COLUMNS), dtype, &device)?;
    let projection = dense(dense_weight.clone())?;
    let even = quant(GgmlDType::Q8_0, TEST_QUANT_ROWS, TEST_COLUMNS, &device)?;
    assert!(try_fused_quantized_pair(&input, &even, &projection)?.is_some());
    for rows in [1, TEST_QUANT_ROWS - 1, TEST_QUANT_ROWS + 1] {
        let odd = quant(GgmlDType::Q8_0, rows, TEST_COLUMNS, &device)?;
        assert!(try_fused_quantized_pair(&input, &odd, &projection)?.is_none());
        assert!(try_fused_quantized_pair(&input, &projection, &odd)?.is_none());
    }
    let multiple = Tensor::ones((2, TEST_COLUMNS), dtype, &device)?;
    assert!(try_fused_quantized_pair(&multiple, &even, &projection)?.is_none());
    let unaligned = Tensor::ones(TEST_DENSE_ROWS * TEST_COLUMNS + 1, dtype, &device)?
        .narrow(0, 1, TEST_DENSE_ROWS * TEST_COLUMNS)?
        .reshape((TEST_DENSE_ROWS, TEST_COLUMNS))?;
    assert!(try_fused_quantized_pair(&input, &even, &dense(unaligned)?)?.is_none());
    let strided = Tensor::ones((TEST_COLUMNS, TEST_DENSE_ROWS), dtype, &device)?.t()?;
    assert!(try_fused_quantized_pair(&input, &even, &dense(strided)?)?.is_none());
    let unaligned =
        Tensor::ones((1, TEST_COLUMNS + 1), dtype, &device)?.narrow(1, 1, TEST_COLUMNS)?;
    assert!(try_fused_quantized_pair(&unaligned, &even, &projection)?.is_none());
    let unsupported = quant(GgmlDType::Q8K, TEST_QUANT_ROWS, TEST_COLUMNS, &device)?;
    assert!(try_fused_quantized_pair(&input, &unsupported, &projection)?.is_none());
    Ok(())
}

#[test]
#[ignore = "requires a CUDA device"]
fn mixed_pair_orders_secondary_stream_allocations() -> Result<()> {
    let device = Device::new_cuda_with_stream(0)?;
    let cuda = device.as_cuda_device()?;
    let producer = cuda
        .cuda_stream()
        .context()
        .new_stream()
        .map_err(candle_core::Error::wrap)?;
    let projection = quant(GgmlDType::Q4K, TEST_QUANT_ROWS, TEST_COLUMNS, &device)?;
    for salt in [3, 17, 43] {
        let delay = producer
            .alloc_zeros::<u8>(STREAM_DELAY_BYTES)
            .map_err(candle_core::Error::wrap)?;
        let input = data(TEST_COLUMNS, salt);
        let weight = data(TEST_DENSE_ROWS * TEST_COLUMNS, salt);
        let input_storage = producer
            .clone_htod(&input)
            .map_err(candle_core::Error::wrap)?;
        let weight_storage = producer
            .clone_htod(&weight)
            .map_err(candle_core::Error::wrap)?;
        let input = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(input_storage, cuda.clone())),
            (1, TEST_COLUMNS),
        ));
        let weight = Tensor::from((
            Storage::Cuda(CudaStorage::wrap_cuda_slice(weight_storage, cuda.clone())),
            (TEST_DENSE_ROWS, TEST_COLUMNS),
        ));
        let dense = dense(weight)?;
        let actual = try_fused_quantized_pair(&input, &projection, &dense)?
            .expect("same-context allocations");
        assert_eq!(values(&actual.0)?, values(&projection.forward(&input)?)?);
        assert_eq!(values(&actual.1)?, values(&dense.forward(&input)?)?);
        std::hint::black_box(delay);
    }
    Ok(())
}
