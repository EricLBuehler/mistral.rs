use super::*;
use candle_core::cuda::cudarc::driver::sys;
use half::{bf16, f16};

const OUTPUT_DIM: usize = 32_769;
const INPUT_DIM: usize = 3_074;
const BIAS_INPUT_DIM: usize = 3_072;
const EVEN_OFFSET: usize = 2;
const CUDA_SEED: u64 = 0x61d2_89a3;
const RANDOM_STDDEV: f32 = 0.0625;
const ACCUMULATION_TOLERANCE: f32 = 1e-5;

fn sm121_device() -> Result<Option<Device>> {
    let device = Device::new_cuda(0)?;
    let cuda = device.as_cuda_device()?;
    let stream = cuda.cuda_stream();
    let major = stream
        .context()
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(candle_core::Error::msg)?;
    let minor = stream
        .context()
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(candle_core::Error::msg)?;
    Ok((major == 12 && minor == 1).then_some(device))
}

fn linear(weight: Tensor, bias: Option<Tensor>) -> Result<UnquantLinear> {
    <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(Linear::new(weight, bias)))
}

fn values(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()
}

fn assert_finite_close(actual: &Tensor, reference: &Tensor, exact: bool) -> Result<()> {
    assert_eq!(actual.dims(), reference.dims());
    assert_eq!(actual.dtype(), reference.dtype());
    let epsilon = match actual.dtype() {
        DType::BF16 => bf16::EPSILON.to_f32(),
        DType::F16 => f16::EPSILON.to_f32(),
        _ => unreachable!(),
    };
    let actual = values(actual)?;
    let reference = values(reference)?;
    assert_eq!(actual.len(), reference.len());
    for (index, (&actual, &reference)) in actual.iter().zip(&reference).enumerate() {
        assert!(
            actual.is_finite() && reference.is_finite(),
            "nonfinite index={index}"
        );
        let tolerance = if exact {
            0.0
        } else {
            epsilon * reference.abs() + ACCUMULATION_TOLERANCE
        };
        assert!(
            (actual - reference).abs() <= tolerance,
            "index={index} actual={actual} reference={reference} tolerance={tolerance}"
        );
    }
    Ok(())
}

fn padded_view(tensor: &Tensor, offset: usize) -> Result<Tensor> {
    let padding = Tensor::zeros(EVEN_OFFSET, tensor.dtype(), tensor.device())?;
    Tensor::cat(&[&padding, &tensor.flatten_all()?, &padding], 0)?
        .narrow(0, offset, tensor.elem_count())?
        .reshape(tensor.dims())
}

fn dyadic_fixture(device: &Device, dtype: DType) -> Result<(Tensor, Tensor, Tensor)> {
    let signs = Tensor::from_vec(
        (0..OUTPUT_DIM)
            .map(|index| {
                let mixed = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
                if (mixed ^ (mixed >> 29)) & 1 == 0 {
                    1f32
                } else {
                    -1f32
                }
            })
            .collect::<Vec<_>>(),
        OUTPUT_DIM,
        device,
    )?
    .to_dtype(dtype)?;
    let delta = match dtype {
        DType::BF16 => 1.0 / 512.0,
        DType::F16 => 1.0 / 4096.0,
        _ => unreachable!(),
    };
    let mut columns = vec![0f32; BIAS_INPUT_DIM];
    columns[0] = 1.0;
    columns[1] = delta;
    let columns = Tensor::from_vec(columns, (1, BIAS_INPUT_DIM), device)?.to_dtype(dtype)?;
    let weight = signs.unsqueeze(1)?.broadcast_mul(&columns)?;
    let input = Tensor::ones((1, BIAS_INPUT_DIM), dtype, device)?;
    Ok((input, weight, signs.neg()?))
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn wide_gemv_random_tails_preserve_ranks_and_statistics() -> Result<()> {
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        device.set_seed(CUDA_SEED)?;
        let weight = Tensor::randn(0f32, RANDOM_STDDEV, (OUTPUT_DIM, INPUT_DIM), &device)?
            .to_dtype(dtype)?;
        let input = Tensor::randn(0f32, RANDOM_STDDEV, (1, INPUT_DIM), &device)?.to_dtype(dtype)?;
        let layer = linear(weight.clone(), None)?;
        layer.begin_track_stats()?;
        for (index, shape) in [
            vec![1, INPUT_DIM],
            vec![1, 1, INPUT_DIM],
            vec![1, 1, 1, INPUT_DIM],
        ]
        .iter()
        .enumerate()
        {
            let input = input.reshape(shape.as_slice())?;
            assert!(crate::gemv::should_use_wide_gemv(&input, &weight));
            let actual = layer.forward(&input)?;
            let mut expected_shape = shape.clone();
            *expected_shape.last_mut().unwrap() = OUTPUT_DIM;
            let reference = input
                .reshape((1, INPUT_DIM))?
                .matmul(&weight.t()?)?
                .reshape(expected_shape)?;
            assert_finite_close(&actual, &reference, false)?;
            assert_eq!(layer.stats_snapshot(), Some((index + 1, index + 1)));
        }
        let _ = layer.end_track_stats()?;
        assert!(layer.stats_snapshot().is_none());
    }
    device.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn wide_gemv_bias_rounds_after_matmul_with_aligned_offsets() -> Result<()> {
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        let (input, weight, bias) = dyadic_fixture(&device, dtype)?;
        let input = padded_view(&input, EVEN_OFFSET)?;
        let weight = padded_view(&weight, EVEN_OFFSET)?;
        assert_eq!(input.layout().start_offset(), EVEN_OFFSET);
        assert_eq!(weight.layout().start_offset(), EVEN_OFFSET);
        assert!(crate::gemv::should_use_wide_gemv(&input, &weight));
        let sentinel = Tensor::ones(OUTPUT_DIM, dtype, &device)?;
        let strided_bias = Tensor::stack(&[&bias, &sentinel], 1)?
            .narrow(1, 0, 1)?
            .squeeze(1)?;
        assert!(!strided_bias.is_contiguous());
        let layer = linear(weight.clone(), Some(strided_bias))?;
        let reference = input.matmul(&weight.t()?)?.broadcast_add(&bias)?;
        assert!(values(&reference)?.iter().all(|&value| value == 0.0));
        assert_finite_close(&layer.forward(&input)?, &reference, true)?;
        let prematurely_biased = crate::gemv::gemv(&input, &weight, Some(&bias))?;
        let prematurely_biased = values(&prematurely_biased)?;
        assert!(prematurely_biased.iter().all(|value| value.is_finite()));
        assert!(prematurely_biased.iter().any(|&value| value != 0.0));
    }
    device.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn wide_gemv_rejects_unsupported_views_and_preserves_gemm_fallback() -> Result<()> {
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        let (input, weight, _) = dyadic_fixture(&device, dtype)?;
        let check_fallback = |input: &Tensor, weight: &Tensor| -> Result<()> {
            assert!(!crate::gemv::should_use_wide_gemv(input, weight));
            let layer = linear(weight.clone(), None)?;
            match input.matmul(&weight.t()?) {
                Ok(reference) => assert_finite_close(&layer.forward(input)?, &reference, true),
                Err(reference) => {
                    assert_eq!(
                        layer.forward(input).unwrap_err().to_string(),
                        reference.to_string()
                    );
                    Ok(())
                }
            }
        };
        let odd_input = padded_view(&input, 1)?;
        check_fallback(&odd_input, &weight)?;
        let odd_weight = padded_view(&weight, 1)?;
        check_fallback(&input, &odd_weight)?;
        let input_columns =
            Tensor::stack(&[&input.flatten_all()?, &input.flatten_all()?.neg()?], 1)?;
        let strided_input = input_columns.narrow(1, 0, 1)?.t()?;
        assert!(!strided_input.is_contiguous());
        check_fallback(&strided_input, &weight)?;
        let strided_input3 = strided_input.unsqueeze(0)?;
        assert!(!crate::gemv::should_use_wide_gemv(&strided_input3, &weight));
        let reference = strided_input
            .contiguous()?
            .matmul(&weight.t()?)?
            .unsqueeze(0)?;
        assert_finite_close(
            &linear(weight.clone(), None)?.forward(&strided_input3)?,
            &reference,
            true,
        )?;
        let transposed_storage = weight.t()?.contiguous()?;
        let strided_weight = transposed_storage.t()?;
        assert!(!strided_weight.is_contiguous());
        check_fallback(&input, &strided_weight)?;
        assert!(!crate::gemv::should_use_wide_gemv(
            &input,
            &weight.unsqueeze(0)?
        ));
        assert!(!crate::gemv::should_use_wide_gemv(
            &input.flatten_all()?,
            &weight
        ));
        assert!(!crate::gemv::should_use_wide_gemv(
            &input.to_dtype(DType::F32)?,
            &weight
        ));
        assert!(!crate::gemv::should_use_wide_gemv(
            &input.narrow(1, 0, BIAS_INPUT_DIM - 2)?,
            &weight
        ));
        assert!(!crate::gemv::should_use_wide_gemv(
            &input.broadcast_as((2, BIAS_INPUT_DIM))?,
            &weight
        ));
        assert!(!crate::gemv::should_use_wide_gemv(
            &input.narrow(0, 0, 0)?,
            &weight
        ));
        let alias = Device::new_cuda(0)?;
        let alias_input = Tensor::ones((1, BIAS_INPUT_DIM), dtype, &alias)?;
        assert!(!alias.same_device(&device));
        assert!(!crate::gemv::should_use_wide_gemv(&alias_input, &weight));
        check_fallback(&alias_input, &weight)?;
    }
    device.synchronize()?;
    Ok(())
}

#[test]
#[ignore = "requires an SM121 CUDA device"]
fn wide_gemv_graph_replay_reads_changed_inputs_and_preserves_bias() -> Result<()> {
    let Some(device) = sm121_device()? else {
        return Ok(());
    };
    for dtype in [DType::BF16, DType::F16] {
        let (source, weight, bias) = dyadic_fixture(&device, dtype)?;
        let alternate = source.neg()?;
        let input = Tensor::zeros((1, BIAS_INPUT_DIM), dtype, &device)?;
        let input4 = input.reshape((1, 1, 1, BIAS_INPUT_DIM))?;
        let layer = linear(padded_view(&weight, EVEN_OFFSET)?, Some(bias.clone()))?;
        assert!(crate::gemv::should_use_wide_gemv(&input4, &layer.w));
        let cuda = device.as_cuda_device()?;
        let _htod_cache_guard = cuda.enable_cuda_graph_htod_cache();
        input.slice_set(&source, 0, 0)?;
        let _ = layer.forward(&input4)?;
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
        let output = layer.forward(&input4);
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let output = output?;
        let graph = graph
            .map_err(candle_core::Error::msg)?
            .ok_or_else(|| candle_core::Error::msg("wide GEMV capture returned no graph"))?;
        for negative in [false, true, false] {
            input.slice_set(if negative { &alternate } else { &source }, 0, 0)?;
            graph.launch().map_err(candle_core::Error::msg)?;
            device.synchronize()?;
            let reference = if negative {
                (&bias * 2.0)?
            } else {
                bias.zeros_like()?
            };
            assert_finite_close(&output, &reference.reshape((1, 1, 1, OUTPUT_DIM))?, true)?;
        }
        drop(graph);
        drop(output);
        input.slice_set(&source, 0, 0)?;
        device.synchronize()?;
    }
    Ok(())
}
