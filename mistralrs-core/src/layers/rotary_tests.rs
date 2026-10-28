use super::*;
use candle_core::Device;

#[test]
fn partial_rope_rotates_leading_dims_only() -> Result<()> {
    let device = Device::Cpu;
    let x = Tensor::arange(0f32, 8., &device)?.reshape((1, 1, 1, 8))?;
    let cos = Tensor::new(&[[[0f32, 1.0]]], &device)?;
    let sin = Tensor::new(&[[[1f32, 0.0]]], &device)?;
    let out = partial_rotary_neox(&x, &cos, &sin)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    // pairs (0, 2) rotate by 90 degrees, (1, 3) stay, dims 4.. pass through
    assert_eq!(out, [-2.0, 1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    Ok(())
}

#[test]
fn partial_rotary_norm_cpu_fallback_preserves_untouched_columns() -> Result<()> {
    let input = Tensor::arange(0f32, 8., &Device::Cpu)?.reshape((1, 1, 1, 8))?;
    let weight = Tensor::ones(8, DType::F32, &Device::Cpu)?;
    let cos = Tensor::new(&[[[0f32, 1.]]], &Device::Cpu)?;
    let sin = Tensor::new(&[[[1f32, 0.]]], &Device::Cpu)?;
    let output = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, 1e-6)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let inverse = (17.5f64 + f64::from(1e-6f32)).sqrt().recip();
    for (actual, expected) in output.into_iter().zip([-2., 1., 0., 3., 4., 5., 6., 7.]) {
        assert!((actual as f64 - expected * inverse).abs() < 2e-7);
    }
    Ok(())
}

#[test]
fn partial_rotary_norm_accepts_leading_batch_dimensions() -> Result<()> {
    for shape in [vec![3, 8], vec![2, 3, 8], vec![2, 1, 2, 3, 8]] {
        let elements = shape.iter().product::<usize>();
        let input =
            Tensor::arange(0f32, elements as f32, &Device::Cpu)?.reshape(shape.as_slice())?;
        let weight = Tensor::ones(8, DType::F32, &Device::Cpu)?;
        let mut cache_shape = shape[..shape.len() - 2].to_vec();
        cache_shape.push(2);
        let cos = Tensor::ones(cache_shape.as_slice(), DType::F32, &Device::Cpu)?;
        let sin = Tensor::zeros(cache_shape.as_slice(), DType::F32, &Device::Cpu)?;
        let output = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, 1e-6)?;
        let expected = candle_nn::ops::rms_norm(&input, &weight, 1e-6)?;
        assert_eq!(output.dims(), shape);
        assert_eq!(
            output.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
mod cuda {
    use super::*;
    use candle_core::{
        cuda::cudarc::driver::{sys, CudaGraph},
        CudaStorage, Storage, Var,
    };
    const EPSILON: f32 = 1e-6;
    const STREAM_DELAY_BYTES: usize = 32 * 1024 * 1024;
    const STREAM_WIDTH: usize = 128;
    const STREAM_HEADS: usize = 4;
    const STREAM_ROTARY_HALF: usize = 32;
    fn data(n: usize, mut state: u64) -> Vec<f32> {
        (0..n)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 40) as f32 / (1u32 << 24) as f32 * 4. - 2.
            })
            .collect()
    }
    fn tensor(shape: &[usize], salt: u64, dtype: DType, device: &Device) -> Result<Tensor> {
        Tensor::from_vec(data(shape.iter().product(), salt), shape, device)?.to_dtype(dtype)
    }
    fn original(input: &Tensor, weight: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        original_eps(input, weight, cos, sin, EPSILON)
    }
    fn original_eps(
        input: &Tensor,
        weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        epsilon: f32,
    ) -> Result<Tensor> {
        let norm = candle_nn::ops::rms_norm(&input.contiguous()?, weight, epsilon)?;
        let half = cos.dim(D::Minus1)?;
        let width = input.dim(D::Minus1)?;
        let c = cos.to_dtype(DType::F32)?.unsqueeze(D::Minus2)?;
        let s = sin.to_dtype(DType::F32)?.unsqueeze(D::Minus2)?;
        let norm = norm.to_dtype(DType::F32)?;
        let x = norm.narrow(D::Minus1, 0, half)?;
        let y = norm.narrow(D::Minus1, half, half)?;
        let mut parts = vec![
            (x.broadcast_mul(&c)? - y.broadcast_mul(&s)?)?,
            (y.broadcast_mul(&c)? + x.broadcast_mul(&s)?)?,
        ];
        if width > half * 2 {
            parts.push(norm.narrow(D::Minus1, half * 2, width - half * 2)?);
        }
        Tensor::cat(&parts, D::Minus1)?.to_dtype(input.dtype())
    }
    fn values(x: &Tensor) -> Result<Vec<f32>> {
        x.to_dtype(DType::F32)?.flatten_all()?.to_vec1()
    }
    fn check(actual: &Tensor, expected: &Tensor, name: &str) -> Result<()> {
        assert_eq!(actual.dims(), expected.dims());
        assert_eq!(actual.dtype(), expected.dtype());
        let (a, e) = (values(actual)?, values(expected)?);
        let mut bad = 0;

        for (i, (a, e)) in a.iter().zip(&e).enumerate() {
            if a.to_bits() != e.to_bits() {
                bad += 1;
                if bad < 3 {
                    eprintln!("{name} {i}: {a} != {e}");
                }
            }
        }

        assert_eq!(bad, 0, "{name}");
        Ok(())
    }
    fn capture(
        device: &Device,
        call: impl FnOnce() -> Result<Vec<Tensor>>,
    ) -> Result<(CudaGraph, Vec<Tensor>)> {
        device.synchronize()?;
        let stream = device.as_cuda_device()?.cuda_stream();
        let tracking = stream.context().is_event_tracking();
        if tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        let begin = stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED);
        if let Err(e) = begin {
            if tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(candle_core::Error::msg(e));
        }
        let out = call();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        Ok((graph.map_err(candle_core::Error::msg)?.unwrap(), out?))
    }
    fn round(x: f32, dtype: DType) -> f32 {
        match dtype {
            DType::F16 => half::f16::from_f32(x).to_f32(),
            DType::BF16 => half::bf16::from_f32(x).to_f32(),
            _ => x,
        }
    }
    fn oracle(
        input: &Tensor,
        weight: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        output: &Tensor,
    ) -> Result<()> {
        let heads = input.dim(D::Minus2)?;
        let width = input.dim(D::Minus1)?;
        let half = cos.dim(D::Minus1)?;
        let x = values(input)?;
        let w = values(weight)?;
        let c = values(cos)?;
        let s = values(sin)?;
        let out = values(output)?;
        let mut squared = 0f64;
        let mut magnitude = 0f64;
        for (row, xs) in x.chunks_exact(width).enumerate() {
            let inv = (xs.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / width as f64
                + EPSILON as f64)
                .sqrt()
                .recip();
            let n: Vec<_> = xs
                .iter()
                .zip(&w)
                .map(|(&v, &w)| round((v as f64 * inv * w as f64) as f32, input.dtype()))
                .collect();
            for col in 0..width {
                let r = if col < half {
                    let p = row / heads * half + col;
                    n[col] * c[p] - n[col + half] * s[p]
                } else if col < half * 2 {
                    let p = row / heads * half + col - half;
                    n[col] * c[p] + n[col - half] * s[p]
                } else {
                    n[col]
                };
                let r = round(r, input.dtype());
                let error = (out[row * width + col] - r) as f64;
                squared += error * error;
                magnitude += (r as f64).powi(2);
            }
        }
        let rrms = (squared / magnitude.max(1e-30)).sqrt();

        assert!(
            rrms < match input.dtype() {
                DType::BF16 => 0.008,
                DType::F16 => 0.001,
                _ => 2e-6,
            }
        );
        Ok(())
    }

    fn corner_cases(device: &Device) -> Result<()> {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let smallest = match dtype {
                DType::BF16 => half::bf16::from_bits(1).to_f32(),
                DType::F16 => half::f16::from_bits(1).to_f32(),
                _ => f32::from_bits(1),
            };
            for width in [4, 33, 128, 1024] {
                for (pattern, scale, epsilon) in [
                    (0, 0., EPSILON),
                    (1, smallest, EPSILON),
                    (2, 1e-20, f32::MIN_POSITIVE),
                    (3, 0., f32::from_bits(1)),
                    (4, if dtype == DType::F16 { 100. } else { 1e10 }, EPSILON),
                ] {
                    let input = Tensor::from_vec(
                        (0..width)
                            .map(|i| {
                                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                                sign * scale * (1 + i % 7) as f32
                            })
                            .collect::<Vec<_>>(),
                        (1, 1, 1, width),
                        device,
                    )?
                    .to_dtype(dtype)?;
                    let weight = Tensor::from_vec(
                        (0..width)
                            .map(|i| if i % 3 == 0 { -1.0f32 } else { 1.0 })
                            .collect::<Vec<_>>(),
                        width,
                        device,
                    )?
                    .to_dtype(dtype)?;
                    let half = width / 4;
                    let cos = Tensor::from_vec(
                        (0..half)
                            .map(|i| [0.0f32, -0.0, 1.0, -1.0][i % 4])
                            .collect::<Vec<_>>(),
                        (1, 1, half),
                        device,
                    )?
                    .to_dtype(dtype)?;
                    let sin = Tensor::from_vec(
                        (0..half)
                            .map(|i| [1.0f32, -1.0, -0.0, 0.0][i % 4])
                            .collect::<Vec<_>>(),
                        (1, 1, half),
                        device,
                    )?
                    .to_dtype(dtype)?;
                    check(
                        &rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, epsilon)?,
                        &original_eps(&input, &weight, &cos, &sin, epsilon)?,
                        &format!("corner-{dtype:?}-{width}-{pattern}"),
                    )?;
                }
            }
        }
        Ok(())
    }
    fn graph_cases(device: &Device) -> Result<()> {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for (batch, sequence, heads, width, half) in [
                (1, 1, 4, 128, 32),
                (1, 1, 7, 64, 32),
                (2, 3, 3, 65, 17),
                (1, 17, 8, 256, 64),
                (1, 1, 2, 1023, 511),
                (1, 2, 2, 1024, 512),
                (1, 1, 2, 1283, 321),
            ] {
                for packed in [false, true] {
                    let cols = if packed { width + 3 } else { width };
                    let backing = Var::zeros((batch, sequence, heads, cols), dtype, device)?;
                    let input = backing.narrow(3, if packed { 1 } else { 0 }, width)?;
                    let weight = Var::zeros(width, dtype, device)?;
                    let cos = Var::zeros((batch, sequence, half), dtype, device)?;
                    let sin = Var::zeros((batch, sequence, half), dtype, device)?;
                    let _ = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?;
                    let (graph, outputs) = capture(device, || {
                        Ok(vec![rms_norm_partial_rotary_neox(
                            &input, &weight, &cos, &sin, EPSILON,
                        )?])
                    })?;
                    for turn in 0..3u64 {
                        backing.set(&tensor(
                            &[batch, sequence, heads, cols],
                            271 + turn,
                            dtype,
                            device,
                        )?)?;
                        weight.set(&tensor(&[width], 911 + turn, dtype, device)?)?;
                        cos.set(&tensor(
                            &[batch, sequence, half],
                            8181 + turn,
                            dtype,
                            device,
                        )?)?;
                        sin.set(&tensor(
                            &[batch, sequence, half],
                            1313 + turn,
                            dtype,
                            device,
                        )?)?;
                        graph.launch().map_err(candle_core::Error::msg)?;
                        check(&outputs[0], &original(&input,&weight,&cos,&sin)?, &format!("{dtype:?}-{batch}-{sequence}-{heads}-{width}-{half}-{packed}-{turn}"))?;
                        oracle(&input, &weight, &cos, &sin, &outputs[0])?;
                    }
                }
            }
        }
        Ok(())
    }

    fn rank_and_stride_cases(device: &Device) -> Result<()> {
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            for (shape, half) in [
                (vec![5, 64], 32),
                (vec![7, 3, 96], 24),
                (vec![2, 5, 3, 129], 33),
            ] {
                let rank = shape.len();
                let width = shape[rank - 1];
                let mut cache_shape = shape[..rank - 2].to_vec();
                cache_shape.push(half);
                let cache_elements = cache_shape.iter().product::<usize>();
                for transposed in [false, true] {
                    let mut backing_shape = shape.clone();
                    if transposed {
                        backing_shape.swap(rank - 2, rank - 1);
                    }
                    let backing = Var::zeros(backing_shape.as_slice(), dtype, device)?;
                    let input = if transposed {
                        backing.transpose(rank - 2, rank - 1)?
                    } else {
                        backing.as_tensor().clone()
                    };
                    let weights = Var::zeros(width + 1, dtype, device)?;
                    let weight = weights.narrow(0, 1, width)?;
                    let cosine = Var::zeros(cache_elements + 1, dtype, device)?;
                    let sine = Var::zeros(cache_elements + 1, dtype, device)?;
                    let cos = cosine
                        .narrow(0, 1, cache_elements)?
                        .reshape(cache_shape.as_slice())?;
                    let sin = sine
                        .narrow(0, 1, cache_elements)?
                        .reshape(cache_shape.as_slice())?;
                    let _ = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?;
                    let (graph, outputs) = capture(device, || {
                        Ok(vec![rms_norm_partial_rotary_neox(
                            &input, &weight, &cos, &sin, EPSILON,
                        )?])
                    })?;
                    for turn in 0..3 {
                        backing.set(&tensor(&backing_shape, 319 + turn, dtype, device)?)?;
                        weights.set(&tensor(&[width + 1], 773 + turn, dtype, device)?)?;
                        cosine.set(&tensor(&[cache_elements + 1], 17 + turn, dtype, device)?)?;
                        sine.set(&tensor(&[cache_elements + 1], 29 + turn, dtype, device)?)?;
                        graph.launch().map_err(candle_core::Error::msg)?;
                        check(
                            &outputs[0],
                            &original(&input, &weight, &cos, &sin)?,
                            &format!("rank-{rank}-{dtype:?}-{transposed}-{turn}"),
                        )?;
                        oracle(&input, &weight, &cos, &sin, &outputs[0])?;
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn rounded_norm_rope_preserves_rounding_and_changed_graph_views() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        let _cache = device.as_cuda_device()?.enable_cuda_graph_htod_cache();
        corner_cases(&device)?;
        rank_and_stride_cases(&device)?;
        graph_cases(&device)
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn rounded_norm_rope_keeps_fallbacks_and_secondary_stream_inputs() -> Result<()> {
        let device = Device::new_cuda_with_stream(0)?;
        for dtype in [DType::BF16, DType::F16] {
            let input = tensor(&[1, 3, 2, 65], 73, dtype, &device)?;
            let weight = tensor(&[65], 777, dtype, &device)?;
            let cos = tensor(&[1, 3, 17], 333, DType::F32, &device)?;
            let sin = tensor(&[1, 3, 17], 555, DType::F32, &device)?;
            assert!(crate::cuda::rounded_rope::try_rms_norm_partial_rotary_neox(
                &input, &weight, &cos, &sin, EPSILON,
            )?
            .is_none());
            check(
                &rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?,
                &original(&input, &weight, &cos, &sin)?,
                "mixed cache dtype fallback",
            )?;
        }
        for (input_shape, cache_shape, transpose_cache) in [
            (vec![2, 1, 3, 2, 64], vec![2, 1, 3, 16], false),
            (vec![2, 3, 2, 64], vec![1, 3, 16], false),
            (vec![1, 3, 2, 64], vec![1, 16, 3], true),
        ] {
            let input = tensor(&input_shape, 37, DType::F32, &device)?;
            let weight = tensor(&[64], 83, DType::F32, &device)?;
            let mut cos = tensor(&cache_shape, 337, DType::F32, &device)?;
            let mut sin = tensor(&cache_shape, 557, DType::F32, &device)?;
            if transpose_cache {
                cos = cos.transpose(1, 2)?;
                sin = sin.transpose(1, 2)?;
            }
            assert!(crate::cuda::rounded_rope::try_rms_norm_partial_rotary_neox(
                &input, &weight, &cos, &sin, EPSILON,
            )?
            .is_none());
            check(
                &rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?,
                &original(&input, &weight, &cos, &sin)?,
                "layout fallback",
            )?;
        }
        let cuda = device.as_cuda_device()?;
        let producer = cuda
            .cuda_stream()
            .context()
            .new_stream()
            .map_err(candle_core::Error::wrap)?;
        for turn in [3, 17, 43] {
            let delay = producer
                .alloc_zeros::<u8>(STREAM_DELAY_BYTES)
                .map_err(candle_core::Error::wrap)?;
            let upload = |shape: &[usize], salt| -> Result<Tensor> {
                let storage = producer
                    .clone_htod(&data(shape.iter().product(), salt))
                    .map_err(candle_core::Error::wrap)?;
                Ok(Tensor::from((
                    Storage::Cuda(CudaStorage::wrap_cuda_slice(storage, cuda.clone())),
                    candle_core::Shape::from_dims(shape),
                )))
            };
            let input = upload(&[1, 1, STREAM_HEADS, STREAM_WIDTH], turn)?;
            let weight = upload(&[STREAM_WIDTH], turn + 999)?;
            let cos = upload(&[1, 1, STREAM_ROTARY_HALF], turn + 333)?;
            let sin = upload(&[1, 1, STREAM_ROTARY_HALF], turn + 555)?;
            let actual = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?;
            check(
                &actual,
                &original(&input, &weight, &cos, &sin)?,
                "secondary stream inputs",
            )?;
            std::hint::black_box(delay);
        }
        Ok(())
    }
}
