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
#[test]
#[ignore = "requires a CUDA device"]
fn rounded_norm_rope_preserves_rounding_and_offset_views() -> Result<()> {
    const EPSILON: f32 = 1e-6;
    let device = Device::new_cuda_with_stream(0)?;
    for dtype in [DType::BF16, DType::F16, DType::F32] {
        for (batch, sequence, heads, width, half) in [(1, 1, 4, 128, 32), (2, 3, 3, 65, 17)] {
            let count = batch * sequence * heads * (width + 1);
            let input = Tensor::from_vec(
                (0..count)
                    .map(|i| (i as f32 * 0.17).sin())
                    .collect::<Vec<_>>(),
                (batch, sequence, heads, width + 1),
                &device,
            )?
            .to_dtype(dtype)?
            .narrow(3, 1, width)?;
            let weight = Tensor::ones(width + 1, dtype, &device)?.narrow(0, 1, width)?;
            let cos = Tensor::full(0.6f32, (batch, sequence, half), &device)?.to_dtype(dtype)?;
            let sin = Tensor::full(0.8f32, (batch, sequence, half), &device)?.to_dtype(dtype)?;
            let expected = partial_rotary_neox(
                &candle_nn::ops::rms_norm(&input.contiguous()?, &weight, EPSILON)?,
                &cos,
                &sin,
            )?;
            let actual = rms_norm_partial_rotary_neox(&input, &weight, &cos, &sin, EPSILON)?;
            assert_eq!(actual.dims(), input.dims());
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
                expected
                    .to_dtype(DType::F32)?
                    .flatten_all()?
                    .to_vec1::<f32>()?,
            );
        }
    }
    Ok(())
}
