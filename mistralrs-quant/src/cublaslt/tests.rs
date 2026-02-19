use super::*;
use candle_core::{DType, Device, IndexOp};
use float8::F8E4M3;

#[test]
fn test_fused_batch_matmul_f8e4m3_nobias() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Use 128x128 matrices for FP8 tensor core compatibility across GPU architectures
    // Use batch_size=1 to simplify testing on new architectures
    // Ensure all tensors are contiguous
    let a = Tensor::randn(0., 1., (1, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let b = Tensor::randn(0., 1., (1, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;

    fn quantize(data: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let data = data.to_dtype(DType::F32)?.contiguous()?;
        let mut absmax = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        let scale = (max_v / absmax)?.clamp(1e-12, f64::INFINITY)?;
        let qw = crate::scalar_fp8::ops::dtype_to_fp8(&data.broadcast_mul(&scale)?.contiguous()?)?;
        Ok((qw.contiguous()?, scale))
    }
    let (qa, a_scale) = quantize(&a, DType::F8E4M3)?;
    let (qb, b_scale) = quantize(&b, DType::F8E4M3)?;

    let cublaslt = CublasLt::new(&device)?;

    // Test without residual add (out=None, beta=None)
    let res = fused_batch_matmul_f8(
        &qa.contiguous()?,
        &qb.contiguous()?,
        &a_scale.recip()?,
        &b_scale.recip()?,
        &a_scale,
        None, // No residual
        None, // alpha
        None, // beta
        None, // bias
        None, // act
        cublaslt,
    )?
    .i((0..1, 0..2, 0..2))?;
    let expected = b.matmul(&a.t()?)?.i((0..1, 0..2, 0..2))?;

    let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
    let abs_diff = abs_diff.to_vec3::<f32>()?;
    // FP8 quantization has inherent error; use tolerance based on relative error
    // For 128x128 matmul with FP8 inputs, ~5% relative error is acceptable
    let range = 1.0;
    assert!(abs_diff
        .iter()
        .all(|x| x.iter().all(|y| y.iter().all(|x| *x <= range))));
    Ok(())
}

#[test]
fn test_fused_batch_matmul_f8e4m3_out_bf16() -> Result<()> {
    let device = Device::new_cuda(0)?;

    // Use 128x128 matrices for FP8 tensor core compatibility across GPU architectures
    // Ensure all tensors are contiguous
    let a = Tensor::randn(0., 1., (2, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let b = Tensor::randn(0., 1., (2, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let c = Tensor::randn(0., 1., (2, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;

    fn quantize(data: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let data = data.to_dtype(DType::F32)?.contiguous()?;
        let mut absmax = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        let scale = (max_v / absmax)?.clamp(1e-12, f64::INFINITY)?;
        let qw = crate::scalar_fp8::ops::dtype_to_fp8(&data.broadcast_mul(&scale)?.contiguous()?)?;
        Ok((qw.contiguous()?, scale))
    }
    let (qa, a_scale) = quantize(&a, DType::F8E4M3)?;
    let (qb, b_scale) = quantize(&b, DType::F8E4M3)?;

    let cublaslt = CublasLt::new(&device)?;

    let res = fused_batch_matmul_f8(
        &qa.contiguous()?,
        &qb.contiguous()?,
        &a_scale.recip()?,
        &b_scale.recip()?,
        &a_scale,
        Some(&c.to_dtype(DType::BF16)?.contiguous()?),
        None,
        Some(1.),
        None,
        None,
        cublaslt,
    )?
    .i((0..2, 0..2, 0..2))?;
    let expected = b.matmul(&a.t()?)?.add(&c)?.i((0..2, 0..2, 0..2))?;

    let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
    let abs_diff = abs_diff.to_vec3::<f32>()?;

    // FP8 quantization has inherent error; use tolerance based on relative error
    // For 128x128 matmul with FP8 inputs, ~5% relative error is acceptable
    let range = 1.0;
    assert!(abs_diff
        .iter()
        .all(|x| x.iter().all(|y| y.iter().all(|x| *x <= range))));
    Ok(())
}
