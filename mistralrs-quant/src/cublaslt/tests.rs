use std::f32::consts::PI;

use super::*;
use candle_core::{DType, Device, IndexOp};
use float8::F8E4M3;

#[test]
fn test_fused_batch_matmul_f8e4m3_nobias() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let a = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
    let b = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
    let c = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;

    fn quantize(data: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let data = data.to_dtype(DType::F32)?;
        let mut absmax = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        let scale = (max_v / absmax)?.clamp(1e-12, f64::INFINITY)?;
        let qw = data.broadcast_mul(&scale)?.to_dtype(DType::F8E4M3)?;
        Ok((qw, scale))
    }
    let (qa, a_scale) = quantize(&a, DType::F8E4M3)?;
    let (qb, b_scale) = quantize(&b, DType::F8E4M3)?;
    println!("{a_scale}");

    let cublaslt = CublasLt::new(&device)?;

    let res = fused_batch_matmul_f8(
        &qa,
        &qb,
        &a_scale.recip()?,
        &b_scale.recip()?,
        &a_scale,
        Some(&c.to_dtype(DType::BF16)?),
        None,
        Some(1.),
        None,
        None,
        F8MatmulOutType::BF16,
        cublaslt,
    )?
    .i((0..2, 0..2, 0..2))?;
    let expected = b.matmul(&a.t()?)?.add(&c)?.i((0..2, 0..2, 0..2))?;

    let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
    let absmax = abs_diff.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
    let abs_diff = abs_diff.to_vec3::<f32>()?;
    let range = 3e-01;
    assert!(abs_diff
        .iter()
        .all(|x| x.iter().all(|y| y.iter().all(|x| *x <= range))));
    Ok(())
}

#[test]
fn test_fused_batch_matmul_f8e4m3_out_bf16() -> Result<()> {
    let device = Device::new_cuda(0)?;

    let a = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
    let b = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;
    let c = Tensor::randn(0., 1., (16, 16, 16), &device)?.to_dtype(DType::F32)?;

    fn quantize(data: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        let data = data.to_dtype(DType::F32)?;
        let mut absmax = data.clone();
        while !absmax.dims().is_empty() {
            absmax = absmax.max(0)?;
        }
        let max_v = F8E4M3::MAX.to_f64().round();
        let scale = (max_v / absmax)?.clamp(1e-12, f64::INFINITY)?;
        let qw = data.broadcast_mul(&scale)?.to_dtype(DType::F8E4M3)?;
        Ok((qw, scale))
    }
    let (qa, a_scale) = quantize(&a, DType::F8E4M3)?;
    let (qb, b_scale) = quantize(&b, DType::F8E4M3)?;

    let cublaslt = CublasLt::new(&device)?;

    let res = fused_batch_matmul_f8(
        &qa,
        &qb,
        &a_scale.recip()?,
        &b_scale.recip()?,
        &a_scale,
        Some(&c.to_dtype(DType::BF16)?),
        None,
        Some(1.),
        None,
        None,
        F8MatmulOutType::BF16,
        cublaslt,
    )?
    .i((0..2, 0..2, 0..2))?;
    let expected = b.matmul(&a.t()?)?.add(&c)?.i((0..2, 0..2, 0..2))?;

    let abs_diff = (res.to_dtype(DType::F32)? - expected)?.abs()?;
    let absmax = abs_diff.max(0)?.max(0)?.max(0)?.to_scalar::<f32>()?;
    let abs_diff = abs_diff.to_vec3::<f32>()?;

    let range = 3e-01;
    assert!(abs_diff
        .iter()
        .all(|x| x.iter().all(|y| y.iter().all(|x| *x <= range))));
    Ok(())
}
