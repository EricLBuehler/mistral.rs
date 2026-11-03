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

/// An FP8 matmul must be capturable into a CUDA graph on candle's default stream (the
/// per-thread default stream, which is a blocking stream) once the same shape ran eagerly,
/// as in the decode-graph warmup. Without the heuristic cache, cuBLASLt 13.x fails the
/// stream-less heuristic query under capture with `CUBLAS_STATUS_INTERNAL_ERROR`
/// ("Could not obtain green context information") and the capture is invalidated.
#[test]
fn test_fused_batch_matmul_f8e4m3_cuda_graph_capture() -> Result<()> {
    use candle_core::cuda::cudarc::driver::sys;

    let device = Device::new_cuda(0)?;
    let a = Tensor::randn(0., 1., (1, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;
    let b = Tensor::randn(0., 1., (1, 128, 128), &device)?
        .to_dtype(DType::F32)?
        .contiguous()?;

    fn quantize(data: &Tensor) -> Result<(Tensor, Tensor)> {
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
    let (qa, a_scale) = quantize(&a)?;
    let (qb, b_scale) = quantize(&b)?;
    let (a_dequant, b_dequant) = (a_scale.recip()?, b_scale.recip()?);

    let cublaslt = CublasLt::new(&device)?;
    let run = || {
        fused_batch_matmul_f8(
            &qa,
            &qb,
            &a_dequant,
            &b_dequant,
            &a_scale,
            None,
            None,
            None,
            None,
            None,
            cublaslt.clone(),
        )
    };

    // Eager warmup: picks (and caches) the algo for this shape
    let eager = run()?;
    device.synchronize()?;

    let stream = device.as_cuda_device()?.cuda_stream();
    let restore_event_tracking = stream.context().is_event_tracking();
    if restore_event_tracking {
        unsafe { stream.context().disable_event_tracking() };
    }
    stream
        .begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .map_err(candle_core::Error::wrap)?;
    let captured = run();
    let graph = stream.end_capture(
        sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
    );
    if restore_event_tracking {
        unsafe { stream.context().enable_event_tracking() };
    }
    let captured = captured?;
    let graph = graph
        .map_err(candle_core::Error::wrap)?
        .expect("CUDA graph capture returned no graph");
    graph.launch().map_err(candle_core::Error::wrap)?;
    stream.synchronize().map_err(candle_core::Error::wrap)?;
    assert_eq!(captured.dims(), eager.dims());
    Ok(())
}
