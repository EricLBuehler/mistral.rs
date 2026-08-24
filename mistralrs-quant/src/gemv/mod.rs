//! Custom GEMV (General Matrix-Vector multiplication) for decode-phase inference.
//!
//! This module provides an optimized GEMV kernel that replaces cuBLAS for
//! small output workloads where cuBLAS GEMM overhead is significant.
//!
//! Key optimizations:
//! - Vectorized loads (half2, nv_bfloat162, float2)
//! - __ldg() for read-only cache path (L2 cache handles x reuse)
//! - Warp-level reduction using XOR shuffle
//! - Static shared memory for block-level reduction
//! - Supports batch sizes 1-8 efficiently

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
use candle_core::{
    cuda::cudarc::driver::DevicePtrMut, CudaDevice, CudaStorage, DType, Result, Shape, Storage,
    Tensor,
};

#[cfg(feature = "cuda")]
use crate::utils::{get_cuda_device, slice_ptr};

#[cfg(feature = "cuda")]
use half::{bf16, f16};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;
#[cfg(feature = "cuda")]
use std::{collections::HashMap, sync::Mutex};

/// Maximum batch size supported by the GEMV kernel
pub const MAX_GEMV_BATCH_SIZE: usize = 8;
#[cfg(any(feature = "cuda", test))]
const MAX_GEMV_OUTPUT_ELEMENTS: usize = 4_096;
#[cfg(any(feature = "cuda", test))]
const SM90_SPLIT_K_MIN_BATCH_REDUCTION: usize = 32_768;
#[cfg(any(feature = "cuda", test))]
const SM90_SPLIT_K_MAX_GEMV_CTA_WAVES: usize = 4;

#[cfg(any(feature = "cuda", test))]
#[derive(Clone, Copy)]
struct GemvDeviceInfo {
    compute_major: i32,
    multiprocessor_count: usize,
}

#[cfg(any(feature = "cuda", test))]
fn should_use_gemv_shape(
    batch_size: usize,
    output_dim: usize,
    input_dim: usize,
    is_bf16: bool,
    device: Option<GemvDeviceInfo>,
) -> bool {
    if batch_size > MAX_GEMV_BATCH_SIZE
        || batch_size.saturating_mul(output_dim) > MAX_GEMV_OUTPUT_ELEMENTS
    {
        return false;
    }
    let Some(device) = device else {
        return true;
    };
    let gemv_cta_waves = output_dim.div_ceil(device.multiprocessor_count);
    let sm90_split_k = device.compute_major == 9
        && is_bf16
        && batch_size == MAX_GEMV_BATCH_SIZE
        && batch_size.saturating_mul(input_dim) >= SM90_SPLIT_K_MIN_BATCH_REDUCTION
        && gemv_cta_waves <= SM90_SPLIT_K_MAX_GEMV_CTA_WAVES;
    !sm90_split_k
}

#[cfg(feature = "cuda")]
static GEMV_DEVICE_INFO: LazyLock<Mutex<HashMap<candle_core::cuda::DeviceId, GemvDeviceInfo>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

#[cfg(feature = "cuda")]
fn gemv_device_info(device: &CudaDevice) -> Option<GemvDeviceInfo> {
    use candle_core::cuda::cudarc::driver::sys::CUdevice_attribute;

    if let Some(info) = GEMV_DEVICE_INFO.lock().unwrap().get(&device.id()).copied() {
        return Some(info);
    }
    let stream = device.cuda_stream();
    let context = stream.context();
    let compute_major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .ok()?;
    let multiprocessor_count = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
        .ok()
        .and_then(|count| usize::try_from(count).ok())
        .filter(|&count| count != 0)?;
    let info = GemvDeviceInfo {
        compute_major,
        multiprocessor_count,
    };
    GEMV_DEVICE_INFO.lock().unwrap().insert(device.id(), info);
    Some(info)
}

/// Controller for enabling/disabling custom GEMV kernel.
pub struct GemvController {
    enabled: AtomicBool,
}

impl GemvController {
    /// Enable or disable the custom GEMV kernel.
    pub fn set_enabled(&self, value: bool) {
        self.enabled.store(value, Ordering::SeqCst);
    }

    /// Check if the custom GEMV kernel is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::SeqCst)
    }
}

/// Global controller for the custom GEMV kernel.
pub static GEMV_CONTROLLER: LazyLock<GemvController> = LazyLock::new(|| GemvController {
    enabled: AtomicBool::new(true),
});

/// Check if custom GEMV should be used instead of cuBLAS.
///
/// Returns true if:
/// - GEMV is enabled via controller
/// - Tensors are on CUDA device
/// - Batch size is 1-8
/// - The shape and device favor GEMV over a split-K GEMM
/// - Data type is supported (BF16, F16, F32)
/// - K dimension is even (required for vectorized loads)
#[cfg(feature = "cuda")]
pub fn should_use_gemv(x: &Tensor, w: &Tensor) -> bool {
    // Check if enabled
    if !GEMV_CONTROLLER.is_enabled() {
        return false;
    }

    let candle_core::Device::Cuda(device) = x.device() else {
        return false;
    };

    // Check batch size (1-8 supported)
    let x_dims = x.dims();
    let batch_size: usize = x_dims[..x_dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    // Must be supported dtype
    let supported = matches!(x.dtype(), DType::BF16 | DType::F16 | DType::F32);
    if !supported {
        return false;
    }

    // Must match dtypes
    if x.dtype() != w.dtype() {
        return false;
    }

    // K must be even for vectorized loads
    let k = x.dim(x.rank() - 1).unwrap_or(0);
    if !k.is_multiple_of(2) {
        return false;
    }

    let output_dim = w.dims().first().copied().unwrap_or(usize::MAX);
    if !should_use_gemv_shape(
        batch_size,
        output_dim,
        k,
        x.dtype() == DType::BF16,
        gemv_device_info(device),
    ) {
        return false;
    }

    // Check that K dimensions match
    let w_k = w.dim(w.rank() - 1).unwrap_or(0);
    if k != w_k {
        return false;
    }

    true
}

/// Fallback for non-CUDA builds
#[cfg(not(feature = "cuda"))]
pub fn should_use_gemv(_x: &candle_core::Tensor, _w: &candle_core::Tensor) -> bool {
    false
}

/// Execute custom GEMV: Y = X @ W^T + bias
///
/// # Arguments
/// * `x` - Input tensor [B, K] where B is batch size (1-8)
/// * `w` - Weight matrix tensor [M, K]
/// * `bias` - Optional bias tensor [M]
///
/// # Returns
/// * Output tensor [B, M]
#[cfg(feature = "cuda")]
pub fn gemv(x: &Tensor, w: &Tensor, bias: Option<&Tensor>) -> Result<Tensor> {
    let dev = get_cuda_device(x)?;

    // Get dimensions
    let (m, k) = w.dims2()?;

    // Calculate batch size from input shape
    let x_dims = x.dims();
    let batch_size: usize = x_dims[..x_dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);

    if batch_size > MAX_GEMV_BATCH_SIZE {
        candle_core::bail!(
            "GEMV batch size {} exceeds maximum {}",
            batch_size,
            MAX_GEMV_BATCH_SIZE
        );
    }

    // Check K dimension
    let x_k = x.dim(x.rank() - 1)?;
    if x_k != k {
        candle_core::bail!("GEMV dimension mismatch: x has K={} but W has K={}", x_k, k);
    }

    // Validate bias if present
    if let Some(b) = bias {
        let b_len = b.elem_count();
        if b_len != m {
            candle_core::bail!(
                "GEMV bias dimension mismatch: bias has {} elements but M={}",
                b_len,
                m
            );
        }
    }

    // Output shape matches input batch dims with last dim = M
    let output_shape = {
        let mut shape = x.dims().to_vec();
        *shape.last_mut().unwrap() = m;
        shape
    };

    // Dispatch based on dtype
    match x.dtype() {
        DType::BF16 => gemv_bf16(dev, x, w, bias, batch_size, m, k, &output_shape),
        DType::F16 => gemv_f16(dev, x, w, bias, batch_size, m, k, &output_shape),
        DType::F32 => gemv_f32(dev, x, w, bias, batch_size, m, k, &output_shape),
        dt => candle_core::bail!("GEMV unsupported dtype: {:?}", dt),
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemv_bf16(
    dev: &CudaDevice,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    batch_size: usize,
    m: usize,
    k: usize,
    output_shape: &[usize],
) -> Result<Tensor> {
    // Allocate output: [B, M]
    let mut y_buf = unsafe { dev.alloc::<bf16>(batch_size * m)? };

    // Get weight pointer
    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<bf16>()?, w_l.start_offset());

    let x_contig;
    let (x_s, x_l) = if batch_size == 1 && x.layout().stride()[x.rank() - 1] == 1 {
        x.storage_and_layout()
    } else {
        x_contig = x.contiguous()?;
        x_contig.storage_and_layout()
    };
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<bf16>()?, x_l.start_offset());

    let stream = dev.cuda_stream();
    let (y_ptr, y_guard) = y_buf.device_ptr_mut(&stream);

    // Get bias storage
    let bias_storage = bias.map(|b| b.storage_and_layout());
    let (bias_ptr, has_bias, _bias_guard) = if let Some((ref b_arc, b_l)) = bias_storage {
        let Storage::Cuda(b_s) = &**b_arc else {
            candle_core::bail!("Expected CUDA storage for bias");
        };
        let (b_ptr, b_guard) = slice_ptr(b_s.as_cuda_slice::<bf16>()?, b_l.start_offset());
        (b_ptr, true, Some(b_guard))
    } else {
        (0u64, false, None)
    };

    unsafe {
        ffi::launch_gemv_bf16(
            w_ptr as *const bf16,
            x_ptr as *const bf16,
            bias_ptr as *const bf16,
            y_ptr as *mut bf16,
            m as i32,
            k as i32,
            batch_size as i32,
            has_bias,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    drop(y_guard);

    let y_storage = CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    let y = Tensor::from((Storage::Cuda(y_storage), Shape::from(output_shape)));

    Ok(y)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemv_f16(
    dev: &CudaDevice,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    batch_size: usize,
    m: usize,
    k: usize,
    output_shape: &[usize],
) -> Result<Tensor> {
    let mut y_buf = unsafe { dev.alloc::<f16>(batch_size * m)? };

    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<f16>()?, w_l.start_offset());

    let x_contig;
    let (x_s, x_l) = if batch_size == 1 && x.layout().stride()[x.rank() - 1] == 1 {
        x.storage_and_layout()
    } else {
        x_contig = x.contiguous()?;
        x_contig.storage_and_layout()
    };
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<f16>()?, x_l.start_offset());

    let stream = dev.cuda_stream();
    let (y_ptr, y_guard) = y_buf.device_ptr_mut(&stream);

    let bias_storage = bias.map(|b| b.storage_and_layout());
    let (bias_ptr, has_bias, _bias_guard) = if let Some((ref b_arc, b_l)) = bias_storage {
        let Storage::Cuda(b_s) = &**b_arc else {
            candle_core::bail!("Expected CUDA storage for bias");
        };
        let (b_ptr, b_guard) = slice_ptr(b_s.as_cuda_slice::<f16>()?, b_l.start_offset());
        (b_ptr, true, Some(b_guard))
    } else {
        (0u64, false, None)
    };

    unsafe {
        ffi::launch_gemv_f16(
            w_ptr as *const f16,
            x_ptr as *const f16,
            bias_ptr as *const f16,
            y_ptr as *mut f16,
            m as i32,
            k as i32,
            batch_size as i32,
            has_bias,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    drop(y_guard);

    let y_storage = CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    let y = Tensor::from((Storage::Cuda(y_storage), Shape::from(output_shape)));

    Ok(y)
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn gemv_f32(
    dev: &CudaDevice,
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    batch_size: usize,
    m: usize,
    k: usize,
    output_shape: &[usize],
) -> Result<Tensor> {
    let mut y_buf = unsafe { dev.alloc::<f32>(batch_size * m)? };

    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<f32>()?, w_l.start_offset());

    let x_contig;
    let (x_s, x_l) = if batch_size == 1 && x.layout().stride()[x.rank() - 1] == 1 {
        x.storage_and_layout()
    } else {
        x_contig = x.contiguous()?;
        x_contig.storage_and_layout()
    };
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<f32>()?, x_l.start_offset());

    let stream = dev.cuda_stream();
    let (y_ptr, y_guard) = y_buf.device_ptr_mut(&stream);

    let bias_storage = bias.map(|b| b.storage_and_layout());
    let (bias_ptr, has_bias, _bias_guard) = if let Some((ref b_arc, b_l)) = bias_storage {
        let Storage::Cuda(b_s) = &**b_arc else {
            candle_core::bail!("Expected CUDA storage for bias");
        };
        let (b_ptr, b_guard) = slice_ptr(b_s.as_cuda_slice::<f32>()?, b_l.start_offset());
        (b_ptr, true, Some(b_guard))
    } else {
        (0u64, false, None)
    };

    unsafe {
        ffi::launch_gemv_f32(
            w_ptr as *const f32,
            x_ptr as *const f32,
            bias_ptr as *const f32,
            y_ptr as *mut f32,
            m as i32,
            k as i32,
            batch_size as i32,
            has_bias,
            stream.cu_stream() as *mut std::ffi::c_void,
        );
    }

    drop(y_guard);

    let y_storage = CudaStorage::wrap_cuda_slice(y_buf, dev.clone());
    let y = Tensor::from((Storage::Cuda(y_storage), Shape::from(output_shape)));

    Ok(y)
}

/// Fallback for non-CUDA builds
#[cfg(not(feature = "cuda"))]
pub fn gemv(
    _x: &candle_core::Tensor,
    _w: &candle_core::Tensor,
    _bias: Option<&candle_core::Tensor>,
) -> candle_core::Result<candle_core::Tensor> {
    candle_core::bail!("GEMV requires CUDA feature");
}

#[cfg(test)]
mod policy_tests {
    use super::*;

    #[test]
    fn gemv_shape_policy_tracks_output_work() {
        for batch_size in [1, 2, 4, 8] {
            let boundary = MAX_GEMV_OUTPUT_ELEMENTS / batch_size;
            assert!(should_use_gemv_shape(
                batch_size, boundary, 1024, true, None
            ));
            assert!(!should_use_gemv_shape(
                batch_size,
                boundary + 1,
                1024,
                true,
                None
            ));
        }
    }

    #[test]
    fn gemv_shape_policy_keeps_tiny_projections() {
        for batch_size in [1, 2, 4, 8] {
            assert!(should_use_gemv_shape(batch_size, 96, 1024, true, None));
        }
    }

    #[test]
    fn sm90_long_reduction_uses_split_k_gemm_for_small_cta_grids() {
        let sm90 = Some(GemvDeviceInfo {
            compute_major: 9,
            multiprocessor_count: 132,
        });
        assert!(!should_use_gemv_shape(8, 96, 5120, true, sm90));
        assert!(!should_use_gemv_shape(8, 512, 5120, true, sm90));
        assert!(should_use_gemv_shape(4, 96, 5120, true, sm90));
        assert!(should_use_gemv_shape(8, 96, 2048, true, sm90));
        assert!(should_use_gemv_shape(8, 96, 5120, false, sm90));
        assert!(should_use_gemv_shape(
            8,
            96,
            5120,
            true,
            Some(GemvDeviceInfo {
                compute_major: 8,
                multiprocessor_count: 108,
            })
        ));
        assert!(should_use_gemv_shape(
            8,
            96,
            5120,
            true,
            Some(GemvDeviceInfo {
                compute_major: 9,
                multiprocessor_count: 16,
            })
        ));
    }

    #[test]
    fn gemv_shape_policy_rejects_large_and_unsupported_batches() {
        assert!(!should_use_gemv_shape(1, 5_120, 1024, true, None));
        assert!(!should_use_gemv_shape(8, 248_320, 1024, true, None));
        assert!(!should_use_gemv_shape(16, 96, 1024, true, None));
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use candle_core::{cuda::cudarc::driver::sys, Device};
    use std::hint::black_box;

    const BENCH_BATCH_SIZES: &[usize] = &[1, 2, 4, 8, 16];
    const BENCH_OUTPUT_DIMS: &[usize] = &[
        96, 128, 256, 512, 1_024, 2_048, 4_096, 5_120, 16_384, 248_320,
    ];
    const BENCH_INPUT_DIM: usize = 5_120;
    const BENCH_WARMUP: usize = 5;
    const BENCH_ITERATIONS: usize = 20;
    const GDN_GATE_BENCH_LAYERS: usize = 128;
    const GDN_GATE_OUTPUT_DIM: usize = 96;
    const CORRECTNESS_TOLERANCE: f32 = 32.0;

    fn cuda_error(context: &str, error: impl std::fmt::Display) -> candle_core::Error {
        candle_core::Error::Msg(format!("{context}: {error}"))
    }

    fn benchmark_iterations() -> usize {
        std::env::var("MISTRALRS_GEMV_BENCH_ITERATIONS")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(BENCH_ITERATIONS)
    }

    fn benchmark_batch_size() -> usize {
        std::env::var("MISTRALRS_GEMV_BENCH_BATCH")
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(MAX_GEMV_BATCH_SIZE)
    }

    fn measure_gpu_us(
        device: &Device,
        iterations: usize,
        mut launch: impl FnMut() -> Result<Tensor>,
    ) -> Result<f64> {
        for _ in 0..BENCH_WARMUP {
            black_box(launch()?);
        }
        device.synchronize()?;

        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| cuda_error("CUDA start event failed", error))?;
        let mut outputs = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            outputs.push(launch()?);
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| cuda_error("CUDA end event failed", error))?;
        end.synchronize()
            .map_err(|error| cuda_error("CUDA event synchronization failed", error))?;
        black_box(&outputs);

        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|error| cuda_error("CUDA event timing failed", error))?;
        Ok(f64::from(elapsed_ms) * 1_000.0 / iterations as f64)
    }

    fn measure_cuda_graph_us<T>(
        device: &Device,
        iterations: usize,
        capture: impl FnOnce() -> Result<T>,
    ) -> Result<f64> {
        let cuda = device.as_cuda_device()?;
        let stream = cuda.cuda_stream();
        let restore_event_tracking = stream.context().is_event_tracking();
        if restore_event_tracking {
            unsafe { stream.context().disable_event_tracking() };
        }
        if let Err(error) =
            stream.begin_capture(sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        {
            if restore_event_tracking {
                unsafe { stream.context().enable_event_tracking() };
            }
            return Err(cuda_error("CUDA graph capture failed", error));
        }
        let captured = capture();
        let graph = stream.end_capture(
            sys::CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
        );
        if restore_event_tracking {
            unsafe { stream.context().enable_event_tracking() };
        }
        let captured = captured?;
        let graph = graph
            .map_err(|error| cuda_error("CUDA graph instantiation failed", error))?
            .ok_or_else(|| candle_core::Error::msg("CUDA graph capture returned no graph"))?;

        for _ in 0..BENCH_WARMUP {
            graph
                .launch()
                .map_err(|error| cuda_error("CUDA graph launch failed", error))?;
        }
        stream
            .synchronize()
            .map_err(|error| cuda_error("CUDA graph warmup failed", error))?;

        let start = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| cuda_error("CUDA start event failed", error))?;
        for _ in 0..iterations {
            graph
                .launch()
                .map_err(|error| cuda_error("CUDA graph launch failed", error))?;
        }
        let end = stream
            .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| cuda_error("CUDA end event failed", error))?;
        end.synchronize()
            .map_err(|error| cuda_error("CUDA event synchronization failed", error))?;
        black_box((&captured, &graph));

        let elapsed_ms = start
            .elapsed_ms(&end)
            .map_err(|error| cuda_error("CUDA event timing failed", error))?;
        Ok(f64::from(elapsed_ms) * 1_000.0 / iterations as f64)
    }

    #[test]
    #[ignore = "reports the CUDA GEMV/GEMM crossover on the installed GPU"]
    fn benchmark_bf16_gemv_gemm_crossover() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let iterations = benchmark_iterations();
        println!("batch,output_dim,input_dim,gemv_us,gemm_us,fastest,selected");

        for &output_dim in BENCH_OUTPUT_DIMS {
            let weight = Tensor::ones((output_dim, BENCH_INPUT_DIM), DType::BF16, &device)?;
            let weight_t = weight.t()?;
            for &batch_size in BENCH_BATCH_SIZES {
                let input = Tensor::ones((batch_size, BENCH_INPUT_DIM), DType::BF16, &device)?;
                let gemm_output = input.matmul(&weight_t)?;
                let gemv_us = if batch_size <= MAX_GEMV_BATCH_SIZE {
                    let gemv_output = gemv(&input, &weight, None)?;
                    let max_diff = (gemv_output.to_dtype(DType::F32)?
                        - gemm_output.to_dtype(DType::F32)?)?
                    .abs()?
                    .max_all()?
                    .to_scalar::<f32>()?;
                    assert!(
                        max_diff <= CORRECTNESS_TOLERANCE,
                        "batch={batch_size} output_dim={output_dim} max_diff={max_diff}"
                    );
                    Some(measure_gpu_us(&device, iterations, || {
                        gemv(&input, &weight, None)
                    })?)
                } else {
                    None
                };
                let gemm_us = measure_gpu_us(&device, iterations, || input.matmul(&weight_t))?;
                let (gemv, fastest) = match gemv_us {
                    Some(gemv_us) => (
                        format!("{gemv_us:.3}"),
                        if gemv_us < gemm_us { "gemv" } else { "gemm" },
                    ),
                    None => ("unsupported".to_string(), "gemm"),
                };
                let selected = if should_use_gemv_shape(
                    batch_size,
                    output_dim,
                    BENCH_INPUT_DIM,
                    true,
                    gemv_device_info(device.as_cuda_device()?),
                ) {
                    "gemv"
                } else {
                    "gemm"
                };
                println!(
                    "{batch_size},{output_dim},{BENCH_INPUT_DIM},{gemv},{gemm_us:.3},{fastest},{selected}"
                );
            }
        }
        Ok(())
    }

    #[test]
    #[ignore = "reports the CUDA graph crossover for cold GDN gate projection weights"]
    fn benchmark_bf16_gdn_gate_projection_cuda_graph_crossover() -> Result<()> {
        let device = Device::new_cuda(0)?;
        let iterations = benchmark_iterations();
        let weights = (0..GDN_GATE_BENCH_LAYERS)
            .map(|_| Tensor::ones((GDN_GATE_OUTPUT_DIM, BENCH_INPUT_DIM), DType::BF16, &device))
            .collect::<Result<Vec<_>>>()?;
        device.synchronize()?;

        println!("batch,gemv_us_per_projection,gemm_us_per_projection,fastest");
        let batch_size = benchmark_batch_size();
        let input = Tensor::ones((batch_size, 1, BENCH_INPUT_DIM), DType::BF16, &device)?;
        let gemv_us = measure_cuda_graph_us(&device, iterations, || {
            weights
                .iter()
                .map(|weight| gemv(&input, weight, None))
                .collect::<Result<Vec<_>>>()
        })? / GDN_GATE_BENCH_LAYERS as f64;
        let gemm_us = measure_cuda_graph_us(&device, iterations, || {
            let input = input.reshape((batch_size, BENCH_INPUT_DIM))?;
            weights
                .iter()
                .map(|weight| input.matmul(&weight.t()?))
                .collect::<Result<Vec<_>>>()
        })? / GDN_GATE_BENCH_LAYERS as f64;
        let fastest = if gemv_us < gemm_us { "gemv" } else { "gemm" };
        println!("{batch_size},{gemv_us:.3},{gemm_us:.3},{fastest}");
        Ok(())
    }
}
