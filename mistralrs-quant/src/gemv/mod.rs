//! Custom GEMV (General Matrix-Vector multiplication) for decode-phase inference.
//!
//! This module provides an optimized GEMV kernel that replaces cuBLAS for
//! small batch sizes (1-8) where cuBLAS GEMM overhead is significant.
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
    cuda::cudarc::driver::DevicePtr, CudaDevice, CudaStorage, DType, Result, Shape, Storage, Tensor,
};

#[cfg(feature = "cuda")]
use crate::utils::{get_cuda_device, slice_ptr};

#[cfg(feature = "cuda")]
use half::{bf16, f16};

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::LazyLock;

/// Maximum batch size supported by the GEMV kernel
pub const MAX_GEMV_BATCH_SIZE: usize = 8;

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
/// - Data type is supported (BF16, F16, F32)
/// - K dimension is even (required for vectorized loads)
#[cfg(feature = "cuda")]
pub fn should_use_gemv(x: &Tensor, w: &Tensor) -> bool {
    // Check if enabled
    if !GEMV_CONTROLLER.is_enabled() {
        return false;
    }

    // Only for CUDA tensors
    if !x.device().is_cuda() {
        return false;
    }

    // Check batch size (1-8 supported)
    let x_dims = x.dims();
    let batch_size: usize = x_dims[..x_dims.len().saturating_sub(1)]
        .iter()
        .product::<usize>()
        .max(1);
    if batch_size > MAX_GEMV_BATCH_SIZE {
        return false;
    }

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
    if k % 2 != 0 {
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
    let y_buf = unsafe { dev.alloc::<bf16>(batch_size * m)? };

    // Get weight pointer
    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<bf16>()?, w_l.start_offset());

    // Get input pointer (contiguous)
    let x_contig = x.contiguous()?;
    let (x_s, x_l) = x_contig.storage_and_layout();
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<bf16>()?, x_l.start_offset());

    let (y_ptr, y_guard) = y_buf.device_ptr(y_buf.stream());

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

    let stream = dev.cuda_stream();

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
    let y_buf = unsafe { dev.alloc::<f16>(batch_size * m)? };

    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<f16>()?, w_l.start_offset());

    let x_contig = x.contiguous()?;
    let (x_s, x_l) = x_contig.storage_and_layout();
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<f16>()?, x_l.start_offset());

    let (y_ptr, y_guard) = y_buf.device_ptr(y_buf.stream());

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

    let stream = dev.cuda_stream();

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
    let y_buf = unsafe { dev.alloc::<f32>(batch_size * m)? };

    let (w_s, w_l) = w.storage_and_layout();
    let Storage::Cuda(w_s) = &*w_s else {
        candle_core::bail!("Expected CUDA storage for weights");
    };
    let (w_ptr, _w_guard) = slice_ptr(w_s.as_cuda_slice::<f32>()?, w_l.start_offset());

    let x_contig = x.contiguous()?;
    let (x_s, x_l) = x_contig.storage_and_layout();
    let Storage::Cuda(x_s) = &*x_s else {
        candle_core::bail!("Expected CUDA storage for input");
    };
    let (x_ptr, _x_guard) = slice_ptr(x_s.as_cuda_slice::<f32>()?, x_l.start_offset());

    let (y_ptr, y_guard) = y_buf.device_ptr(y_buf.stream());

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

    let stream = dev.cuda_stream();

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
