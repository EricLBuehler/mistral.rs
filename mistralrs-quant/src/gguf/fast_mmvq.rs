//! Fast mat-vec-q path for GGUF quantized weights on CUDA.
//!
//! This module is the Rust-side entry point for
//! `kernels/mmvq_gguf/mmvq_gguf.cu`: a port of llama.cpp's current
//! `mul_mat_vec_q` kernel template covering all 10 GGUF quant types
//! (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K) with BF16
//! and F32 output.
//!
//! Unlike the legacy candle-core quantized matmul path this function:
//!
//! - Accepts BF16 (or F32) input end-to-end — no BF16↔F32 cast pair around
//!   each linear layer.
//! - Runs a padding-aware BF16 → Q8_1 quantize kernel so the scratch
//!   allocation doesn't need to be zeroed.
//! - Reuses a persistent per-device Q8_1 scratch buffer instead of an
//!   alloc/alloc_zeros/free churn on every call.
//! - Reads the weight data directly via `candle_core::quantized::QTensor::
//!   device_ptr()` — zero copy, zero duplication. candle keeps owning the
//!   allocation.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{
    quantized::{GgmlDType, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

use super::ffi;
use crate::utils::slice_ptr;

const Q8_1_BLOCK_SIZE: usize = 32;
const Q8_1_TYPE_SIZE: usize = 36; // 2 halves (4 bytes) + QK8_1 int8 = 4 + 32 = 36
const MATRIX_ROW_PADDING: usize = 512;

#[inline]
fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

#[inline]
fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Quant types whose weights can flow through this fast path.
///
/// Exactly the 10 types the `mmvq_gguf.cu` kernel compiles for.
pub fn supports(dtype: GgmlDType) -> bool {
    matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    )
}

/// Maximum batch size supported by the mmvq kernel family (one extern `cuda1`
/// through `cuda8` per (dtype, output-dtype)).
pub const MMVQ_MAX_BATCH: usize = 8;

// ---------------------------------------------------------------------------
// Per-device Q8_1 scratch workspace
//
// One shared buffer per CUDA device, protected by a mutex. Grows monotonically
// via a watermark: if a caller requests more bytes than the current capacity,
// we replace the allocation with a larger one. No shrinking.
// ---------------------------------------------------------------------------

struct WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

static WORKSPACE: OnceLock<Mutex<HashMap<candle_core::cuda::DeviceId, WorkspaceSlot>>> =
    OnceLock::new();

fn workspace_ensure(dev: &CudaDevice, bytes: usize) -> Result<CudaSlice<u8>> {
    let map = WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let mut guard = map.lock().unwrap();
    let slot = match guard.get_mut(&device_key) {
        Some(slot) => slot,
        None => {
            let slice = unsafe { dev.alloc::<u8>(bytes.max(1))? };
            guard.insert(
                device_key,
                WorkspaceSlot {
                    slice,
                    cap: bytes.max(1),
                },
            );
            guard.get_mut(&device_key).unwrap()
        }
    };
    if slot.cap < bytes {
        slot.slice = unsafe { dev.alloc::<u8>(bytes)? };
        slot.cap = bytes;
    }
    Ok(slot.slice.clone())
}

// ---------------------------------------------------------------------------
// Per-(weight-dtype, output-dtype) launcher dispatch
//
// Two `extern "C" fn` function pointer types matching the launcher
// signatures in `ffi.rs`, plus small tables that map a `GgmlDType` to the
// right launcher for each output dtype.
// ---------------------------------------------------------------------------

type PlainLauncher = unsafe extern "C" fn(
    vx: *const std::ffi::c_void,
    vy: *const std::ffi::c_void,
    dst: *mut std::ffi::c_void,
    ncols_x: i32,
    nrows_x: i32,
    stride_col_y: i32,
    stride_col_dst: i32,
    b_size: i32,
    stream: *mut std::ffi::c_void,
);

fn plain_launcher_bf16(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_bf16_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_bf16_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_bf16_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_bf16_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_bf16_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_bf16_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_bf16_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_bf16_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_bf16_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_bf16_plain,
        _ => return None,
    };
    Some(f)
}

fn plain_launcher_f32(dtype: GgmlDType) -> Option<PlainLauncher> {
    let f: PlainLauncher = match dtype {
        GgmlDType::Q4_0 => ffi::launch_mmvq_gguf_q4_0_f32_plain,
        GgmlDType::Q4_1 => ffi::launch_mmvq_gguf_q4_1_f32_plain,
        GgmlDType::Q5_0 => ffi::launch_mmvq_gguf_q5_0_f32_plain,
        GgmlDType::Q5_1 => ffi::launch_mmvq_gguf_q5_1_f32_plain,
        GgmlDType::Q8_0 => ffi::launch_mmvq_gguf_q8_0_f32_plain,
        GgmlDType::Q2K => ffi::launch_mmvq_gguf_q2_k_f32_plain,
        GgmlDType::Q3K => ffi::launch_mmvq_gguf_q3_k_f32_plain,
        GgmlDType::Q4K => ffi::launch_mmvq_gguf_q4_k_f32_plain,
        GgmlDType::Q5K => ffi::launch_mmvq_gguf_q5_k_f32_plain,
        GgmlDType::Q6K => ffi::launch_mmvq_gguf_q6_k_f32_plain,
        _ => return None,
    };
    Some(f)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Compute `w @ xs^T` where `w` is a Q8_1-quantizable GGUF weight tensor and
/// `xs` is a contiguous BF16 / F32 activation on the same CUDA device.
///
/// Supported input shapes:
/// * `[b, k]`     with b in `1..=8`
/// * `[b, m, k]`  with `b * m` in `1..=8`
///
/// Output has the same leading dimensions as `xs` with the last axis replaced
/// by `w.shape().dims2()?.0` (nrows of the weight).
///
/// The output dtype matches the input dtype (BF16 → BF16, F32 → F32).
pub fn plain(w: &QTensor, xs: &Tensor) -> Result<Tensor> {
    let dtype = w.dtype();
    if !supports(dtype) {
        candle_core::bail!("fast_mmvq: unsupported quant dtype {dtype:?}");
    }
    let Device::Cuda(dev) = w.device() else {
        candle_core::bail!("fast_mmvq: weight must live on CUDA");
    };
    let (nrows, ncols) = w.shape().dims2()?;

    let (b_size, k) = match xs.dims() {
        [b, k] => (*b, *k),
        [b, m, k] => (*b * *m, *k),
        other => candle_core::bail!("fast_mmvq: unexpected input rank {other:?}"),
    };
    if k != ncols {
        candle_core::bail!(
            "fast_mmvq: shape mismatch — weight [{nrows}, {ncols}] vs input tail {k}"
        );
    }
    if b_size == 0 || b_size > MMVQ_MAX_BATCH {
        candle_core::bail!(
            "fast_mmvq: batch size {b_size} out of supported range 1..={MMVQ_MAX_BATCH}"
        );
    }
    let input_ty = xs.dtype();
    if !matches!(input_ty, DType::BF16 | DType::F32) {
        candle_core::bail!("fast_mmvq: input dtype must be BF16 or F32, got {input_ty:?}");
    }

    let xs = xs.contiguous()?;
    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Cuda(xs_cuda) = &*xs_storage else {
        candle_core::bail!("fast_mmvq: input must live on CUDA");
    };
    // `contiguous()` may return `self` for already-contiguous inputs, so
    // the layout can still carry a non-zero start offset. Respect it by
    // passing it through to `slice_ptr` when taking device pointers.
    let xs_offset = xs_layout.start_offset();

    let stream_ptr = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks_per_row = k_padded / Q8_1_BLOCK_SIZE;
    let dst_row_bytes = num_blocks_per_row * Q8_1_TYPE_SIZE;
    let scratch_bytes = b_size * dst_row_bytes;

    // Grab the persistent Q8_1 scratch buffer — may grow it if the caller
    // exceeds the current watermark.
    let scratch = workspace_ensure(&dev, scratch_bytes)?;
    let stride_col_y = (k_padded / Q8_1_BLOCK_SIZE) as i32;
    let stride_col_dst = nrows as i32;
    let weight_ptr = w.device_ptr()? as *const std::ffi::c_void;

    // The borrow/move dance: each kernel launch needs raw device pointers
    // that hold guards for the duration of the launch. We have to drop all
    // guards explicitly before moving the output `CudaSlice` into a
    // `CudaStorage` wrapper (since `wrap_cuda_slice` takes by value).
    match input_ty {
        DType::BF16 => {
            let slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            let out = unsafe { dev.alloc::<half::bf16>(nrows * b_size)? };

            // Quantize xs → scratch, then launch the matmul. Drop guards
            // before moving `out`.
            {
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                let (scratch_ptr, _scratch_guard) = slice_ptr(&scratch, 0);
                let (out_ptr, _out_guard) = slice_ptr(&out, 0);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_bf16(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr as *mut std::ffi::c_void,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher =
                        plain_launcher_bf16(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                // Explicit drops (would also happen at end of scope, but make
                // the ordering obvious).
                drop(_xs_guard);
                drop(_scratch_guard);
                drop(_out_guard);
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            let mut out_dims = xs.dims().to_vec();
            let last = out_dims.len() - 1;
            out_dims[last] = nrows;
            return Ok(Tensor::from((
                Storage::Cuda(out_storage),
                Shape::from(out_dims),
            )));
        }
        DType::F32 => {
            let slice = xs_cuda.as_cuda_slice::<f32>()?;
            let out = unsafe { dev.alloc::<f32>(nrows * b_size)? };

            {
                let (xs_ptr, _xs_guard) = slice_ptr(slice, xs_offset);
                let (scratch_ptr, _scratch_guard) = slice_ptr(&scratch, 0);
                let (out_ptr, _out_guard) = slice_ptr(&out, 0);

                unsafe {
                    ffi::launch_mmvq_gguf_quantize_q8_1_f32(
                        xs_ptr as *const std::ffi::c_void,
                        scratch_ptr as *mut std::ffi::c_void,
                        k as i32,
                        k_padded as i32,
                        b_size as i32,
                        stream_ptr,
                    );
                    let launcher =
                        plain_launcher_f32(dtype).expect("supports() checked");
                    launcher(
                        weight_ptr,
                        scratch_ptr as *const std::ffi::c_void,
                        out_ptr as *mut std::ffi::c_void,
                        k as i32,
                        nrows as i32,
                        stride_col_y,
                        stride_col_dst,
                        b_size as i32,
                        stream_ptr,
                    );
                }
                drop(_xs_guard);
                drop(_scratch_guard);
                drop(_out_guard);
            }

            let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
            let mut out_dims = xs.dims().to_vec();
            let last = out_dims.len() - 1;
            out_dims[last] = nrows;
            return Ok(Tensor::from((
                Storage::Cuda(out_storage),
                Shape::from(out_dims),
            )));
        }
        _ => unreachable!(),
    }
}
