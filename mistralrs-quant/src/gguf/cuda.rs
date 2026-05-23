//! CUDA implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This provides a direct implementation using local CUDA kernels for
//! quantized matrix-vector multiplication with expert indexing.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use super::ffi;
use crate::utils::slice_ptr;
use candle_core::{
    cuda::cudarc::driver::{CudaSlice, DevicePtr},
    quantized::{GgmlDType, QMatMul, QTensor},
    CudaDevice, CudaStorage, DType, Device, Result, Shape, Storage, Tensor,
};

// Constants matching candle's quantized CUDA implementation
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

struct DispatchWorkspaceSlot {
    slice: CudaSlice<u32>,
    cap: usize,
}

type DispatchWsMap =
    Mutex<HashMap<candle_core::cuda::DeviceId, &'static Mutex<DispatchWorkspaceSlot>>>;

static MOE_DISPATCH_WORKSPACE: OnceLock<DispatchWsMap> = OnceLock::new();

struct U8WorkspaceSlot {
    slice: CudaSlice<u8>,
    cap: usize,
}

struct F32WorkspaceSlot {
    slice: CudaSlice<f32>,
    cap: usize,
}

type U8WsMap = Mutex<HashMap<candle_core::cuda::DeviceId, &'static Mutex<U8WorkspaceSlot>>>;
type F32WsMap = Mutex<HashMap<candle_core::cuda::DeviceId, &'static Mutex<F32WorkspaceSlot>>>;

static MOE_DECODE_Q8_WORKSPACE: OnceLock<U8WsMap> = OnceLock::new();
static MOE_DECODE_F32_WORKSPACE: OnceLock<F32WsMap> = OnceLock::new();

fn dispatch_workspace_ensure(
    dev: &CudaDevice,
    len: usize,
) -> Result<(u64, std::sync::MutexGuard<'static, DispatchWorkspaceSlot>)> {
    let len = len.max(1);
    let map = MOE_DISPATCH_WORKSPACE.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<DispatchWorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u32>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(DispatchWorkspaceSlot {
                    slice,
                    cap: len,
                })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<u32>(len)? };
        slot.cap = len;
    }
    let ptr = slot.slice.device_ptr(slot.slice.stream()).0;
    Ok((ptr, slot))
}

fn u8_workspace_ensure(
    ws: &'static OnceLock<U8WsMap>,
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, U8WorkspaceSlot>> {
    let len = len.max(1);
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<U8WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<u8>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(U8WorkspaceSlot { slice, cap: len })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<u8>(len)? };
        slot.cap = len;
    }
    Ok(slot)
}

fn f32_workspace_ensure(
    ws: &'static OnceLock<F32WsMap>,
    dev: &CudaDevice,
    len: usize,
) -> Result<std::sync::MutexGuard<'static, F32WorkspaceSlot>> {
    let len = len.max(1);
    let map = ws.get_or_init(|| Mutex::new(HashMap::new()));
    let device_key = dev.id();
    let device_mtx: &'static Mutex<F32WorkspaceSlot> = {
        let mut guard = map.lock().unwrap();
        match guard.get(&device_key).copied() {
            Some(mtx) => mtx,
            None => {
                let slice = unsafe { dev.alloc::<f32>(len)? };
                let leaked = Box::leak(Box::new(Mutex::new(F32WorkspaceSlot { slice, cap: len })));
                guard.insert(device_key, leaked);
                leaked
            }
        }
    };
    let mut slot = device_mtx.lock().unwrap();
    if slot.cap < len {
        slot.slice = unsafe { dev.alloc::<f32>(len)? };
        slot.cap = len;
    }
    Ok(slot)
}

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

/// Quantize f32 input to Q8_1 format for use with quantized matmul kernels.
fn quantize_q8_1(
    src: &CudaSlice<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;

    // Get stream pointer
    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    const CHUNK_SIZE: usize = 65535;
    let mut rows_processed = 0;
    while rows_processed < total_rows {
        let remaining_rows = total_rows - rows_processed;
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        let src_start_elem = rows_processed * k;

        let q8_1_block_size = GgmlDType::Q8_1.block_size();
        let q8_1_type_size = GgmlDType::Q8_1.type_size();
        let num_blocks_per_row = kx_padded / q8_1_block_size;
        let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

        let dst_start_byte = rows_processed * dst_row_size_bytes;

        let (src_ptr, _src_guard) = slice_ptr(src, src_start_elem);
        let (dst_ptr, _dst_guard) = slice_ptr(dst, dst_start_byte);

        unsafe {
            ffi::launch_quantize_q8_1(
                src_ptr as *const f32,
                dst_ptr as *mut std::ffi::c_void,
                k as i32,
                kx_padded as i32,
                num_blocks as i32,
                rows_in_chunk as i32,
                stream,
            );
        }

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn q8_1_bytes(num_rows: usize, k_padded: usize) -> usize {
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();
    let num_blocks_per_row = k_padded / q8_1_block_size;
    num_rows * num_blocks_per_row * q8_1_type_size
}

/// Perform indexed MoE forward pass with fused Q8_1 input quantization.
///
/// # Arguments
/// * `weight_ptr` - Raw device pointer to quantized weight data
/// * `w_shape` - Weight shape [num_experts, n, k]
/// * `w_dtype` - Weight quantization dtype
/// * `input` - Input tensor [batch, topk_or_1, k]
/// * `in_shape` - Input shape
/// * `ids` - Expert indices CudaSlice [batch * topk]
/// * `idx_shape` - Indices shape
/// * `dev` - CUDA device
#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight_ptr: u64,
    w_shape: &Shape,
    w_dtype: GgmlDType,
    input: &Tensor,
    in_shape: &Shape,
    ids: &CudaSlice<u32>,
    idx_shape: &Shape,
    dev: &CudaDevice,
) -> Result<(CudaStorage, Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    let topk = idx_shape.dims()[1];
    assert!(batch == idx_shape.dims()[0], "batch dim not match!");

    let total_rows = batch * input_dim1;
    let input = input.reshape((total_rows, k))?;
    let (input_quant, quant_k, k_padded) = quantize_input_q8_1(&input, dev)?;
    assert!(quant_k == k, "K mismatch");

    // Output buffer - zero-initialize to prevent NaN from uninitialized memory
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    // Get stream pointer
    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    let n_i32 = n as i32;
    let k_i32 = k as i32;
    let batch_i32 = batch as i32;
    let topk_i32 = topk as i32;
    let k_padded_i32 = k_padded as i32;
    let input_dim1_i32 = input_dim1 as i32;

    let (inputs_ptr, _inputs_guard) = slice_ptr(&input_quant, 0);
    let (indices_ptr, _indices_guard) = slice_ptr(ids, 0);
    let (outputs_ptr, _outputs_guard) = slice_ptr(&out, 0);

    unsafe {
        let weights_ptr = weight_ptr as *const std::ffi::c_void;

        match w_dtype {
            GgmlDType::Q4_0 => {
                ffi::launch_indexed_moe_forward_q4_0_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q4_1 => {
                ffi::launch_indexed_moe_forward_q4_1_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q5_0 => {
                ffi::launch_indexed_moe_forward_q5_0_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q5_1 => {
                ffi::launch_indexed_moe_forward_q5_1_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q8_1 => {
                ffi::launch_indexed_moe_forward_q8_1_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q2K => {
                ffi::launch_indexed_moe_forward_q2k_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q3K => {
                ffi::launch_indexed_moe_forward_q3k_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q4K => {
                ffi::launch_indexed_moe_forward_q4k_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q5K => {
                ffi::launch_indexed_moe_forward_q5k_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q6K => {
                ffi::launch_indexed_moe_forward_q6k_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            GgmlDType::Q8_0 => {
                ffi::launch_indexed_moe_forward_q8_0_q8_1(
                    weights_ptr,
                    inputs_ptr as *const std::ffi::c_void,
                    indices_ptr as *const u32,
                    outputs_ptr as *mut f32,
                    n_i32,
                    k_i32,
                    batch_i32,
                    topk_i32,
                    k_padded_i32,
                    input_dim1_i32,
                    stream,
                );
            }
            _ => candle_core::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
        }
    }

    // Drop guards before moving out
    drop(_inputs_guard);
    drop(_indices_guard);
    drop(_outputs_guard);

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;

    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

/// Perform indexed MoE forward pass on a QTensor.
///
/// # Arguments
/// * `qtensor` - The quantized weight tensor [num_experts, n, k]
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qtensor_indexed_moe_forward(qtensor: &QTensor, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    let dtype = qtensor.dtype();

    // Check supported dtypes
    if !matches!(
        dtype,
        GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q8_1
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
    ) {
        candle_core::bail!(
            "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
            dtype
        );
    }

    // Ensure tensors are on CUDA
    let Device::Cuda(dev) = qtensor.device() else {
        candle_core::bail!("indexed_moe_forward requires CUDA device for weights");
    };

    let (x_storage, _x_layout) = x.storage_and_layout();
    let Storage::Cuda(_) = &*x_storage else {
        candle_core::bail!("indexed_moe_forward requires CUDA device for input");
    };

    let (ids_storage, _ids_layout) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("indexed_moe_forward requires CUDA device for indices");
    };

    // Get weight device pointer directly (no copy)
    let weight_ptr = qtensor.device_ptr()? as u64;

    let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;

    let (storage, out_shape) = indexed_moe_forward_fused_q8_1_input(
        weight_ptr,
        qtensor.shape(),
        dtype,
        x,
        x.shape(),
        ids_slice,
        ids.shape(),
        &dev,
    )?;

    Ok(Tensor::from((Storage::Cuda(storage), out_shape)))
}

/// Perform indexed MoE forward pass on a QMatMul.
///
/// This is the main entry point for GGUF quantized MoE forward.
///
/// # Arguments
/// * `qmatmul` - The quantized weight matrix
/// * `x` - Input tensor [batch, topk_or_1, k]
/// * `ids` - Expert indices tensor [batch, topk]
///
/// # Returns
/// Output tensor [batch, topk, n]
pub fn qmatmul_indexed_moe_forward(qmatmul: &QMatMul, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
    match qmatmul {
        QMatMul::QTensor(qtensor) => qtensor_indexed_moe_forward(qtensor, x, ids),
        QMatMul::Tensor(_) | QMatMul::TensorF16(_) => {
            candle_core::bail!(
                "indexed_moe_forward is only supported for quantized tensors (QTensor)"
            )
        }
    }
}

// ============== Grouped MoE (prefill-optimized) ==============

/// Build expert dispatch tables on GPU using u32 buffers.
///
/// Returns (expert_bounds, sorted_token_ids, sorted_source_ids) as u32
/// CudaSlices.
/// Values are always non-negative so u32 and i32 are interchangeable.
/// We use u32 so the sorted_token_ids can be wrapped directly into
/// Candle tensors (which don't support i32).
pub fn moe_dispatch_build(
    topk_ids_flat: &CudaSlice<u32>,
    total_assignments: usize,
    num_experts: usize,
    topk: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<u32>, CudaSlice<u32>, CudaSlice<u32>)> {
    let expert_bounds = unsafe { dev.alloc::<u32>(num_experts + 1) }?;
    let sorted_token_ids = unsafe { dev.alloc::<u32>(total_assignments) }?;
    let sorted_source_ids = unsafe { dev.alloc::<u32>(total_assignments) }?;

    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let (dispatch_ws_ptr, _dispatch_ws_guard) = dispatch_workspace_ensure(dev, 2 * num_experts)?;

    {
        // Cast u32 pointers to i32 for the CUDA kernel (same bit pattern)
        let (topk_ptr, _topk_guard) = slice_ptr(topk_ids_flat, 0);
        let (bounds_ptr, _bounds_guard) = slice_ptr(&expert_bounds, 0);
        let (sorted_ptr, _sorted_guard) = slice_ptr(&sorted_token_ids, 0);
        let (source_ptr, _source_guard) = slice_ptr(&sorted_source_ids, 0);
        let counts_ptr = dispatch_ws_ptr as *mut i32;
        let cursors_ptr = unsafe { counts_ptr.add(num_experts) };

        unsafe {
            ffi::launch_moe_dispatch(
                topk_ptr as *const i32,
                bounds_ptr as *mut i32,
                sorted_ptr as *mut i32,
                source_ptr as *mut i32,
                total_assignments as i32,
                num_experts as i32,
                topk as i32,
                counts_ptr,
                cursors_ptr,
                stream,
            );
        }
    }

    Ok((expert_bounds, sorted_token_ids, sorted_source_ids))
}

/// Reduce flat per-assignment MoE outputs into per-token outputs.
///
/// # Safety
///
/// `topk_weights` must be a valid CUDA device pointer to at least
/// `num_tokens * topk` contiguous `f32` values on `dev`. The pointed allocation
/// must remain alive until all work queued on `dev`'s stream by this function
/// has completed.
pub unsafe fn moe_weighted_reduce_flat(
    inputs: &Tensor,
    topk_weights: *const f32,
    num_tokens: usize,
    topk: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (total_assignments, hidden) = inputs.dims2()?;
    if total_assignments != num_tokens * topk {
        candle_core::bail!(
            "moe_weighted_reduce_flat: input rows {total_assignments} do not match num_tokens={num_tokens} * topk={topk}"
        );
    }
    if inputs.dtype() != DType::F32 {
        candle_core::bail!(
            "moe_weighted_reduce_flat: input dtype must be F32, got {:?}",
            inputs.dtype()
        );
    }

    let inputs = inputs.contiguous()?;
    let (storage, layout) = inputs.storage_and_layout();
    let Storage::Cuda(cuda) = &*storage else {
        candle_core::bail!("moe_weighted_reduce_flat: input must live on CUDA");
    };
    let input_slice = cuda.as_cuda_slice::<f32>()?;
    let out = unsafe { dev.alloc::<f32>(num_tokens * hidden)? };
    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    {
        let (input_ptr, _ig) = slice_ptr(input_slice, layout.start_offset());
        let (out_ptr, _og) = slice_ptr(&out, 0);
        unsafe {
            ffi::launch_moe_weighted_reduce_flat(
                input_ptr as *const f32,
                topk_weights,
                out_ptr as *mut f32,
                num_tokens as i32,
                hidden as i32,
                topk as i32,
                stream,
            );
        }
    }

    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        Shape::from((num_tokens, hidden)),
    )))
}

/// Quantize input to Q8_1 format, returning the quantized buffer.
///
/// Supports F32, BF16, and F16 inputs directly without dtype conversion.
pub fn quantize_input_q8_1(xs: &Tensor, dev: &CudaDevice) -> Result<(CudaSlice<u8>, usize, usize)> {
    let xs_contig = xs.contiguous()?;
    let num_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes = q8_1_bytes(num_rows, k_padded);

    // SAFETY: quantize kernel writes all elements up to k_padded per row
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
    quantize_input_q8_1_into(&xs_contig, &mut input_quant, dev)?;

    Ok((input_quant, k, k_padded))
}

fn quantize_input_q8_1_into(
    xs_contig: &Tensor,
    input_quant: &mut CudaSlice<u8>,
    dev: &CudaDevice,
) -> Result<(usize, usize)> {
    let num_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes = q8_1_bytes(num_rows, k_padded);
    if input_quant.len() < y_size_in_bytes {
        candle_core::bail!(
            "quantize_input_q8_1_into: output buffer too small: {} < {y_size_in_bytes}",
            input_quant.len()
        );
    }
    // Use fused half->Q8_1 kernels when input is BF16/F16 (avoids separate cast kernel)
    if xs_contig.dtype() == candle_core::DType::BF16 || xs_contig.dtype() == candle_core::DType::F16
    {
        let (xs_storage, xs_layout) = xs_contig.storage_and_layout();
        let xs_cuda = match &*xs_storage {
            Storage::Cuda(c) => c,
            _ => candle_core::bail!("expected CUDA tensor"),
        };
        assert!(xs_layout.start_offset() == 0);
        let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
        let (out_ptr, _og) = slice_ptr(input_quant, 0);
        if xs_contig.dtype() == candle_core::DType::BF16 {
            let xs_slice = xs_cuda.as_cuda_slice::<half::bf16>()?;
            unsafe {
                ffi::launch_quantize_q8_1_bf16(
                    xs_slice.slice(0..).device_ptr(xs_slice.stream()).0 as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    k_padded as i32,
                    num_rows as i32,
                    stream,
                );
            }
        } else {
            let xs_slice = xs_cuda.as_cuda_slice::<half::f16>()?;
            unsafe {
                ffi::launch_quantize_q8_1_f16(
                    xs_slice.slice(0..).device_ptr(xs_slice.stream()).0 as *const std::ffi::c_void,
                    out_ptr as *mut std::ffi::c_void,
                    k as i32,
                    k_padded as i32,
                    num_rows as i32,
                    stream,
                );
            }
        }
    } else {
        let xs_f32 = xs_contig.to_dtype(candle_core::DType::F32)?;
        let (xs_storage, xs_layout) = xs_f32.storage_and_layout();
        let xs_cuda = match &*xs_storage {
            Storage::Cuda(c) => c,
            _ => candle_core::bail!("expected CUDA tensor"),
        };
        let xs_slice = xs_cuda.as_cuda_slice::<f32>()?;
        assert!(xs_layout.start_offset() == 0);
        quantize_q8_1(xs_slice, input_quant, k, num_rows, dev)?;
    }

    Ok((k, k_padded))
}

/// Run grouped MoE GEMM with pre-quantized Q8_1 input.
///
/// Avoids re-quantizing the input when the same input is used for multiple projections.
#[allow(clippy::too_many_arguments)]
pub fn grouped_moe_gemm_prequantized(
    qtensor: &QTensor,
    input_quant: &CudaSlice<u8>,
    k: usize,
    k_padded: usize,
    expert_bounds: &CudaSlice<u32>,
    sorted_token_ids: &CudaSlice<u32>,
    topk_weights: Option<(*const f32, usize)>, // (ptr, guard_token) - raw pointer
    total_assignments: usize,
    topk: usize,
    num_experts: usize,
    input_dim1: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let dtype = qtensor.dtype();
    let (_, n, k_w) = qtensor.shape().dims3()?;
    assert!(k == k_w, "K mismatch");

    let has_topk_weights = topk_weights.is_some();
    let num_tokens = total_assignments / topk;
    let out_rows = if has_topk_weights {
        num_tokens
    } else {
        total_assignments
    };
    let out = dev.alloc_zeros::<f32>(out_rows * n)?;

    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;
    let weight_ptr = qtensor.device_ptr()? as *const std::ffi::c_void;

    let topk_w_ptr = topk_weights.map(|(p, _)| p).unwrap_or(std::ptr::null());

    {
        let (inputs_ptr, _ig) = slice_ptr(input_quant, 0);
        let (bounds_ptr, _bg) = slice_ptr(expert_bounds, 0);
        let (sorted_ptr, _sg) = slice_ptr(sorted_token_ids, 0);
        let (out_ptr, _og) = slice_ptr(&out, 0);

        unsafe {
            let launch_fn = match dtype {
                GgmlDType::Q8_0 => ffi::launch_moe_grouped_gemm_q8_0,
                GgmlDType::Q4_0 => ffi::launch_moe_grouped_gemm_q4_0,
                GgmlDType::Q4_1 => ffi::launch_moe_grouped_gemm_q4_1,
                GgmlDType::Q5_0 => ffi::launch_moe_grouped_gemm_q5_0,
                GgmlDType::Q5_1 => ffi::launch_moe_grouped_gemm_q5_1,
                GgmlDType::Q8_1 => ffi::launch_moe_grouped_gemm_q8_1,
                GgmlDType::Q2K => ffi::launch_moe_grouped_gemm_q2k,
                GgmlDType::Q3K => ffi::launch_moe_grouped_gemm_q3k,
                GgmlDType::Q4K => ffi::launch_moe_grouped_gemm_q4k,
                GgmlDType::Q5K => ffi::launch_moe_grouped_gemm_q5k,
                GgmlDType::Q6K => ffi::launch_moe_grouped_gemm_q6k,
                _ => candle_core::bail!("unsupported dtype: {dtype:?}"),
            };

            launch_fn(
                weight_ptr,
                inputs_ptr as *const std::ffi::c_void,
                bounds_ptr as *const i32,
                sorted_ptr as *const i32,
                topk_w_ptr,
                out_ptr as *mut f32,
                n as i32,
                k as i32,
                k_padded as i32,
                num_experts as i32,
                topk as i32,
                input_dim1 as i32,
                stream,
            );
        }
    }

    let out_shape: Shape = vec![out_rows, n].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(out, dev.clone())),
        out_shape,
    )))
}

/// Activation type IDs matching the CUDA kernel's act_type parameter.
pub const ACT_GELU_PYTORCH_TANH: i32 = 0;
pub const ACT_SILU: i32 = 1;

/// Perform fused MoE decode: gate+up+activation → quantize → down+aggregate.
///
/// This fuses the entire MoE decode into 4 kernel launches:
///   1. quantize input to Q8_1
///   2. fused gate+up+activation+multiply
///   3. quantize intermediate to Q8_1
///   4. fused down+topk_weights+atomicAdd aggregation
///
/// Returns the aggregated output tensor of shape [batch, hidden_size].
#[allow(clippy::too_many_arguments)]
/// # Safety
/// `topk_weights_ptr` must be a valid device pointer to `batch * topk` f32 values.
pub unsafe fn indexed_moe_fused_decode(
    gate_qt: &QTensor,
    up_qt: &QTensor,
    down_qt: &QTensor,
    xs_flat: &Tensor,
    topk_ids: &CudaSlice<u32>,
    topk_weights_ptr: *const f32,
    batch: usize,
    topk: usize,
    act_type: i32,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let gate_up_dtype = gate_qt.dtype();
    assert!(up_qt.dtype() == gate_up_dtype);
    let down_dtype = down_qt.dtype();

    let (_, n_gate, k_gate) = gate_qt.shape().dims3()?;
    let (_, n_down, k_down) = down_qt.shape().dims3()?;

    // gate and up: [E, intermediate, hidden], down: [E, hidden, intermediate]
    let hidden_size = k_gate;
    let intermediate_size = n_gate;
    assert!(n_down == hidden_size);
    assert!(k_down == intermediate_size);

    let xs_contig = xs_flat.contiguous()?;
    let input_rows = xs_contig.dim(0)?;
    let k = xs_contig.dim(1)?;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let input_q8_bytes = q8_1_bytes(input_rows, k_padded);
    let intermediate_rows = batch * topk;
    let k_down_padded = pad(intermediate_size, MATRIX_ROW_PADDING);
    let intermediate_q8_bytes = q8_1_bytes(intermediate_rows, k_down_padded);

    let mut q8_workspace = u8_workspace_ensure(
        &MOE_DECODE_Q8_WORKSPACE,
        dev,
        input_q8_bytes.max(intermediate_q8_bytes),
    )?;

    // Step 1: Quantize input to Q8_1 (shared between gate and up)
    quantize_input_q8_1_into(&xs_contig, &mut q8_workspace.slice, dev)?;

    let stream = dev.cuda_stream().cu_stream() as *mut std::ffi::c_void;

    // Step 2: Fused gate+up+activation+multiply
    let gate_up_outsize = batch * topk * intermediate_size;
    let gate_up_workspace = f32_workspace_ensure(&MOE_DECODE_F32_WORKSPACE, dev, gate_up_outsize)?;

    let gate_ptr = gate_qt.device_ptr()? as *const std::ffi::c_void;
    let up_ptr = up_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = slice_ptr(&q8_workspace.slice, 0);
        let (ids_ptr, _idg) = slice_ptr(topk_ids, 0);
        let (out_ptr, _og) = slice_ptr(&gate_up_workspace.slice, 0);

        type FusedGateUpFn = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const u32,
            *mut f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut std::ffi::c_void,
        );

        let launch_fn: FusedGateUpFn = match gate_up_dtype {
            GgmlDType::Q8_0 => ffi::launch_moe_gemv_fused_gate_up_q8_0_q8_1,
            GgmlDType::Q4_0 => ffi::launch_moe_gemv_fused_gate_up_q4_0_q8_1,
            GgmlDType::Q4_1 => ffi::launch_moe_gemv_fused_gate_up_q4_1_q8_1,
            GgmlDType::Q5_0 => ffi::launch_moe_gemv_fused_gate_up_q5_0_q8_1,
            GgmlDType::Q5_1 => ffi::launch_moe_gemv_fused_gate_up_q5_1_q8_1,
            GgmlDType::Q8_1 => ffi::launch_moe_gemv_fused_gate_up_q8_1_q8_1,
            GgmlDType::Q2K => ffi::launch_moe_gemv_fused_gate_up_q2k_q8_1,
            GgmlDType::Q3K => ffi::launch_moe_gemv_fused_gate_up_q3k_q8_1,
            GgmlDType::Q4K => ffi::launch_moe_gemv_fused_gate_up_q4k_q8_1,
            GgmlDType::Q5K => ffi::launch_moe_gemv_fused_gate_up_q5k_q8_1,
            GgmlDType::Q6K => ffi::launch_moe_gemv_fused_gate_up_q6k_q8_1,
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {gate_up_dtype:?}"),
        };

        unsafe {
            launch_fn(
                gate_ptr,
                up_ptr,
                inp_ptr as *const std::ffi::c_void,
                ids_ptr as *const u32,
                out_ptr as *mut f32,
                intermediate_size as i32,
                k as i32,
                batch as i32,
                topk as i32,
                k_padded as i32,
                act_type,
                stream,
            );
        }
    }

    // Step 3: Quantize the intermediate output for down projection
    // gate_up_out is [batch * topk, intermediate_size] in f32
    quantize_q8_1(
        &gate_up_workspace.slice,
        &mut q8_workspace.slice,
        intermediate_size,
        intermediate_rows,
        dev,
    )?;

    // Step 4: Fused down+aggregate with topk_weights and atomicAdd
    let final_outsize = batch * hidden_size;
    let final_out = dev.alloc_zeros::<f32>(final_outsize)?;

    let down_ptr = down_qt.device_ptr()? as *const std::ffi::c_void;

    {
        let (inp_ptr, _ig) = slice_ptr(&q8_workspace.slice, 0);
        let (ids_ptr, _idg) = slice_ptr(topk_ids, 0);
        let (out_ptr, _og) = slice_ptr(&final_out, 0);

        type DownAggregateFn = unsafe extern "C" fn(
            *const std::ffi::c_void,
            *const std::ffi::c_void,
            *const u32,
            *const f32,
            *mut f32,
            i32,
            i32,
            i32,
            i32,
            i32,
            *mut std::ffi::c_void,
        );

        let launch_fn: DownAggregateFn = match down_dtype {
            GgmlDType::Q8_0 => ffi::launch_moe_gemv_down_aggregate_q8_0_q8_1,
            GgmlDType::Q4_0 => ffi::launch_moe_gemv_down_aggregate_q4_0_q8_1,
            GgmlDType::Q4_1 => ffi::launch_moe_gemv_down_aggregate_q4_1_q8_1,
            GgmlDType::Q5_0 => ffi::launch_moe_gemv_down_aggregate_q5_0_q8_1,
            GgmlDType::Q5_1 => ffi::launch_moe_gemv_down_aggregate_q5_1_q8_1,
            GgmlDType::Q8_1 => ffi::launch_moe_gemv_down_aggregate_q8_1_q8_1,
            GgmlDType::Q2K => ffi::launch_moe_gemv_down_aggregate_q2k_q8_1,
            GgmlDType::Q3K => ffi::launch_moe_gemv_down_aggregate_q3k_q8_1,
            GgmlDType::Q4K => ffi::launch_moe_gemv_down_aggregate_q4k_q8_1,
            GgmlDType::Q5K => ffi::launch_moe_gemv_down_aggregate_q5k_q8_1,
            GgmlDType::Q6K => ffi::launch_moe_gemv_down_aggregate_q6k_q8_1,
            _ => candle_core::bail!("unsupported dtype for fused MoE decode: {down_dtype:?}"),
        };

        unsafe {
            launch_fn(
                down_ptr,
                inp_ptr as *const std::ffi::c_void,
                ids_ptr as *const u32,
                topk_weights_ptr,
                out_ptr as *mut f32,
                hidden_size as i32,
                intermediate_size as i32,
                batch as i32,
                topk as i32,
                k_down_padded as i32,
                stream,
            );
        }
    }

    let out_shape: Shape = vec![batch, hidden_size].into();
    Ok(Tensor::from((
        Storage::Cuda(CudaStorage::wrap_cuda_slice(final_out, dev.clone())),
        out_shape,
    )))
}
