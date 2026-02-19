//! CUDA implementation of indexed MoE forward for GGUF quantized weights.
//!
//! This provides a direct implementation using local CUDA kernels for
//! quantized matrix-vector multiplication with expert indexing.

use super::ffi;
use crate::utils::slice_ptr;
use candle_core::{
    cuda::cudarc::driver::CudaSlice,
    quantized::{GgmlDType, QMatMul, QTensor},
    CudaDevice, CudaStorage, Device, Result, Shape, Storage, Tensor,
};

// Constants matching candle's quantized CUDA implementation
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

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

/// Perform indexed MoE forward pass with fused Q8_1 input quantization.
///
/// # Arguments
/// * `weight_ptr` - Raw device pointer to quantized weight data
/// * `w_shape` - Weight shape [num_experts, n, k]
/// * `w_dtype` - Weight quantization dtype
/// * `input` - Input CudaSlice [batch * topk_or_1 * k]
/// * `in_shape` - Input shape
/// * `ids` - Expert indices CudaSlice [batch * topk]
/// * `idx_shape` - Indices shape
/// * `dev` - CUDA device
#[allow(clippy::too_many_arguments)]
fn indexed_moe_forward_fused_q8_1_input(
    weight_ptr: u64,
    w_shape: &Shape,
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
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

    // Quantize input into Q8_1
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };

    quantize_q8_1(input, &mut input_quant, k, total_rows, dev)?;

    // Output buffer
    let outsize = batch * topk * n;
    let out = unsafe { dev.alloc::<f32>(outsize)? };

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
        GgmlDType::Q8_0
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
    let Storage::Cuda(x_cuda) = &*x_storage else {
        candle_core::bail!("indexed_moe_forward requires CUDA device for input");
    };

    let (ids_storage, _ids_layout) = ids.storage_and_layout();
    let Storage::Cuda(ids_cuda) = &*ids_storage else {
        candle_core::bail!("indexed_moe_forward requires CUDA device for indices");
    };

    // Get weight device pointer directly (no copy)
    let weight_ptr = qtensor.device_ptr()? as u64;

    let input_storage = x_cuda.as_cuda_slice::<f32>()?;
    let ids_slice = ids_cuda.as_cuda_slice::<u32>()?;

    let (storage, out_shape) = indexed_moe_forward_fused_q8_1_input(
        weight_ptr,
        qtensor.shape(),
        dtype,
        input_storage,
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
