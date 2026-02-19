use candle_core::{CudaStorage, Device, IndexOp, Result, Shape, Storage, Tensor};
use half::{bf16, f16};

use super::ffi;
use crate::utils::slice_ptr;

/// Perform MXFP4 matmul: output = input @ weight.T + bias
///
/// Args:
///   input: [M, K] in f16/bf16
///   weight: [N, K/2] packed u8 (2 FP4 values per byte)
///   scale: [N, K/32] u8 E8M0 scales
///   bias: Optional [N] in f16/bf16
///
/// Returns: [M, N]
pub fn mxfp4_matmul(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    if !ffi::HAVE_MXFP4_GEMM_KERNELS {
        candle_core::bail!("MXFP4 GEMM kernels not available");
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();

    if input_dims.len() != 2 {
        candle_core::bail!("Expected input to be rank 2, got {:?}", input_dims);
    }

    let m = input_dims[0];
    let k = input_dims[1];
    let n = weight_dims[0];

    if weight_dims[1] != k / 2 {
        candle_core::bail!(
            "Weight shape mismatch: expected [N, K/2] = [{}, {}], got {:?}",
            n,
            k / 2,
            weight_dims
        );
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("Expected CUDA device"),
    };

    let input_l = input.layout();
    let weight_l = weight.layout();
    let scale_l = scale.layout();

    let input_storage = input.storage_and_layout().0;
    let weight_storage = weight.storage_and_layout().0;
    let scale_storage = scale.storage_and_layout().0;

    let weight_s = match &*weight_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u8>()?,
        _ => candle_core::bail!("Expected CUDA storage for weight"),
    };
    let scale_s = match &*scale_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u8>()?,
        _ => candle_core::bail!("Expected CUDA storage for scale"),
    };

    let (weight_ptr, _weight_guard) = slice_ptr(weight_s, weight_l.start_offset());
    let (scale_ptr, _scale_guard) = slice_ptr(scale_s, scale_l.start_offset());

    let has_bias = bias.is_some();

    match input.dtype() {
        candle_core::DType::F16 => {
            let output = dev.alloc_zeros::<f16>(m * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };
            let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
            let (output_ptr, _output_guard) = slice_ptr(&output, 0);

            let bias_ptr = if has_bias {
                let b = bias.unwrap();
                let b_l = b.layout();
                let b_storage = b.storage_and_layout().0;
                let b_s = match &*b_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                    _ => candle_core::bail!("Expected CUDA storage for bias"),
                };
                let (ptr, _guard) = slice_ptr(b_s, b_l.start_offset());
                ptr as *const f16
            } else {
                std::ptr::null()
            };

            unsafe {
                if ffi::HAVE_MXFP4_WMMA_KERNELS {
                    ffi::launch_mxfp4_matmul_wmma_f16(
                        input_ptr as *const f16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        output_ptr as *mut f16,
                        m as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        dev.cuda_stream().cu_stream(),
                    );
                } else {
                    ffi::launch_mxfp4_matmul_f16(
                        input_ptr as *const f16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        output_ptr as *mut f16,
                        m as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }

            drop(_output_guard);
            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                Shape::from((m, n)),
            )))
        }
        candle_core::DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>(m * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };
            let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
            let (output_ptr, _output_guard) = slice_ptr(&output, 0);

            let bias_ptr = if has_bias {
                let b = bias.unwrap();
                let b_l = b.layout();
                let b_storage = b.storage_and_layout().0;
                let b_s = match &*b_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                    _ => candle_core::bail!("Expected CUDA storage for bias"),
                };
                let (ptr, _guard) = slice_ptr(b_s, b_l.start_offset());
                ptr as *const bf16
            } else {
                std::ptr::null()
            };

            unsafe {
                if ffi::HAVE_MXFP4_WMMA_KERNELS {
                    ffi::launch_mxfp4_matmul_wmma_bf16(
                        input_ptr as *const bf16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        output_ptr as *mut bf16,
                        m as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        dev.cuda_stream().cu_stream(),
                    );
                } else {
                    ffi::launch_mxfp4_matmul_bf16(
                        input_ptr as *const bf16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        output_ptr as *mut bf16,
                        m as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }

            drop(_output_guard);
            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                Shape::from((m, n)),
            )))
        }
        _ => candle_core::bail!("Unsupported dtype for MXFP4 matmul: {:?}", input.dtype()),
    }
}

/// Fallback: groups tokens by expert on CPU and calls mxfp4_matmul per expert.
/// Used when the fused CUDA kernel's shared memory requirement exceeds the device limit.
#[allow(clippy::too_many_arguments)]
fn mxfp4_grouped_moe_gemm(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    indices: &Tensor,
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    n: usize,
    k: usize,
    input_has_topk_dim: bool,
) -> Result<Tensor> {
    let dev = input.device().clone();
    let total_work = num_tokens * topk;

    let indices_flat: Vec<u32> = indices.flatten_all()?.to_vec1()?;

    let mut expert_groups: Vec<Vec<(usize, usize)>> = vec![Vec::new(); num_experts];
    for t in 0..num_tokens {
        for s in 0..topk {
            let flat_idx = t * topk + s;
            let expert_id = indices_flat[flat_idx] as usize;
            let input_row = if input_has_topk_dim { flat_idx } else { t };
            if expert_id < num_experts {
                expert_groups[expert_id].push((flat_idx, input_row));
            }
        }
    }

    let flat_input = if input_has_topk_dim {
        input.reshape((total_work, k))?
    } else {
        input.clone()
    };

    let mut sorted_input_indices: Vec<u32> = Vec::with_capacity(total_work);
    let mut output_positions: Vec<usize> = Vec::with_capacity(total_work);
    let mut expert_offsets: Vec<(usize, usize, usize)> = Vec::new();

    let mut pos = 0;
    for (expert_id, items) in expert_groups.iter().enumerate() {
        if items.is_empty() {
            continue;
        }
        let start = pos;
        for &(flat_out_idx, input_row) in items {
            sorted_input_indices.push(input_row as u32);
            output_positions.push(flat_out_idx);
            pos += 1;
        }
        expert_offsets.push((expert_id, start, items.len()));
    }

    let perm = Tensor::from_vec(sorted_input_indices, (total_work,), &dev)?;
    let sorted_input = flat_input.index_select(&perm, 0)?;

    let mut result_chunks: Vec<Tensor> = Vec::with_capacity(expert_offsets.len());
    for &(expert_id, start, count) in &expert_offsets {
        let batch = sorted_input.narrow(0, start, count)?;
        let expert_w = weight.i(expert_id)?;
        let expert_s = scale.i(expert_id)?;
        let expert_b = bias.map(|b| b.i(expert_id)).transpose()?;
        let result = mxfp4_matmul(&batch, &expert_w, &expert_s, expert_b.as_ref())?;
        result_chunks.push(result);
    }

    let sorted_output = Tensor::cat(&result_chunks, 0)?;

    let mut inv_perm = vec![0u32; total_work];
    for (sorted_pos, &flat_out_idx) in output_positions.iter().enumerate() {
        inv_perm[flat_out_idx] = sorted_pos as u32;
    }
    let inv_perm_t = Tensor::from_vec(inv_perm, (total_work,), &dev)?;
    let output = sorted_output.index_select(&inv_perm_t, 0)?;

    output.reshape((num_tokens, topk, n))
}

/// Perform MXFP4 indexed MoE GEMM: for each (token, expert_slot), compute input @ weight[expert].T + bias[expert]
///
/// For prefill (num_tokens > 1), uses a fused CUDA kernel that discovers tokens per expert
/// on-GPU and does tiled GEMM with indirect I/O (WMMA tensor cores when available).
/// Falls back to Rust-side grouped GEMM when shared memory requirements exceed the device limit.
/// For decode (num_tokens == 1), uses the per-token scalar kernel.
///
/// Args:
///   input: [num_tokens, K] or [num_tokens, topk, K] in f16/bf16
///   weight: [num_experts, N, K/2] packed u8
///   scale: [num_experts, N, K/32] u8 E8M0 scales
///   bias: Optional [num_experts, N] in f16/bf16
///   indices: [num_tokens, topk] u32 expert indices
///
/// Returns: [num_tokens, topk, N]
pub fn mxfp4_indexed_moe_gemm(
    input: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    bias: Option<&Tensor>,
    indices: &Tensor,
) -> Result<Tensor> {
    if !ffi::HAVE_MXFP4_GEMM_KERNELS {
        candle_core::bail!("MXFP4 GEMM kernels not available");
    }

    let input = if input.is_contiguous() {
        input.clone()
    } else {
        input.contiguous()?
    };
    let weight = if weight.is_contiguous() {
        weight.clone()
    } else {
        weight.contiguous()?
    };
    let scale = if scale.is_contiguous() {
        scale.clone()
    } else {
        scale.contiguous()?
    };
    let indices = if indices.is_contiguous() {
        indices.clone()
    } else {
        indices.contiguous()?
    };

    let input_dims = input.dims();
    let weight_dims = weight.dims();
    let indices_dims = indices.dims();

    let (num_tokens, topk, k, input_has_topk_dim) = if input_dims.len() == 2 {
        (input_dims[0], indices_dims[1], input_dims[1], false)
    } else if input_dims.len() == 3 {
        (input_dims[0], input_dims[1], input_dims[2], true)
    } else {
        candle_core::bail!("Expected input to be rank 2 or 3, got {:?}", input_dims);
    };

    let num_experts = weight_dims[0];
    let n = weight_dims[1];

    if weight_dims[2] != k / 2 {
        candle_core::bail!(
            "Weight shape mismatch: expected [num_experts, N, K/2], got {:?}",
            weight_dims
        );
    }

    // Check if the fused CUDA kernel's shared memory requirement fits on this device.
    // The fused kernel stores a token list (num_tokens * topk * sizeof(int)) in shared memory
    // plus the kernel's own tile buffers:
    //   WMMA: A_sh(64*32*2) + B_sh(64*32*2) + alignment_pad + C_sh(64*64*4) = 24576 bytes
    //   Plain: s_input(64*33*4) + s_weight(64*33*4) + counter(4) = 17428 bytes
    let token_list_bytes = num_tokens * topk * 4;
    let base_smem: usize = if ffi::HAVE_MXFP4_WMMA_KERNELS {
        24576
    } else {
        17428
    };
    let needed_smem = token_list_bytes + base_smem;
    let max_smem = unsafe { ffi::mxfp4_get_max_smem_optin() } as usize;
    let use_fused = num_tokens > 1 && needed_smem <= max_smem;

    // If the fused kernel doesn't fit in shared memory, fall back to Rust-side grouped GEMM
    // which calls mxfp4_matmul per expert (still uses WMMA tensor cores, just not fused).
    if num_tokens > 1 && !use_fused {
        return mxfp4_grouped_moe_gemm(
            &input,
            &weight,
            &scale,
            bias,
            &indices,
            num_tokens,
            topk,
            num_experts,
            n,
            k,
            input_has_topk_dim,
        );
    }

    let dev = match input.device() {
        Device::Cuda(dev) => dev,
        _ => candle_core::bail!("Expected CUDA device"),
    };

    let input_l = input.layout();
    let weight_l = weight.layout();
    let scale_l = scale.layout();
    let indices_l = indices.layout();

    let input_storage = input.storage_and_layout().0;
    let weight_storage = weight.storage_and_layout().0;
    let scale_storage = scale.storage_and_layout().0;
    let indices_storage = indices.storage_and_layout().0;

    let weight_s = match &*weight_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u8>()?,
        _ => candle_core::bail!("Expected CUDA storage for weight"),
    };
    let scale_s = match &*scale_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u8>()?,
        _ => candle_core::bail!("Expected CUDA storage for scale"),
    };
    let indices_s = match &*indices_storage {
        Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("Expected CUDA storage for indices"),
    };

    let (weight_ptr, _weight_guard) = slice_ptr(weight_s, weight_l.start_offset());
    let (scale_ptr, _scale_guard) = slice_ptr(scale_s, scale_l.start_offset());
    let (indices_ptr, _indices_guard) = slice_ptr(indices_s, indices_l.start_offset());

    let has_bias = bias.is_some();

    match input.dtype() {
        candle_core::DType::F16 => {
            let output = dev.alloc_zeros::<f16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };
            let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
            let (output_ptr, _output_guard) = slice_ptr(&output, 0);

            let bias_ptr = if has_bias {
                let b = bias.unwrap();
                let b_storage = b.storage_and_layout().0;
                let b_s = match &*b_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<f16>()?,
                    _ => candle_core::bail!("Expected CUDA storage for bias"),
                };
                let (ptr, _guard) = slice_ptr(b_s, 0);
                ptr as *const f16
            } else {
                std::ptr::null()
            };

            unsafe {
                if use_fused {
                    if ffi::HAVE_MXFP4_WMMA_KERNELS {
                        ffi::launch_mxfp4_moe_grouped_gemm_wmma_f16(
                            input_ptr as *const f16,
                            weight_ptr as *const u8,
                            scale_ptr as *const u8,
                            bias_ptr,
                            indices_ptr as *const u32,
                            output_ptr as *mut f16,
                            num_tokens as i32,
                            topk as i32,
                            num_experts as i32,
                            n as i32,
                            k as i32,
                            has_bias,
                            input_has_topk_dim,
                            dev.cuda_stream().cu_stream(),
                        );
                    } else {
                        ffi::launch_mxfp4_moe_grouped_gemm_f16(
                            input_ptr as *const f16,
                            weight_ptr as *const u8,
                            scale_ptr as *const u8,
                            bias_ptr,
                            indices_ptr as *const u32,
                            output_ptr as *mut f16,
                            num_tokens as i32,
                            topk as i32,
                            num_experts as i32,
                            n as i32,
                            k as i32,
                            has_bias,
                            input_has_topk_dim,
                            dev.cuda_stream().cu_stream(),
                        );
                    }
                } else {
                    ffi::launch_mxfp4_indexed_moe_gemm_f16(
                        input_ptr as *const f16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        indices_ptr as *const u32,
                        output_ptr as *mut f16,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            drop(_output_guard);

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                Shape::from((num_tokens, topk, n)),
            )))
        }
        candle_core::DType::BF16 => {
            let output = dev.alloc_zeros::<bf16>(num_tokens * topk * n)?;

            let input_s = match &*input_storage {
                Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                _ => candle_core::bail!("Expected CUDA storage for input"),
            };
            let (input_ptr, _input_guard) = slice_ptr(input_s, input_l.start_offset());
            let (output_ptr, _output_guard) = slice_ptr(&output, 0);

            let bias_ptr = if has_bias {
                let b = bias.unwrap();
                let b_storage = b.storage_and_layout().0;
                let b_s = match &*b_storage {
                    Storage::Cuda(cuda_storage) => cuda_storage.as_cuda_slice::<bf16>()?,
                    _ => candle_core::bail!("Expected CUDA storage for bias"),
                };
                let (ptr, _guard) = slice_ptr(b_s, 0);
                ptr as *const bf16
            } else {
                std::ptr::null()
            };

            unsafe {
                if use_fused {
                    if ffi::HAVE_MXFP4_WMMA_KERNELS {
                        ffi::launch_mxfp4_moe_grouped_gemm_wmma_bf16(
                            input_ptr as *const bf16,
                            weight_ptr as *const u8,
                            scale_ptr as *const u8,
                            bias_ptr,
                            indices_ptr as *const u32,
                            output_ptr as *mut bf16,
                            num_tokens as i32,
                            topk as i32,
                            num_experts as i32,
                            n as i32,
                            k as i32,
                            has_bias,
                            input_has_topk_dim,
                            dev.cuda_stream().cu_stream(),
                        );
                    } else {
                        ffi::launch_mxfp4_moe_grouped_gemm_bf16(
                            input_ptr as *const bf16,
                            weight_ptr as *const u8,
                            scale_ptr as *const u8,
                            bias_ptr,
                            indices_ptr as *const u32,
                            output_ptr as *mut bf16,
                            num_tokens as i32,
                            topk as i32,
                            num_experts as i32,
                            n as i32,
                            k as i32,
                            has_bias,
                            input_has_topk_dim,
                            dev.cuda_stream().cu_stream(),
                        );
                    }
                } else {
                    ffi::launch_mxfp4_indexed_moe_gemm_bf16(
                        input_ptr as *const bf16,
                        weight_ptr as *const u8,
                        scale_ptr as *const u8,
                        bias_ptr,
                        indices_ptr as *const u32,
                        output_ptr as *mut bf16,
                        num_tokens as i32,
                        topk as i32,
                        num_experts as i32,
                        n as i32,
                        k as i32,
                        has_bias,
                        input_has_topk_dim,
                        dev.cuda_stream().cu_stream(),
                    );
                }
            }
            drop(_output_guard);

            let output_storage = CudaStorage::wrap_cuda_slice(output, dev.clone());
            Ok(Tensor::from((
                Storage::Cuda(output_storage),
                Shape::from((num_tokens, topk, n)),
            )))
        }
        _ => candle_core::bail!(
            "Unsupported dtype for MXFP4 indexed MoE GEMM: {:?}",
            input.dtype()
        ),
    }
}
