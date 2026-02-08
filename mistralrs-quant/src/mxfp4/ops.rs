use candle_core::{CudaStorage, Device, Result, Shape, Storage, Tensor};
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

/// Perform MXFP4 indexed MoE GEMM: for each (token, expert_slot), compute input @ weight[expert].T + bias[expert]
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
