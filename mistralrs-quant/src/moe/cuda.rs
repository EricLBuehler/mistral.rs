//! Plain-CUDA MoE ops (kernels in `kernels/moe/*.cu`): token alignment, fused GeLU-tanh + multiply, and cross-expert sum.

use candle_core::cuda::cudarc::driver::CudaSlice;
use candle_core::{CudaDevice, DType, Result, Storage, Tensor};
use half::bf16;

use crate::utils::{slice_ptr_mut_on_stream, slice_ptr_on_stream};

mod ffi {
    use candle_core::cuda::cudarc::driver::sys::CUstream;
    use core::ffi::c_void;

    extern "C" {
        pub fn launch_moe_align(
            topk_ids: *const i32,
            sorted_token_ids: *mut i32,
            expert_ids: *mut i32,
            num_tokens_post_pad: *mut i32,
            cumsum: *mut i32,
            num_experts: i32,
            block_size: i32,
            numel: i32,
            max_num_tokens_padded: i32,
            stream: CUstream,
        );

        pub fn launch_gelu_tanh_and_mul_bf16(
            out: *mut c_void,
            input: *const c_void,
            num_tokens: i32,
            d: i32,
            stream: CUstream,
        );

        pub fn launch_moe_sum_bf16(
            out: *mut c_void,
            input: *const c_void,
            num_tokens: i32,
            hidden: i32,
            topk: i32,
            stream: CUstream,
        );
    }
}

/// Padded length (EM) of `sorted_token_ids`.
pub fn moe_align_em(
    num_tokens: usize,
    topk: usize,
    num_experts: usize,
    block_size: usize,
) -> usize {
    let numel = num_tokens * topk;
    let em = numel + num_experts * (block_size - 1);
    if numel < num_experts {
        (numel * block_size).min(em)
    } else {
        em
    }
}

/// Aligns tokens into per-expert blocks. Returns (sorted_token_ids[EM], expert_ids[nblocks], num_tokens_post_pad[1], EM); topk_ids is a contiguous u32 slice of [num_tokens*topk] expert ids (< 2^31).
#[allow(clippy::type_complexity)]
pub fn moe_align(
    topk_ids_u32: &CudaSlice<u32>,
    num_tokens: usize,
    num_experts: usize,
    topk: usize,
    block_size: i32,
    dev: &CudaDevice,
) -> Result<(CudaSlice<i32>, CudaSlice<i32>, CudaSlice<i32>, usize)> {
    let numel = num_tokens * topk;
    let bs = block_size as usize;
    let em = moe_align_em(num_tokens, topk, num_experts, bs);
    let nblocks = em.div_ceil(bs);

    let mut sids = unsafe { dev.alloc::<i32>(em)? };
    let mut eids = unsafe { dev.alloc::<i32>(nblocks)? };
    let mut ntpp = unsafe { dev.alloc::<i32>(1)? };
    let mut cumsum = unsafe { dev.alloc::<i32>(num_experts + 1)? };

    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();
    let (tk_ptr, _tk_guard) = slice_ptr_on_stream(topk_ids_u32, 0, &stream);
    let (s_ptr, s_guard) = slice_ptr_mut_on_stream(&mut sids, 0, &stream);
    let (e_ptr, e_guard) = slice_ptr_mut_on_stream(&mut eids, 0, &stream);
    let (n_ptr, n_guard) = slice_ptr_mut_on_stream(&mut ntpp, 0, &stream);
    let (c_ptr, c_guard) = slice_ptr_mut_on_stream(&mut cumsum, 0, &stream);
    unsafe {
        ffi::launch_moe_align(
            tk_ptr as *const i32,
            s_ptr as *mut i32,
            e_ptr as *mut i32,
            n_ptr as *mut i32,
            c_ptr as *mut i32,
            num_experts as i32,
            block_size,
            numel as i32,
            em as i32,
            cu_stream,
        );
    }
    drop((s_guard, e_guard, n_guard, c_guard));
    Ok((sids, eids, ntpp, em))
}

/// Fused GeLU-tanh(gate) * up: input [num_tokens, 2*d] -> [num_tokens, d] bf16.
pub fn gelu_tanh_and_mul(input: &Tensor, d: usize, dev: &CudaDevice) -> Result<Tensor> {
    let (num_tokens, two_d) = input.dims2()?;
    assert_eq!(two_d, 2 * d, "gelu_tanh_and_mul expects last dim == 2*d");
    assert_eq!(input.dtype(), DType::BF16, "cutile gelu path is bf16-only");

    let mut out = unsafe { dev.alloc::<bf16>(num_tokens * d)? };
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_slice = match &*in_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("input must be cuda"),
    };
    let (in_ptr, _in_guard) = slice_ptr_on_stream(in_slice, in_layout.start_offset(), &stream);
    let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    unsafe {
        ffi::launch_gelu_tanh_and_mul_bf16(
            out_ptr as *mut core::ffi::c_void,
            in_ptr as *const core::ffi::c_void,
            num_tokens as i32,
            d as i32,
            cu_stream,
        );
    }
    drop(out_guard);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_tokens, d))))
}

pub fn moe_sum_bf16(
    input: &Tensor,
    num_tokens: usize,
    topk: usize,
    dev: &CudaDevice,
) -> Result<Tensor> {
    let (total_assignments, hidden) = input.dims2()?;
    assert_eq!(
        total_assignments,
        num_tokens * topk,
        "moe_sum_bf16 input rows mismatch"
    );
    assert_eq!(input.dtype(), DType::BF16, "moe_sum_bf16 is bf16-only");

    let mut out = unsafe { dev.alloc::<bf16>(num_tokens * hidden)? };
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();

    let (in_storage, in_layout) = input.storage_and_layout();
    let in_slice = match &*in_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<bf16>()?,
        _ => candle_core::bail!("input must be cuda"),
    };
    let (in_ptr, _in_guard) = slice_ptr_on_stream(in_slice, in_layout.start_offset(), &stream);
    let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);
    unsafe {
        ffi::launch_moe_sum_bf16(
            out_ptr as *mut core::ffi::c_void,
            in_ptr as *const core::ffi::c_void,
            num_tokens as i32,
            hidden as i32,
            topk as i32,
            cu_stream,
        );
    }
    drop(out_guard);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((Storage::Cuda(storage), (num_tokens, hidden))))
}
