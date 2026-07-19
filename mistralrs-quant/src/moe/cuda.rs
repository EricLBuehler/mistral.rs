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

        pub fn launch_silu_and_mul_bf16(
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

        pub fn launch_hunyuan_moe_capacity_mask(
            ids: *const c_void,
            weights: *const c_void,
            masked_weights: *mut c_void,
            n_tokens: i32,
            n_experts: i32,
            top_k: i32,
            expert_capacity: i32,
            stream: CUstream,
        );
    }
}

/// Applies HunYuan's official per-expert capacity rule to top-k routing weights.
pub fn hunyuan_moe_apply_capacity_mask(
    topk_ids: &Tensor,
    topk_weights: &Tensor,
    num_experts: usize,
    top_k: usize,
) -> Result<Tensor> {
    if !topk_ids.device().is_cuda() || !topk_weights.device().is_cuda() {
        candle_core::bail!("hunyuan_moe_apply_capacity_mask requires CUDA tensors");
    }
    if topk_ids.dtype() != DType::U32 {
        candle_core::bail!("hunyuan_moe_apply_capacity_mask topk_ids must be U32");
    }
    if topk_weights.dtype() != DType::F32 {
        candle_core::bail!("hunyuan_moe_apply_capacity_mask topk_weights must be F32");
    }
    if top_k == 0 || num_experts == 0 {
        candle_core::bail!("hunyuan_moe_apply_capacity_mask got empty routing config");
    }
    if topk_ids.shape() != topk_weights.shape() {
        candle_core::bail!("hunyuan_moe_apply_capacity_mask ids/weights shape mismatch");
    }
    let dims = topk_ids.dims();
    if dims.last().copied() != Some(top_k) {
        candle_core::bail!(
            "hunyuan_moe_apply_capacity_mask expected last dim top_k={top_k}, got {:?}",
            dims.last()
        );
    }

    let n_tokens = topk_ids.elem_count() / top_k;
    // Top-k cannot select the same expert twice, so this case cannot overflow.
    if n_tokens <= top_k {
        return Ok(topk_weights.clone());
    }
    let expert_capacity = top_k.max(top_k * n_tokens / num_experts);
    let ids = topk_ids.contiguous()?;
    let weights = topk_weights.contiguous()?;

    let (ids_storage, ids_layout) = ids.storage_and_layout();
    let ids_slice = match &*ids_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("hunyuan_moe_apply_capacity_mask requires CUDA ids"),
    };
    let (weights_storage, weights_layout) = weights.storage_and_layout();
    let weights_slice = match &*weights_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("hunyuan_moe_apply_capacity_mask requires CUDA weights"),
    };

    let dev = topk_weights.device().as_cuda_device()?;
    let mut out = unsafe { dev.alloc::<f32>(topk_weights.elem_count()) }?;
    let stream = dev.cuda_stream();
    let cu_stream = stream.cu_stream();
    let (ids_ptr, _ids_guard) = slice_ptr_on_stream(ids_slice, ids_layout.start_offset(), &stream);
    let (weights_ptr, _weights_guard) =
        slice_ptr_on_stream(weights_slice, weights_layout.start_offset(), &stream);
    let (out_ptr, out_guard) = slice_ptr_mut_on_stream(&mut out, 0, &stream);

    unsafe {
        ffi::launch_hunyuan_moe_capacity_mask(
            ids_ptr as *const core::ffi::c_void,
            weights_ptr as *const core::ffi::c_void,
            out_ptr as *mut core::ffi::c_void,
            n_tokens as i32,
            num_experts as i32,
            top_k as i32,
            expert_capacity as i32,
            cu_stream,
        );
    }
    drop(out_guard);

    let storage = candle_core::CudaStorage::wrap_cuda_slice(out, dev.clone());
    Ok(Tensor::from((
        Storage::Cuda(storage),
        candle_core::Shape::from_dims(topk_weights.dims()),
    )))
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

/// Gated activation kind for the fused act-and-mul kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GatedAct {
    GeluTanh,
    Silu,
}

/// Fused act(gate) * up: input [num_tokens, 2*d] -> [num_tokens, d] bf16.
pub fn act_and_mul(input: &Tensor, d: usize, act: GatedAct, dev: &CudaDevice) -> Result<Tensor> {
    match act {
        GatedAct::GeluTanh => gelu_tanh_and_mul(input, d, dev),
        GatedAct::Silu => silu_and_mul(input, d, dev),
    }
}

/// Fused SiLU(gate) * up: input [num_tokens, 2*d] -> [num_tokens, d] bf16.
pub fn silu_and_mul(input: &Tensor, d: usize, dev: &CudaDevice) -> Result<Tensor> {
    let (num_tokens, two_d) = input.dims2()?;
    assert_eq!(two_d, 2 * d, "silu_and_mul expects last dim == 2*d");
    assert_eq!(input.dtype(), DType::BF16, "silu_and_mul is bf16-only");

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
        ffi::launch_silu_and_mul_bf16(
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

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    #[test]
    fn test_hunyuan_moe_capacity_mask_cuda() -> candle_core::Result<()> {
        use super::hunyuan_moe_apply_capacity_mask;
        use candle_core::{Device, Tensor};

        let device = Device::new_cuda(0)?;
        let ids = Tensor::new(
            vec![
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
                vec![0u32, 1u32],
            ],
            &device,
        )?;
        let weights = Tensor::new(
            vec![
                vec![1f32, 2f32],
                vec![3f32, 4f32],
                vec![5f32, 6f32],
                vec![7f32, 8f32],
            ],
            &device,
        )?;

        let masked = hunyuan_moe_apply_capacity_mask(&ids, &weights, 4, 2)?;
        assert_eq!(
            masked.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            vec![
                vec![1f32, 2f32],
                vec![3f32, 4f32],
                vec![0f32, 0f32],
                vec![0f32, 0f32],
            ]
        );

        let decode_ids = Tensor::new(vec![vec![3u32, 1u32]], &device)?;
        let decode_weights = Tensor::new(vec![vec![0.25f32, 0.75f32]], &device)?;
        let decode_masked = hunyuan_moe_apply_capacity_mask(&decode_ids, &decode_weights, 4, 2)?;
        assert_eq!(
            decode_masked.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            vec![vec![0.25f32, 0.75f32]]
        );

        let top1_ids = Tensor::new(
            vec![vec![0u32], vec![0u32], vec![0u32], vec![0u32]],
            &device,
        )?;
        let top1_weights = Tensor::ones((4, 1), candle_core::DType::F32, &device)?;
        let top1_masked = hunyuan_moe_apply_capacity_mask(&top1_ids, &top1_weights, 4, 1)?;
        assert_eq!(
            top1_masked.to_device(&Device::Cpu)?.to_vec2::<f32>()?,
            vec![vec![1f32], vec![0f32], vec![0f32], vec![0f32]]
        );

        Ok(())
    }
}
