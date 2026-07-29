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

        pub fn moe_gemm_wmma_wna16(
            input: *const c_void,
            weights: *const u32,
            weight_scales: *const c_void,
            sorted_token_ids: *const i32,
            expert_ids: *const i32,
            topk_weights: *const f32,
            output: *mut c_void,
            expert_counts: *mut i32,
            expert_offsets: *mut i32,
            num_experts: i32,
            topk: i32,
            size_m: i32,
            size_n: i32,
            size_k: i32,
            bits: i32,
            group_size: i32,
            zero_point: i32,
            data_type: i32,
            is_prefill: bool,
            stream: i64,
        );

        pub fn moe_gemv_wna16(
            input: *const c_void,
            weights: *const u32,
            weight_scales: *const c_void,
            sorted_token_ids: *const i32,
            expert_ids: *const i32,
            topk_weights: *const f32,
            output: *mut c_void,
            num_experts: i32,
            topk: i32,
            size_m: i32,
            size_n: i32,
            size_k: i32,
            bits: i32,
            group_size: i32,
            zero_point: i32,
            data_type: i32,
            stream: i64,
        );
    }
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn moe_gemm_wna16(
    input: &Tensor,
    weights: &Tensor,
    weight_scales: &Tensor,
    topk_weights: Option<&Tensor>,
    sorted_token_ids: &Tensor,
    expert_ids: &Tensor,
    topk: usize,
    bits: usize,
    group_size: usize,
    is_prefill: bool,
    zero_point: usize,
) -> Result<Tensor> {
    use candle_core::cuda::cudarc::driver::{CudaStream, DevicePtrMut, SyncOnDrop};
    use core::ffi::c_void;
    let dev = input.device().as_cuda_device()?;

    fn tensor_ptr<'a>(
        storage: &'a candle_core::CudaStorage,
        dtype: candle_core::DType,
        offset: usize,
        stream: &'a CudaStream,
    ) -> Result<(u64, SyncOnDrop<'a>)> {
        let (ptr, guard) = match dtype {
            candle_core::DType::F16 => {
                let slice = storage.as_cuda_slice::<half::f16>()?;
                slice_ptr_on_stream(slice, offset, stream)
            }
            candle_core::DType::BF16 => {
                let slice = storage.as_cuda_slice::<half::bf16>()?;
                slice_ptr_on_stream(slice, offset, stream)
            }
            candle_core::DType::F32 => {
                let slice = storage.as_cuda_slice::<f32>()?;
                slice_ptr_on_stream(slice, offset, stream)
            }
            candle_core::DType::U32 => {
                let slice = storage.as_cuda_slice::<u32>()?;
                slice_ptr_on_stream(slice, offset, stream)
            }
            candle_core::DType::I32 => {
                let slice = storage.as_cuda_slice::<i32>()?;
                slice_ptr_on_stream(slice, offset, stream)
            }
            dtype => candle_core::bail!("unsupported WNA16 tensor dtype {dtype:?}"),
        };
        Ok((ptr, guard))
    }

    fn cuda_fwd<
        T: candle_core::cuda_backend::CudaDType + candle_core::cuda::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        weight_scales: &Tensor,
        topk_weights: Option<&Tensor>,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        topk: usize,
        bits: usize,
        group_size: usize,
        is_prefill: bool,
        zero_point: usize,
        dev: &candle_core::CudaDevice,
    ) -> Result<Tensor> {
        let (input_rows, size_k) = input.dims2()?;
        let (num_experts, size_n, packed_k) = weights.dims3()?;
        if num_experts > 1024 {
            candle_core::bail!("WNA16 MoE supports at most 1024 experts");
        }
        let output_rows = if topk_weights.is_some() {
            input_rows
        } else {
            input_rows * topk
        };
        let expected_packed_k = (size_k + (32 / bits) - 1) / (32 / bits);
        if !matches!(bits, 4 | 8) || packed_k != expected_packed_k {
            candle_core::bail!("invalid WNA16 packed shape or bit width");
        }
        if group_size == 0 || !size_k.is_multiple_of(group_size) {
            candle_core::bail!("WNA16 requires K divisible by group_size");
        }
        let expected_scale_shape = (num_experts, size_n, size_k / group_size);
        if weight_scales.dims3()? != expected_scale_shape {
            candle_core::bail!(
                "invalid WNA16 scale shape: expected {:?}, got {:?}",
                expected_scale_shape,
                weight_scales.dims()
            );
        }
        if weight_scales.dtype() != candle_core::DType::F32 {
            candle_core::bail!("WNA16 scales must be F32, got {:?}", weight_scales.dtype());
        }
        let data_type = match input.dtype() {
            candle_core::DType::F16 => 0,
            candle_core::DType::BF16 => 1,
            dtype => candle_core::bail!("WNA16 only supports F16/BF16 input, got {dtype:?}"),
        };
        let stream = dev.cuda_stream();
        let (input_storage, input_layout) = input.storage_and_layout();
        let (weights_storage, weights_layout) = weights.storage_and_layout();
        let (scales_storage, scales_layout) = weight_scales.storage_and_layout();
        let (sorted_storage, sorted_layout) = sorted_token_ids.storage_and_layout();
        let (experts_storage, experts_layout) = expert_ids.storage_and_layout();
        let Storage::Cuda(input_storage) = &*input_storage else {
            candle_core::bail!("WNA16 input must be on CUDA")
        };
        let Storage::Cuda(weights_storage) = &*weights_storage else {
            candle_core::bail!("WNA16 weights must be on CUDA")
        };
        let Storage::Cuda(scales_storage) = &*scales_storage else {
            candle_core::bail!("WNA16 scales must be on CUDA")
        };
        let Storage::Cuda(sorted_storage) = &*sorted_storage else {
            candle_core::bail!("WNA16 sorted ids must be on CUDA")
        };
        let Storage::Cuda(experts_storage) = &*experts_storage else {
            candle_core::bail!("WNA16 expert ids must be on CUDA")
        };
        let (input_ptr, _input_guard) = tensor_ptr(
            input_storage,
            input.dtype(),
            input_layout.start_offset(),
            &stream,
        )?;
        let (weights_ptr, _weights_guard) = tensor_ptr(
            weights_storage,
            weights.dtype(),
            weights_layout.start_offset(),
            &stream,
        )?;
        let (scales_ptr, _scales_guard) = tensor_ptr(
            scales_storage,
            weight_scales.dtype(),
            scales_layout.start_offset(),
            &stream,
        )?;
        let (sorted_ptr, _sorted_guard) = tensor_ptr(
            sorted_storage,
            sorted_token_ids.dtype(),
            sorted_layout.start_offset(),
            &stream,
        )?;
        let (experts_ptr, _experts_guard) = tensor_ptr(
            experts_storage,
            expert_ids.dtype(),
            experts_layout.start_offset(),
            &stream,
        )?;
        let topk_storage = topk_weights.map(Tensor::storage_and_layout);
        let (topk_ptr, _topk_guard) =
            if let Some((topk_storage, topk_layout)) = topk_storage.as_ref() {
                let topk_weights = topk_weights.expect("top-k storage without top-k tensor");
                let Storage::Cuda(topk_storage) = &**topk_storage else {
                    candle_core::bail!("WNA16 top-k weights must be on CUDA")
                };
                let (ptr, guard) = tensor_ptr(
                    topk_storage,
                    topk_weights.dtype(),
                    topk_layout.start_offset(),
                    &stream,
                )?;
                (ptr as *const f32, Some(guard))
            } else {
                (std::ptr::null(), None)
            };
        let (input_ptr, weights_ptr, scales_ptr, sorted_ptr, experts_ptr) = (
            input_ptr as *const c_void,
            weights_ptr as *const u32,
            scales_ptr as *const c_void,
            sorted_ptr as *const i32,
            experts_ptr as *const i32,
        );
        let stream_id = stream.cu_stream() as i64;
        let mut output = unsafe { dev.alloc::<T>(output_rows * size_n) }?;
        let (output_ptr, _output_guard) = output.device_ptr_mut(&stream);

        if is_prefill {
            let mut counts = unsafe { dev.alloc::<i32>(num_experts) }?;
            let mut offsets = unsafe { dev.alloc::<i32>(num_experts + 1) }?;
            unsafe {
                ffi::moe_gemm_wmma_wna16(
                    input_ptr,
                    weights_ptr,
                    scales_ptr,
                    sorted_ptr,
                    experts_ptr,
                    topk_ptr,
                    output_ptr as *mut c_void,
                    counts.device_ptr_mut(&stream).0 as *mut i32,
                    offsets.device_ptr_mut(&stream).0 as *mut i32,
                    num_experts as i32,
                    topk as i32,
                    output_rows as i32,
                    size_n as i32,
                    size_k as i32,
                    bits as i32,
                    group_size as i32,
                    zero_point as i32,
                    data_type,
                    true,
                    stream_id,
                );
            }
        } else {
            unsafe {
                ffi::moe_gemv_wna16(
                    input_ptr,
                    weights_ptr,
                    scales_ptr,
                    sorted_ptr,
                    experts_ptr,
                    topk_ptr,
                    output_ptr as *mut c_void,
                    num_experts as i32,
                    topk as i32,
                    output_rows as i32,
                    size_n as i32,
                    size_k as i32,
                    bits as i32,
                    group_size as i32,
                    zero_point as i32,
                    data_type,
                    stream_id,
                );
            }
        }

        drop(_output_guard);
        let storage = candle_core::CudaStorage::wrap_cuda_slice(output, dev.clone());
        Ok(Tensor::from((
            Storage::Cuda(storage),
            (output_rows, size_n),
        )))
    }

    match input.dtype() {
        candle_core::DType::F16 => cuda_fwd::<half::f16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            topk,
            bits,
            group_size,
            is_prefill,
            zero_point,
            dev,
        ),
        candle_core::DType::BF16 => cuda_fwd::<half::bf16>(
            input,
            weights,
            weight_scales,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            topk,
            bits,
            group_size,
            is_prefill,
            zero_point,
            dev,
        ),
        dtype => candle_core::bail!("WNA16 only supports F16/BF16 input, got {dtype:?}"),
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
    if two_d != 2 * d {
        candle_core::bail!("silu_and_mul expects last dim == 2*d");
    }
    if input.dtype() != DType::BF16 {
        candle_core::bail!("silu_and_mul is bf16-only");
    }

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
    if two_d != 2 * d {
        candle_core::bail!("gelu_tanh_and_mul expects last dim == 2*d");
    }
    if input.dtype() != DType::BF16 {
        candle_core::bail!("cutile gelu path is bf16-only");
    }

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
