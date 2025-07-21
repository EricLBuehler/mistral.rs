#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod cuda {
    use candle_core::cuda::cudarc::driver::DeviceSlice;
    use candle_core::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core::{DType, Result, Storage, Tensor};
    use half::{bf16, f16};
    use std::cmp::min;

    use crate::moe::ffi;

    pub fn apply_topk_softmax_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        gating_output: &Tensor,
        topk_weight: &Tensor,
        topk_indices: &Tensor,
        token_expert_indices: &Tensor,
    ) -> Result<()> {
        let (g, g_l) = gating_output.storage_and_layout();
        let g: &candle_core::CudaStorage = match &*g {
            Storage::Cuda(g) => g,
            _ => candle_core::bail!("gating_output must be a cuda tensor"),
        };

        let (w, w_l) = topk_weight.storage_and_layout();
        let w = match &*w {
            Storage::Cuda(w) => w,
            _ => candle_core::bail!("topk_weight must be a cuda tensor"),
        };

        let (i, i_l) = topk_indices.storage_and_layout();
        let i = match &*i {
            Storage::Cuda(i) => i,
            _ => candle_core::bail!("topk_indices must be a cuda tensor"),
        };

        let (ei, ei_l) = token_expert_indices.storage_and_layout();
        let ei: &candle_core::CudaStorage = match &*ei {
            Storage::Cuda(ei) => ei,
            _ => candle_core::bail!("token_expert_indices must be a cuda tensor"),
        };

        let g_rank = g_l.stride().len();
        let w_rank = w_l.stride().len();
        let i_rank = i_l.stride().len();
        let ei_rank = ei_l.stride().len();

        if g_rank != 2 || w_rank != 2 || i_rank != 2 || ei_rank != 2 {
            candle_core::bail!(
            "apply_topk_softmax_inplace expects input tensors of rank 2 (w: {w_l:?}, i: {i_l:?}, ei: {ei_l:?}, g: {g_l:?})"
        )
        }

        // Get cuda slices for all tensors
        let g = g.as_cuda_slice::<T>()?;
        let w = w.as_cuda_slice::<T>()?;
        let i = i.as_cuda_slice::<u32>()?;
        let ei = ei.as_cuda_slice::<u32>()?;

        // Get cuda views for all tensors
        let g = g.slice(g_l.start_offset()..);
        let w = w.slice(w_l.start_offset()..);
        let i = i.slice(i_l.start_offset()..);
        let ei = ei.slice(ei_l.start_offset()..);

        let (num_tokens, top_k) = w_l.shape().dims2()?;
        let (_, num_experts) = g_l.shape().dims2()?;

        let is_pow2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
        if !is_pow2 || num_experts > 256 {
            candle_core::bail!(
            "num_experts should be power of 2 and smaller than 256 (num_experts: {num_experts:?})"
        )
        }

        if (num_tokens, top_k) != i_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch topk_indices {:?}, expected {:?}",
                i_l.shape(),
                (num_tokens, top_k)
            )
        }

        if (num_tokens, top_k) != ei_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch token_expert_indices {:?}, expected {:?}",
                ei_l.shape(),
                (num_tokens, top_k)
            )
        }

        let (gate_ptr, _gate_guard) = g.device_ptr(g.stream());
        let (weight_ptr, _weight_guard) = w.device_ptr(w.stream());
        let (indices_ptr, _indices_guard) = i.device_ptr(i.stream());
        let (expert_indices_ptr, _expert_indices_guard) = ei.device_ptr(ei.stream());

        unsafe {
            ffi::topk_softmax(
                gate_ptr as *const core::ffi::c_void,
                weight_ptr as *const core::ffi::c_void,
                indices_ptr as *const core::ffi::c_void,
                expert_indices_ptr as *const core::ffi::c_void,
                num_experts as i32,
                num_tokens as i64,
                top_k as i32,
            )
        }

        Ok(())
    }

    pub fn apply_topk_softmax_inplace(
        gating_output: &Tensor,
        topk_weight: &Tensor,
        topk_indices: &Tensor,
        token_expert_indices: &Tensor,
    ) -> Result<()> {
        match topk_weight.dtype() {
            DType::F16 => apply_topk_softmax_::<f16>(
                gating_output,
                topk_weight,
                topk_indices,
                token_expert_indices,
            ),
            DType::BF16 => apply_topk_softmax_::<bf16>(
                gating_output,
                topk_weight,
                topk_indices,
                token_expert_indices,
            ),
            DType::F32 => apply_topk_softmax_::<f32>(
                gating_output,
                topk_weight,
                topk_indices,
                token_expert_indices,
            ),
            dt => {
                candle_core::bail!(
                    "apply_topk_softmax_inplace is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }

    pub fn apply_moe_sum_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        output: &Tensor,
        num_token: usize,
        topk: usize,
        dtype: u32,
    ) -> Result<()> {
        let (i, i_l) = input.storage_and_layout();
        let i: &candle_core::CudaStorage = match &*i {
            Storage::Cuda(i) => i,
            _ => candle_core::bail!("input must be a cuda tensor"),
        };

        let (o, o_l) = output.storage_and_layout();
        let o: &candle_core::CudaStorage = match &*o {
            Storage::Cuda(o) => o,
            _ => candle_core::bail!("output must be a cuda tensor"),
        };

        let i_rank = i_l.stride().len();
        let o_rank = o_l.stride().len();

        if i_rank != 3 {
            candle_core::bail!("input should be rank 3 (input: {i_l:?})")
        }

        if o_rank != 2 {
            candle_core::bail!("output should be rank 2 (input: {o_l:?})")
        }

        // Get cuda slices for all tensors
        let i = i.as_cuda_slice::<T>()?;
        let o = o.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let i = i.slice(i_l.start_offset()..);
        let o = o.slice(o_l.start_offset()..);

        let (num_tokens, _, hidden_size) = i_l.shape().dims3()?;

        if (num_tokens, hidden_size) != o_l.shape().dims2()? {
            candle_core::bail!(
                "shape mismatch output {:?}, expected {:?}",
                o_l.shape(),
                (num_tokens, hidden_size)
            )
        }

        let (input_ptr, _input_guard) = i.device_ptr(i.stream());
        let (output_ptr, _output_guard) = o.device_ptr(o.stream());

        unsafe {
            ffi::moe_sum(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *const core::ffi::c_void,
                hidden_size as i32,
                num_token as i64,
                topk as i32,
                dtype,
            )
        }

        Ok(())
    }

    pub fn apply_moe_sum_inplace(
        input: &Tensor,
        output: &Tensor,
        num_token: usize,
        topk: usize,
        dtype: u32,
    ) -> Result<()> {
        match input.dtype() {
            DType::F16 => apply_moe_sum_::<f16>(input, output, num_token, topk, dtype),
            DType::BF16 => apply_moe_sum_::<bf16>(input, output, num_token, topk, dtype),
            DType::F32 => apply_moe_sum_::<f32>(input, output, num_token, topk, dtype),
            dt => {
                candle_core::bail!(
                    "apply_moe_sum_inplace is only supported for f32, f16 and bf16 ({dt:?})"
                )
            }
        }
    }

    fn get_moe_wna16_config(
        num_valid_tokens: usize,
        size_n: usize,
        size_k: usize,
        block_size_m: usize,
        group_size: usize,
        num_experts: usize,
        top_k: usize,
    ) -> (usize, usize) {
        let mut block_size_n = 128;
        let mut block_size_k = 128;

        if block_size_k <= group_size {
            block_size_k = group_size;
        }

        let num_n_blocks = size_k / block_size_k;
        let num_k_blocks = size_n / block_size_k;
        let mut num_m_blocks = num_valid_tokens.div_ceil(block_size_m) + num_experts;

        if num_valid_tokens / top_k <= block_size_m {
            num_m_blocks = min(num_m_blocks, num_valid_tokens)
        }

        let mut num_blocks = num_m_blocks * num_n_blocks * num_k_blocks;

        if size_k % 256 == 0 && num_blocks >= 256 && block_size_k < 256 {
            block_size_k = 256;
            num_blocks /= 256 / block_size_k;
        }

        if num_m_blocks <= 16
            && size_k % (block_size_k * 2) == 0
            && block_size_k <= 512
            && num_blocks >= 512
        {
            block_size_k *= 2;
            num_blocks /= 2;
        }

        if num_blocks > 1024 {
            block_size_n = 256;
            num_blocks /= 2;
        }

        if size_n <= 1024 && num_blocks >= 1024 {
            block_size_n = 1024
        }

        (block_size_n, block_size_k)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_moe_wna16_gemm_<
        T: candle_core::cuda_backend::CudaDType
            + candle_core::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        output: &Tensor,
        b_qweight: &Tensor,
        b_scales: &Tensor,
        b_qzeros: &Tensor,
        topk_weights: &Tensor,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        num_tokens_post_pad: &Tensor,
        top_k: usize,
        bit: usize,
        dtype: u32,
    ) -> Result<()> {
        let (i, i_l) = input.storage_and_layout();
        let i: &candle_core::CudaStorage = match &*i {
            Storage::Cuda(i) => i,
            _ => candle_core::bail!("input must be a cuda tensor"),
        };

        let (o, o_l) = output.storage_and_layout();
        let o: &candle_core::CudaStorage = match &*o {
            Storage::Cuda(o) => o,
            _ => candle_core::bail!("output must be a cuda tensor"),
        };

        let (qw, qw_l) = b_qweight.storage_and_layout();
        let qw: &candle_core::CudaStorage = match &*qw {
            Storage::Cuda(qw) => qw,
            _ => candle_core::bail!("b_qweight must be a cuda tensor"),
        };

        let (s, s_l) = b_scales.storage_and_layout();
        let s: &candle_core::CudaStorage = match &*s {
            Storage::Cuda(s) => s,
            _ => candle_core::bail!("b_scales must be a cuda tensor"),
        };

        let (qz, qz_l) = b_qzeros.storage_and_layout();
        let qz: &candle_core::CudaStorage = match &*qz {
            Storage::Cuda(qz) => qz,
            _ => candle_core::bail!("b_qzeros must be a cuda tensor"),
        };

        let (tw, tw_l) = topk_weights.storage_and_layout();
        let tw: &candle_core::CudaStorage = match &*tw {
            Storage::Cuda(tw) => tw,
            _ => candle_core::bail!("topk_weights must be a cuda tensor"),
        };

        let (sti, sti_l) = sorted_token_ids.storage_and_layout();
        let sti: &candle_core::CudaStorage = match &*sti {
            Storage::Cuda(sti) => sti,
            _ => candle_core::bail!("sorted_token_ids must be a cuda tensor"),
        };

        let (ei, ei_l) = expert_ids.storage_and_layout();
        let ei: &candle_core::CudaStorage = match &*ei {
            Storage::Cuda(ei) => ei,
            _ => candle_core::bail!("expert_ids must be a cuda tensor"),
        };

        let (nt, nt_l) = num_tokens_post_pad.storage_and_layout();
        let nt: &candle_core::CudaStorage = match &*nt {
            Storage::Cuda(nt) => nt,
            _ => candle_core::bail!("num_tokens_post_pad must be a cuda tensor"),
        };

        // Get cuda slices for all tensors
        let i = i.as_cuda_slice::<T>()?;
        let o = o.as_cuda_slice::<T>()?;
        let qw = qw.as_cuda_slice::<T>()?;
        let s = s.as_cuda_slice::<T>()?;

        // Get cuda views for all tensors
        let i = i.slice(i_l.start_offset()..);
        let o = o.slice(o_l.start_offset()..);
        let qw = qw.slice(qw_l.start_offset()..);
        let s = s.slice(s_l.start_offset()..);

        let (input_ptr, _input_guard) = i.device_ptr(i.stream());
        let (output_ptr, _output_guard) = o.device_ptr(o.stream());
        let (b_qweight_ptr, _b_qweight_guard) = qw.device_ptr(qw.stream());
        let (b_scales_ptr, _b_scales_guard) = s.device_ptr(s.stream());

        let (size_m, size_k) = input.shape().dims2()?;
        let (num_experts, size_n) = b_qweight.shape().dims2()?;
        let (_, _, x) = b_scales.shape().dims3()?;
        let group_size = size_k / x;
        let (em, _) = sorted_token_ids.shape().dims2()?;
        let block_size_m = 64;
        let num_token = size_m * top_k;

        let (block_size_n, block_size_k) = get_moe_wna16_config(
            num_token,
            size_n,
            size_k,
            block_size_m,
            group_size,
            num_experts,
            top_k,
        );

        unsafe {
            ffi::moe_wna16_gemm(
                input_ptr as *const core::ffi::c_void,
                output_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                b_scales_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                b_qweight_ptr as *const core::ffi::c_void,
                top_k as i64,
                block_size_m as i64,
                block_size_n as i64,
                block_size_k as i64,
                bit as i64,
                num_experts as i32,
                size_m as i32,
                size_n as i32,
                size_k as i32,
                group_size as i32,
                em as i64,
                false,
                false,
                dtype,
            )
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn apply_moe_wna16_gemm_inplace(
        input: &Tensor,
        output: &Tensor,
        b_qweight: &Tensor,
        b_scales: &Tensor,
        b_qzeros: &Tensor,
        topk_weights: &Tensor,
        sorted_token_ids: &Tensor,
        expert_ids: &Tensor,
        num_tokens_post_pad: &Tensor,
        top_k: usize,
        bit: usize,
        dtype: u32,
    ) -> Result<()> {
        match input.dtype() {
            DType::F16 => apply_moe_wna16_gemm_::<f16>(
                input,
                output,
                b_qweight,
                b_scales,
                b_qzeros,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_pad,
                top_k,
                bit,
                dtype,
            ),
            DType::BF16 => apply_moe_wna16_gemm_::<f16>(
                input,
                output,
                b_qweight,
                b_scales,
                b_qzeros,
                topk_weights,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_pad,
                top_k,
                bit,
                dtype,
            ),
            dt => {
                candle_core::bail!(
                    "apply_moe_wna16_gemm_inplace is only supported for f16 and bf16 ({dt:?})"
                )
            }
        }
    }
}
