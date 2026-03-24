use candle_core::{Result, Tensor};

#[cfg(feature = "cuda")]
pub fn moe_gemm(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        let (num_experts, size_n, size_k) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} last dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_wmma only accept f16/bf16 inputs!")
            }
        };

        let (input, input_l) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let input_offset = input_l.start_offset();

        let (weights, weights_l) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let weights_offset = weights_l.start_offset();

        let (sorted_token_ids, sti_l) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let sti_offset = sti_l.start_offset();

        let (experts_ids, ei_l) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };
        let ei_offset = ei_l.start_offset();

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, tw_l) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let tw_offset = tw_l.start_offset();
            let topk_w_ptr = topk_weights
                .slice(tw_offset..)
                .device_ptr(topk_weights.stream())
                .0 as *const f32;
            topk_w_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        // Threshold for using GEMV kernel (optimized for small batch sizes)
        const GEMV_THRESHOLD: i32 = 8;

        let num_experts_i32 = i32::try_from(num_experts).expect("num_experts too large for i32");
        let topk_i32 = i32::try_from(topk).expect("topk too large for i32");
        let size_m_i32 = i32::try_from(size_m).expect("size_m too large for i32");
        let size_n_i32 = i32::try_from(size_n).expect("size_n too large for i32");
        let size_k_i32 = i32::try_from(size_k).expect("size_k too large for i32");

        // Select kernel based on prefill/decode and batch size
        // - Prefill (larger batches): use WMMA-based kernel for tensor core acceleration
        // - Decode with small M (<=8): use GEMV kernel optimized for warp reductions
        // - Decode with larger M: use standard moe_gemm kernel
        let moe_func = if is_prefill {
            crate::cuda::ffi::moe_gemm_wmma
        } else if size_m_i32 <= GEMV_THRESHOLD {
            crate::cuda::ffi::moe_gemv
        } else {
            crate::cuda::ffi::moe_gemm
        };

        unsafe {
            moe_func(
                input.slice(input_offset..).device_ptr(input.stream()).0 as *const c_void, // [size_m, size_k]
                weights
                    .slice(weights_offset..)
                    .device_ptr(weights.stream())
                    .0 as *const c_void, // [num_experts, size_n, size_k]
                sorted_token_ids
                    .slice(sti_offset..)
                    .device_ptr(sorted_token_ids.stream())
                    .0 as *const i32,
                experts_ids
                    .slice(ei_offset..)
                    .device_ptr(experts_ids.stream())
                    .0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                num_experts_i32,
                topk_i32,
                size_m_i32,
                size_n_i32,
                size_k_i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                stream,
            );
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from((candle::Storage::Cuda(output), (size_m, size_n)));

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm only accept f16/bf16 inputs!")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn moe_gemm(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm is not implemented on this platform!")
}

/// MoE GEMM for transposed weight layout [num_experts, K, N] instead of [num_experts, N, K].
/// This is used for stacked weight format where gate_up_proj is [E, hidden, inter*2]
/// and down_proj is [E, inter, hidden].
#[cfg(feature = "cuda")]
pub fn moe_gemm_transposed(
    input: &Tensor,
    weights: &Tensor,
    topk_weights: &Option<Tensor>,
    sorted_token_ids: &Tensor,
    experts_ids: &Tensor,
    topk: usize,
    is_prefill: bool,
) -> Result<Tensor> {
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    use candle_core as candle;
    use candle_core::DType;
    use half::{bf16, f16};

    fn cuda_fwd<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        input: &Tensor,
        weights: &Tensor,
        topk_weights: &Option<Tensor>,
        sorted_token_ids: &Tensor,
        experts_ids: &Tensor,
        topk: usize,
        is_prefill: bool,
    ) -> Result<Tensor> {
        let (mut size_m, size_k1) = input.dims2()?;
        if topk_weights.is_none() {
            size_m *= topk;
        }
        // For transposed layout: [num_experts, K, N]
        let (num_experts, size_k, size_n) = weights.dims3()?;
        assert!(
            size_k == size_k1,
            "input {:?} and weight {:?} K dim mismatch!",
            size_k1,
            size_k
        );
        let dev = input.device().as_cuda_device()?;
        let data_type = match input.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            _ => {
                candle_core::bail!("moe_gemm_transposed only accept f16/bf16 inputs!")
            }
        };

        let (input, input_l) = input.storage_and_layout();
        let input = match &*input {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("input must be a cuda tensor"),
        };
        let input_offset = input_l.start_offset();

        let (weights, weights_l) = weights.storage_and_layout();
        let weights = match &*weights {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<T>()?,
            _ => candle::bail!("weight must be a cuda tensor"),
        };
        let weights_offset = weights_l.start_offset();

        let (sorted_token_ids, sti_l) = sorted_token_ids.storage_and_layout();
        let sorted_token_ids = match &*sorted_token_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("sorted_token_ids must be a cuda tensor"),
        };
        let sti_offset = sti_l.start_offset();

        let (experts_ids, ei_l) = experts_ids.storage_and_layout();
        let experts_ids = match &*experts_ids {
            candle::Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
            _ => candle::bail!("experts_ids must be a cuda tensor"),
        };
        let ei_offset = ei_l.start_offset();

        let topk_weights_ptr = if let Some(topk_weights) = &topk_weights {
            let (topk_weights, tw_l) = topk_weights.storage_and_layout();
            let topk_weights = match &*topk_weights {
                candle::Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
                _ => candle::bail!("topk_weights must be a cuda tensor"),
            };
            let tw_offset = tw_l.start_offset();
            let topk_w_ptr = topk_weights
                .slice(tw_offset..)
                .device_ptr(topk_weights.stream())
                .0 as *const f32;
            topk_w_ptr
        } else {
            std::ptr::null()
        };

        let output = unsafe { dev.alloc::<T>(size_m * size_n) }?;

        let stream = dev.cuda_stream().cu_stream() as i64;
        use core::ffi::c_void;

        // Threshold for using GEMV kernel (optimized for small batch sizes)
        const GEMV_THRESHOLD: i32 = 8;

        let num_experts_i32 = i32::try_from(num_experts).expect("num_experts too large for i32");
        let topk_i32 = i32::try_from(topk).expect("topk too large for i32");
        let size_m_i32 = i32::try_from(size_m).expect("size_m too large for i32");
        let size_n_i32 = i32::try_from(size_n).expect("size_n too large for i32");
        let size_k_i32 = i32::try_from(size_k).expect("size_k too large for i32");

        // Select kernel based on prefill/decode and batch size
        // - Prefill (larger batches): use WMMA-based kernel for tensor core acceleration
        // - Decode with small M (<=8): use GEMV kernel optimized for warp reductions
        // - Decode with larger M: use standard moe_gemm_transposed kernel
        let moe_func = if is_prefill {
            crate::cuda::ffi::moe_gemm_wmma_transposed
        } else if size_m_i32 <= GEMV_THRESHOLD {
            crate::cuda::ffi::moe_gemv_transposed
        } else {
            crate::cuda::ffi::moe_gemm_transposed
        };

        unsafe {
            moe_func(
                input.slice(input_offset..).device_ptr(input.stream()).0 as *const c_void, // [size_m, size_k]
                weights
                    .slice(weights_offset..)
                    .device_ptr(weights.stream())
                    .0 as *const c_void, // [num_experts, size_k, size_n]
                sorted_token_ids
                    .slice(sti_offset..)
                    .device_ptr(sorted_token_ids.stream())
                    .0 as *const i32,
                experts_ids
                    .slice(ei_offset..)
                    .device_ptr(experts_ids.stream())
                    .0 as *const i32,
                topk_weights_ptr,
                output.device_ptr(output.stream()).0 as *mut c_void, // [size_m, size_n]
                num_experts_i32,
                topk_i32,
                size_m_i32,
                size_n_i32,
                size_k_i32,
                data_type as i32, // 0=float16, 1=bf16 (for input/output)
                stream,
            );
        }

        let output = candle::CudaStorage::wrap_cuda_slice(output, dev.clone());
        let output = Tensor::from((candle::Storage::Cuda(output), (size_m, size_n)));

        Ok(output)
    }

    match input.dtype() {
        DType::F16 => cuda_fwd::<f16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        DType::BF16 => cuda_fwd::<bf16>(
            input,
            weights,
            topk_weights,
            sorted_token_ids,
            experts_ids,
            topk,
            is_prefill,
        ),
        _ => {
            candle_core::bail!("moe_gemm_transposed only accept f16/bf16 inputs!")
        }
    }
}

#[cfg(not(feature = "cuda"))]
#[allow(unused)]
pub fn moe_gemm_transposed(
    _: &Tensor,
    _: &Tensor,
    _: &Option<Tensor>,
    _: &Tensor,
    _: &Tensor,
    _: usize,
    _: bool,
) -> Result<Tensor> {
    candle_core::bail!("moe_gemm_transposed is not implemented on this platform!")
}
