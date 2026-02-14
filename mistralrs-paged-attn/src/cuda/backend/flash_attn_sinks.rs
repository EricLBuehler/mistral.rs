use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use candle::backend::BackendStorage;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use candle_core::cuda::cudarc::driver::{DevicePtr, DeviceSlice};
use half::{bf16, f16};
use std::ffi::{c_int, c_uint};

struct FlashAttnSinks {
    key: Tensor,
    value: Tensor,
    sinks: Option<Tensor>,
    softmax_scale: f32,
    window_size: usize,
}

impl FlashAttnSinks {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let (batch_size, num_heads, q_len, head_dim) = q_l.shape().dims4()?;

        // Extract K storage
        let (k_s, k_l) = self.key.storage_and_layout();
        let k_cuda = match &*k_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks: key must be a cuda tensor"),
        };
        let (_, num_kv_heads, kv_len, _) = k_l.shape().dims4()?;

        // Extract V storage
        let (v_s, v_l) = self.value.storage_and_layout();
        let v_cuda = match &*v_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks: value must be a cuda tensor"),
        };

        // Validate head_dim
        if !(head_dim == 64
            || head_dim == 80
            || head_dim == 96
            || head_dim == 112
            || head_dim == 128
            || head_dim == 192
            || head_dim == 256)
        {
            candle::bail!(
                "flash_attn_sinks: head_dim must be one of 64, 80, 96, 112, 128, 192, 256, got {head_dim}"
            );
        }

        // Get CUDA slices
        let q_slice = q.as_cuda_slice::<T>()?;
        let q_view = q_slice.slice(q_l.start_offset()..);
        let (q_ptr, _q_guard) = q_view.device_ptr(q_view.stream());

        let k_slice = k_cuda.as_cuda_slice::<T>()?;
        let k_view = k_slice.slice(k_l.start_offset()..);
        let (k_ptr, _k_guard) = k_view.device_ptr(k_view.stream());

        let v_slice = v_cuda.as_cuda_slice::<T>()?;
        let v_view = v_slice.slice(v_l.start_offset()..);
        let (v_ptr, _v_guard) = v_view.device_ptr(v_view.stream());

        // Sinks pointer (always f32, can be null)
        let sinks_ptr: *const f32 = if let Some(sinks) = &self.sinks {
            let (s_s, s_l) = sinks.storage_and_layout();
            let s_cuda = match &*s_s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("flash_attn_sinks: sinks must be a cuda tensor"),
            };
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let (s_ptr, _s_guard) = slice_ptr(s_slice, s_l.start_offset());
            s_ptr as *const f32
        } else {
            std::ptr::null()
        };

        // Allocate output
        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;
        let (out_ptr, out_guard) = out.device_ptr(out.stream());

        // Dispatch by dtype
        let launch_fn = match dtype {
            DType::F16 => ffi::flash_attn_sinks_f16,
            DType::BF16 => ffi::flash_attn_sinks_bf16,
            DType::F32 => ffi::flash_attn_sinks_f32,
            dt => candle::bail!("flash_attn_sinks: unsupported dtype {dt:?}"),
        };

        unsafe {
            launch_fn(
                q_ptr as *const std::ffi::c_void,
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                sinks_ptr,
                self.softmax_scale,
                batch_size as c_int,
                q_len as c_int,
                kv_len as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                self.window_size as c_int,
                dev.cuda_stream().cu_stream(),
            );
        }

        drop(out_guard);
        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out_storage, out_shape))
    }
}

impl candle::CustomOp1 for FlashAttnSinks {
    fn name(&self) -> &'static str {
        "flash-attn-sinks"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn-sinks")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("flash-attn-sinks only supports f32/f16/bf16 ({dt:?})"),
        }
    }
}

struct FlashAttnSinksVarlen {
    key: Tensor,   // [total_kv, num_kv_heads, D]
    value: Tensor, // [total_kv, num_kv_heads, D]
    sinks: Option<Tensor>,
    cu_seqlens_q: Tensor, // [B+1] u32
    cu_seqlens_k: Tensor, // [B+1] u32
    softmax_scale: f32,
    window_size: usize,
}

impl FlashAttnSinksVarlen {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType
            + candle::cuda_backend::cudarc::driver::DeviceRepr
            + candle::cuda_backend::cudarc::driver::ValidAsZeroBits,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        let dev = q.device();
        let out_shape = q_l.shape().clone();
        let (batch_size, num_heads, max_q_len, head_dim) = q_l.shape().dims4()?;

        // Extract K storage [total_kv, num_kv_heads, D]
        let (k_s, k_l) = self.key.storage_and_layout();
        let k_cuda = match &*k_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks_varlen: key must be a cuda tensor"),
        };
        let (_, num_kv_heads, _) = k_l.shape().dims3()?;

        // Extract V storage
        let (v_s, v_l) = self.value.storage_and_layout();
        let v_cuda = match &*v_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks_varlen: value must be a cuda tensor"),
        };

        // Validate head_dim
        if !(head_dim == 64
            || head_dim == 80
            || head_dim == 96
            || head_dim == 112
            || head_dim == 128
            || head_dim == 192
            || head_dim == 256)
        {
            candle::bail!(
                "flash_attn_sinks_varlen: head_dim must be one of 64, 80, 96, 112, 128, 192, 256, got {head_dim}"
            );
        }

        // Get CUDA slices
        let q_slice = q.as_cuda_slice::<T>()?;
        let q_view = q_slice.slice(q_l.start_offset()..);
        let (q_ptr, _q_guard) = q_view.device_ptr(q_view.stream());

        let k_slice = k_cuda.as_cuda_slice::<T>()?;
        let k_view = k_slice.slice(k_l.start_offset()..);
        let (k_ptr, _k_guard) = k_view.device_ptr(k_view.stream());

        let v_slice = v_cuda.as_cuda_slice::<T>()?;
        let v_view = v_slice.slice(v_l.start_offset()..);
        let (v_ptr, _v_guard) = v_view.device_ptr(v_view.stream());

        // cu_seqlens_q pointer (u32)
        let (csq_s, csq_l) = self.cu_seqlens_q.storage_and_layout();
        let csq_cuda = match &*csq_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks_varlen: cu_seqlens_q must be a cuda tensor"),
        };
        let csq_slice = csq_cuda.as_cuda_slice::<u32>()?;
        let (csq_ptr, _csq_guard) = slice_ptr(csq_slice, csq_l.start_offset());

        // cu_seqlens_k pointer (u32)
        let (csk_s, csk_l) = self.cu_seqlens_k.storage_and_layout();
        let csk_cuda = match &*csk_s {
            Storage::Cuda(s) => s,
            _ => candle::bail!("flash_attn_sinks_varlen: cu_seqlens_k must be a cuda tensor"),
        };
        let csk_slice = csk_cuda.as_cuda_slice::<u32>()?;
        let (csk_ptr, _csk_guard) = slice_ptr(csk_slice, csk_l.start_offset());

        // Sinks pointer
        let sinks_ptr: *const f32 = if let Some(sinks) = &self.sinks {
            let (s_s, s_l) = sinks.storage_and_layout();
            let s_cuda = match &*s_s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("flash_attn_sinks_varlen: sinks must be a cuda tensor"),
            };
            let s_slice = s_cuda.as_cuda_slice::<f32>()?;
            let (s_ptr, _s_guard) = slice_ptr(s_slice, s_l.start_offset());
            s_ptr as *const f32
        } else {
            std::ptr::null()
        };

        // Allocate output (zero-initialized for padding rows)
        let elem_count = out_shape.elem_count();
        let out = dev.alloc_zeros::<T>(elem_count)?;
        let (out_ptr, out_guard) = out.device_ptr(out.stream());

        let launch_fn = match dtype {
            DType::F16 => ffi::flash_attn_sinks_varlen_f16,
            DType::BF16 => ffi::flash_attn_sinks_varlen_bf16,
            DType::F32 => ffi::flash_attn_sinks_varlen_f32,
            dt => candle::bail!("flash_attn_sinks_varlen: unsupported dtype {dt:?}"),
        };

        unsafe {
            launch_fn(
                q_ptr as *const std::ffi::c_void,
                k_ptr as *const std::ffi::c_void,
                v_ptr as *const std::ffi::c_void,
                out_ptr as *mut std::ffi::c_void,
                sinks_ptr,
                csq_ptr as *const c_uint,
                csk_ptr as *const c_uint,
                self.softmax_scale,
                batch_size as c_int,
                max_q_len as c_int,
                num_heads as c_int,
                num_kv_heads as c_int,
                head_dim as c_int,
                self.window_size as c_int,
                dev.cuda_stream().cu_stream(),
            );
        }

        drop(out_guard);
        let out_storage = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out_storage, out_shape))
    }
}

impl candle::CustomOp1 for FlashAttnSinksVarlen {
    fn name(&self) -> &'static str {
        "flash-attn-sinks-varlen"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for flash-attn-sinks-varlen")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("flash-attn-sinks-varlen only supports f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// Fused flash attention with per-head sinks for prefill.
///
/// Uses FlashAttention-2 online softmax, never materializes the N x N
/// attention matrix. Per-head sinks contribute to the softmax denominator
/// without an associated value (virtual probability-mass-absorbing token).
///
/// Causal masking is always applied. If `window_size > 0`, a sliding window
/// restricts attention to the last `window_size` KV positions.
///
/// # Arguments
///
/// * `q` - Query tensor `[batch_size, num_heads, seq_len, head_dim]`
/// * `k` - Key tensor `[batch_size, num_kv_heads, seq_len, head_dim]`
/// * `v` - Value tensor `[batch_size, num_kv_heads, seq_len, head_dim]`
/// * `sinks` - Optional per-head sink values `[num_heads]` (will be cast to f32)
/// * `softmax_scale` - Scaling factor (typically `1 / sqrt(head_dim)`)
/// * `window_size` - Sliding window size (0 = full causal attention)
///
/// Returns `[batch_size, num_heads, seq_len, head_dim]`
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let op = FlashAttnSinks {
        key: k.clone(),
        value: v.clone(),
        sinks: sinks.map(|s| s.to_dtype(DType::F32)).transpose()?,
        softmax_scale,
        window_size,
    };
    q.apply_op1(op)
}

/// Fused varlen flash attention with per-head sinks for prefill.
///
/// Handles variable-length sequences within a batch. Q is padded,
/// K/V are packed (concatenated across sequences).
///
/// # Arguments
///
/// * `q` - Query tensor `[batch_size, num_heads, max_q_len, head_dim]` (padded)
/// * `k` - Key tensor `[total_kv, num_kv_heads, head_dim]` (packed)
/// * `v` - Value tensor `[total_kv, num_kv_heads, head_dim]` (packed)
/// * `sinks` - Optional per-head sink values `[num_heads]` (will be cast to f32)
/// * `cu_seqlens_q` - Cumulative Q sequence lengths `[batch_size + 1]` (u32)
/// * `cu_seqlens_k` - Cumulative KV sequence lengths `[batch_size + 1]` (u32)
/// * `softmax_scale` - Scaling factor (typically `1 / sqrt(head_dim)`)
/// * `window_size` - Sliding window size (0 = full causal attention)
///
/// Returns `[batch_size, num_heads, max_q_len, head_dim]` (padding rows are zero)
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_sinks_varlen(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    sinks: Option<&Tensor>,
    cu_seqlens_q: &Tensor,
    cu_seqlens_k: &Tensor,
    softmax_scale: f32,
    window_size: usize,
) -> Result<Tensor> {
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;

    let op = FlashAttnSinksVarlen {
        key: k.clone(),
        value: v.clone(),
        sinks: sinks.map(|s| s.to_dtype(DType::F32)).transpose()?,
        cu_seqlens_q: cu_seqlens_q.clone(),
        cu_seqlens_k: cu_seqlens_k.clone(),
        softmax_scale,
        window_size,
    };
    q.apply_op1(op)
}
