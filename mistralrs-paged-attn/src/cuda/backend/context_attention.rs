use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi::{
    context_attention_fwd_bf16, context_attention_fwd_f16, context_attention_fwd_f32,
    context_attention_fwd_v2_bf16, context_attention_fwd_v2_f16, context_attention_fwd_v2_f32,
};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use candle_core::cuda::cudarc::driver::DeviceSlice;
use float8::F8E4M3;
use half::{bf16, f16};
use std::ffi::c_int;

struct ContextAttention {
    softmax_scale: f32,

    key: Tensor,
    value: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    query_lens: Tensor,
    query_start_locs: Tensor,
    seq_ids: Tensor,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    sliding_window: i32,
    sinks: Option<Tensor>,
    max_total_kv_len: usize,
}

impl ContextAttention {
    fn cuda_fwd_t<
        T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
    >(
        &self,
        q: &CudaStorage,
        q_l: &Layout,
    ) -> Result<(CudaStorage, Shape)> {
        let dtype = q.dtype();
        let cache_dtype = match self.key_cache.dtype() {
            DType::F16 => 0,
            DType::BF16 => 1,
            DType::F32 => 2,
            DType::F8E4M3 => 3,
            dtype => candle::bail!("cache dtype {dtype:?} is not supported"),
        };

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        // Extract K tensor
        let (k, k_l) = self.key.storage_and_layout();
        let k = match &*k {
            Storage::Cuda(k) => k,
            _ => candle::bail!("key must be a cuda tensor"),
        };

        // Extract V tensor
        let (v, v_l) = self.value.storage_and_layout();
        let v = match &*v {
            Storage::Cuda(v) => v,
            _ => candle::bail!("value must be a cuda tensor"),
        };

        // Extract key_cache
        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle::bail!("key_cache must be a cuda tensor"),
        };

        // Extract value_cache
        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle::bail!("value_cache must be a cuda tensor"),
        };

        // Extract block_tables
        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Cuda(bt) => bt,
            _ => candle::bail!("block_tables must be a cuda tensor"),
        };

        // Extract context_lens
        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Cuda(cl) => cl,
            _ => candle::bail!("context_lens must be a cuda tensor"),
        };

        // Extract query_lens
        let (ql, ql_l) = self.query_lens.storage_and_layout();
        let ql = match &*ql {
            Storage::Cuda(ql) => ql,
            _ => candle::bail!("query_lens must be a cuda tensor"),
        };

        // Extract query_start_locs
        let (qsl, qsl_l) = self.query_start_locs.storage_and_layout();
        let qsl = match &*qsl {
            Storage::Cuda(qsl) => qsl,
            _ => candle::bail!("query_start_locs must be a cuda tensor"),
        };

        // Extract seq_ids
        let (si, si_l) = self.seq_ids.storage_and_layout();
        let si = match &*si {
            Storage::Cuda(si) => si,
            _ => candle::bail!("seq_ids must be a cuda tensor"),
        };

        // Validate ranks
        let q_rank = q_l.stride().len();
        let k_rank = k_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle::bail!("context-attention expects `q` tensor to be of rank 3 (q: {q_l:?})")
        }
        if k_rank != 3 {
            candle::bail!("context-attention expects `k` tensor to be of rank 3 (k: {k_l:?})")
        }
        if kc_rank != 5 {
            candle::bail!(
                "context-attention expects `key_cache` tensor to be of rank 5 (key_cache: {kc_l:?})"
            )
        }
        if vc_rank != 4 {
            candle::bail!(
                "context-attention expects `value_cache` tensor to be of rank 4 (value_cache: {vc_l:?})"
            )
        }

        // Get cuda slices
        let q = q.as_cuda_slice::<T>()?;
        let k_slice = k.as_cuda_slice::<T>()?;
        let v_slice = v.as_cuda_slice::<T>()?;
        let (kc_ptr, _kc_guard) = if cache_dtype == 3 {
            slice_ptr(kc.as_cuda_slice::<F8E4M3>()?, kc_l.start_offset())
        } else {
            slice_ptr(kc.as_cuda_slice::<T>()?, kc_l.start_offset())
        };
        let (vc_ptr, _vc_guard) = if cache_dtype == 3 {
            slice_ptr(vc.as_cuda_slice::<F8E4M3>()?, vc_l.start_offset())
        } else {
            slice_ptr(vc.as_cuda_slice::<T>()?, vc_l.start_offset())
        };
        let bt = bt.as_cuda_slice::<u32>()?;
        let cl = cl.as_cuda_slice::<u32>()?;
        let ql_slice = ql.as_cuda_slice::<u32>()?;
        let qsl = qsl.as_cuda_slice::<u32>()?;
        let si = si.as_cuda_slice::<u32>()?;

        // Get views with offsets
        let q = q.slice(q_l.start_offset()..);
        let k_slice = k_slice.slice(k_l.start_offset()..);
        let v_slice = v_slice.slice(v_l.start_offset()..);
        let bt = bt.slice(bt_l.start_offset()..);
        let cl = cl.slice(cl_l.start_offset()..);
        let ql_slice = ql_slice.slice(ql_l.start_offset()..);
        let qsl = qsl.slice(qsl_l.start_offset()..);
        let si = si.slice(si_l.start_offset()..);

        // FP8 scale pointers
        let (k_scale_ptr, v_scale_ptr) =
            if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {
                if !crate::cuda::USE_FP8 {
                    candle::bail!("FP8 is not supported on this system.");
                }

                let (ks, ks_l) = k_scale.storage_and_layout();
                let ks = match &*ks {
                    Storage::Cuda(ks) => ks,
                    _ => candle::bail!("k_scale must be a cuda tensor"),
                };
                let ks = ks.as_cuda_slice::<f32>()?;
                let (ks, _ks_guard) = slice_ptr(ks, ks_l.start_offset());

                let (vs, vs_l) = v_scale.storage_and_layout();
                let vs = match &*vs {
                    Storage::Cuda(vs) => vs,
                    _ => candle::bail!("v_scale must be a cuda tensor"),
                };
                let vs = vs.as_cuda_slice::<f32>()?;
                let (vs, _vs_guard) = slice_ptr(vs, vs_l.start_offset());

                (ks as *const f32, vs as *const f32)
            } else {
                (std::ptr::null(), std::ptr::null())
            };

        // Sinks pointer
        let sinks_ptr = if let Some(sinks) = self.sinks.as_ref() {
            let (s, s_l) = sinks.storage_and_layout();
            let s = match &*s {
                Storage::Cuda(s) => s,
                _ => candle::bail!("sinks must be a cuda tensor"),
            };
            let s = s.as_cuda_slice::<f32>()?;
            let (s_ptr, _s_guard) = slice_ptr(s, s_l.start_offset());
            s_ptr as *const f32
        } else {
            std::ptr::null()
        };

        // Dimensions
        let (total_new_tokens, num_heads, head_size) = q_l.shape().dims3()?;
        if !(head_size == 64
            || head_size == 80
            || head_size == 96
            || head_size == 112
            || head_size == 128
            || head_size == 192
            || head_size == 256)
        {
            candle_core::bail!("`head_size` must be one of 64, 80, 96, 112, 128, 192 or 256");
        }

        let (_, max_num_blocks_per_seq) = bt_l.shape().dims2()?;
        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;

        if head_size_kc != head_size / x {
            candle::bail!(
                "shape mismatch key_cache {:?}, expected head_size/x = {}",
                kc_l.shape(),
                head_size / x
            )
        }

        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;

        let (out_ptr, out_guard) = out.device_ptr(out.stream());
        let (q_ptr, _q_guard) = q.device_ptr(q.stream());
        let (k_ptr, _k_guard) = k_slice.device_ptr(k_slice.stream());
        let (v_ptr, _v_guard) = v_slice.device_ptr(v_slice.stream());
        let (bt_ptr, _bt_guard) = bt.device_ptr(bt.stream());
        let (cl_ptr, _cl_guard) = cl.device_ptr(cl.stream());
        let (ql_ptr, _ql_guard) = ql_slice.device_ptr(ql_slice.stream());
        let (qsl_ptr, _qsl_guard) = qsl.device_ptr(qsl.stream());
        let (si_ptr, _si_guard) = si.device_ptr(si.stream());

        // v1/v2 dispatch: prefer v1 (monolithic) when the shared memory logits
        // array fits, since it avoids the partition+reduce overhead of v2.
        // V1 needs: max(padded_kv * 4, (NUM_WARPS/2) * head_size * 4) bytes.
        // 48KB is the baseline dynamic shared memory on all CUDA GPUs;
        // SET_MaxDynamicSharedMemorySize in the launcher can raise it further.
        const NUM_THREADS: usize = 128;
        const WARP_SIZE: usize = 32;
        const V1_SHARED_MEM_LIMIT: usize = 48 * 1024;
        let padded_kv = self.max_total_kv_len.div_ceil(block_size) * block_size;
        let v1_logits_bytes = padded_kv * std::mem::size_of::<f32>();
        let v1_outputs_bytes =
            (NUM_THREADS / WARP_SIZE / 2) * head_size * std::mem::size_of::<f32>();
        let v1_shared_mem = std::cmp::max(v1_logits_bytes, v1_outputs_bytes);
        let use_v1 = v1_shared_mem <= V1_SHARED_MEM_LIMIT;

        let partition_size = 512usize;
        let max_num_partitions = self.max_total_kv_len.div_ceil(partition_size);

        if use_v1 {
            let context_attention_fwd_func = match dtype {
                DType::F16 => context_attention_fwd_f16,
                DType::BF16 => context_attention_fwd_bf16,
                DType::F32 => context_attention_fwd_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };

            unsafe {
                context_attention_fwd_func(
                    out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    v_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    ql_ptr as *const i32,
                    qsl_ptr as *const i32,
                    si_ptr as *const i32,
                    max_num_blocks_per_seq as c_int,
                    total_new_tokens as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    block_size as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    self.sliding_window,
                    sinks_ptr,
                    self.max_total_kv_len as c_int,
                )
            }
        } else {
            // v2: allocate temp buffers for partitioned attention
            let exp_sums_shape = Shape::from((total_new_tokens, num_heads, max_num_partitions));
            let tmp_out_shape =
                Shape::from((total_new_tokens, num_heads, max_num_partitions, head_size));
            let tmp_out_buf = unsafe { dev.alloc::<T>(tmp_out_shape.elem_count()) }?;
            let exp_sums_buf = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }?;
            let max_logits_buf = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }?;

            let (tmp_out_ptr, _tmp_out_guard) = tmp_out_buf.device_ptr(tmp_out_buf.stream());
            let (exp_sums_ptr, _exp_sums_guard) = exp_sums_buf.device_ptr(exp_sums_buf.stream());
            let (max_logits_ptr, _max_logits_guard) =
                max_logits_buf.device_ptr(max_logits_buf.stream());

            let context_attention_fwd_v2_func = match dtype {
                DType::F16 => context_attention_fwd_v2_f16,
                DType::BF16 => context_attention_fwd_v2_bf16,
                DType::F32 => context_attention_fwd_v2_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };

            unsafe {
                context_attention_fwd_v2_func(
                    out_ptr as *const std::ffi::c_void,
                    exp_sums_ptr as *const f32,
                    max_logits_ptr as *const f32,
                    tmp_out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    k_ptr as *const std::ffi::c_void,
                    v_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    ql_ptr as *const i32,
                    qsl_ptr as *const i32,
                    si_ptr as *const i32,
                    max_num_blocks_per_seq as c_int,
                    total_new_tokens as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    block_size as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    self.sliding_window,
                    sinks_ptr,
                    self.max_total_kv_len as c_int,
                    max_num_partitions as c_int,
                )
            }
        }

        drop(out_guard);

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl candle::CustomOp1 for ContextAttention {
    fn name(&self) -> &'static str {
        "context-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for context-attention")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("context-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// Context attention forward for prefix caching.
///
/// When prefix cache hits occur, only new tokens need to be computed. Each query token
/// attends to cached context tokens (from paged KV cache via block table) and preceding
/// new tokens (from input K/V with causal mask).
///
/// # Arguments
///
/// * `query` - Query tensor `[total_new_tokens, num_heads, head_size]`
/// * `key` - Key tensor for new tokens `[total_new_tokens, num_kv_heads, head_size]`
/// * `value` - Value tensor for new tokens `[total_new_tokens, num_kv_heads, head_size]`
/// * `key_cache` - Paged key cache `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
/// * `value_cache` - Paged value cache `[num_blocks, num_kv_heads, head_size, block_size]`
/// * `block_tables` - Block table `[num_seqs, max_num_blocks_per_seq]`
/// * `context_lens` - Number of cached tokens per sequence `[num_seqs]`
/// * `query_lens` - Number of new tokens per sequence `[num_seqs]`
/// * `query_start_locs` - Cumulative start offsets `[num_seqs + 1]`
/// * `seq_ids` - Batch index for each new token `[total_new_tokens]`
/// * `softmax_scale` - Scaling factor for QK products
/// * `sliding_window` - Sliding window size (0 = disabled)
/// * `max_total_kv_len` - Maximum total KV length across all sequences (for shared memory)
#[allow(clippy::too_many_arguments)]
pub fn context_attention_fwd(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    query_lens: &Tensor,
    query_start_locs: &Tensor,
    seq_ids: &Tensor,
    softmax_scale: f32,
    sliding_window: i32,
    sinks: Option<&Tensor>,
    max_total_kv_len: usize,
) -> Result<Tensor> {
    let op = ContextAttention {
        softmax_scale,
        key: key.clone(),
        value: value.clone(),
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        query_lens: query_lens.clone(),
        query_start_locs: query_start_locs.clone(),
        seq_ids: seq_ids.clone(),
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
        sliding_window,
        sinks: sinks
            .map(|s| s.to_dtype(candle_core::DType::F32))
            .transpose()?,
        max_total_kv_len,
    };
    query.apply_op1(op)
}
