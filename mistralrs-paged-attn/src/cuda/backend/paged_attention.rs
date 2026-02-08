use crate::cuda::backend::slice_ptr;
use crate::cuda::ffi;
use crate::cuda::ffi::{
    paged_attention_v1_bf16, paged_attention_v1_f16, paged_attention_v1_f32,
    paged_attention_v2_bf16, paged_attention_v2_f16, paged_attention_v2_f32,
};
use candle::backend::BackendStorage;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{CpuStorage, CudaStorage, DType, Layout, Result, Shape, Storage, Tensor};
use candle_core as candle;
use candle_core::cuda::cudarc::driver::DeviceSlice;
use float8::F8E4M3;
use half::{bf16, f16};
use std::ffi::c_int;

struct PagedAttention {
    softmax_scale: f32,
    softcapping: f32,

    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
    alibi_slopes: Option<Tensor>,
    max_context_len: usize,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    sinks: Option<Tensor>,
}

impl PagedAttention {
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

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Cuda(kc) => kc,
            _ => candle::bail!("key_cache must be a cuda tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Cuda(vc) => vc,
            _ => candle::bail!("value_cache must be a cuda tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Cuda(bt) => bt,
            _ => candle::bail!("block_tables must be a cuda tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Cuda(cl) => cl,
            _ => candle::bail!("context_lens must be a cuda tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        if kc_rank != 5 {
            candle::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
            )
        }

        // Get cuda slices for all tensors
        let q = q.as_cuda_slice::<T>()?;
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
        let cl = cl.as_cuda_slice::<u32>()?; // Should be i32!
        let bt = bt.as_cuda_slice::<u32>()?; // Should be i32!

        // Get cuda views for all tensors
        let q = q.slice(q_l.start_offset()..);
        let cl = cl.slice(cl_l.start_offset()..);
        let bt = bt.slice(bt_l.start_offset()..);

        let alibi_s_ptr = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Cuda(alibi_s) => alibi_s,
                _ => candle::bail!("context_lens must be a cuda tensor"),
            };
            let alibi_s = alibi_s.as_cuda_slice::<f32>()?;
            let (alibi_s_ptr, _alibi_s_guard) = slice_ptr(alibi_s, alibi_s_l.start_offset());
            alibi_s_ptr as *const std::ffi::c_void
        } else {
            std::ptr::null()
        };

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

        let (num_seqs, num_heads, head_size) = q_l.shape().dims3()?;
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

        let (num_seqs_bt, max_num_blocks_per_seq) = bt_l.shape().dims2()?;

        if num_seqs_bt != num_seqs {
            candle::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc != head_size / x {
            candle::bail!(
                "shape mismatch value_cache {:?}, expected {:?}",
                vc_l.shape(),
                (num_blocks, num_kv_heads, head_size / x, block_size, x)
            )
        }

        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_seqs) != cl_l.shape().dims1()? {
            candle::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let partition_size = 512;
        let max_num_partitions = self.max_context_len.div_ceil(partition_size);
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();
        let out = unsafe { dev.alloc::<T>(elem_count) }?;

        let (out_ptr, out_guard) = out.device_ptr(out.stream());
        let (q_ptr, _q_guard) = q.device_ptr(q.stream());
        let (bt_ptr, _bt_guard) = bt.device_ptr(bt.stream());
        let (cl_ptr, _cl_guard) = cl.device_ptr(cl.stream());

        if use_v1 {
            let paged_attention_v1_func = match dtype {
                DType::F16 => paged_attention_v1_f16,
                DType::BF16 => paged_attention_v1_bf16,
                DType::F32 => paged_attention_v1_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };
            unsafe {
                paged_attention_v1_func(
                    out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    alibi_s_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    self.softcapping,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    block_size as c_int,
                    self.max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    sinks_ptr,
                )
            }
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out = unsafe { dev.alloc::<T>(tmp_out_shape.elem_count()) }?;
            let exp_sums = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }?;
            let max_logits = unsafe { dev.alloc::<f32>(exp_sums_shape.elem_count()) }?;

            let (tmp_out_ptr, _tmp_out_guard) = tmp_out.device_ptr(tmp_out.stream());
            let (exp_sums_ptr, _exp_sums_guard) = exp_sums.device_ptr(exp_sums.stream());
            let (max_logits_ptr, _max_logits_guard) = max_logits.device_ptr(max_logits.stream());

            let paged_attention_v2_func = match dtype {
                DType::F16 => paged_attention_v2_f16,
                DType::BF16 => paged_attention_v2_bf16,
                DType::F32 => paged_attention_v2_f32,
                dtype => candle::bail!("dtype {dtype:?} is not supported"),
            };
            unsafe {
                paged_attention_v2_func(
                    out_ptr as *const std::ffi::c_void,
                    exp_sums_ptr as *const f32,
                    max_logits_ptr as *const f32,
                    tmp_out_ptr as *const std::ffi::c_void,
                    q_ptr as *const std::ffi::c_void,
                    kc_ptr as *const std::ffi::c_void,
                    vc_ptr as *const std::ffi::c_void,
                    alibi_s_ptr,
                    num_kv_heads as c_int,
                    self.softmax_scale,
                    self.softcapping,
                    bt_ptr as *const i32,
                    cl_ptr as *const i32,
                    block_size as c_int,
                    self.max_context_len as c_int,
                    num_seqs as c_int,
                    num_heads as c_int,
                    head_size as c_int,
                    max_num_blocks_per_seq as c_int,
                    q_stride as c_int,
                    kv_block_stride as c_int,
                    kv_head_stride as c_int,
                    dev.cuda_stream().cu_stream(),
                    cache_dtype,
                    k_scale_ptr,
                    v_scale_ptr,
                    sinks_ptr,
                )
            }
        }

        drop(out_guard);

        let out = CudaStorage::wrap_cuda_slice(out, dev.clone());
        Ok((out, out_shape))
    }
}

impl candle::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle::bail!("no cpu support for paged-attention")
    }

    fn cuda_fwd(&self, q: &CudaStorage, q_l: &Layout) -> Result<(CudaStorage, Shape)> {
        match q.dtype() {
            DType::F32 => self.cuda_fwd_t::<f32>(q, q_l),
            DType::F16 => self.cuda_fwd_t::<f16>(q, q_l),
            DType::BF16 => self.cuda_fwd_t::<bf16>(q, q_l),
            dt => candle::bail!("paged-attention is only supported for f32/f16/bf16 ({dt:?})"),
        }
    }
}

/// PagedAttention layer.
///
/// This implements scaled dot-product attention, `softmax(Q @ K^T . softmax_scale) @ V`.
/// Multi-query and grouped-query attention are supported by using tensors key_cache and value_cache
/// with fewer heads than q, the number of heads in k and v has to be divisible by the number of heads in q.
///
/// # Arguments
///
/// * `q` - Query tensor with shape `(num_sequences, num_heads_q, head_size)`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads_kv, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads_kv, head_size, block_size)`.
/// * `block_tables` - Padded table associating blocks to each sequence of shape `(num_sequences, max_context_len // block_size)`
/// * `context_lens` - Tensor associating lengths to each sequence of shape `(num_sequences)`
/// * `max_context_len` - Max of `context_len`
/// * `softmax_scale` - scaling factor
/// * `softcapping`- Softcapping value as in Gemma 2. Using 1.0 means do nothing.
/// * `alibi_slopes`- Optional alibi slopes, `(num_heads_q)`.
///
/// The resulting tensor has dimensions `(num_sequences, num_heads_q, head_size)`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention(
    q: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    alibi_slopes: Option<&Tensor>,
    max_context_len: usize,
    softmax_scale: f32,
    softcapping: f32,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    let op = PagedAttention {
        softmax_scale,
        key_cache: key_cache.clone(),
        value_cache: value_cache.clone(),
        block_tables: block_tables.clone(),
        context_lens: context_lens.clone(),
        max_context_len,
        softcapping,
        alibi_slopes: alibi_slopes.cloned(),
        k_scale: k_scale.cloned(),
        v_scale: v_scale.cloned(),
        sinks: sinks
            .map(|s| s.to_dtype(candle_core::DType::F32))
            .transpose()?,
    };
    q.apply_op1(op)
}

fn update_cache<
    T: candle::cuda_backend::CudaDType + candle::cuda_backend::cudarc::driver::DeviceRepr,
>(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    let dtype = key.dtype();

    let internal_type = match dtype {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        dtype => candle::bail!("dtype {dtype:?} is not supported"),
    };

    let cache_dtype = match key_cache.dtype() {
        DType::F16 => 0,
        DType::BF16 => 1,
        DType::F32 => 2,
        DType::F8E4M3 => 3,
        dtype => candle::bail!("cache dtype {dtype:?} is not supported"),
    };

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Cuda(k) => k,
        _ => candle::bail!("key must be a cuda tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Cuda(v) => v,
        _ => candle::bail!("value must be a cuda tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Cuda(kc) => kc,
        _ => candle::bail!("key_cache must be a cuda tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Cuda(vc) => vc,
        _ => candle::bail!("value_cache must be a cuda tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Cuda(s) => s,
        _ => candle::bail!("slot_mapping must be a cuda tensor"),
    };

    let k_rank = k_l.stride().len();
    let v_rank = v_l.stride().len();
    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if k_rank != 3 || v_rank != 3 {
        candle::bail!("paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})")
    }

    if kc_rank != 5 {
        candle::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
        )
    }

    let dev = k.device();

    // Get cuda slices for all tensors
    let k = k.as_cuda_slice::<T>()?;
    let v = v.as_cuda_slice::<T>()?;
    let s = s.as_cuda_slice::<i64>()?;

    // For FP8 cache, we need to get as u8 slices instead
    let ((kc_ptr, _kc_guard), (vc_ptr, _vc_guard)) = if cache_dtype == 3 {
        if !crate::cuda::USE_FP8 {
            candle::bail!("FP8 is not supported on this system.");
        }

        let kc = kc.as_cuda_slice::<F8E4M3>()?;
        let vc = vc.as_cuda_slice::<F8E4M3>()?;
        (
            slice_ptr(kc, kc_l.start_offset()),
            slice_ptr(vc, vc_l.start_offset()),
        )
    } else {
        let kc = kc.as_cuda_slice::<T>()?;
        let vc = vc.as_cuda_slice::<T>()?;
        (
            slice_ptr(kc, kc_l.start_offset()),
            slice_ptr(vc, vc_l.start_offset()),
        )
    };

    // Get cuda views for all tensors
    let k = k.slice(k_l.start_offset()..);
    let v = v.slice(v_l.start_offset()..);
    let s = s.slice(s_l.start_offset()..);

    let (k_scale_ptr, v_scale_ptr) = if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {
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

    let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
        candle::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        candle::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = k_l.stride()[0] as c_int;
    let value_stride = v_l.stride()[0] as c_int;

    let (k_ptr, _k_guard) = k.device_ptr(k.stream());
    let (v_ptr, _v_guard) = v.device_ptr(v.stream());
    let (s_ptr, _s_guard) = s.device_ptr(s.stream());

    unsafe {
        ffi::reshape_and_cache(
            k_ptr as *const core::ffi::c_void,
            v_ptr as *const core::ffi::c_void,
            kc_ptr as *const core::ffi::c_void,
            vc_ptr as *const core::ffi::c_void,
            s_ptr as *const core::ffi::c_long,
            num_tokens as c_int,
            num_heads as c_int,
            head_size as c_int,
            block_size as c_int,
            x as c_int,
            key_stride,
            value_stride,
            dev.cuda_stream().cu_stream(),
            internal_type,
            cache_dtype,
            k_scale_ptr,
            v_scale_ptr,
        )
    }
    Ok(())
}

/// Insert key and values at the provided slot mapping inside the key value paged cache
///
/// # Arguments
///
/// * `key` - Key tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `value` - Value tensor of shape `(num_tokens, num_heads, head_size)`.
/// * `key_cache` - Key cache paged tensor of shape `(num_blocks, num_heads, head_size / x, block_size, x)`
///   with `x` being the size of an element in bytes.
/// * `value_cache` - Value cache paged tensor of shape `(num_blocks, num_heads, head_size, block_size)`.
/// * `slot_mapping` - Mapping associating a slot to each token of shape `(num_tokens)`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    key_cache: &Tensor,
    value_cache: &Tensor,
    slot_mapping: &Tensor,
) -> Result<()> {
    match key.dtype() {
        DType::F16 => update_cache::<f16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        DType::BF16 => update_cache::<bf16>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        DType::F32 => update_cache::<f32>(
            key,
            value,
            k_scale,
            v_scale,
            key_cache,
            value_cache,
            slot_mapping,
        ),
        dt => {
            candle::bail!("reshape_and_cache is only supported for f32, f16 and bf16 ({dt:?})")
        }
    }
}
