use candle_core::{
    backend::BackendStorage, CpuStorage, DType, Layout, MetalStorage, Result, Shape, Storage,
    Tensor,
};

use crate::metal::kernels::{self, PagedAttentionDType};

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

impl candle_core::CustomOp1 for PagedAttention {
    fn name(&self) -> &'static str {
        "paged-attention"
    }

    fn cpu_fwd(&self, _: &CpuStorage, _: &Layout) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("no cpu support for paged-attention")
    }

    fn metal_fwd(&self, q: &MetalStorage, q_l: &Layout) -> Result<(MetalStorage, Shape)> {
        let ty = match q.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };
        let cache_ty = match self.key_cache.dtype() {
            DType::F16 => PagedAttentionDType::F16,
            DType::BF16 => PagedAttentionDType::BF16,
            DType::F32 => PagedAttentionDType::F32,
            DType::F8E4M3 => PagedAttentionDType::F8E4M3,
            dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
        };

        let dev = q.device();
        let out_shape = q_l.shape().clone();

        let (kc, kc_l) = self.key_cache.storage_and_layout();
        let kc = match &*kc {
            Storage::Metal(kc) => kc,
            _ => candle_core::bail!("key_cache must be a metal tensor"),
        };

        let (vc, vc_l) = self.value_cache.storage_and_layout();
        let vc = match &*vc {
            Storage::Metal(vc) => vc,
            _ => candle_core::bail!("value_cache must be a metal tensor"),
        };

        let (bt, bt_l) = self.block_tables.storage_and_layout();
        let bt = match &*bt {
            Storage::Metal(bt) => bt,
            _ => candle_core::bail!("block_tables must be a metal tensor"),
        };

        let (cl, cl_l) = self.context_lens.storage_and_layout();
        let cl = match &*cl {
            Storage::Metal(cl) => cl,
            _ => candle_core::bail!("context_lens must be a metal tensor"),
        };

        let q_rank = q_l.stride().len();
        let kc_rank = kc_l.stride().len();
        let vc_rank = vc_l.stride().len();

        if q_rank != 3 {
            candle_core::bail!(
                "paged-attention expects `q` tensor to be of rank 3 \
                (q: {q_l:?})"
            )
        }

        if kc_rank != 5 {
            candle_core::bail!(
                "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
            )
        }

        if vc_rank != 4 {
            candle_core::bail!(
                "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
            )
        }

        let alibi_storage_and_offset = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            let (alibi_s, alibi_s_l) = alibi_slopes.storage_and_layout();
            let alibi_s = match &*alibi_s {
                Storage::Metal(alibi_s) => alibi_s,
                _ => candle_core::bail!("context_lens must be a metal tensor"),
            };
            Some((
                alibi_s.clone(),
                alibi_s_l.start_offset() * alibi_s.dtype().size_in_bytes(),
            ))
        } else {
            None
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
            candle_core::bail!(
                "shape mismatch block_tables {:?}, expected {:?}",
                bt_l.shape(),
                (num_seqs, max_num_blocks_per_seq)
            )
        }

        let (num_blocks, num_kv_heads, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
        if head_size_kc != head_size / x {
            candle_core::bail!(
                "shape mismatch value_cache {:?}, expected {:?}",
                vc_l.shape(),
                (num_blocks, num_kv_heads, head_size / x, block_size, x)
            )
        }

        if (num_blocks, num_kv_heads, head_size, block_size) != vc_l.shape().dims4()? {
            candle_core::bail!(
                "shape mismatch key_cache {:?} and value_cache {:?}",
                kc_l.shape(),
                vc_l.shape()
            )
        }

        if (num_seqs) != cl_l.shape().dims1()? {
            candle_core::bail!(
                "shape mismatch context_lens {:?}, expected {:?}",
                cl_l.shape(),
                (num_seqs)
            )
        }

        let k_v_scale = if let (Some(k_scale), Some(v_scale)) = (&self.k_scale, &self.v_scale) {
            if k_scale.elem_count() != 1 || v_scale.elem_count() != 1 {
                candle_core::bail!("k_scale and v_scale must be scalars");
            }
            if k_scale.dtype() != DType::F32 || v_scale.dtype() != DType::F32 {
                candle_core::bail!("k_scale and v_scale must be f32");
            }

            let (k_scale, _) = k_scale.storage_and_layout();
            let k_scale = match &*k_scale {
                Storage::Metal(k_scale) => k_scale,
                _ => candle_core::bail!("k_scale must be a metal tensor"),
            };

            let (v_scale, _) = v_scale.storage_and_layout();
            let v_scale = match &*v_scale {
                Storage::Metal(v_scale) => v_scale,
                _ => candle_core::bail!("v_scale must be a metal tensor"),
            };

            Some((k_scale.buffer().clone(), v_scale.buffer().clone()))
        } else {
            None
        };

        let sinks_storage_and_offset = if let Some(sinks) = self.sinks.as_ref() {
            let (s, s_l) = sinks.storage_and_layout();
            let s = match &*s {
                Storage::Metal(s) => s,
                _ => candle_core::bail!("sinks must be a metal tensor"),
            };
            Some((
                s.buffer().clone(),
                s_l.start_offset() * s.dtype().size_in_bytes(),
            ))
        } else {
            None
        };

        let q_stride = q_l.stride()[0];
        let kv_block_stride = kc_l.stride()[0];
        let kv_head_stride = kc_l.stride()[1];

        let partition_size = 512;
        let max_num_partitions = self.max_context_len.div_ceil(partition_size);
        let use_v1 = (max_num_partitions == 1 || num_seqs * num_heads > 512)
            && partition_size % block_size == 0;

        let elem_count = out_shape.elem_count();

        let encoder = dev.command_encoder()?;
        encoder.set_label("paged-attention");

        let out = dev.new_buffer(elem_count, q.dtype(), "paged-attention-out")?;

        if use_v1 {
            kernels::call_paged_attention_v1(
                dev.device(),
                &encoder,
                &kernels::Kernels::new(),
                ty,
                cache_ty,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                k_v_scale,
                alibi_storage_and_offset,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
                sinks_storage_and_offset
                    .as_ref()
                    .map(|(b, o)| (b as &_, *o)),
            )
            .map_err(candle_core::Error::wrap)?;
        } else {
            let tmp_out_shape = Shape::from((num_seqs, num_heads, max_num_partitions, head_size));
            let exp_sums_shape = Shape::from((num_seqs, num_heads, max_num_partitions));
            let tmp_out = dev.new_buffer(
                tmp_out_shape.elem_count(),
                q.dtype(),
                "paged-attention-tmpout",
            )?;
            let exp_sums = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-expsums",
            )?;
            let max_logits = dev.new_buffer(
                exp_sums_shape.elem_count(),
                DType::F32,
                "paged-attention-maxlogits",
            )?;

            kernels::call_paged_attention_v2(
                dev.device(),
                &encoder,
                &kernels::Kernels::new(),
                ty,
                cache_ty,
                &exp_sums,
                &max_logits,
                q.buffer(),
                q_l.start_offset() * q.dtype().size_in_bytes(),
                kc.buffer(),
                kc_l.start_offset() * kc.dtype().size_in_bytes(),
                vc.buffer(),
                vc_l.start_offset() * vc.dtype().size_in_bytes(),
                bt.buffer(),
                bt_l.start_offset() * bt.dtype().size_in_bytes(),
                cl.buffer(),
                cl_l.start_offset() * cl.dtype().size_in_bytes(),
                k_v_scale,
                alibi_storage_and_offset,
                &tmp_out,
                &out,
                num_kv_heads as i32,
                self.softmax_scale,
                self.softcapping,
                block_size as i32,
                self.max_context_len as i32,
                num_seqs as i32,
                num_heads as i32,
                head_size as i32,
                max_num_blocks_per_seq as i32,
                q_stride as i32,
                kv_block_stride as i32,
                kv_head_stride as i32,
                sinks_storage_and_offset
                    .as_ref()
                    .map(|(b, o)| (b as &_, *o)),
            )
            .map_err(candle_core::Error::wrap)?;
        }

        let newstorage =
            candle_core::MetalStorage::new(out, q.device().clone(), elem_count, q.dtype());
        Ok((newstorage, out_shape))
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
    let kv_ty = match key.dtype() {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
    };
    let cache_ty = match key_cache.dtype() {
        DType::F16 => PagedAttentionDType::F16,
        DType::BF16 => PagedAttentionDType::BF16,
        DType::F32 => PagedAttentionDType::F32,
        DType::F8E4M3 => PagedAttentionDType::F8E4M3,
        dtype => candle_core::bail!("dtype {dtype:?} is not supported"),
    };

    let (k, k_l) = key.storage_and_layout();
    let k = match &*k {
        Storage::Metal(k) => k,
        _ => candle_core::bail!("key must be a metal tensor"),
    };

    let (v, v_l) = value.storage_and_layout();
    let v = match &*v {
        Storage::Metal(v) => v,
        _ => candle_core::bail!("value must be a metal tensor"),
    };

    let (kc, kc_l) = key_cache.storage_and_layout();
    let kc = match &*kc {
        Storage::Metal(kc) => kc,
        _ => candle_core::bail!("key_cache must be a metal tensor"),
    };

    let (vc, vc_l) = value_cache.storage_and_layout();
    let vc = match &*vc {
        Storage::Metal(vc) => vc,
        _ => candle_core::bail!("value_cache must be a metal tensor"),
    };

    let (s, s_l) = slot_mapping.storage_and_layout();
    let s = match &*s {
        Storage::Metal(s) => s,
        _ => candle_core::bail!("slot_mapping must be a metal tensor"),
    };

    let k_v_scale = if let (Some(k_scale), Some(v_scale)) = (k_scale, v_scale) {
        if k_scale.elem_count() != 1 || v_scale.elem_count() != 1 {
            candle_core::bail!("k_scale and v_scale must be scalars");
        }
        if k_scale.dtype() != DType::F32 || v_scale.dtype() != DType::F32 {
            candle_core::bail!("k_scale and v_scale must be f32");
        }

        let (k_scale, _) = k_scale.storage_and_layout();
        let k_scale = match &*k_scale {
            Storage::Metal(k_scale) => k_scale,
            _ => candle_core::bail!("k_scale must be a metal tensor"),
        };

        let (v_scale, _) = v_scale.storage_and_layout();
        let v_scale = match &*v_scale {
            Storage::Metal(v_scale) => v_scale,
            _ => candle_core::bail!("v_scale must be a metal tensor"),
        };

        Some((k_scale.buffer().clone(), v_scale.buffer().clone()))
    } else {
        None
    };

    let k_rank = k_l.stride().len();
    let v_rank = v_l.stride().len();
    let kc_rank = kc_l.stride().len();
    let vc_rank = vc_l.stride().len();

    if k_rank != 3 || v_rank != 3 {
        candle_core::bail!(
            "paged-attention expects input tensors of rank 3 (k: {k_l:?}, v: {v_l:?})"
        )
    }

    if kc_rank != 5 {
        candle_core::bail!(
            "paged-attention expects `key_cache` tensor to be of rank 5 \
                (key_cache: {kc_l:?})"
        )
    }

    if vc_rank != 4 {
        candle_core::bail!(
            "paged-attention expects `value_cache` tensor to be of rank 4 \
                (value_cache: {vc_l:?})"
        )
    }

    let (num_tokens, num_heads, head_size) = k_l.shape().dims3()?;
    if (num_tokens, num_heads, head_size) != v_l.shape().dims3()? {
        candle_core::bail!("shape mismatch k {:?} and v {:?}", k_l.shape(), v_l.shape())
    }

    let (num_blocks, num_heads_kc, head_size_kc, block_size, x) = kc_l.shape().dims5()?;
    if num_heads_kc != num_heads || head_size_kc != head_size / x {
        candle_core::bail!(
            "shape mismatch value_cache {:?}, expected {:?}",
            vc_l.shape(),
            (num_blocks, num_heads, head_size / x, block_size, x)
        )
    }

    if (num_blocks, num_heads, head_size, block_size) != vc_l.shape().dims4()? {
        candle_core::bail!(
            "shape mismatch key_cache {:?} and value_cache {:?}",
            kc_l.shape(),
            vc_l.shape()
        )
    }

    if (num_tokens) != s_l.shape().dims1()? {
        candle_core::bail!(
            "shape mismatch slot_mapping {:?}, expected {:?}",
            s_l.shape(),
            (num_tokens)
        )
    }

    let key_stride = k_l.stride()[0] as i32;
    let value_stride = v_l.stride()[0] as i32;

    let dev = key.device().as_metal_device()?;

    let encoder = dev.command_encoder()?;
    encoder.set_label("reshape-and-cache");

    kernels::call_reshape_and_cache(
        dev.device(),
        &encoder,
        &kernels::Kernels::new(),
        kv_ty,
        cache_ty,
        k.buffer(),
        k_l.start_offset() * key.dtype().size_in_bytes(),
        v.buffer(),
        v_l.start_offset() * value.dtype().size_in_bytes(),
        kc.buffer(),
        kc_l.start_offset() * key_cache.dtype().size_in_bytes(),
        vc.buffer(),
        vc_l.start_offset() * value_cache.dtype().size_in_bytes(),
        s.buffer(),
        s_l.start_offset() * slot_mapping.dtype().size_in_bytes(),
        k_v_scale,
        num_tokens as i32,
        num_heads as i32,
        head_size as i32,
        block_size as i32,
        x as i32,
        key_stride,
        value_stride,
    )
    .map_err(candle_core::Error::wrap)?;

    Ok(())
}
