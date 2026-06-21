use candle_core::{backend::BackendStorage, DType, MetalStorage, Result, Shape, Storage, Tensor};
use mistralrs_quant::metal_kernels::{
    call_flash_attn_ext_bf16_dk512, call_flash_attn_ext_vec_bf16_dk512,
    flash_attn_ext_blk_scratch_size, Kernels, FA_NCPSG,
};

const HEAD_DIM: usize = 512;

/// Returns `Ok(None)` when inputs don't match the supported shape so the
/// caller can fall through to the next attention path.
pub(crate) fn try_flash_attn_ext_bf16_dk512(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
) -> Result<Option<Tensor>> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Ok(None);
    }
    if mask.dtype() != DType::F16 && mask.dtype() != DType::BF16 {
        return Ok(None);
    }
    let q_dims = q.dims4()?;
    let k_dims = k.dims4()?;
    let v_dims = v.dims4()?;
    if q_dims.3 != HEAD_DIM || k_dims.3 != HEAD_DIM || v_dims.3 != HEAD_DIM {
        return Ok(None);
    }
    // Unaligned K seq goes through the pad kernel (handled below).
    let mask = match mask.rank() {
        2 => mask.unsqueeze(0)?.unsqueeze(0)?,
        3 => mask.unsqueeze(0)?,
        4 => mask.clone(),
        _ => return Ok(None),
    };

    let (b, n_heads_q, q_seq, _) = q_dims;
    let (b_kv, _n_heads_kv, k_seq, _) = k_dims;
    if b != b_kv {
        return Ok(None);
    }

    // Kernel takes arbitrary strides via ne_/nb_, but we materialize up-front
    // to avoid threading broadcast/non-contig handling through the dispatcher.
    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let mask = if mask.dtype() == DType::F16 {
        mask.contiguous()?
    } else {
        mask.to_dtype(DType::F16)?.contiguous()?
    };

    let q_s = q.storage_and_layout().0;
    let Storage::Metal(q_s) = &*q_s else {
        return Ok(None);
    };
    let k_s = k.storage_and_layout().0;
    let Storage::Metal(k_s) = &*k_s else {
        return Ok(None);
    };
    let v_s = v.storage_and_layout().0;
    let Storage::Metal(v_s) = &*v_s else {
        return Ok(None);
    };
    let m_s = mask.storage_and_layout().0;
    let Storage::Metal(m_s) = &*m_s else {
        return Ok(None);
    };

    let device = q_s.device().clone();
    let out_shape = vec![b, n_heads_q, q_seq, HEAD_DIM];
    let out_buf = device.new_buffer(out_shape.iter().product(), DType::BF16, "fa-ext-out")?;

    let mask_shape = mask.dims();
    let mask_stride = mask.stride();
    let blk_bytes =
        flash_attn_ext_blk_scratch_size(q_seq, k_seq, mask_shape[1].max(1), mask_shape[0].max(1));
    let blk_scratch = device.new_buffer(blk_bytes, DType::U8, "fa-ext-blk")?;

    // When k_seq isn't a multiple of NCPSG we need a pad scratch holding the
    // padded tail of K/V (and optionally a tail mask).
    let n_heads_kv = k_dims.1;
    let pad_scratch = if k_seq % FA_NCPSG != 0 {
        let head_bytes = HEAD_DIM * 2;
        let kv_pad_bytes = head_bytes * FA_NCPSG * n_heads_kv.max(1) * b.max(1);
        let mask_pad_bytes =
            2 * FA_NCPSG * q_seq.max(1) * mask_shape[1].max(1) * mask_shape[0].max(1);
        Some(device.new_buffer(2 * kv_pad_bytes + mask_pad_bytes, DType::U8, "fa-ext-pad")?)
    } else {
        None
    };

    let encoder = device.command_encoder()?;
    encoder.set_label("flash-attn-ext-bf16-dk512");

    call_flash_attn_ext_bf16_dk512(
        device.device(),
        &encoder,
        &Kernels::new(),
        (
            q_s.buffer(),
            q.layout().start_offset() * q.dtype().size_in_bytes(),
        ),
        (
            k_s.buffer(),
            k.layout().start_offset() * k.dtype().size_in_bytes(),
        ),
        (
            v_s.buffer(),
            v.layout().start_offset() * v.dtype().size_in_bytes(),
        ),
        (
            m_s.buffer(),
            mask.layout().start_offset() * mask.dtype().size_in_bytes(),
        ),
        &out_buf,
        &blk_scratch,
        pad_scratch.as_deref(),
        q.dims(),
        q.stride(),
        k.dims(),
        k.stride(),
        v.stride(),
        mask_shape,
        mask_stride,
        scale,
    )
    .map_err(candle_core::Error::wrap)?;

    let out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            out_shape.iter().product(),
            DType::BF16,
        )),
        Shape::from(out_shape),
    ));
    Ok(Some(out))
}

/// Decode-specialized DK=DV=512 BF16 flash attention. Handles q_seq>=1 with
/// arbitrary k_seq. mask=None signals "no mask" (kernel detects via `mask == q`).
pub(crate) fn try_flash_attn_ext_vec_bf16_dk512(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
) -> Result<Option<Tensor>> {
    if q.dtype() != DType::BF16 || k.dtype() != DType::BF16 || v.dtype() != DType::BF16 {
        return Ok(None);
    }
    if let Some(m) = mask {
        if m.dtype() != DType::F16 && m.dtype() != DType::BF16 {
            return Ok(None);
        }
    }
    let q_dims = q.dims4()?;
    let k_dims = k.dims4()?;
    let v_dims = v.dims4()?;
    if q_dims.3 != HEAD_DIM || k_dims.3 != HEAD_DIM || v_dims.3 != HEAD_DIM {
        return Ok(None);
    }
    let mask = if let Some(m) = mask {
        Some(match m.rank() {
            2 => m.unsqueeze(0)?.unsqueeze(0)?,
            3 => m.unsqueeze(0)?,
            4 => m.clone(),
            _ => return Ok(None),
        })
    } else {
        None
    };

    let (b, n_heads_q, q_seq, _) = q_dims;
    let (b_kv, _n_heads_kv, _k_seq, _) = k_dims;
    if b != b_kv {
        return Ok(None);
    }

    let q = q.contiguous()?;
    let k = k.contiguous()?;
    let v = v.contiguous()?;
    let mask = if let Some(m) = mask {
        Some(if m.dtype() == DType::F16 {
            m.contiguous()?
        } else {
            m.to_dtype(DType::F16)?.contiguous()?
        })
    } else {
        None
    };

    let q_s = q.storage_and_layout().0;
    let Storage::Metal(q_s) = &*q_s else {
        return Ok(None);
    };
    let k_s = k.storage_and_layout().0;
    let Storage::Metal(k_s) = &*k_s else {
        return Ok(None);
    };
    let v_s = v.storage_and_layout().0;
    let Storage::Metal(v_s) = &*v_s else {
        return Ok(None);
    };

    // When no mask is provided, signal "no mask" to the kernel by passing q
    // as the mask buffer (kernel checks `mask != q` to gate the mask reads).
    let mask_storage_and_layout = mask.as_ref().map(|m| m.storage_and_layout());
    let mask_metal = match mask_storage_and_layout.as_ref() {
        Some((s, _)) => match &**s {
            Storage::Metal(ms) => Some(ms),
            _ => return Ok(None),
        },
        None => None,
    };

    let device = q_s.device().clone();
    let out_shape = vec![b, n_heads_q, q_seq, HEAD_DIM];
    let out_buf = device.new_buffer(out_shape.iter().product(), DType::BF16, "fa-vec-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("flash-attn-ext-vec-bf16-dk512");

    let (mask_buf, mask_offset, mask_dims, mask_stride) = match (mask_metal, mask.as_ref()) {
        (Some(ms), Some(m)) => (
            ms.buffer(),
            m.layout().start_offset() * m.dtype().size_in_bytes(),
            m.dims().to_vec(),
            m.stride().to_vec(),
        ),
        _ => (
            // Pass q as a dummy mask buffer so the kernel's `mask != q` check
            // resolves to "no mask". The kernel won't actually read from it.
            q_s.buffer(),
            0,
            vec![1, 1, 1, k_dims.2],
            vec![k_dims.2, k_dims.2, k_dims.2, 1],
        ),
    };

    call_flash_attn_ext_vec_bf16_dk512(
        device.device(),
        &encoder,
        &Kernels::new(),
        (
            q_s.buffer(),
            q.layout().start_offset() * q.dtype().size_in_bytes(),
        ),
        (
            k_s.buffer(),
            k.layout().start_offset() * k.dtype().size_in_bytes(),
        ),
        (
            v_s.buffer(),
            v.layout().start_offset() * v.dtype().size_in_bytes(),
        ),
        (mask_buf, mask_offset),
        &out_buf,
        q.dims(),
        q.stride(),
        k.dims(),
        k.stride(),
        v.stride(),
        &mask_dims,
        &mask_stride,
        scale,
    )
    .map_err(candle_core::Error::wrap)?;

    let out = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out_buf,
            device.clone(),
            out_shape.iter().product(),
            DType::BF16,
        )),
        Shape::from(out_shape),
    ));
    Ok(Some(out))
}
