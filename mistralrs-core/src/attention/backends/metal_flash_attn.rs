use candle_core::{backend::BackendStorage, DType, MetalStorage, Result, Shape, Storage, Tensor};
use mistralrs_quant::metal_kernels::{
    call_flash_attn_ext_bf16_dk512, flash_attn_ext_blk_scratch_size, Kernels, FA_NCPSG,
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
    // unaligned K seq needs the pad kernel; fall back for now
    if k_dims.2 % FA_NCPSG != 0 {
        return Ok(None);
    }
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
        None,
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
