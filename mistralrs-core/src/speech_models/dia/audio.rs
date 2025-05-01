use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

/// Builds the (t_idx, indices) tensors once and re-uses them every call.
pub fn build_delay_indices(
    b: usize,
    t: usize,
    c: usize,
    delay_pattern: &[i32],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let delay = Tensor::from_slice(delay_pattern, (c,), dev)?.to_dtype(DType::F32)?;
    let t_idx_bt = Tensor::arange(0f32, t as f32, dev)?
        .reshape((1, t))?
        .repeat(&[b, 1])?
        .to_dtype(DType::F32)?
        .unsqueeze(2)?;
    let t_idx_btc = t_idx_bt.broadcast_sub(&delay.reshape((1, 1, c))?)?;
    // clamp – Candle has a ‘clamp_min_max’ util.
    let t_idx_btc = t_idx_btc.clamp(0, (t - 1) as i32)?;
    // build gather indices [B*T*C, 3]
    let b_idx = Tensor::arange(0f32, b as f32, dev)?
        .to_dtype(DType::F32)?
        .reshape((b, 1, 1))?
        .repeat(&[1, t, c])?
        .reshape(b * t * c)?;
    let t_idx_flat = t_idx_btc.reshape(b * t * c)?;
    let c_idx = Tensor::arange(0f32, c as f32, dev)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, c))?
        .repeat(&[b, t, 1])?
        .reshape(b * t * c)?;
    let stacked = Tensor::stack(&[b_idx, t_idx_flat, c_idx], 1)?;
    Ok((t_idx_btc, stacked))
}

/// Precompute indices for reverting the delay pattern.
pub fn build_revert_indices(
    b: usize,
    t: usize,
    c: usize,
    delay_pattern: &[i32],
    dev: &Device,
) -> Result<(Tensor, Tensor)> {
    let delay = Tensor::from_slice(delay_pattern, (c,), dev)?.to_dtype(DType::F32)?;
    let t_idx_bt = Tensor::arange(0f32, t as f32, dev)?
        .reshape((1, t))?
        .repeat(&[b, 1])?
        .to_dtype(DType::F32)?
        .unsqueeze(2)?;
    let t_idx_btc = t_idx_bt.broadcast_add(&delay.reshape((1, 1, c))?)?;
    // clamp to valid time range [0, t-1]
    let t_idx_btc = t_idx_btc.clamp(0, (t - 1) as i32)?;
    let b_idx = Tensor::arange(0f32, b as f32, dev)?
        .to_dtype(DType::F32)?
        .reshape((b, 1, 1))?
        .repeat(&[b, t, c])?
        .reshape((b * t * c,))?;
    let t_idx_flat = t_idx_btc.reshape((b * t * c,))?;
    let c_idx = Tensor::arange(0f32, c as f32, dev)?
        .to_dtype(DType::F32)?
        .reshape((1, 1, c))?
        .repeat(&[b, t, 1])?
        .reshape((b * t * c,))?;
    let stacked = Tensor::stack(&[b_idx, t_idx_flat, c_idx], 1)?;
    Ok((t_idx_btc, stacked))
}

/// Reverts the delay pattern from the audio tensor.
pub fn revert_audio_delay(
    audio: &Tensor, // [B, T, C]
    pad_value: i32,
    precomp: &(Tensor, Tensor),
    original_t: usize,
) -> Result<Tensor> {
    let (t_idx, gather_idx) = precomp;
    let dev = audio.device();
    let t_idx = t_idx.to_device(dev)?;

    let gather_idx = gather_idx.to_dtype(DType::F32)?;
    // Flatten and gather with a single index_select
    let (b, t, c) = audio.dims3()?;
    let flat_audio = audio.reshape((b * t * c,))?;
    let gather_b_idx = gather_idx.i((.., 0))?.squeeze(D::Minus1)?;
    let gather_t_idx = gather_idx.i((.., 1))?.squeeze(D::Minus1)?;
    let gather_c_idx = gather_idx.i((.., 2))?.squeeze(D::Minus1)?;
    let linear_idx =
        (((&gather_b_idx * (t * c) as f64)? + (&gather_t_idx * c as f64)?)? + &gather_c_idx)?;
    let gathered_flat = flat_audio.index_select(&linear_idx.to_dtype(DType::U32)?, 0)?;
    let gathered = gathered_flat.reshape((b, t, c))?;

    let mask_out = t_idx.ge(original_t as f64)?;
    let pad = Tensor::full(pad_value as u32, gathered.shape(), dev)?;
    let result = mask_out.where_cond(&pad, &gathered)?;
    Ok(result)
}

/// Applies the delay pattern.
pub fn apply_audio_delay(
    audio: &Tensor, // [B, T, C]
    pad_value: i32,
    bos_value: i32,
    precomp: &(Tensor, Tensor),
) -> Result<Tensor> {
    let (t_idx, gather_idx) = precomp;

    let gather_idx = gather_idx.to_dtype(DType::F32)?;
    // Flatten and gather with a single index_select
    let (b, t, c) = audio.dims3()?;
    let flat_audio = audio.reshape((b * t * c,))?;
    let gather_b_idx = gather_idx.i((.., 0))?.squeeze(D::Minus1)?;
    let gather_t_idx = gather_idx.i((.., 1))?.squeeze(D::Minus1)?;
    let gather_c_idx = gather_idx.i((.., 2))?.squeeze(D::Minus1)?;
    let linear_idx =
        (((&gather_b_idx * (t * c) as f64)? + (&gather_t_idx * c as f64)?)? + &gather_c_idx)?;
    let gathered_flat = flat_audio.index_select(&linear_idx.to_dtype(DType::U32)?, 0)?;
    let gathered = gathered_flat.reshape((b, t, c))?;

    let mask_bos = t_idx.lt(0)?;
    let mask_pad = t_idx.ge(audio.dims()[1] as f64)?;
    let bos = Tensor::full(bos_value as f32, gathered.shape(), gathered.device())?;
    let pad = Tensor::full(pad_value as f32, gathered.shape(), gathered.device())?;
    let res = mask_bos.where_cond(&bos, &mask_pad.where_cond(&pad, &gathered)?)?;
    Ok(res)
}
