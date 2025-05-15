// use candle_core::{DType, Device, IndexOp, Result, Tensor, D};

// /// Builds the (t_idx, indices) tensors once and re-uses them every call.
// pub fn build_delay_indices(
//     b: usize,
//     t: usize,
//     c: usize,
//     delay_pattern: &[i32],
//     dev: &Device,
// ) -> Result<(Tensor, Tensor)> {
//     let delay = Tensor::from_slice(delay_pattern, (c,), dev)?.to_dtype(DType::F32)?;
//     let t_idx_bt = Tensor::arange(0f32, t as f32, dev)?
//         .reshape((1, t))?
//         .repeat(&[b, 1])?
//         .to_dtype(DType::F32)?
//         .unsqueeze(2)?;
//     let t_idx_btc = t_idx_bt.broadcast_sub(&delay.reshape((1, 1, c))?)?;
//     // clamp – Candle has a ‘clamp_min_max’ util.
//     let t_idx_btc = t_idx_btc.clamp(0, (t - 1) as i32)?;
//     // build gather indices [B*T*C, 3]
//     let b_idx = Tensor::arange(0f32, b as f32, dev)?
//         .to_dtype(DType::F32)?
//         .reshape((b, 1, 1))?
//         .repeat(&[1, t, c])?
//         .reshape(b * t * c)?;
//     let t_idx_flat = t_idx_btc.reshape(b * t * c)?;
//     let c_idx = Tensor::arange(0f32, c as f32, dev)?
//         .to_dtype(DType::F32)?
//         .reshape((1, 1, c))?
//         .repeat(&[b, t, 1])?
//         .reshape(b * t * c)?;
//     let stacked = Tensor::stack(&[b_idx, t_idx_flat, c_idx], 1)?;
//     Ok((t_idx_btc, stacked))
// }

// /// Precompute indices for reverting the delay pattern.
// pub fn build_revert_indices(
//     b: usize,
//     t: usize,
//     c: usize,
//     delay_pattern: &[i32],
//     dev: &Device,
// ) -> Result<(Tensor, Tensor)> {
//     let delay = Tensor::from_slice(delay_pattern, (c,), dev)?.to_dtype(DType::F32)?;
//     let t_idx_bt = Tensor::arange(0f32, t as f32, dev)?
//         .reshape((1, t))?
//         .repeat(&[b, 1])?
//         .to_dtype(DType::F32)?
//         .unsqueeze(2)?;
//     let t_idx_btc = t_idx_bt.broadcast_add(&delay.reshape((1, 1, c))?)?;
//     // clamp to valid time range [0, t-1]
//     let t_idx_btc = t_idx_btc.clamp(0, (t - 1) as i32)?;
//     let b_idx = Tensor::arange(0f32, b as f32, dev)?
//         .to_dtype(DType::F32)?
//         .reshape((b, 1, 1))?
//         .repeat(&[b, t, c])?
//         .reshape((b * t * c,))?;
//     let t_idx_flat = t_idx_btc.reshape((b * t * c,))?;
//     let c_idx = Tensor::arange(0f32, c as f32, dev)?
//         .to_dtype(DType::F32)?
//         .reshape((1, 1, c))?
//         .repeat(&[b, t, 1])?
//         .reshape((b * t * c,))?;
//     let stacked = Tensor::stack(&[b_idx, t_idx_flat, c_idx], 1)?;
//     Ok((t_idx_btc, stacked))
// }

// /// Reverts the delay pattern from the audio tensor.
// pub fn revert_audio_delay(
//     audio: &Tensor, // [B, T, C]
//     pad_value: i32,
//     precomp: &(Tensor, Tensor),
//     original_t: usize,
// ) -> Result<Tensor> {
//     let (t_idx, gather_idx) = precomp;
//     let dev = audio.device();
//     let t_idx = t_idx.to_device(dev)?;

//     let gather_idx = gather_idx.to_dtype(DType::F32)?;
//     // Flatten and gather with a single index_select
//     let (b, t, c) = audio.dims3()?;
//     let flat_audio = audio.reshape((b * t * c,))?;
//     let gather_b_idx = gather_idx.i((.., 0))?.squeeze(D::Minus1)?;
//     let gather_t_idx = gather_idx.i((.., 1))?.squeeze(D::Minus1)?;
//     let gather_c_idx = gather_idx.i((.., 2))?.squeeze(D::Minus1)?;
//     let linear_idx =
//         (((&gather_b_idx * (t * c) as f64)? + (&gather_t_idx * c as f64)?)? + &gather_c_idx)?;
//     let gathered_flat = flat_audio.index_select(&linear_idx.to_dtype(DType::U32)?, 0)?;
//     let gathered = gathered_flat.reshape((b, t, c))?;

//     let mask_out = t_idx.ge(original_t as f64)?;
//     let pad = Tensor::full(pad_value as f32, gathered.shape(), dev)?;
//     let result = mask_out.where_cond(&pad, &gathered)?;
//     Ok(result)
// }

// /// Applies the delay pattern.
// pub fn apply_audio_delay(
//     audio: &Tensor, // [B, T, C]
//     pad_value: i32,
//     bos_value: i32,
//     precomp: &(Tensor, Tensor),
// ) -> Result<Tensor> {
//     let (t_idx, gather_idx) = precomp;

//     let gather_idx = gather_idx.to_dtype(DType::F32)?;
//     // Flatten and gather with a single index_select
//     let (b, t, c) = audio.dims3()?;
//     let flat_audio = audio.reshape((b * t * c,))?;
//     let gather_b_idx = gather_idx.i((.., 0))?.squeeze(D::Minus1)?;
//     let gather_t_idx = gather_idx.i((.., 1))?.squeeze(D::Minus1)?;
//     let gather_c_idx = gather_idx.i((.., 2))?.squeeze(D::Minus1)?;
//     let linear_idx =
//         (((&gather_b_idx * (t * c) as f64)? + (&gather_t_idx * c as f64)?)? + &gather_c_idx)?;
//     let gathered_flat = flat_audio.index_select(&linear_idx.to_dtype(DType::U32)?, 0)?;
//     let gathered = gathered_flat.reshape((b, t, c))?;

//     let mask_bos = t_idx.lt(0)?;
//     let mask_pad = t_idx.ge(audio.dims()[1] as f64)?;
//     let bos = Tensor::full(bos_value as f32, gathered.shape(), gathered.device())?;
//     let pad = Tensor::full(pad_value as f32, gathered.shape(), gathered.device())?;
//     let res = mask_bos.where_cond(&bos, &mask_pad.where_cond(&pad, &gathered)?)?;
//     Ok(res)
// }

use candle_core::{DType, Device, Result, Tensor};
use std::ops::{Add, Sub};

/// Precompute indices for applying delay to audio channels.
///
/// # Arguments
/// * `b` - Batch size
/// * `t` - Sequence length
/// * `c` - Number of channels
/// * `delay_pattern` - List of delays per channel
///
/// # Returns
/// A tuple of tensors for efficient delayed indexing:
/// * `t_idx_bxtxc` - Time indices adjusted by delay
/// * `indices_btcx3` - Indices for gathering values
pub fn build_delay_indices(
    b: usize,
    t: usize,
    c: usize,
    delay_pattern: &[i64],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Convert delay pattern to tensor
    let delay_arr = Tensor::from_slice(delay_pattern, (c,), device)?;

    // Create time indices tensor [B, T]
    let t_range = Tensor::arange(0i64, t as i64, device)?;
    let t_idx_bxt = t_range.unsqueeze(0)?.expand((b, t))?;

    // Reshape to [B, T, 1] and broadcast-subtract delays to get [B, T, C]
    let t_idx_bxtx1 = t_idx_bxt.unsqueeze(2)?;
    let delay_view = delay_arr.reshape((1, 1, c))?;
    let t_idx_bxtxc = t_idx_bxtx1.broadcast_sub(&delay_view)?;

    // Create batch indices [B, T, C]
    let b_range = Tensor::arange(0i64, b as i64, device)?;
    let b_idx_bxtxc = b_range.reshape((b, 1, 1))?.expand((b, t, c))?;

    // Create channel indices [B, T, C]
    let c_range = Tensor::arange(0i64, c as i64, device)?;
    let c_idx_bxtxc = c_range.reshape((1, 1, c))?.expand((b, t, c))?;

    // Clamp time indices to valid range [0, T-1]
    let t_max =
        Tensor::from_slice(&[t as i64 - 1], (1,), device)?.broadcast_as(t_idx_bxtxc.shape())?;
    let t_zero = Tensor::zeros((1,), DType::I64, device)?.broadcast_as(t_idx_bxtxc.shape())?;
    let t_clamped_bxtxc = t_idx_bxtxc.clamp(&t_zero, &t_max)?;

    // Reshape for gathering: [B*T*C, 3]
    let b_flat = b_idx_bxtxc.flatten_all()?;
    let t_flat = t_clamped_bxtxc.flatten_all()?;
    let c_flat = c_idx_bxtxc.flatten_all()?;

    // Stack the index tensors along dimension 1
    let indices_btcx3 = Tensor::stack(&[b_flat, t_flat, c_flat], 1)?;

    Ok((t_idx_bxtxc, indices_btcx3))
}

/// Applies audio delay pattern using precomputed indices.
///
/// # Arguments
/// * `audio_bxtxc` - Input audio tensor [B, T, C]
/// * `pad_value` - Value to use for padding
/// * `bos_value` - Value to use for beginning-of-sequence
/// * `precomp` - Precomputed indices from build_delay_indices
///
/// # Returns
/// Delayed audio tensor with same shape as input
pub fn apply_audio_delay(
    audio_bxtxc: &Tensor,
    pad_value: i64,
    bos_value: i64,
    precomp: &(Tensor, Tensor),
) -> Result<Tensor> {
    let device = audio_bxtxc.device();
    let (t_idx_bxtxc, indices_btcx3) = precomp;

    // Get tensor dimensions
    let shape = audio_bxtxc.dims();
    assert_eq!(shape.len(), 3, "Expected 3D tensor for audio_bxtxc");

    // Gather values using precomputed indices
    // Note: Candle may not have direct equivalent to PyTorch's advanced indexing,
    // so we would need to implement this differently depending on Candle's capabilities
    // This is a simplified approach - in practice would need to use Candle's indexing methods
    let gathered_flat = gather_nd(audio_bxtxc, indices_btcx3)?;
    let gathered_bxtxc = gathered_flat.reshape(shape)?;

    // Create masks for BOS and PAD
    let zero = Tensor::zeros((1,), DType::I64, device)?;
    let t_len = Tensor::from_slice(&[shape[1] as i64], (1,), device)?;

    let mask_bos = t_idx_bxtxc.broadcast_lt(&zero)?;
    let mask_pad = t_idx_bxtxc.broadcast_ge(&t_len)?;

    // Create scalar tensors
    let bos_tensor =
        Tensor::from_slice(&[bos_value], (1,), device)?.to_dtype(audio_bxtxc.dtype())?;
    let pad_tensor =
        Tensor::from_slice(&[pad_value], (1,), device)?.to_dtype(audio_bxtxc.dtype())?;

    // Apply masks: if mask_bos, use bos_value; if mask_pad, use pad_value; else use gathered value
    let temp = mask_pad.where_cond(
        &pad_tensor.broadcast_as(mask_pad.shape())?,
        &gathered_bxtxc.broadcast_as(mask_pad.shape())?,
    )?;
    let result_bxtxc = mask_bos.where_cond(&bos_tensor.broadcast_as(mask_pad.shape())?, &temp)?;

    Ok(result_bxtxc)
}

/// Precompute indices for reverting delay pattern.
///
/// # Arguments
/// * `b` - Batch size
/// * `t` - Sequence length
/// * `c` - Number of channels
/// * `delay_pattern` - List of delays per channel
///
/// # Returns
/// A tuple of tensors for efficient revert indexing
pub fn build_revert_indices(
    b: usize,
    t: usize,
    c: usize,
    delay_pattern: &[i64],
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    // Convert delay pattern to tensor
    let delay_arr = Tensor::from_slice(delay_pattern, (c,), device)?;

    // Create time indices tensor
    let t_range = Tensor::arange(0i64, t as i64, device)?;
    let t_idx_bt1 = t_range.unsqueeze(0)?.expand((b, t))?;
    let t_idx_bt1 = t_idx_bt1.unsqueeze(2)?;

    // Add delay to time indices and reshape to [B, T, C]
    let delay_view = delay_arr.reshape((1, 1, c))?;
    let t_plus_delay = t_idx_bt1.broadcast_add(&delay_view)?;

    // Clamp to maximum valid time
    let t_max = Tensor::from_slice(&[t as i64 - 1], (1,), device)?;
    let t_idx_bxtxc = t_plus_delay.broadcast_minimum(&t_max)?;

    // Create batch and channel indices
    let b_range = Tensor::arange(0i64, b as i64, device)?;
    let b_idx_bxtxc = b_range.reshape((b, 1, 1))?.expand((b, t, c))?;

    let c_range = Tensor::arange(0i64, c as i64, device)?;
    let c_idx_bxtxc = c_range.reshape((1, 1, c))?.expand((b, t, c))?;

    // Reshape for gathering: [B*T*C, 3]
    let b_flat = b_idx_bxtxc.flatten_all()?;
    let t_flat = t_idx_bxtxc.flatten_all()?;
    let c_flat = c_idx_bxtxc.flatten_all()?;

    // Stack the index tensors
    let indices_btcx3 = Tensor::stack(&[b_flat, t_flat, c_flat], 1)?;

    Ok((t_idx_bxtxc, indices_btcx3))
}

/// Reverts audio delay using precomputed indices.
///
/// # Arguments
/// * `audio_bxtxc` - Delayed audio tensor
/// * `pad_value` - Value to use for padding
/// * `precomp` - Precomputed revert indices
/// * `t` - Original sequence length
///
/// # Returns
/// Reverted audio tensor
pub fn revert_audio_delay(
    audio_bxtxc: &Tensor,
    pad_value: i64,
    precomp: &(Tensor, Tensor),
    t: usize,
) -> Result<Tensor> {
    let (t_idx_bxtxc, indices_btcx3) = precomp;
    let device = audio_bxtxc.device();

    // Get tensor dimensions
    let shape = audio_bxtxc.dims();

    // Gather values using precomputed indices
    let gathered_flat = gather_nd(audio_bxtxc, indices_btcx3)?;
    let gathered_bxtxc = gathered_flat.reshape(shape)?;

    // Create padding mask
    let t_len = Tensor::from_slice(&[t as i64], (1,), device)?;
    let mask_pad = t_idx_bxtxc.broadcast_ge(&t_len)?;

    // Create pad tensor
    let pad_tensor =
        Tensor::from_slice(&[pad_value], (1,), device)?.to_dtype(audio_bxtxc.dtype())?;

    // Apply mask: if out of bounds, use pad_value; else use gathered value
    let result_bxtxc =
        mask_pad.where_cond(&pad_tensor.broadcast_as(mask_pad.shape())?, &gathered_bxtxc)?;

    Ok(result_bxtxc)
}

// Helper function to implement gather_nd-like functionality
// This is a simplified version - actual implementation would depend on Candle's indexing capabilities
pub fn gather_nd(tensor: &Tensor, indices: &Tensor) -> Result<Tensor> {
    // This is a placeholder that would need to be implemented
    // based on Candle's indexing capabilities

    // For now, let's assume we'd implement something like:
    let n_indices = indices.dim(0)?;
    let mut results = Vec::with_capacity(n_indices);

    // For each set of indices
    for i in 0..n_indices {
        let idx = indices.get(i)?;
        let b = idx.get(0)?.to_scalar::<i64>()?;
        let t = idx.get(1)?.to_scalar::<i64>()?;
        let c = idx.get(2)?.to_scalar::<i64>()?;

        // Extract value at [b, t, c]
        let value = tensor.get(b as usize)?.get(t as usize)?.get(c as usize)?;
        results.push(value);
    }

    // Stack results
    Tensor::stack(&results, 0)
}
