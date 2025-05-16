#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use candle_core::{DType, Device, Result, Tensor};

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

pub fn gather_nd(tensor: &Tensor, indices: &Tensor) -> Result<Tensor> {
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
