#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::ops::Add;

use candle_core::{DType, Device, Result, Tensor, WithDType, D};

use crate::pipeline::KvCache;

// https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
pub struct CausalMasker;

// https://github.com/mokeyish/candle-ext/blob/main/src/masked_fill.rs
/// xs are on false (0), value is on true (1)
pub fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}

pub trait PastKvLenCache {
    fn get_past_kv_len(&self) -> Result<usize>;
}

impl PastKvLenCache for Vec<KvCache> {
    fn get_past_kv_len(&self) -> Result<usize> {
        let kv_cache_1 = &self[0];
        Ok(kv_cache_1.current_seq_len())
    }
}

impl PastKvLenCache for &[usize] {
    fn get_past_kv_len(&self) -> Result<usize> {
        if self.windows(2).all(|w| w[0] == w[1]) {
            Ok(self[0])
        } else {
            Ok(0)
        }
    }
}

impl PastKvLenCache for Vec<Option<(Tensor, Tensor)>> {
    fn get_past_kv_len(&self) -> Result<usize> {
        let kv_cache_1 = &self[0];
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        Ok(k_cache_1.dims()[2])
    }
}

impl CausalMasker {
    fn make_mask(&self, tgt_len: usize, past_kv_len: usize, device: &Device) -> Result<Tensor> {
        let offset = tgt_len + past_kv_len;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..offset).map(move |j| u8::from(j + tgt_len > i + offset)))
            .collect();
        Tensor::from_slice(&mask, (tgt_len, offset), device)
    }

    fn make_mask_chunked(
        &self,
        tgt_len: usize,
        past_kv_len: usize,
        chunk_size: usize,
        device: &Device,
    ) -> Result<Tensor> {
        let offset = tgt_len + past_kv_len;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..offset).map(move |j| {
                    // For past key-value positions
                    if j < past_kv_len {
                        return 0;
                    }

                    // Adjust j to account for past_kv_len
                    let j_adj = j - past_kv_len;

                    // Calculate block position (equivalent to block_pos)
                    let i_block = i / chunk_size;
                    let j_block = j_adj / chunk_size;
                    let block_pos = (i_block as isize - j_block as isize).abs();

                    // Calculate token position (equivalent to token_pos)
                    let token_pos = j_adj as isize - i as isize;

                    // Apply mask conditions: same block and causal
                    1 - u8::from((block_pos == 0) && (token_pos <= 0))
                })
            })
            .collect();

        Tensor::from_slice(&mask, (tgt_len, offset), device)
    }

    fn make_swa_mask(
        &self,
        tgt_len: usize,
        seqlen_offset: usize,
        sliding_window: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor> {
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| {
                (0..tgt_len).map(move |j| {
                    if i < j || j + sliding_window < i {
                        f32::NEG_INFINITY
                    } else {
                        0.
                    }
                })
            })
            .collect();
        let mask = Tensor::from_slice(&mask, (tgt_len, tgt_len), device)?;
        let mask = if seqlen_offset > 0 {
            let mask0 = Tensor::zeros((tgt_len, seqlen_offset), DType::F32, device)?;
            Tensor::cat(&[&mask0, &mask], D::Minus1)?
        } else {
            mask
        };
        mask.to_dtype(dtype)
    }

    /// Expands a mask from (bs, seq_len) to (bs, 1, tgt_len, seq_len)
    /// If tgt_len is None, use seq_len
    pub fn expand_mask(
        &self,
        mask: &Tensor,
        dtype: DType,
        tgt_len: Option<usize>,
    ) -> Result<Tensor> {
        let (bs, src_len) = mask.dims2()?;

        let expanded_mask = mask.unsqueeze(1)?.unsqueeze(1)?;
        let expanded_mask = expanded_mask
            .expand((bs, 1, tgt_len.unwrap_or(src_len), src_len))?
            .to_dtype(dtype)?;

        let inverted_mask = expanded_mask.neg()?.add(1.0f64)?;
        masked_fill(
            &inverted_mask,
            &inverted_mask.to_dtype(DType::U8)?,
            f32::MIN,
        )
    }

    pub fn calculate_past_kv_len(
        &self,
        cache: &[Option<(Tensor, Tensor)>],
    ) -> candle_core::Result<usize> {
        let kv_cache_1 = &cache[0];
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        Ok(k_cache_1.dims()[2])
    }

    pub fn make_causal_mask_matrix(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        dtype: DType,
        _n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = self
            .make_mask(tgt_len, past_kv_len, input_ids.device())?
            .to_dtype(DType::U8)?;

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask =
                causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;
            mask
        };

        Ok(Some(causal_mask))
    }

    pub fn make_chunked_mask_matrix(
        &self,
        input_ids: &Tensor,
        chunk_size: usize,
        cache: &dyn PastKvLenCache,
        dtype: DType,
        _n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = self
            .make_mask_chunked(tgt_len, past_kv_len, chunk_size, input_ids.device())?
            .to_dtype(DType::U8)?;

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask =
                causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;
            mask
        };

        Ok(Some(causal_mask))
    }

    pub fn make_sliding_window_causal_mask_matrix(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        sliding_window: Option<usize>,
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            return self.make_causal_mask_matrix(input_ids, cache, dtype, n_attn_heads);
        }
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        let sliding_window = sliding_window.unwrap();
        // Compare the past KV len to the sliding window size. If the past kv len is 0 (no prefix cache), then this will be 0.
        // Otherwise, this will be the number required such that the mask fits the size of the k/v seqlen (usually sliding window)
        let past_kv_len = cache
            .get_past_kv_len()?
            .min(sliding_window.saturating_sub(tgt_len));
        if tgt_len == 1 {
            return Ok(None);
        }

        Ok(Some(self.make_swa_mask(
            tgt_len,
            past_kv_len,
            sliding_window,
            input_ids.device(),
            dtype,
        )?))
    }

    pub fn apply_mask_one_and_zero(
        &self,
        mask: &Option<Tensor>,
        att: Tensor,
        neg_inf: &Tensor,
    ) -> Result<Tensor> {
        match mask {
            None => Ok(att),
            Some(mask) => {
                let mask = mask.broadcast_as(att.shape())?;
                mask.where_cond(
                    &neg_inf
                        .to_device(att.device())?
                        .to_dtype(att.dtype())?
                        .broadcast_as(att.dims())?,
                    &att,
                )
            }
        }
    }
}
