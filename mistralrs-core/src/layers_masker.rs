#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::ops::Add;

use candle_core::{DType, Device, Result, Tensor, WithDType};
use mistralrs_quant::get_use_matmul_via_f16;

use crate::pipeline::KvCache;

// https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_attn_mask_utils.py
pub struct CausalMasker;

// https://github.com/mokeyish/candle-ext/blob/main/src/triangular.rs
fn apply_tril(xs: &Tensor, diagonal: isize) -> Result<Tensor> {
    let device = xs.device();
    let (l, s) = xs.dims2()?;
    let mut xs_tri = vec![];
    for i in 0..l as isize {
        for j in 0..s as isize {
            let cond = i + diagonal < j;
            xs_tri.push(if cond { 0u8 } else { 1u8 });
        }
    }
    xs * Tensor::from_vec(xs_tri, (l, s), device)?.to_dtype(xs.dtype())?
}

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

impl PastKvLenCache for &[Option<(Tensor, Tensor)>] {
    fn get_past_kv_len(&self) -> Result<usize> {
        let kv_cache_1 = &self[0];
        if kv_cache_1.is_none() {
            return Ok(0);
        }
        let k_cache_1 = &kv_cache_1.as_ref().unwrap().0;
        Ok(k_cache_1.dims()[2])
    }
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

impl PastKvLenCache for Option<&[(Tensor, Tensor)]> {
    fn get_past_kv_len(&self) -> Result<usize> {
        match self {
            None => Ok(0),
            Some([(k_cache_1, _), ..]) => Ok(k_cache_1.dims()[2]),
            _ => candle_core::bail!("Unreachable"),
        }
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
        let sliding_window = sliding_window.unwrap();
        let past_kv_len = cache.get_past_kv_len()?;
        let (_b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = {
            let mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let diagonal = past_kv_len as isize - sliding_window as isize - 1;
            let context_mask = apply_tril(&mask.ones_like()?, diagonal)?;

            masked_fill(&mask.to_dtype(DType::F32)?, &context_mask, f32::MIN)?
                .to_dtype(DType::U8)?
        };

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mask = causal_mask.broadcast_as((causal_mask.dims()[0], causal_mask.dims()[1]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)

            masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?
        };

        Ok(Some(causal_mask))
    }

    #[deprecated(
        since = "0.3.5",
        note = "use `make_causal_mask_matrix_as_attn_bias` instead. This is incompatible with `Sdpa`."
    )]
    pub fn make_causal_mask_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = cache.get_past_kv_len()?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = {
            let mut mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;
            mask
        };

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask = causal_mask.broadcast_as((
                causal_mask.dims()[0],
                n_attn_heads,
                causal_mask.dims()[2],
                causal_mask.dims()[3],
            ))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;

            mask
        };

        // IMPORTANT: this must match the logic in attention.rs. Assume the cublaslt handle will be initialized
        if causal_mask.device().is_cuda() && !get_use_matmul_via_f16() {
            causal_mask = causal_mask.unsqueeze(0)?.repeat((n_attn_heads, 1, 1))?;
        }

        Ok(Some(causal_mask))
    }

    #[deprecated(
        since = "0.3.5",
        note = "use `make_causal_mask_matrix_with_sliding_window_as_attn_bias` instead. This is incompatible with `Sdpa`."
    )]
    pub fn make_causal_mask_with_sliding_window_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &dyn PastKvLenCache,
        sliding_window: Option<usize>,
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            #[allow(deprecated)]
            return self.make_causal_mask_as_attn_bias(input_ids, cache, dtype, n_attn_heads);
        }
        let sliding_window = sliding_window.unwrap();
        let past_kv_len = cache.get_past_kv_len()?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }

        let mut causal_mask = {
            let mut mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let diagonal = past_kv_len as isize - sliding_window as isize - 1;
            let context_mask = apply_tril(&mask.ones_like()?, diagonal)?;
            mask = masked_fill(&mask.to_dtype(DType::F32)?, &context_mask, f32::MIN)?;
            mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;

            mask
        };

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        causal_mask = {
            let mut mask = causal_mask.broadcast_as((
                causal_mask.dims()[0],
                n_attn_heads,
                causal_mask.dims()[2],
                causal_mask.dims()[3],
            ))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;
            mask
        };

        // IMPORTANT: this must match the logic in attention.rs. Assume the cublaslt handle will be initialized
        if causal_mask.device().is_cuda() && !get_use_matmul_via_f16() {
            causal_mask = causal_mask.unsqueeze(0)?.repeat((n_attn_heads, 1, 1))?;
        }

        Ok(Some(causal_mask))
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
