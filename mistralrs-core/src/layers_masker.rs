#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{collections::HashMap, sync::Mutex};

use candle_core::{DType, Device, Result, Tensor, WithDType};
use once_cell::sync::Lazy;

// (bs, tgt_len, past_kv_len)
type MaskKey = (usize, usize, usize);
static MASKS: Lazy<Mutex<HashMap<MaskKey, Tensor>>> = Lazy::new(|| Mutex::new(HashMap::new()));

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
fn masked_fill<D: WithDType>(xs: &Tensor, mask: &Tensor, value: D) -> Result<Tensor> {
    let on_true = Tensor::full(value, xs.shape(), xs.device())?.to_dtype(xs.dtype())?;
    let on_false = xs;
    let res = mask
        .broadcast_as(xs.shape())?
        .where_cond(&on_true, on_false)?;
    Ok(res)
}

impl CausalMasker {
    fn make_mask(&self, tgt_len: usize, past_kv_len: usize, device: &Device) -> Result<Tensor> {
        let offset = tgt_len + past_kv_len;
        let mask: Vec<_> = (0..tgt_len)
            .flat_map(|i| (0..offset).map(move |j| u8::from(j + tgt_len > i + offset)))
            .collect();
        Tensor::from_slice(&mask, (tgt_len, offset), device)
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
        return Ok(k_cache_1.dims()[2]);
    }

    pub fn make_causal_mask_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &[Option<(Tensor, Tensor)>],
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        let past_kv_len = self.calculate_past_kv_len(cache)?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }
        let res = MASKS
            .lock()
            .unwrap()
            .get(&(b_sz, tgt_len, past_kv_len))
            .cloned();
        let causal_mask = if let Some(mask) = res {
            return Ok(Some(mask));
        } else {
            let mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;
            Some(mask)
        };

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        let causal_mask: Option<Result<Tensor>> = causal_mask.map(|mask| {
            let mask =
                mask.broadcast_as((mask.dims()[0], n_attn_heads, mask.dims()[2], mask.dims()[3]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            let mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;

            MASKS
                .lock()
                .unwrap()
                .insert((b_sz, tgt_len, past_kv_len), mask.clone());
            Ok(mask)
        });
        let mask: Option<Tensor> = if let Some(mask) = causal_mask {
            Some(mask?)
        } else {
            None
        };
        Ok(mask)
    }

    pub fn make_causal_mask_with_sliding_window_as_attn_bias(
        &self,
        input_ids: &Tensor,
        cache: &[Option<(Tensor, Tensor)>],
        sliding_window: Option<usize>,
        dtype: DType,
        n_attn_heads: usize,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            return self.make_causal_mask_as_attn_bias(input_ids, cache, dtype, n_attn_heads);
        }
        let sliding_window = sliding_window.unwrap();
        let past_kv_len = self.calculate_past_kv_len(cache)?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }
        let res = MASKS
            .lock()
            .unwrap()
            .get(&(b_sz, tgt_len, past_kv_len))
            .cloned();
        let causal_mask = if let Some(mask) = res {
            return Ok(Some(mask));
        } else {
            let mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let diagonal = past_kv_len as isize - sliding_window as isize - 1;
            let context_mask = apply_tril(&mask.ones_like()?, diagonal)?;
            let mask = masked_fill(&mask.to_dtype(DType::F32)?, &context_mask, f32::MIN)?;
            let mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;

            Some(mask)
        };

        let zero = Tensor::new(0.0f32, input_ids.device())?;
        let causal_mask: Option<Result<Tensor>> = causal_mask.map(|mask| {
            let mask =
                mask.broadcast_as((mask.dims()[0], n_attn_heads, mask.dims()[2], mask.dims()[3]))?;
            // Mask: 1 means use from x (add 0.0), 0 means mask out (add -inf)
            let mask = masked_fill(
                &zero.to_dtype(dtype)?.broadcast_as(mask.shape())?,
                &mask,
                f32::NEG_INFINITY,
            )?;

            MASKS
                .lock()
                .unwrap()
                .insert((b_sz, tgt_len, past_kv_len), mask.clone());
            Ok(mask)
        });
        let mask: Option<Tensor> = if let Some(mask) = causal_mask {
            Some(mask?)
        } else {
            None
        };
        Ok(mask)
    }

    #[deprecated(
        since = "0.1.10",
        note = "use `make_causal_mask_as_attn_bias` instead! \
        This is *not* compatible with `ScaledDotProductAttention`"
    )]
    pub fn make_causal_mask(
        &self,
        input_ids: &Tensor,
        cache: &[Option<(Tensor, Tensor)>],
    ) -> Result<Option<Tensor>> {
        let past_kv_len = self.calculate_past_kv_len(cache)?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }
        let res = MASKS
            .lock()
            .unwrap()
            .get(&(b_sz, tgt_len, past_kv_len))
            .cloned();
        if let Some(mask) = res {
            Ok(Some(mask))
        } else {
            let mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;

            MASKS
                .lock()
                .unwrap()
                .insert((b_sz, tgt_len, past_kv_len), mask.clone());
            Ok(Some(mask))
        }
    }

    #[deprecated(
        since = "0.1.10",
        note = "use `make_causal_mask_with_sliding_window_as_attn_bias` instead! \
        This is *not* compatible with `ScaledDotProductAttention`"
    )]
    pub fn make_causal_mask_with_sliding_window(
        &self,
        input_ids: &Tensor,
        cache: &[Option<(Tensor, Tensor)>],
        sliding_window: Option<usize>,
    ) -> Result<Option<Tensor>> {
        if sliding_window.is_none() {
            #[allow(deprecated)]
            return self.make_causal_mask(input_ids, cache);
        }
        let sliding_window = sliding_window.unwrap();
        let past_kv_len = self.calculate_past_kv_len(cache)?;
        let (b_sz, tgt_len) = input_ids.dims2()?;
        if tgt_len == 1 {
            return Ok(None);
        }
        let res = MASKS
            .lock()
            .unwrap()
            .get(&(b_sz, tgt_len, past_kv_len))
            .cloned();
        if let Some(mask) = res {
            Ok(Some(mask))
        } else {
            let mask = self.make_mask(tgt_len, past_kv_len, input_ids.device())?;
            let diagonal = past_kv_len as isize - sliding_window as isize - 1;
            let context_mask = apply_tril(&mask.ones_like()?, diagonal)?;
            let mask = masked_fill(&mask.to_dtype(DType::F32)?, &context_mask, f32::MIN)?;
            let mask = mask
                .expand((b_sz, 1, tgt_len, tgt_len + past_kv_len))?
                .to_dtype(DType::U8)?;

            MASKS
                .lock()
                .unwrap()
                .insert((b_sz, tgt_len, past_kv_len), mask.clone());
            Ok(Some(mask))
        }
    }

    #[deprecated(
        since = "0.1.10",
        note = "use one of the `*_as_attn_bias` functions\
                    to create an attention bias which can be added."
    )]
    pub fn apply_mask(
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
