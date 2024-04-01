use std::collections::HashMap;

use candle_core::{DType, Device, Tensor};

use crate::pa::{
    kernels::{
        attention::{apply_paged_attention_v1_, apply_paged_attention_v2_},
        cache::apply_reshape_and_cache,
    },
    InputMetadata,
};

#[cfg(feature = "flash-attn")]
fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    softmax_scale: f32,
    causal: bool,
) -> candle_core::Result<Tensor> {
    candle_core_flash_attn::flash_attn(q, k, v, softmax_scale, causal)
}

#[cfg(not(feature = "flash-attn"))]
fn flash_attn(_: &Tensor, _: &Tensor, _: &Tensor, _: f32, _: bool) -> candle_core::Result<Tensor> {
    unimplemented!("compile with '--features flash-attn'")
}

fn masked_fill(on_false: &Tensor, mask: &Tensor, on_true: f32) -> candle_core::Result<Tensor> {
    let shape = mask.shape();
    let on_true = Tensor::new(on_true, on_false.device())?.broadcast_as(shape.dims())?;
    let m = mask.where_cond(&on_true, on_false)?;
    Ok(m)
}

const _PARTITION_SIZE: usize = 512;
#[derive(Debug, Clone)]
pub struct PagedAttention {
    num_attention_heads: usize,
    head_dim: usize,
    scale: f32,
    num_kv_heads: usize,
    alibi_slopes: Option<Tensor>,
    sliding_window: Option<usize>,
    num_queries_per_kv: usize,
    head_mapping: Tensor,
    use_flash_attn: bool,
    masks: HashMap<usize, Tensor>,
    device: Device,
}

// const _SUPPORTED_HEAD_SIZES: Vec<usize> = vec![64, 80, 96, 112, 128, 256];

impl PagedAttention {
    pub fn new(
        device: &Device,
        num_attention_heads: usize,
        head_dim: usize,
        scale: f32,
        num_key_value_heads: Option<usize>,
        sliding_window: Option<usize>,
        alibi_slopes: Option<Vec<f64>>,
    ) -> candle_core::Result<Self> {
        let _cuda_device = if let Device::Cuda(cuda_dev) = &device {
            cuda_dev
        } else {
            candle_core::bail!("no cuda device")
        };
        let num_kv_heads = num_key_value_heads.unwrap_or(num_attention_heads);
        let num_queries_per_kv = num_attention_heads / num_kv_heads;
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };

        Ok(Self {
            num_attention_heads,
            head_dim,
            num_kv_heads,
            scale,
            sliding_window,
            num_queries_per_kv,
            head_mapping: Tensor::arange(0u32, num_kv_heads as u32, device)?
                .repeat(num_queries_per_kv)?,
            alibi_slopes,
            use_flash_attn: false,
            masks: HashMap::new(),
            device: device.clone(),
        })
    }

    fn repeat_kv(&self, x: Tensor) -> candle_core::Result<Tensor> {
        let n_rep = self.num_attention_heads / self.num_kv_heads;
        if n_rep == 1 {
            Ok(x)
        } else {
            let (b_sz, n_kv_head, seq_len, head_dim) = x.dims4()?;
            let x = x
                .unsqueeze(2)?
                .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
                .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
            Ok(x)
        }
    }

    pub fn make_mask(&mut self, t: usize) -> candle_core::Result<Tensor> {
        let masks = &mut self.masks;
        if let Some(mask) = masks.get(&t) {
            Ok(mask.clone())
        } else {
            let mask: Vec<_> = (0..t)
                .flat_map(|i| (0..t).map(move |j| u8::from(j > i)))
                .collect();
            let mask = Tensor::from_slice(&mask, (t, t), &self.device)?;
            masks.insert(t, mask.clone());
            Ok(mask)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn prompt_attention(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        _hidden_size: usize,
        query: Tensor, //[batch_size*seq_len, num_attention_heads,  head_size]
        key: Tensor,   //[batch_size*seq_len, num_attention_heads,  head_size]
        value: Tensor, //[batch_size*seq_len, num_attention_heads,  head_size]
    ) -> candle_core::Result<Tensor> {
        let query = query
            .reshape((batch_size, seq_len, self.num_attention_heads, self.head_dim))?
            .transpose(1, 2)?;

        let key = key
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let value: Tensor = value
            .reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let key = self.repeat_kv(key)?;
        let value = self.repeat_kv(value)?;

        let y = if self.use_flash_attn {
            // flash-attn expects (b_sz, seq_len, nheads, head_dim)
            let query = query.transpose(1, 2)?;
            let key = key.transpose(1, 2)?;
            let value = value.transpose(1, 2)?;
            let softmax_scale = 1f32 / (self.head_dim as f32).sqrt();
            flash_attn(&query, &key, &value, softmax_scale, seq_len > 1)?.transpose(1, 2)?
        } else {
            let in_dtype = query.dtype();
            let q = query.to_dtype(DType::F32)?.contiguous()?;
            let k: Tensor = key.to_dtype(DType::F32)?.contiguous()?;
            let v = value.to_dtype(DType::F32)?.contiguous()?;

            let att = (q.matmul(&k.t()?)? / (self.head_dim as f64).sqrt())?;

            let mask = self.make_mask(seq_len)?.broadcast_as(att.shape())?;
            let att = masked_fill(&att, &mask, f32::NEG_INFINITY)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;

            // Convert to contiguous as matmul doesn't support strided vs for now.
            att.matmul(&v.contiguous()?)?.to_dtype(in_dtype)?
        };
        let y = y.transpose(1, 2)?;
        Ok(y)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &mut self,
        query: &Tensor, //[batch_size, seq_len, num_heads * head_size]
        key: &Tensor,   //[batch_size, seq_len, num_heads * head_size]
        value: &Tensor, //[batch_size, seq_len, num_heads * head_size]
        key_cache: Option<&Tensor>,
        value_cache: Option<&Tensor>,
        input_metadata: &mut InputMetadata,
        _dtype: DType,
        log_enable: bool,
    ) -> candle_core::Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = query.shape().dims3()?;

        let query = query.reshape(((), self.num_attention_heads, self.head_dim))?;

        let key = key.reshape(((), self.num_kv_heads, self.head_dim))?;

        let value = value.reshape(((), self.num_kv_heads, self.head_dim))?;

        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            let slot_mapping = input_metadata.slot_mapping.flatten_all()?;
            apply_reshape_and_cache(
                &key,
                &value,
                key_cache.unwrap(),
                value_cache.unwrap(),
                &slot_mapping,
            )?;
        }

        let output: Tensor = if input_metadata.is_prompt {
            // Prompt run.
            self.prompt_attention(batch_size, seq_len, hidden_size, query, key, value)?
        } else {
            match (key_cache, value_cache) {
                (Some(key_cache), Some(value_cache)) => {
                    let _max_context_len = input_metadata.max_context_len.unwrap();
                    self.paged_attention(
                        &query,
                        key_cache,
                        value_cache,
                        input_metadata,
                        log_enable,
                    )?
                }
                _ => query.zeros_like()?,
            }
        };

        output.reshape((batch_size, seq_len, hidden_size))
    }

    fn paged_attention(
        &self,
        query: &Tensor,
        key_cache: &Tensor,
        value_cache: &Tensor,
        input_metadata: &mut InputMetadata,
        log_enable: bool,
    ) -> candle_core::Result<Tensor> {
        let num_seqs = query.shape().dims()[0];
        let num_heads = query.shape().dims()[1];
        let max_context_len = input_metadata.max_context_len.unwrap();
        let max_num_partitions = (max_context_len + _PARTITION_SIZE - 1) / _PARTITION_SIZE;

        // # NOTE(woosuk): We use a simple heuristic to decide whether to use
        // # PagedAttention V1 or V2. If the number of partitions is 1, we use
        // # V1 to avoid the overhead of reduction. Also, if the number of
        // # sequences or heads is large, we use V1 since there is enough work
        // # to parallelize.
        // # TODO(woosuk): Tune this heuristic.
        // # For context len > 8192, use V2 kernel to avoid shared memory shortage.
        let use_v1 =
            max_context_len <= 8192 && (max_num_partitions == 1 || num_seqs * num_heads > 512);
        let output = if use_v1 {
            apply_paged_attention_v1_(
                self.scale,
                max_context_len,
                self.num_kv_heads,
                query,
                key_cache,
                value_cache,
                input_metadata.block_tables.as_ref().unwrap(),
                input_metadata.context_lens.as_ref().unwrap(),
                self.alibi_slopes.as_ref(),
            )?
        } else {
            apply_paged_attention_v2_(
                self.scale,
                max_context_len,
                self.num_kv_heads,
                query,
                key_cache,
                value_cache,
                input_metadata.block_tables.as_ref().unwrap(),
                input_metadata.context_lens.as_ref().unwrap(),
                self.alibi_slopes.as_ref(),
            )?
        };
        Ok(output)
    }
}
