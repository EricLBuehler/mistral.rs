use candle_core::{DType, Device, Result, Tensor};
#[allow(unused_imports)]
use mistralrs_paged_attn::{kv_scale_update, paged_attention, reshape_and_cache};

const KV_SCALE_UPDATE_ITERATION: i32 = 128;
use std::sync::atomic::{AtomicI32, Ordering};

use crate::{
    attention::SdpaParams,
    layers::Sdpa,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

pub struct PagedAttention {
    alibi_slopes: Option<Tensor>,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
    kv_updated_times: AtomicI32,
}

impl PagedAttention {
    pub fn new(head_dim: usize, device: &Device, alibi_slopes: Option<Vec<f32>>) -> Result<Self> {
        let alibi_slopes = if let Some(alibi_slopes) = alibi_slopes {
            assert_eq!(alibi_slopes.len(), head_dim);
            Some(Tensor::new(alibi_slopes, device)?)
        } else {
            None
        };
        Ok(Self {
            alibi_slopes,
            k_scale: Some(Tensor::new(1f32, device)?),
            v_scale: Some(Tensor::new(1f32, device)?),
            kv_updated_times: AtomicI32::new(0),
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    #[allow(clippy::too_many_arguments, clippy::cast_possible_truncation)]
    pub fn forward(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        attention_mask: Option<&Tensor>,
        mut key_cache: Option<Tensor>,
        mut value_cache: Option<Tensor>,
        input_metadata: &PagedAttentionInputMetadata,
        sdpa_params: &SdpaParams,
        flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        if let (Some(k_scale), Some(v_scale), Some(key_cache)) =
            (&self.k_scale, &self.v_scale, &key_cache)
        {
            if self.kv_updated_times.load(Ordering::Relaxed) < KV_SCALE_UPDATE_ITERATION
                && key_cache.dtype() == DType::F8E4M3
            {
                // scale update only used for fp8 kvcache
                kv_scale_update(key, value, k_scale, v_scale)?;
                self.kv_updated_times.fetch_add(1, Ordering::Relaxed);
            }
        }

        let slot_mapping = input_metadata
            .slot_mappings
            .get(&query.device().location())
            .unwrap();
        let dims = slot_mapping.dims();
        let slot_mapping = if dims.len() > 1 {
            &slot_mapping.flatten(0, dims.len())?
        } else {
            slot_mapping
        };

        // For models with per-layer sliding windows (GPT-OSS, Gemma2):
        // - Full-attention layers (sliding_window == None) use the full block tables.
        // - Sliding-window layers (sliding_window == Some) use the windowed block tables.
        // If full_block_tables is not populated, fall back to the regular block_tables.
        let use_full =
            sdpa_params.sliding_window.is_none() && input_metadata.full_block_tables.is_some();

        let block_tables = if use_full {
            input_metadata
                .full_block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .block_tables
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };
        let context_lens = if use_full {
            input_metadata
                .full_context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        } else {
            input_metadata
                .context_lens
                .as_ref()
                .unwrap()
                .get(&query.device().location())
                .unwrap()
        };

        let alibi_slopes = if let Some(alibi_slopes) = self.alibi_slopes.as_ref() {
            Some(alibi_slopes.to_device(query.device())?)
        } else {
            None
        };

        let (batch_size, attention_heads, seq_len, head_size) = query.shape().dims4()?;
        let (_, key_value_heads, _, _) = key.shape().dims4()?;

        // === Prefix cache hit path ===
        if input_metadata.num_cached_tokens.is_some() && attention_mask.is_some() {
            // Write new tokens to cache for future decode steps
            if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
                let k_flat = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let v_flat = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                reshape_and_cache(
                    &k_flat,
                    &v_flat,
                    self.k_scale.as_ref(),
                    self.v_scale.as_ref(),
                    key_cache.as_mut().unwrap(),
                    value_cache.as_mut().unwrap(),
                    slot_mapping,
                )?;
            }

            assert!(
                alibi_slopes.is_none(),
                "alibi slopes not supported in prefix cache path"
            );

            let device = query.device();

            // Gather all K/V from paged cache into contiguous tensors.
            // The gather kernel handles x-unpacking for K, transpose for V,
            // and FP8 dequantization via k_scale/v_scale when applicable.
            let cu_kv = input_metadata
                .cu_seqlens_kv
                .as_ref()
                .expect("cu_seqlens_kv required for prefix cache path")
                .get(&device.location())
                .unwrap();
            let (k_gathered, v_gathered) = mistralrs_paged_attn::gather_kv_cache(
                key_cache.as_ref().unwrap(),
                value_cache.as_ref().unwrap(),
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                block_tables,
                cu_kv,
                query.dtype(),
            )?;

            // gathered: (total_kv, kv_heads, dim) -> (1, kv_heads, total_kv, dim)
            let k_4d = k_gathered.unsqueeze(0)?.transpose(1, 2)?;
            let v_4d = v_gathered.unsqueeze(0)?.transpose(1, 2)?;

            // Build a local FlashParams with packed K cu_seqlens from
            // cu_seqlens_kv (matching the gathered KV layout). The pipeline's
            // flash_params uses padded seqlens_k which doesn't match packed KV.
            // Q seqlens stay padded since Q is still in padded batch layout.
            let prefix_flash_params = flash_params.map(|fp| {
                let max_kv = input_metadata
                    .num_cached_tokens
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(input_metadata.query_lens.as_ref().unwrap().iter())
                    .map(|(&nc, &ql)| (nc + ql) as u32)
                    .max()
                    .unwrap_or(0);
                FlashParams {
                    max_q: fp.max_q,
                    max_k: max_kv,
                    cumulative_seqlens_q: fp.cumulative_seqlens_q.clone(),
                    cumulative_seqlens_k: input_metadata.cu_seqlens_kv.as_ref().unwrap().clone(),
                    causal: fp.causal,
                }
            });

            return Sdpa.run_attention(
                query,
                &k_4d,
                &v_4d,
                attention_mask,
                prefix_flash_params.as_ref(),
                sdpa_params,
            );
        }

        #[allow(clippy::cast_possible_truncation)]
        let att = match attention_mask {
            None => None,
            Some(mask) => Some(Sdpa.run_attention(
                query,
                key,
                value,
                Some(mask),
                flash_params,
                sdpa_params,
            )?),
        };

        // paged-attn expects [batch_size, num_tokens, num_heads, head_size]
        let (query, key, value) = if seq_len > 1 {
            let q = query
                .transpose(1, 2)?
                .reshape(((), attention_heads, head_size))?;
            let k = key
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            let v = value
                .transpose(1, 2)?
                .reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        } else {
            // avoid unnecessary transpose for decoding
            let q = query.reshape(((), attention_heads, head_size))?;
            let k = key.reshape(((), key_value_heads, head_size))?;
            let v = value.reshape(((), key_value_heads, head_size))?;
            (q, k, v)
        };

        // key: Tensor,              // [num_tokens, num_heads, head_size]
        // value: Tensor,            // [num_tokens, num_heads, head_size]
        // key_cache: &mut Tensor,   // [num_blocks, num_heads, head_size/x, block_size, x] 48,32,16,16,8
        // value_cache: &mut Tensor, // [num_blocks, num_heads, head_size, block_size] 48,32,128,16
        // slot_mapping: Tensor,     // [num_tokens]
        if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
            reshape_and_cache(
                &key,
                &value,
                self.k_scale.as_ref(),
                self.v_scale.as_ref(),
                key_cache.as_mut().unwrap(),
                value_cache.as_mut().unwrap(),
                slot_mapping,
            )?;
        }

        if let Some(att) = att {
            // Return result in prefill or first prefix chunk
            return Ok(att);
        }

        //  Args:
        //  output: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  query: shape = [num_generation_tokens, num_heads, head_size]
        //
        //  key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
        //      block_size, x]
        //
        //  value_cache: shape = [num_blocks, num_kv_heads, head_size,
        //      block_size]
        //
        //  input_metadata: metadata for paged attention.
        //
        //  alibi_slopes: shape = [num_heads]
        #[allow(clippy::cast_possible_truncation)]
        let res = paged_attention(
            &query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            key_cache.as_ref().unwrap(),
            value_cache.as_ref().unwrap(),
            block_tables,
            context_lens,
            alibi_slopes.as_ref(),
            if use_full {
                input_metadata.full_max_context_len.unwrap()
            } else {
                input_metadata.max_context_len.unwrap()
            },
            sdpa_params.softmax_scale,
            sdpa_params.softcap.unwrap_or(1.0f32),
            sdpa_params.sinks.as_ref(),
        )?;

        Ok(res)
    }
}
