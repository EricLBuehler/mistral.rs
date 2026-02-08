use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_paged_attn::flash_attn_sinks;
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
        sinks: Option<&Tensor>,
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

        #[allow(clippy::cast_possible_truncation)]
        let att = match attention_mask {
            None => None,
            Some(mask) => {
                if let Some(sinks) = sinks {
                    // Sinks-aware prefill: fused flash attention with per-head sinks.
                    // The sink adds exp(sink_h) to the softmax denominator without a
                    // corresponding value contribution (probability mass absorption).
                    #[cfg(all(feature = "cuda", target_family = "unix"))]
                    {
                        let window = sdpa_params.sliding_window.unwrap_or(0);
                        Some(flash_attn_sinks(
                            query,
                            key,
                            value,
                            Some(sinks),
                            sdpa_params.softmax_scale,
                            window,
                        )?)
                    }
                    #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                    {
                        if query.device().is_metal() {
                            // Fused Metal flash attention with sinks
                            let window = sdpa_params.sliding_window.unwrap_or(0);
                            Some(mistralrs_quant::flash_attn_sinks_metal(
                                query,
                                key,
                                value,
                                Some(sinks),
                                sdpa_params.softmax_scale,
                                window,
                            )?)
                        } else {
                            // CPU fallback: unfused path
                            let n_kv_groups = attention_heads / key_value_heads;
                            let key_expanded =
                                crate::layers::repeat_kv(key.clone(), n_kv_groups)?;
                            let value_expanded =
                                crate::layers::repeat_kv(value.clone(), n_kv_groups)?;
                            let logits = (query.matmul(&key_expanded.transpose(2, 3)?)?
                                * sdpa_params.softmax_scale as f64)?;
                            let logits = logits.broadcast_add(mask)?;
                            let attn_weights =
                                mistralrs_quant::softmax_with_sinks(&logits, sinks, None)?;
                            Some(attn_weights.matmul(&value_expanded)?)
                        }
                    }
                } else {
                    match flash_params {
                        Some(_) => Some(Sdpa.run_attention(
                            query,
                            key,
                            value,
                            Some(mask),
                            flash_params,
                            sdpa_params,
                        )?),
                        None => Some(Sdpa.run_attention_noflash(
                            query,
                            key,
                            value,
                            Some(mask),
                            sdpa_params,
                        )?),
                    }
                }
            }
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
            sinks,
        )?;

        Ok(res)
    }
}
