use candle_core::{DType, Device, Result, Tensor};
#[cfg(all(feature = "cuda", target_family = "unix"))]
use mistralrs_paged_attn::flash_attn_sinks;
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

        // === Prefix cache hit path ===
        // When num_cached_tokens is set, Q/K/V contain ONLY new (non-cached) tokens.
        // We write new tokens to cache via reshape_and_cache, then run
        // context_attention_fwd which reads cached tokens from the paged KV cache
        // and new tokens from the input tensors.
        if let Some(num_cached) = &input_metadata.num_cached_tokens {
            if attention_mask.is_some() {
                let query_lens_data = input_metadata
                    .query_lens
                    .as_ref()
                    .expect("query_lens required when num_cached_tokens is set");

                // Reshape Q/K/V from [batch, heads, seq, dim] to [total_tokens, heads, dim]
                let q_flat = query
                    .transpose(1, 2)?
                    .reshape(((), attention_heads, head_size))?;
                let k_flat = key
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;
                let v_flat = value
                    .transpose(1, 2)?
                    .reshape(((), key_value_heads, head_size))?;

                // Write new tokens to cache for future decode steps
                if key_cache.as_ref().is_some_and(|_| value_cache.is_some()) {
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

                let device = query.device();

                // Fast path: gather all K/V from paged cache into contiguous tensors,
                // then use flash_attn_varlen. Much faster than the context_attention_fwd
                // kernel because flash attention is heavily optimized with tiling.
                // Flash attention automatically right-aligns Q when seqlen_q < seqlen_k
                // (causal), so Q token 0 correctly attends to all cached context tokens.
                #[cfg(feature = "flash-attn")]
                if sinks.is_none() && key_cache.as_ref().unwrap().dtype() != DType::F8E4M3 {
                    let kc = key_cache.as_ref().unwrap();
                    let vc = value_cache.as_ref().unwrap();
                    let (_, _, _, block_size_dim, _) = kc.dims5()?;

                    let mut all_k = Vec::new();
                    let mut all_v = Vec::new();
                    let mut cu_seqlens_q_vec = vec![0u32];
                    let mut cu_seqlens_k_vec = vec![0u32];
                    let mut max_sq = 0usize;
                    let mut max_sk = 0usize;

                    for (i, (&nc, &ql)) in
                        num_cached.iter().zip(query_lens_data.iter()).enumerate()
                    {
                        let total_kv = nc + ql;
                        let n_blocks = total_kv.div_ceil(block_size_dim);

                        let block_ids = block_tables
                            .narrow(0, i, 1)?
                            .squeeze(0)?
                            .narrow(0, 0, n_blocks)?;

                        // K cache: [n_blk, kv_h, hd/x, bs, x] -> [total_kv, kv_h, hd]
                        let k_seq = kc
                            .index_select(&block_ids, 0)?
                            .permute((0, 3, 1, 2, 4))?
                            .contiguous()?
                            .reshape((
                                n_blocks * block_size_dim,
                                key_value_heads,
                                head_size,
                            ))?
                            .narrow(0, 0, total_kv)?;

                        // V cache: [n_blk, kv_h, hd, bs] -> [total_kv, kv_h, hd]
                        let v_seq = vc
                            .index_select(&block_ids, 0)?
                            .permute((0, 3, 1, 2))?
                            .contiguous()?
                            .reshape((
                                n_blocks * block_size_dim,
                                key_value_heads,
                                head_size,
                            ))?
                            .narrow(0, 0, total_kv)?;

                        all_k.push(k_seq);
                        all_v.push(v_seq);
                        cu_seqlens_q_vec
                            .push(cu_seqlens_q_vec.last().unwrap() + ql as u32);
                        cu_seqlens_k_vec
                            .push(cu_seqlens_k_vec.last().unwrap() + total_kv as u32);
                        max_sq = max_sq.max(ql);
                        max_sk = max_sk.max(total_kv);
                    }

                    let k_all = Tensor::cat(&all_k, 0)?;
                    let v_all = Tensor::cat(&all_v, 0)?;
                    let cu_q = Tensor::new(&cu_seqlens_q_vec[..], device)?;
                    let cu_k = Tensor::new(&cu_seqlens_k_vec[..], device)?;

                    let window_left = sdpa_params.sliding_window;
                    let window_right = Some(0); // causal

                    let result = if let Some(softcap) = sdpa_params.softcap {
                        candle_flash_attn::flash_attn_varlen_alibi_windowed_softcap(
                            &q_flat,
                            &k_all,
                            &v_all,
                            alibi_slopes.as_ref(),
                            &cu_q,
                            &cu_k,
                            max_sq,
                            max_sk,
                            sdpa_params.softmax_scale,
                            window_left,
                            window_right,
                            softcap,
                        )?
                    } else if let Some(ref slopes) = alibi_slopes {
                        candle_flash_attn::flash_attn_varlen_alibi_windowed(
                            &q_flat,
                            &k_all,
                            &v_all,
                            slopes,
                            &cu_q,
                            &cu_k,
                            max_sq,
                            max_sk,
                            sdpa_params.softmax_scale,
                            window_left,
                            window_right,
                        )?
                    } else {
                        candle_flash_attn::flash_attn_varlen_windowed(
                            &q_flat,
                            &k_all,
                            &v_all,
                            &cu_q,
                            &cu_k,
                            max_sq,
                            max_sk,
                            sdpa_params.softmax_scale,
                            window_left,
                            window_right,
                        )?
                    };

                    // flash_attn_varlen output: [total_new_tokens, num_heads, head_size]
                    let result = result
                        .reshape((batch_size, seq_len, attention_heads, head_size))?
                        .transpose(1, 2)?;
                    return Ok(result);
                }

                // Fallback: context_attention_fwd kernel (for sinks, fp8 cache,
                // or when flash-attn feature is not enabled)
                let sliding_window_val = sdpa_params.sliding_window.map(|w| w as i32).unwrap_or(0);

                #[cfg(all(feature = "cuda", target_family = "unix"))]
                let result = {
                    // Build auxiliary tensors for the CUDA fused kernel
                    let context_lens_t = Tensor::new(
                        &num_cached.iter().map(|&c| c as u32).collect::<Vec<_>>()[..],
                        device,
                    )?;
                    let query_lens_t = Tensor::new(
                        &query_lens_data
                            .iter()
                            .map(|&q| q as u32)
                            .collect::<Vec<_>>()[..],
                        device,
                    )?;

                    // Cumulative start offsets: [0, q0, q0+q1, ...]
                    let mut start_locs = Vec::with_capacity(num_cached.len() + 1);
                    start_locs.push(0u32);
                    for &ql in query_lens_data {
                        start_locs.push(start_locs.last().unwrap() + ql as u32);
                    }
                    let query_start_locs = Tensor::new(&start_locs[..], device)?;

                    // Batch index for each new token: [0,0,...,1,1,...,2,2,...]
                    let total_new: usize = query_lens_data.iter().sum();
                    let mut seq_ids_data = Vec::with_capacity(total_new);
                    for (i, &ql) in query_lens_data.iter().enumerate() {
                        seq_ids_data.extend(std::iter::repeat_n(i as u32, ql));
                    }
                    let seq_ids = Tensor::new(&seq_ids_data[..], device)?;

                    let max_total_kv = num_cached
                        .iter()
                        .zip(query_lens_data.iter())
                        .map(|(&c, &q)| c + q)
                        .max()
                        .unwrap_or(0);

                    mistralrs_paged_attn::context_attention_fwd(
                        &q_flat,
                        &k_flat,
                        &v_flat,
                        self.k_scale.as_ref(),
                        self.v_scale.as_ref(),
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        block_tables,
                        &context_lens_t,
                        &query_lens_t,
                        &query_start_locs,
                        &seq_ids,
                        sdpa_params.softmax_scale,
                        sliding_window_val,
                        sinks,
                        max_total_kv,
                    )?
                };

                #[cfg(not(all(feature = "cuda", target_family = "unix")))]
                let result = {
                    // CPU/Metal fallback: convert block_tables tensor to Vec<Vec<u32>>
                    let bt = block_tables.to_device(&Device::Cpu)?;
                    let (nseq, nblk) = bt.dims2()?;
                    let bt_data: Vec<u32> = bt.flatten_all()?.to_vec1()?;
                    let block_tables_vec: Vec<Vec<u32>> = (0..nseq)
                        .map(|i| bt_data[i * nblk..(i + 1) * nblk].to_vec())
                        .collect();

                    super::context_attention_cpu::context_attention_fwd_cpu(
                        &q_flat,
                        &k_flat,
                        &v_flat,
                        key_cache.as_ref().unwrap(),
                        value_cache.as_ref().unwrap(),
                        &block_tables_vec,
                        num_cached,
                        query_lens_data,
                        sdpa_params.softmax_scale,
                        sdpa_params.sliding_window,
                        sinks,
                    )?
                };

                // Reshape from [total_new_tokens, num_heads, head_size] to
                // [batch_size, num_heads, seq_len, head_size] to match flash_attn output format.
                let result = result
                    .reshape((batch_size, seq_len, attention_heads, head_size))?
                    .transpose(1, 2)?;

                return Ok(result);
            }
        }

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
                            let key_expanded = crate::layers::repeat_kv(key.clone(), n_kv_groups)?;
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
