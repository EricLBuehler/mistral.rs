use candle_core::{Result, Tensor};

/// CPU fallback for context_attention_fwd.
///
/// This is an unfused implementation using Candle tensor operations. It gathers
/// cached K/V from the paged block table, concatenates with new K/V, and runs
/// standard attention with a causal mask. Intentionally slow — for correctness
/// testing, not production use.
///
/// # Arguments
///
/// * `query` - `[total_new_tokens, num_heads, head_size]`
/// * `key` - `[total_new_tokens, num_kv_heads, head_size]`
/// * `value` - `[total_new_tokens, num_kv_heads, head_size]`
/// * `key_cache` - `[num_blocks, num_kv_heads, head_size/x, block_size, x]`
/// * `value_cache` - `[num_blocks, num_kv_heads, head_size, block_size]`
/// * `block_tables_data` - `[num_seqs][max_blocks_per_seq]` physical block IDs
/// * `context_lens` - Number of cached tokens per sequence
/// * `query_lens` - Number of new tokens per sequence
/// * `softmax_scale` - Scaling factor for QK products
/// * `sliding_window` - Optional sliding window size
/// * `sinks` - Optional per-head sink values `[num_heads]`
#[allow(dead_code, clippy::too_many_arguments, clippy::cast_possible_truncation)]
pub fn context_attention_fwd_cpu(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables_data: &[Vec<u32>],
    context_lens: &[usize],
    query_lens: &[usize],
    softmax_scale: f32,
    sliding_window: Option<usize>,
    sinks: Option<&Tensor>,
) -> Result<Tensor> {
    let device = query.device();
    let dtype = query.dtype();
    let (_, num_heads, head_size) = query.dims3()?;
    let (_, num_kv_heads, _) = key.dims3()?;
    let n_groups = num_heads / num_kv_heads;
    let num_seqs = context_lens.len();

    // Reshape key_cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
    //   -> permute(0,3,1,2,4) -> [num_blocks, block_size, num_kv_heads, head_size/x, x]
    //   -> reshape -> [num_blocks * block_size, num_kv_heads, head_size]
    let kc_shape = key_cache.dims5()?;
    let (num_blocks, _kv_h, hd_x, block_size, x) = kc_shape;
    let flat_kc = key_cache
        .permute((0, 3, 1, 2, 4))?
        .reshape((num_blocks * block_size, num_kv_heads, hd_x * x))?
        .contiguous()?;

    // Reshape value_cache: [num_blocks, num_kv_heads, head_size, block_size]
    //   -> permute(0,3,1,2) -> [num_blocks, block_size, num_kv_heads, head_size]
    //   -> reshape -> [num_blocks * block_size, num_kv_heads, head_size]
    let flat_vc = value_cache
        .permute((0, 3, 1, 2))?
        .reshape((num_blocks * block_size, num_kv_heads, head_size))?
        .contiguous()?;

    let mut outputs = Vec::with_capacity(num_seqs);
    let mut q_offset = 0usize;

    for seq_idx in 0..num_seqs {
        let ctx_len = context_lens[seq_idx];
        let q_len = query_lens[seq_idx];
        let total_kv = ctx_len + q_len;

        // Get this sequence's new tokens
        let q_i = query.narrow(0, q_offset, q_len)?; // [q_len, num_heads, head_size]
        let new_k = key.narrow(0, q_offset, q_len)?; // [q_len, num_kv_heads, head_size]
        let new_v = value.narrow(0, q_offset, q_len)?;

        // Gather cached K/V from block table
        let full_k;
        let full_v;
        if ctx_len > 0 {
            let block_table = &block_tables_data[seq_idx];
            let mut slot_indices = Vec::with_capacity(ctx_len);
            for t in 0..ctx_len {
                let block_idx = t / block_size;
                let block_offset = t % block_size;
                let physical_block = block_table[block_idx];
                slot_indices.push(physical_block * block_size as u32 + block_offset as u32);
            }
            let slot_tensor = Tensor::new(slot_indices.as_slice(), device)?;
            let cached_k = flat_kc.index_select(&slot_tensor, 0)?;
            let cached_v = flat_vc.index_select(&slot_tensor, 0)?;
            full_k = Tensor::cat(&[&cached_k, &new_k], 0)?;
            full_v = Tensor::cat(&[&cached_v, &new_v], 0)?;
        } else {
            full_k = new_k;
            full_v = new_v;
        }

        // GQA: expand KV heads -> [total_kv, num_heads, head_size]
        let full_k = if n_groups > 1 {
            let (tkv, nkv, hs) = full_k.dims3()?;
            full_k
                .unsqueeze(2)?
                .expand((tkv, nkv, n_groups, hs))?
                .reshape((tkv, num_heads, hs))?
        } else {
            full_k
        };
        let full_v = if n_groups > 1 {
            let (tkv, nkv, hs) = full_v.dims3()?;
            full_v
                .unsqueeze(2)?
                .expand((tkv, nkv, n_groups, hs))?
                .reshape((tkv, num_heads, hs))?
        } else {
            full_v
        };

        // Transpose for batched matmul: [num_heads, seq, head_size]
        let q_t = q_i.transpose(0, 1)?; // [num_heads, q_len, head_size]
        let k_t = full_k.transpose(0, 1)?; // [num_heads, total_kv, head_size]
        let v_t = full_v.transpose(0, 1)?; // [num_heads, total_kv, head_size]

        // Compute attention scores: [num_heads, q_len, total_kv]
        let scores = (q_t.matmul(&k_t.transpose(1, 2)?)? * softmax_scale as f64)?;

        // Build causal mask: [q_len, total_kv]
        // Query at local position q attends to kv positions [0, ctx_len + q]
        let mask = build_context_causal_mask(q_len, total_kv, ctx_len, sliding_window, device)?
            .to_dtype(dtype)?;

        // Apply mask: [num_heads, q_len, total_kv]
        let scores = scores.broadcast_add(&mask)?;

        // Softmax (with sinks if needed)
        let attn_weights = if let Some(sinks) = sinks {
            // softmax_with_sinks expects 4D: [batch, heads, q_len, kv_len]
            let scores_4d = scores.unsqueeze(0)?;
            let w = mistralrs_quant::softmax_with_sinks(&scores_4d, sinks, None)?;
            w.squeeze(0)?
        } else {
            candle_nn::ops::softmax_last_dim(&scores)?
        };

        // Weighted sum: [num_heads, q_len, head_size]
        let out_i = attn_weights.matmul(&v_t)?;
        // Transpose back: [q_len, num_heads, head_size]
        let out_i = out_i.transpose(0, 1)?;
        outputs.push(out_i);

        q_offset += q_len;
    }

    Tensor::cat(&outputs, 0)
}

/// Build a causal mask for context attention.
///
/// Returns a `[q_len, total_kv]` tensor where valid positions are 0.0
/// and masked positions are -inf. Query token at local position `q` can attend to:
/// - All context positions `[0, ctx_len)` (already computed)
/// - New tokens `[ctx_len, ctx_len + q]` (causal within new tokens)
///
/// Optionally restricted by sliding window on the query's global position.
#[allow(dead_code)]
fn build_context_causal_mask(
    q_len: usize,
    total_kv: usize,
    ctx_len: usize,
    sliding_window: Option<usize>,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; q_len * total_kv];
    let neg_inf = f32::NEG_INFINITY;

    for q in 0..q_len {
        let q_global = ctx_len + q;
        let causal_limit = ctx_len + q + 1; // can attend up to and including self
        for kv in 0..total_kv {
            let idx = q * total_kv + kv;
            if kv >= causal_limit {
                // Future token — mask out
                mask_data[idx] = neg_inf;
            } else if let Some(window) = sliding_window {
                // Check sliding window: kv position must be within window of query
                if q_global.saturating_sub(kv) >= window {
                    mask_data[idx] = neg_inf;
                }
            }
        }
    }

    Tensor::from_vec(mask_data, (q_len, total_kv), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_build_context_causal_mask_no_context() {
        // No context, 4 new tokens, no sliding window
        let mask = build_context_causal_mask(4, 4, 0, None, &Device::Cpu).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: [0, -inf, -inf, -inf] - can attend to self only
        // Row 1: [0, 0, -inf, -inf]
        // Row 2: [0, 0, 0, -inf]
        // Row 3: [0, 0, 0, 0]
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert_eq!(data[4], 0.0);
        assert_eq!(data[5], 0.0);
        assert!(data[6].is_infinite());
        assert_eq!(data[15], 0.0); // last element
    }

    #[test]
    fn test_build_context_causal_mask_with_context() {
        // 3 context tokens, 2 new tokens
        let mask = build_context_causal_mask(2, 5, 3, None, &Device::Cpu).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0 (q=0): [0, 0, 0, 0, -inf] - attends to all 3 ctx + self
        // Row 1 (q=1): [0, 0, 0, 0, 0] - attends to all
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
        assert!(data[4].is_infinite());
        for i in 5..10 {
            assert_eq!(data[i], 0.0);
        }
    }

    #[test]
    fn test_build_context_causal_mask_sliding_window() {
        // 4 context tokens, 2 new tokens, window=3
        // q=0 global pos=4: can see [2,3,4] (positions within window of 3)
        // q=1 global pos=5: can see [3,4,5]
        let mask = build_context_causal_mask(2, 6, 4, Some(3), &Device::Cpu).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: pos 0,1 masked (too far), 2,3,4 visible, 5 causal-masked
        assert!(data[0].is_infinite()); // pos 0: 4-0=4 >= 3
        assert!(data[1].is_infinite()); // pos 1: 4-1=3 >= 3
        assert_eq!(data[2], 0.0); // pos 2: 4-2=2 < 3
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_infinite()); // causal mask
    }

    #[test]
    fn test_context_attention_fwd_cpu_no_context() {
        let device = Device::Cpu;
        let dtype = DType::F32;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_size = 4;
        let block_size = 4;
        let x = 4; // head_size / x = 1

        // 2 new tokens, no context
        let q = Tensor::randn(0f32, 1.0, (2, num_heads, head_size), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (2, num_kv_heads, head_size), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (2, num_kv_heads, head_size), &device).unwrap();

        // Empty cache (1 block, but unused)
        let key_cache =
            Tensor::zeros((1, num_kv_heads, head_size / x, block_size, x), dtype, &device)
                .unwrap();
        let value_cache =
            Tensor::zeros((1, num_kv_heads, head_size, block_size), dtype, &device).unwrap();

        let block_tables: Vec<Vec<u32>> = vec![vec![0]];
        let context_lens = vec![0usize];
        let query_lens = vec![2usize];

        let out = context_attention_fwd_cpu(
            &q,
            &k,
            &v,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            &query_lens,
            1.0 / (head_size as f32).sqrt(),
            None,
            None,
        )
        .unwrap();

        assert_eq!(out.dims(), &[2, num_heads, head_size]);
    }

    #[test]
    fn test_context_attention_fwd_cpu_with_context() {
        let device = Device::Cpu;
        let num_heads = 2;
        let num_kv_heads = 2;
        let head_size = 4;
        let block_size = 4;
        let x = 4;

        // 1 new token, 2 cached tokens
        let q = Tensor::randn(0f32, 1.0, (1, num_heads, head_size), &device).unwrap();
        let k = Tensor::randn(0f32, 1.0, (1, num_kv_heads, head_size), &device).unwrap();
        let v = Tensor::randn(0f32, 1.0, (1, num_kv_heads, head_size), &device).unwrap();

        // Cache with some data in block 0
        let key_cache =
            Tensor::randn(0f32, 1.0, (2, num_kv_heads, head_size / x, block_size, x), &device)
                .unwrap();
        let value_cache =
            Tensor::randn(0f32, 1.0, (2, num_kv_heads, head_size, block_size), &device).unwrap();

        let block_tables: Vec<Vec<u32>> = vec![vec![0, 1]];
        let context_lens = vec![2usize];
        let query_lens = vec![1usize];

        let out = context_attention_fwd_cpu(
            &q,
            &k,
            &v,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            &query_lens,
            1.0 / (head_size as f32).sqrt(),
            None,
            None,
        )
        .unwrap();

        assert_eq!(out.dims(), &[1, num_heads, head_size]);
    }
}
