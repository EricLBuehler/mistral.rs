use candle_core::{Result, Tensor};

use super::mla::gather_mla_cache;

/// Softmax over the last dimension (no candle-nn dependency).
fn softmax_last_dim(t: &Tensor) -> Result<Tensor> {
    let max = t.max_keepdim(candle_core::D::Minus1)?;
    let shifted = t.broadcast_sub(&max)?;
    let exp = shifted.exp()?;
    let sum = exp.sum_keepdim(candle_core::D::Minus1)?;
    exp.broadcast_div(&sum)
}

/// Build a causal mask for context attention (MLA variant).
///
/// Returns `[q_len, total_kv]` with 0.0 for valid and -inf for masked positions.
/// Query at local position q can attend to kv positions `[0, ctx_len + q]`.
fn build_causal_mask(
    q_len: usize,
    total_kv: usize,
    ctx_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let mut mask_data = vec![0.0f32; q_len * total_kv];
    let neg_inf = f32::NEG_INFINITY;
    for q in 0..q_len {
        let causal_limit = ctx_len + q + 1;
        for kv in causal_limit..total_kv {
            mask_data[q * total_kv + kv] = neg_inf;
        }
    }
    Tensor::from_vec(mask_data, (q_len, total_kv), device)
}

/// Gather cached MLA tokens from paged cache.
///
/// Uses the CUDA kernel when on GPU, falls back to index_select on CPU.
fn gather_cached_mla(
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    ctx_lens: &[i32],
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let total_ctx: i32 = ctx_lens.iter().sum();
    if total_ctx == 0 {
        candle_core::bail!("gather_cached_mla called with 0 total context tokens");
    }

    if device.is_cuda() {
        // Build metadata for the CUDA gather kernel
        let mut cu_seq = vec![0i32];
        let mut tok_to_seq = Vec::new();
        for (i, &cl) in ctx_lens.iter().enumerate() {
            cu_seq.push(cu_seq.last().unwrap() + cl);
            for _ in 0..cl {
                tok_to_seq.push(i as i32);
            }
        }
        let cu_seq_t = Tensor::new(cu_seq.as_slice(), device)?;
        let tok_to_seq_t = Tensor::new(tok_to_seq.as_slice(), device)?;
        gather_mla_cache(ckv_cache, kpe_cache, block_tables, &cu_seq_t, &tok_to_seq_t)
    } else {
        // CPU fallback: manual gather via index_select
        let (_, block_size, kv_lora_rank) = ckv_cache.dims3()?;
        let (_, _, kpe_head_dim) = kpe_cache.dims3()?;
        let num_blocks = ckv_cache.dims3()?.0;

        // Flatten cache: [num_blocks, block_size, dim] -> [num_blocks*block_size, dim]
        let flat_ckv = ckv_cache.reshape((num_blocks * block_size, kv_lora_rank))?;
        let flat_kpe = kpe_cache.reshape((num_blocks * block_size, kpe_head_dim))?;

        // Read block tables to CPU
        let bt_2d = block_tables.to_dtype(candle_core::DType::I64)?;
        let bt_data: Vec<Vec<i64>> = (0..ctx_lens.len())
            .map(|i| bt_2d.get(i).unwrap().to_vec1::<i64>().unwrap_or_default())
            .collect();

        let mut slot_indices = Vec::with_capacity(total_ctx as usize);
        for (seq_idx, &cl) in ctx_lens.iter().enumerate() {
            for t in 0..cl as usize {
                let block_idx = t / block_size;
                let block_offset = t % block_size;
                let physical_block = bt_data[seq_idx][block_idx] as u32;
                slot_indices.push(physical_block * block_size as u32 + block_offset as u32);
            }
        }
        let slot_tensor = Tensor::new(slot_indices.as_slice(), device)?;
        let ckv_gathered = flat_ckv.index_select(&slot_tensor, 0)?;
        let kpe_gathered = flat_kpe.index_select(&slot_tensor, 0)?;
        Ok((ckv_gathered, kpe_gathered))
    }
}

/// Unfused MLA context attention forward.
///
/// Performs attention in the MLA latent space for prefill with cached prefix.
/// Absorbs `w_uk` into the query to avoid decompressing cached KV:
///   score = (q_nope @ w_uk) · ckv + q_pe · kpe
///   output = softmax(score) @ ckv   (in latent space, kv_lora_rank dim)
///
/// The caller must then project the output by `w_uv_t` to get the final
/// attention output in value-head space.
///
/// # Arguments
///
/// * `q_nope` - `[total_new_tokens, num_heads, qk_nope_head_dim]`
/// * `q_pe`   - `[total_new_tokens, num_heads, qk_rope_head_dim]`
/// * `ckv_new` - `[total_new_tokens, kv_lora_rank]` compressed KV for new tokens
/// * `kpe_new` - `[total_new_tokens, kpe_head_dim]` positional K for new tokens
/// * `ckv_cache` - `[num_blocks, block_size, kv_lora_rank]`
/// * `kpe_cache` - `[num_blocks, block_size, kpe_head_dim]`
/// * `block_tables` - `[num_seqs, max_blocks_per_seq]` i32
/// * `context_lens` - `[num_seqs]` i32, number of cached tokens per sequence
/// * `query_lens`   - `[num_seqs]` i32, number of new tokens per sequence
/// * `w_uk` - `[num_heads, qk_nope_head_dim, kv_lora_rank]`
/// * `softmax_scale` - scaling factor for QK products
///
/// # Returns
///
/// `[total_new_tokens, num_heads, kv_lora_rank]` — attention output in latent space
#[allow(clippy::too_many_arguments)]
pub fn context_attention_fwd_mla(
    q_nope: &Tensor,
    q_pe: &Tensor,
    ckv_new: &Tensor,
    kpe_new: &Tensor,
    ckv_cache: &Tensor,
    kpe_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    query_lens: &Tensor,
    w_uk: &Tensor,
    softmax_scale: f32,
) -> Result<Tensor> {
    let device = q_nope.device();
    let dtype = q_nope.dtype();
    let (_, num_heads, _qk_nope_head_dim) = q_nope.dims3()?;
    let (_, _, qk_rope_head_dim) = q_pe.dims3()?;
    let (_, kv_lora_rank) = ckv_new.dims2()?;

    // Read metadata to CPU
    let ctx_lens: Vec<i32> = context_lens.to_vec1()?;
    let q_lens: Vec<i32> = query_lens.to_vec1()?;
    let num_seqs = ctx_lens.len();

    // Step 1: Absorb w_uk into query
    // q_nope: [total_new, num_heads, qk_nope_head_dim]
    // w_uk:   [num_heads, qk_nope_head_dim, kv_lora_rank]
    // Result: [total_new, num_heads, kv_lora_rank]
    let q_nope_t = q_nope.transpose(0, 1)?; // [num_heads, total_new, qk_nope_head_dim]
    let ql_nope = q_nope_t.matmul(w_uk)?; // [num_heads, total_new, kv_lora_rank]
    let ql_nope = ql_nope.transpose(0, 1)?.contiguous()?; // [total_new, num_heads, kv_lora_rank]

    // Step 2: Gather cached ckv/kpe if any context exists
    let total_ctx: i32 = ctx_lens.iter().sum();
    let (ckv_cached, kpe_cached) = if total_ctx > 0 {
        let (ckv_g, kpe_g) =
            gather_cached_mla(ckv_cache, kpe_cache, block_tables, &ctx_lens, device)?;
        (Some(ckv_g), Some(kpe_g))
    } else {
        (None, None)
    };

    // Step 3: Per-sequence attention in latent space
    let mut outputs = Vec::with_capacity(num_seqs);
    let mut q_offset = 0usize;
    let mut ctx_offset = 0usize;

    for seq_idx in 0..num_seqs {
        let ctx_len = ctx_lens[seq_idx] as usize;
        let q_len = q_lens[seq_idx] as usize;
        let total_kv = ctx_len + q_len;

        // This sequence's latent queries
        let ql_i = ql_nope.narrow(0, q_offset, q_len)?; // [q_len, num_heads, kv_lora_rank]
        let qpe_i = q_pe.narrow(0, q_offset, q_len)?; // [q_len, num_heads, qk_rope_head_dim]

        // This sequence's new ckv/kpe
        let ckv_new_i = ckv_new.narrow(0, q_offset, q_len)?; // [q_len, kv_lora_rank]
        let kpe_new_i = kpe_new.narrow(0, q_offset, q_len)?; // [q_len, kpe_head_dim]

        // Build full ckv/kpe (cached + new)
        let full_ckv;
        let full_kpe;
        if ctx_len > 0 {
            let ckv_ctx = ckv_cached
                .as_ref()
                .unwrap()
                .narrow(0, ctx_offset, ctx_len)?;
            let kpe_ctx = kpe_cached
                .as_ref()
                .unwrap()
                .narrow(0, ctx_offset, ctx_len)?;
            full_ckv = Tensor::cat(&[&ckv_ctx, &ckv_new_i], 0)?; // [total_kv, kv_lora_rank]
            full_kpe = Tensor::cat(&[&kpe_ctx, &kpe_new_i], 0)?; // [total_kv, kpe_head_dim]
            ctx_offset += ctx_len;
        } else {
            full_ckv = ckv_new_i;
            full_kpe = kpe_new_i;
        }

        // Transpose queries to [num_heads, q_len, dim]
        let ql_t = ql_i.transpose(0, 1)?; // [num_heads, q_len, kv_lora_rank]
        let qpe_t = qpe_i.transpose(0, 1)?; // [num_heads, q_len, qk_rope_head_dim]

        // Expand ckv/kpe across heads: [total_kv, dim] -> [num_heads, total_kv, dim]
        let ckv_exp = full_ckv
            .unsqueeze(0)?
            .expand((num_heads, total_kv, kv_lora_rank))?;
        let kpe_exp = full_kpe
            .unsqueeze(0)?
            .expand((num_heads, total_kv, qk_rope_head_dim))?;

        // Score = ql_nope @ ckv^T + q_pe @ kpe^T -> [num_heads, q_len, total_kv]
        let score_nope = ql_t.matmul(&ckv_exp.transpose(1, 2)?)?;
        let score_pe = qpe_t.matmul(&kpe_exp.transpose(1, 2)?)?;
        let scores = ((score_nope + score_pe)? * softmax_scale as f64)?;

        // Apply causal mask
        let mask = build_causal_mask(q_len, total_kv, ctx_len, device)?.to_dtype(dtype)?;
        let scores = scores.broadcast_add(&mask)?;

        // Softmax
        let attn_weights = softmax_last_dim(&scores)?;

        // Output in latent space: attn_weights @ ckv -> [num_heads, q_len, kv_lora_rank]
        let out_i = attn_weights.matmul(&ckv_exp)?;
        let out_i = out_i.transpose(0, 1)?; // [q_len, num_heads, kv_lora_rank]

        outputs.push(out_i);
        q_offset += q_len;
    }

    Tensor::cat(&outputs, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn test_context_attention_fwd_mla_no_context() {
        let device = Device::Cpu;
        let num_heads = 2;
        let qk_nope_head_dim = 8;
        let qk_rope_head_dim = 4;
        let kv_lora_rank = 16;
        let kpe_head_dim = 4;
        let block_size = 4;

        // 3 new tokens, no context
        let q_nope = Tensor::randn(0f32, 1.0, (3, num_heads, qk_nope_head_dim), &device).unwrap();
        let q_pe = Tensor::randn(0f32, 1.0, (3, num_heads, qk_rope_head_dim), &device).unwrap();
        let ckv_new = Tensor::randn(0f32, 1.0, (3, kv_lora_rank), &device).unwrap();
        let kpe_new = Tensor::randn(0f32, 1.0, (3, kpe_head_dim), &device).unwrap();

        let ckv_cache = Tensor::zeros((2, block_size, kv_lora_rank), DType::F32, &device).unwrap();
        let kpe_cache = Tensor::zeros((2, block_size, kpe_head_dim), DType::F32, &device).unwrap();

        let block_tables = Tensor::new(&[0i32, 1], &device)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        let context_lens = Tensor::new(&[0i32], &device).unwrap();
        let query_lens = Tensor::new(&[3i32], &device).unwrap();
        let w_uk = Tensor::randn(
            0f32,
            0.1,
            (num_heads, qk_nope_head_dim, kv_lora_rank),
            &device,
        )
        .unwrap();

        let out = context_attention_fwd_mla(
            &q_nope,
            &q_pe,
            &ckv_new,
            &kpe_new,
            &ckv_cache,
            &kpe_cache,
            &block_tables,
            &context_lens,
            &query_lens,
            &w_uk,
            1.0 / (qk_nope_head_dim as f32 + qk_rope_head_dim as f32).sqrt(),
        )
        .unwrap();

        assert_eq!(out.dims(), &[3, num_heads, kv_lora_rank]);
    }

    #[test]
    fn test_context_attention_fwd_mla_with_context() {
        let device = Device::Cpu;
        let num_heads = 2;
        let qk_nope_head_dim = 8;
        let qk_rope_head_dim = 4;
        let kv_lora_rank = 16;
        let kpe_head_dim = 4;
        let block_size = 4;

        // 1 new token, 2 cached tokens
        let q_nope = Tensor::randn(0f32, 1.0, (1, num_heads, qk_nope_head_dim), &device).unwrap();
        let q_pe = Tensor::randn(0f32, 1.0, (1, num_heads, qk_rope_head_dim), &device).unwrap();
        let ckv_new = Tensor::randn(0f32, 1.0, (1, kv_lora_rank), &device).unwrap();
        let kpe_new = Tensor::randn(0f32, 1.0, (1, kpe_head_dim), &device).unwrap();

        // Cache with some data
        let ckv_cache = Tensor::randn(0f32, 1.0, (2, block_size, kv_lora_rank), &device).unwrap();
        let kpe_cache = Tensor::randn(0f32, 1.0, (2, block_size, kpe_head_dim), &device).unwrap();

        let block_tables = Tensor::new(&[0i32, 1], &device)
            .unwrap()
            .reshape((1, 2))
            .unwrap();
        let context_lens = Tensor::new(&[2i32], &device).unwrap();
        let query_lens = Tensor::new(&[1i32], &device).unwrap();
        let w_uk = Tensor::randn(
            0f32,
            0.1,
            (num_heads, qk_nope_head_dim, kv_lora_rank),
            &device,
        )
        .unwrap();

        let out = context_attention_fwd_mla(
            &q_nope,
            &q_pe,
            &ckv_new,
            &kpe_new,
            &ckv_cache,
            &kpe_cache,
            &block_tables,
            &context_lens,
            &query_lens,
            &w_uk,
            1.0 / (qk_nope_head_dim as f32 + qk_rope_head_dim as f32).sqrt(),
        )
        .unwrap();

        assert_eq!(out.dims(), &[1, num_heads, kv_lora_rank]);
    }

    #[test]
    fn test_context_attention_fwd_mla_multi_seq() {
        let device = Device::Cpu;
        let num_heads = 2;
        let qk_nope_head_dim = 8;
        let qk_rope_head_dim = 4;
        let kv_lora_rank = 16;
        let kpe_head_dim = 4;
        let block_size = 4;

        // Seq 0: 2 new tokens, 3 cached
        // Seq 1: 1 new token, 0 cached
        let total_new = 3;
        let q_nope =
            Tensor::randn(0f32, 1.0, (total_new, num_heads, qk_nope_head_dim), &device).unwrap();
        let q_pe =
            Tensor::randn(0f32, 1.0, (total_new, num_heads, qk_rope_head_dim), &device).unwrap();
        let ckv_new = Tensor::randn(0f32, 1.0, (total_new, kv_lora_rank), &device).unwrap();
        let kpe_new = Tensor::randn(0f32, 1.0, (total_new, kpe_head_dim), &device).unwrap();

        let ckv_cache = Tensor::randn(0f32, 1.0, (4, block_size, kv_lora_rank), &device).unwrap();
        let kpe_cache = Tensor::randn(0f32, 1.0, (4, block_size, kpe_head_dim), &device).unwrap();

        let block_tables = Tensor::new(&[0i32, 1, 2, 3], &device)
            .unwrap()
            .reshape((2, 2))
            .unwrap();
        let context_lens = Tensor::new(&[3i32, 0], &device).unwrap();
        let query_lens = Tensor::new(&[2i32, 1], &device).unwrap();
        let w_uk = Tensor::randn(
            0f32,
            0.1,
            (num_heads, qk_nope_head_dim, kv_lora_rank),
            &device,
        )
        .unwrap();

        let out = context_attention_fwd_mla(
            &q_nope,
            &q_pe,
            &ckv_new,
            &kpe_new,
            &ckv_cache,
            &kpe_cache,
            &block_tables,
            &context_lens,
            &query_lens,
            &w_uk,
            1.0 / (qk_nope_head_dim as f32 + qk_rope_head_dim as f32).sqrt(),
        )
        .unwrap();

        assert_eq!(out.dims(), &[total_new, num_heads, kv_lora_rank]);
    }

    #[test]
    fn test_build_causal_mask_mla() {
        // 2 new tokens, 3 cached tokens
        let mask = build_causal_mask(2, 5, 3, &Device::Cpu).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0 (q=0): [0, 0, 0, 0, -inf] — attends to all 3 ctx + self
        // Row 1 (q=1): [0, 0, 0, 0, 0] — attends to all
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 0.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 0.0);
        assert!(data[4].is_infinite() && data[4] < 0.0);
        for i in 5..10 {
            assert_eq!(data[i], 0.0);
        }
    }

    #[test]
    fn test_causal_mask_no_context() {
        let mask = build_causal_mask(3, 3, 0, &Device::Cpu).unwrap();
        let data: Vec<f32> = mask.flatten_all().unwrap().to_vec1().unwrap();
        // Row 0: [0, -inf, -inf]
        // Row 1: [0, 0, -inf]
        // Row 2: [0, 0, 0]
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite());
        assert!(data[2].is_infinite());
        assert_eq!(data[3], 0.0);
        assert_eq!(data[4], 0.0);
        assert!(data[5].is_infinite());
        assert_eq!(data[6], 0.0);
        assert_eq!(data[7], 0.0);
        assert_eq!(data[8], 0.0);
    }
}
