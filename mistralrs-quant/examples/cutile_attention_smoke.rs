use candle_core::{Device, Result, Tensor};
use half::bf16;
use mistralrs_quant::cutile::{cutile_paged_attention_decode, cutile_paged_attention_prefill};

fn bf16_data(len: usize, scale: f32, phase: usize) -> Vec<bf16> {
    (0..len)
        .map(|i| {
            let v = (((i * 17 + phase * 31) % 97) as f32 - 48.0) * scale;
            bf16::from_f32(v)
        })
        .collect()
}

fn attention_ref(
    query: &[bf16],
    key_cache: &[bf16],
    value_cache: &[bf16],
    num_q_heads: usize,
    num_kv_heads: usize,
    block_size: usize,
    head_dim: usize,
    query_len: usize,
    seq_len: usize,
    causal: bool,
    scale: f32,
) -> Vec<bf16> {
    let q_group = num_q_heads / num_kv_heads;
    let mut out = vec![bf16::from_f32(0.0); query_len * num_q_heads * head_dim];
    for q_tok in 0..query_len {
        for q_head in 0..num_q_heads {
            let kv_head = q_head / q_group;
            let max_key = if causal { q_tok + 1 } else { seq_len };
            let mut scores = vec![0.0f32; max_key];
            for k_tok in 0..max_key {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q = query[(q_tok * num_q_heads + q_head) * head_dim + d].to_f32();
                    let k = key_cache[((kv_head * block_size + k_tok) * head_dim) + d].to_f32();
                    dot += q * k;
                }
                scores[k_tok] = dot * scale;
            }
            let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for score in scores.iter_mut() {
                *score = (*score - max_score).exp();
                denom += *score;
            }
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for k_tok in 0..max_key {
                    let v = value_cache[((kv_head * block_size + k_tok) * head_dim) + d].to_f32();
                    acc += scores[k_tok] / denom * v;
                }
                out[(q_tok * num_q_heads + q_head) * head_dim + d] = bf16::from_f32(acc);
            }
        }
    }
    out
}

fn assert_close(name: &str, got: &Tensor, expected: &[bf16]) -> Result<()> {
    let got = got
        .to_device(&Device::Cpu)?
        .flatten_all()?
        .to_vec1::<bf16>()?;
    let mut max_abs = 0.0f32;
    for (a, b) in got.iter().zip(expected) {
        max_abs = max_abs.max((a.to_f32() - b.to_f32()).abs());
    }
    if max_abs > 0.03125 {
        candle_core::bail!("{name} max abs diff {max_abs}");
    }
    println!("{name} max_abs_diff={max_abs:.6}");
    Ok(())
}

fn main() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let block_size = 32usize;
    let head_dim = 512usize;
    let num_q_heads = 8usize;
    let num_kv_heads = 2usize;
    let seq_len = 4usize;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let key_data = bf16_data(num_kv_heads * block_size * head_dim, 0.004, 1);
    let value_data = bf16_data(num_kv_heads * block_size * head_dim, 0.01, 2);
    let key_cache = Tensor::from_vec(
        key_data.clone(),
        (1usize, num_kv_heads, block_size, head_dim),
        &device,
    )?;
    let value_cache = Tensor::from_vec(
        value_data.clone(),
        (1usize, num_kv_heads, block_size, head_dim),
        &device,
    )?;
    let block_tables = Tensor::from_vec(vec![0u32], (1usize, 1usize), &device)?;
    let context_lens = Tensor::from_vec(vec![seq_len as u32], (1usize,), &device)?;

    let query_prefill_data = bf16_data(seq_len * num_q_heads * head_dim, 0.004, 3);
    let query_prefill = Tensor::from_vec(
        query_prefill_data.clone(),
        (seq_len, num_q_heads, head_dim),
        &device,
    )?;
    let q_indptr = Tensor::from_vec(vec![0i32, seq_len as i32], (2usize,), &device)?;
    let prefill = cutile_paged_attention_prefill(
        &query_prefill,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        &q_indptr,
        scale,
    )?;
    let expected_prefill = attention_ref(
        &query_prefill_data,
        &key_data,
        &value_data,
        num_q_heads,
        num_kv_heads,
        block_size,
        head_dim,
        seq_len,
        seq_len,
        true,
        scale,
    );
    assert_close("prefill", &prefill, &expected_prefill)?;

    let query_decode_data = bf16_data(num_q_heads * head_dim, 0.004, 4);
    let query_decode = Tensor::from_vec(
        query_decode_data.clone(),
        (1usize, num_q_heads, head_dim),
        &device,
    )?;
    let decode = cutile_paged_attention_decode(
        &query_decode,
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        scale,
    )?;
    let expected_decode = attention_ref(
        &query_decode_data,
        &key_data,
        &value_data,
        num_q_heads,
        num_kv_heads,
        block_size,
        head_dim,
        1,
        seq_len,
        false,
        scale,
    );
    assert_close("decode", &decode, &expected_decode)?;
    Ok(())
}
