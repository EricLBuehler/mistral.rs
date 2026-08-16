// GPU test of the F4 KV cache kernels (reshape_and_cache + paged_attention),
// comparing the kernel round-trip against a CPU reference of the same format.
#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Tensor};
use half::f16;

const NUM_HEADS: usize = 2;
const HEAD_SIZE: usize = 64;
const BLOCK_SIZE: usize = 32;
const NUM_TOKENS: usize = 48;
const NUM_BLOCKS: usize = 2;

// llrintf (round half to even), matching the CUDA kernels.
fn rint_half_even(x: f32) -> f32 {
    let f = x.floor();
    let diff = x - f;
    if diff < 0.5 {
        f
    } else if diff > 0.5 {
        f + 1.0
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

fn quant(v: f32, scale: f32) -> u8 {
    let s = if scale > 0.0 { scale } else { 1.0 };
    let q = rint_half_even(v / s).clamp(-8.0, 7.0) as i32 + 8;
    q as u8
}

fn dequant(nib: u8, scale: f32) -> f32 {
    ((nib as i32 - 8) as f32) * scale
}

fn f16f(v: f32) -> f32 {
    f16::from_f32(v).to_f32()
}

fn ref_write(kv: &[f32], slot: &[usize], k_cache: &mut [u8], v_cache: &mut [u8], k_scale: &mut [f32], v_scale: &mut [f32]) {
    for (token, &slot_idx) in slot.iter().enumerate() {
        let token_values: Vec<f32> = (0..NUM_HEADS * HEAD_SIZE)
            .map(|d| f16f(kv[token * NUM_HEADS * HEAD_SIZE + d]))
            .collect();
        let block = slot_idx / BLOCK_SIZE;
        let t = slot_idx % BLOCK_SIZE;
        for h in 0..NUM_HEADS {
            for xrow in 0..HEAD_SIZE / 32 {
                let max = (0..32)
                    .map(|x| token_values[h * HEAD_SIZE + xrow * 32 + x].abs())
                    .fold(0.0f32, f32::max);
                let scale = max / 8.0;
                k_scale[((block * NUM_HEADS + h) * (HEAD_SIZE / 32) + xrow) * BLOCK_SIZE + t] = scale;
                for xoff in 0..32 {
                    let d = xrow * 32 + xoff;
                    let byte = ((block * NUM_HEADS + h) * (HEAD_SIZE / 32) + xrow) * (BLOCK_SIZE * 16) + t * 16 + xoff / 2;
                    let nib = quant(token_values[h * HEAD_SIZE + d], scale);
                    let pos = if xoff % 2 == 0 {
                        (nib & 0x0F) | (k_cache[byte] & 0xF0)
                    } else {
                        ((nib & 0x0F) << 4) | (k_cache[byte] & 0x0F)
                    };
                    k_cache[byte] = pos;
                }
            }
            let max = (0..HEAD_SIZE)
                .map(|d| token_values[h * HEAD_SIZE + d].abs())
                .fold(0.0f32, f32::max);
            let scale = max / 8.0;
            v_scale[(block * NUM_HEADS + h) * BLOCK_SIZE + t] = scale;
            for d in 0..HEAD_SIZE {
                let byte = ((block * NUM_HEADS + h) * (HEAD_SIZE / 2) + d / 2) * BLOCK_SIZE + t;
                let nib = quant(token_values[h * HEAD_SIZE + d], scale);
                let pos = if d % 2 == 0 {
                    (nib & 0x0F) | (v_cache[byte] & 0xF0)
                } else {
                    ((nib & 0x0F) << 4) | (v_cache[byte] & 0x0F)
                };
                v_cache[byte] = pos;
            }
        }
    }
}

fn ref_attention(
    q: &[f32],
    k_cache: &[u8],
    v_cache: &[u8],
    k_scale: &[f32],
    v_scale: &[f32],
    block_table: &[u32],
    context_len: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; NUM_HEADS * HEAD_SIZE];
    for h in 0..NUM_HEADS {
        let mut scores = vec![0f32; context_len];
        for t in 0..context_len {
            let block = (block_table[t / BLOCK_SIZE] as usize) % NUM_BLOCKS;
            let tt = t % BLOCK_SIZE;
            let mut dot = 0f32;
            for d in 0..HEAD_SIZE {
                let xrow = d / 32;
                let xoff = d % 32;
                let byte = ((block * NUM_HEADS + h) * (HEAD_SIZE / 32) + xrow) * (BLOCK_SIZE * 16) + tt * 16 + xoff / 2;
                let nib = if xoff % 2 == 0 { k_cache[byte] & 0x0F } else { (k_cache[byte] >> 4) & 0x0F };
                let scale = k_scale[((block * NUM_HEADS + h) * (HEAD_SIZE / 32) + xrow) * BLOCK_SIZE + tt];
                let k = dequant(nib, scale);
                dot += q[h * HEAD_SIZE + d] * k;
            }
            scores[t] = dot * softmax_scale;
        }
        let max = scores.iter().cloned().fold(f32::MIN, f32::max);
        let mut denom = 0f32;
        for s in &mut scores {
            *s = (*s - max).exp();
            denom += *s;
        }
        for t in 0..context_len {
            let block = (block_table[t / BLOCK_SIZE] as usize) % NUM_BLOCKS;
            let tt = t % BLOCK_SIZE;
            let w = scores[t] / denom;
            for d in 0..HEAD_SIZE {
                let byte = ((block * NUM_HEADS + h) * (HEAD_SIZE / 2) + d / 2) * BLOCK_SIZE + tt;
                let nib = if d % 2 == 0 { v_cache[byte] & 0x0F } else { (v_cache[byte] >> 4) & 0x0F };
                let scale = v_scale[(block * NUM_HEADS + h) * BLOCK_SIZE + tt];
                out[h * HEAD_SIZE + d] += w * dequant(nib, scale);
            }
        }
    }
    out
}

#[test]
fn f4_kernels_match_cpu_reference() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let kv: Vec<f32> = (0..NUM_TOKENS * NUM_HEADS * HEAD_SIZE)
        .map(|i| (((i as u64 * 2654435761) % 1000) as f32 / 500.0 - 1.0) * 1.5)
        .collect();
    let kv_f16: Vec<f16> = kv.iter().map(|&v| f16::from_f32(v)).collect();

    let slot: Vec<i64> = (0..NUM_TOKENS as i64).collect();
    let block_table: Vec<u32> = (0..NUM_BLOCKS as u32).collect();

    // CPU reference writes.
    let mut k_cache_ref = vec![0u8; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 32) * BLOCK_SIZE * 16];
    let mut v_cache_ref = vec![0u8; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 2) * BLOCK_SIZE];
    let mut k_scale_ref = vec![0f32; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 32) * BLOCK_SIZE];
    let mut v_scale_ref = vec![0f32; NUM_BLOCKS * NUM_HEADS * BLOCK_SIZE];
    let slot_us: Vec<usize> = slot.iter().map(|&s| s as usize).collect();
    ref_write(&kv, &slot_us, &mut k_cache_ref, &mut v_cache_ref, &mut k_scale_ref, &mut v_scale_ref);

    // GPU tensors.
    let key = Tensor::from_vec(kv_f16.clone(), (NUM_TOKENS, NUM_HEADS, HEAD_SIZE), &device)?;
    let value = key.clone();
    let slot_mapping = Tensor::from_vec(slot, NUM_TOKENS, &device)?;
    let key_cache = unsafe {
        Tensor::empty(
            (NUM_BLOCKS, NUM_HEADS, HEAD_SIZE / 32, BLOCK_SIZE, 16),
            DType::U8,
            &device,
        )?
    };
    let value_cache = unsafe {
        Tensor::empty(
            (NUM_BLOCKS, NUM_HEADS, HEAD_SIZE / 2, BLOCK_SIZE),
            DType::U8,
            &device,
        )?
    };
    let k_scale = Tensor::zeros(
        (NUM_BLOCKS, NUM_HEADS, HEAD_SIZE / 32, BLOCK_SIZE),
        DType::F32,
        &device,
    )?;
    let v_scale = Tensor::zeros((NUM_BLOCKS, NUM_HEADS, BLOCK_SIZE), DType::F32, &device)?;

    mistralrs_paged_attn::reshape_and_cache(
        &key,
        &value,
        Some(&k_scale),
        Some(&v_scale),
        &key_cache,
        &value_cache,
        &slot_mapping,
    )?;

    // The reshape kernel writes scales inline; verify the packed bytes match.
    let kc_gpu = key_cache.flatten_all()?.to_vec1::<u8>()?;
    let vc_gpu = value_cache.flatten_all()?.to_vec1::<u8>()?;
    let ks_gpu = k_scale.flatten_all()?.to_vec1::<f32>()?;
    let vs_gpu = v_scale.flatten_all()?.to_vec1::<f32>()?;
    for i in 0..k_cache_ref.len() {
        assert_eq!(kc_gpu[i], k_cache_ref[i], "K cache byte {i}");
    }
    for i in 0..v_cache_ref.len() {
        assert_eq!(vc_gpu[i], v_cache_ref[i], "V cache byte {i}");
    }
    for i in 0..k_scale_ref.len() {
        assert!(
            (ks_gpu[i] - k_scale_ref[i]).abs() < 1e-6,
            "K scale {i}: {} vs {}",
            ks_gpu[i],
            k_scale_ref[i]
        );
    }
    for i in 0..v_scale_ref.len() {
        assert!(
            (vs_gpu[i] - v_scale_ref[i]).abs() < 1e-6,
            "V scale {i}: {} vs {}",
            vs_gpu[i],
            v_scale_ref[i]
        );
    }

    // Decode one token against the full context via the real kernels.
    let q: Vec<f32> = (0..NUM_HEADS * HEAD_SIZE)
        .map(|i| ((i * 97) % 50) as f32 / 50.0 - 0.5)
        .collect();
    let q_f16: Vec<f16> = q.iter().map(|&v| f16::from_f32(v)).collect();
    let query = Tensor::from_vec(q_f16, (1, NUM_HEADS, HEAD_SIZE), &device)?;
    let block_tables = Tensor::from_vec(vec![0u32, 1], (1, NUM_BLOCKS), &device)?;
    let context_lens = Tensor::from_vec(vec![NUM_TOKENS as u32], 1, &device)?;
    let out = mistralrs_paged_attn::paged_attention(
        &query,
        Some(&k_scale),
        Some(&v_scale),
        &key_cache,
        &value_cache,
        &block_tables,
        &context_lens,
        None,
        NUM_TOKENS,
        1.0 / (HEAD_SIZE as f32).sqrt(),
        1.0,
        None,
    )?;
    let out_gpu = out.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;

    let out_ref = ref_attention(
        &q,
        &k_cache_ref,
        &v_cache_ref,
        &k_scale_ref,
        &v_scale_ref,
        &block_table,
        NUM_TOKENS,
        1.0 / (HEAD_SIZE as f32).sqrt(),
    );
    for i in 0..out_gpu.len() {
        assert!(
            (out_gpu[i] - out_ref[i]).abs() < 1e-2,
            "attn out {i}: gpu {} vs ref {}",
            out_gpu[i],
            out_ref[i]
        );
    }
    Ok(())
}
