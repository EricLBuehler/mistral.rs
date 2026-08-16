// CPU-side verification of the F4 KV cache byte layout and kernel indexing.
// Mirrors the CUDA kernels exactly (reshape_and_cache_f4_kernel,
// pagedattention.cuh F4 branches) so indexing bugs are caught without a GPU.
use candle_core::{Device, Tensor};

const NUM_HEADS: usize = 2;
const HEAD_SIZE: usize = 256;
const BLOCK_SIZE: usize = 32;
const NUM_BLOCKS: usize = 4;

fn quant(v: f32, scale: f32) -> u8 {
    let s = if scale > 0.0 { scale } else { 1.0 };
    let q = (v / s).round().clamp(-8.0, 7.0) as i32 + 8;
    q as u8
}

fn dequant(nib: u8, scale: f32) -> f32 {
    ((nib as i32 - 8) as f32) * scale
}

// K cache: (num_blocks, num_heads, hd/32, block_size, 16) u8; value d of
// token t, head h, xrow = d/32, xoff = d%32 lives at
//   ((block*H + h)*8 + xrow)*(BS*16) + t*16 + xoff/2, nibble xoff%2.
// k_scale: (num_blocks, num_heads, hd/32, block_size) f32, per (h, xrow, t).
fn write_k_ref(
    k_cache: &mut [u8],
    k_scale: &mut [f32],
    values: &[f32], // [tokens][heads][head_size]
    slot: &[usize],
) {
    for (token, &slot_idx) in slot.iter().enumerate() {
        let block = slot_idx / BLOCK_SIZE;
        let t = slot_idx % BLOCK_SIZE;
        for h in 0..NUM_HEADS {
            let mut cell_max = [0f32; HEAD_SIZE / 32];
            for d in 0..HEAD_SIZE {
                let xrow = d / 32;
                cell_max[xrow] = cell_max[xrow].max(values[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d].abs());
            }
            for d in 0..HEAD_SIZE {
                let xrow = d / 32;
                let xoff = d % 32;
                let scale = cell_max[xrow] / 8.0;
                let byte = ((block * NUM_HEADS + h) * 8 + xrow) * (BLOCK_SIZE * 16) + t * 16 + xoff / 2;
                let nib = quant(values[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d], scale);
                let pos = if xoff % 2 == 0 {
                    (nib & 0x0F) | (k_cache[byte] & 0xF0)
                } else {
                    ((nib & 0x0F) << 4) | (k_cache[byte] & 0x0F)
                };
                k_cache[byte] = pos;
                if xoff == 0 {
                    k_scale[((block * NUM_HEADS + h) * 8 + xrow) * BLOCK_SIZE + t] = scale;
                }
            }
        }
    }
}

// V cache: (num_blocks, num_heads, hd/2, block_size) u8; value d of token t
// lives at ((block*H + h)*(hd/2) + d/2)*BS + t, nibble d%2.
// v_scale: (num_blocks, num_heads, block_size) f32, per (h, t).
fn write_v_ref(
    v_cache: &mut [u8],
    v_scale: &mut [f32],
    values: &[f32],
    slot: &[usize],
) {
    for (token, &slot_idx) in slot.iter().enumerate() {
        let block = slot_idx / BLOCK_SIZE;
        let t = slot_idx % BLOCK_SIZE;
        for h in 0..NUM_HEADS {
            let max = (0..HEAD_SIZE)
                .map(|d| values[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d].abs())
                .fold(0.0f32, f32::max);
            let scale = max / 8.0;
            v_scale[(block * NUM_HEADS + h) * BLOCK_SIZE + t] = scale;
            for d in 0..HEAD_SIZE {
                let byte = ((block * NUM_HEADS + h) * (HEAD_SIZE / 2) + d / 2) * BLOCK_SIZE + t;
                let nib = quant(values[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d], scale);
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

// Attention-kernel reads.
fn read_k(k_cache: &[u8], k_scale: &[f32], block: usize, head: usize, token: usize, d: usize) -> f32 {
    let xrow = d / 32;
    let xoff = d % 32;
    let byte = ((block * NUM_HEADS + head) * 8 + xrow) * (BLOCK_SIZE * 16) + token * 16 + xoff / 2;
    let nib = if xoff % 2 == 0 {
        k_cache[byte] & 0x0F
    } else {
        (k_cache[byte] >> 4) & 0x0F
    };
    let scale = k_scale[((block * NUM_HEADS + head) * 8 + xrow) * BLOCK_SIZE + token];
    dequant(nib, scale)
}

fn read_v(v_cache: &[u8], v_scale: &[f32], block: usize, head: usize, token: usize, d: usize) -> f32 {
    let byte = ((block * NUM_HEADS + head) * (HEAD_SIZE / 2) + d / 2) * BLOCK_SIZE + token;
    let nib = if d % 2 == 0 {
        v_cache[byte] & 0x0F
    } else {
        (v_cache[byte] >> 4) & 0x0F
    };
    let scale = v_scale[(block * NUM_HEADS + head) * BLOCK_SIZE + token];
    dequant(nib, scale)
}

#[test]
fn f4_kv_layout_round_trip_close_to_source() {
    let k_cache = vec![0u8; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 32) * BLOCK_SIZE * 16];
    let v_cache = vec![0u8; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 2) * BLOCK_SIZE];
    let k_scale = vec![0f32; NUM_BLOCKS * NUM_HEADS * (HEAD_SIZE / 32) * BLOCK_SIZE];
    let v_scale = vec![0f32; NUM_BLOCKS * NUM_HEADS * BLOCK_SIZE];
    let mut k_cache = k_cache;
    let mut v_cache = v_cache;
    let mut k_scale = k_scale;
    let mut v_scale = v_scale;

    // 100 tokens across 4 blocks.
    let n_tokens = 100;
    let slot: Vec<usize> = (0..n_tokens).collect();
    let k_vals: Vec<f32> = (0..n_tokens * NUM_HEADS * HEAD_SIZE)
        .map(|i| (((i * 7919) % 1000) as f32 / 500.0 - 1.0) * 2.0)
        .collect();
    let v_vals: Vec<f32> = (0..n_tokens * NUM_HEADS * HEAD_SIZE)
        .map(|i| ((i * 104729) % 997) as f32 / 250.0 - 2.0)
        .collect();
    write_k_ref(&mut k_cache, &mut k_scale, &k_vals, &slot);
    write_v_ref(&mut v_cache, &mut v_scale, &v_vals, &slot);

    // Every value round-trips within the 4-bit bound (max_abs/16 per cell).
    for token in 0..n_tokens {
        let block = slot[token] / BLOCK_SIZE;
        let t = slot[token] % BLOCK_SIZE;
        for h in 0..NUM_HEADS {
            for d in 0..HEAD_SIZE {
                let k_orig = k_vals[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d];
                let k_got = read_k(&k_cache, &k_scale, block, h, t, d);
                let cell_max = (0..32)
                    .map(|x| k_vals[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + (d / 32) * 32 + x].abs())
                    .fold(0.0f32, f32::max);
                // Worst case: half a step plus the top-level clamp (q4_0-style).
                let k_bound = cell_max / 8.0;
                assert!(
                    (k_orig - k_got).abs() <= k_bound + 1e-4,
                    "K token {token} h {h} d {d}: {k_orig} -> {k_got}"
                );
                let v_orig = v_vals[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + d];
                let v_got = read_v(&v_cache, &v_scale, block, h, t, d);
                let head_max = (0..HEAD_SIZE)
                    .map(|x| v_vals[token * NUM_HEADS * HEAD_SIZE + h * HEAD_SIZE + x].abs())
                    .fold(0.0f32, f32::max);
                let v_bound = head_max / 8.0;
                assert!(
                    (v_orig - v_got).abs() <= v_bound + 1e-4,
                    "V token {token} h {h} d {d}: {v_orig} -> {v_got}"
                );
            }
        }
    }
    let _ = (Device::Cpu, Tensor::new(1f32, &Device::Cpu));
}
