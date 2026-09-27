//! Decode attention micro-benchmark on Metal: paged kernel vs dense SDPA over the same Q/K/V.
//! Run with `cargo test --release -p mistralrs-core --features metal --lib metal_decode_bench -- --ignored --nocapture`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

const BLOCK_SIZE: usize = 32;
const WARMUP_ITERS: usize = 25;
const TIMED_ITERS: usize = 50;
const CONTEXT_LENS: [usize; 4] = [512, 2048, 8192, 32768];
const BATCHES: [usize; 4] = [1, 4, 8, 16];
// (query heads, kv heads, head dim): Qwen3.8 full attention, then a common GQA-4 shape.
const SHAPES: [(usize, usize, usize); 2] = [(24, 4, 256), (32, 8, 128)];

struct Case {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    batch: usize,
    context: usize,
}

fn time_ms(device: &Device, mut f: impl FnMut() -> Result<Tensor>) -> Result<f64> {
    for _ in 0..WARMUP_ITERS {
        f()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..TIMED_ITERS {
        f()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / TIMED_ITERS as f64)
}

fn paged_ms(device: &Device, c: &Case) -> Result<f64> {
    let dtype = DType::BF16;
    let x = 16 / dtype.size_in_bytes();
    let blocks_per_seq = c.context.div_ceil(BLOCK_SIZE);
    let num_blocks = c.batch * blocks_per_seq;
    let q = Tensor::randn(0f32, 1., (c.batch, c.q_heads, c.head_dim), device)?.to_dtype(dtype)?;
    let key_cache = Tensor::randn(
        0f32,
        1.,
        (num_blocks, c.kv_heads, c.head_dim / x, BLOCK_SIZE, x),
        device,
    )?
    .to_dtype(dtype)?;
    let value_cache = Tensor::randn(
        0f32,
        1.,
        (num_blocks, c.kv_heads, c.head_dim, BLOCK_SIZE),
        device,
    )?
    .to_dtype(dtype)?;
    let block_tables =
        Tensor::arange(0u32, num_blocks as u32, device)?.reshape((c.batch, blocks_per_seq))?;
    let context_lens = Tensor::full(c.context as u32, c.batch, device)?;
    let scale = 1. / (c.head_dim as f32).sqrt();
    time_ms(device, || {
        mistralrs_paged_attn::paged_attention(
            &q,
            None,
            None,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            None,
            c.context,
            scale,
            1.,
            None,
        )
    })
}

fn dense_ms(device: &Device, c: &Case) -> Result<f64> {
    let dtype = DType::BF16;
    let q =
        Tensor::randn(0f32, 1., (c.batch, c.q_heads, 1, c.head_dim), device)?.to_dtype(dtype)?;
    let k = Tensor::randn(
        0f32,
        1.,
        (c.batch, c.kv_heads, c.context, c.head_dim),
        device,
    )?
    .to_dtype(dtype)?;
    let v = Tensor::randn(
        0f32,
        1.,
        (c.batch, c.kv_heads, c.context, c.head_dim),
        device,
    )?
    .to_dtype(dtype)?;
    let scale = 1. / (c.head_dim as f32).sqrt();
    time_ms(device, || {
        candle_nn::ops::sdpa(&q, &k, &v, None, false, scale, 1.)
    })
}

#[test]
#[ignore = "benchmark; needs a Metal GPU"]
fn metal_decode_bench() -> Result<()> {
    let device = Device::new_metal(0)?;
    println!("shape(qh/kvh/d) batch context | paged_ms  GB/s | dense_ms  GB/s | paged/dense");
    for (q_heads, kv_heads, head_dim) in SHAPES {
        for batch in BATCHES {
            for context in CONTEXT_LENS {
                let c = Case {
                    q_heads,
                    kv_heads,
                    head_dim,
                    batch,
                    context,
                };
                let kv_gb = (2 * batch * kv_heads * context * head_dim * 2) as f64 / 1e9;
                let paged = paged_ms(&device, &c)?;
                let dense = dense_ms(&device, &c)?;
                println!(
                    "{q_heads}/{kv_heads}/{head_dim} {batch:>5} {context:>7} | {paged:8.3} {:5.0} | {dense:8.3} {:5.0} | {:6.2}x",
                    kv_gb / (paged / 1e3),
                    kv_gb / (dense / 1e3),
                    paged / dense
                );
            }
        }
    }
    Ok(())
}

/// Packs dense `[kv_heads, len, D]` K/V into one sequence's paged blocks, padding past `len` with NaN.
fn pack_seq(k: &Tensor, v: &Tensor, blocks: usize) -> Result<(Tensor, Tensor)> {
    let (kv_heads, len, d) = k.dims3()?;
    let x = 8;
    let padded = blocks * BLOCK_SIZE;
    let pad = |t: &Tensor| -> Result<Tensor> {
        if padded == len {
            return Ok(t.clone());
        }
        let nan =
            Tensor::full(f32::NAN, (kv_heads, padded - len, d), t.device())?.to_dtype(t.dtype())?;
        Tensor::cat(&[t, &nan], 1)
    };
    let k = pad(k)?
        .reshape((kv_heads, blocks, BLOCK_SIZE, d / x, x))?
        .permute((1, 0, 3, 2, 4))?
        .contiguous()?;
    let v = pad(v)?
        .reshape((kv_heads, blocks, BLOCK_SIZE, d))?
        .permute((1, 0, 3, 2))?
        .contiguous()?;
    Ok((k, v))
}

#[test]
fn paged_decode_matches_dense_sdpa() -> Result<()> {
    let device = Device::new_metal(0)?;
    let dtype = DType::BF16;
    for (q_heads, kv_heads, head_dim, lens) in [
        (24, 4, 256, vec![1usize, 37, 700, 1500]),
        (32, 8, 128, vec![33, 512, 513, 2049]),
        (8, 2, 512, vec![1, 37, 90, 700]),
        (8, 2, 256, vec![1, 37, 90, 700]),
    ] {
        let batch = lens.len();
        let max_len = *lens.iter().max().unwrap();
        let blocks_per_seq = max_len.div_ceil(BLOCK_SIZE);
        let scale = 1. / (head_dim as f32).sqrt();
        let q = Tensor::randn(0f32, 1., (batch, q_heads, head_dim), &device)?.to_dtype(dtype)?;
        let (mut k_blocks, mut v_blocks, mut expected) = (vec![], vec![], vec![]);
        for (b, &len) in lens.iter().enumerate() {
            let k = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), &device)?.to_dtype(dtype)?;
            let v = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), &device)?.to_dtype(dtype)?;
            let (kb, vb) = pack_seq(&k, &v, blocks_per_seq)?;
            k_blocks.push(kb);
            v_blocks.push(vb);
            let gqa = q_heads / kv_heads;
            let qb = q
                .narrow(0, b, 1)?
                .to_dtype(DType::F32)?
                .reshape((kv_heads, gqa, head_dim))?;
            let kf = k.to_dtype(DType::F32)?;
            let att = (qb.matmul(&kf.t()?)? * scale as f64)?;
            let att = candle_nn::ops::softmax_last_dim(&att)?;
            let out = att.matmul(&v.to_dtype(DType::F32)?)?;
            expected.push(out.reshape((1, q_heads, head_dim))?);
        }
        let key_cache = Tensor::cat(&k_blocks, 0)?;
        let value_cache = Tensor::cat(&v_blocks, 0)?;
        let block_tables = Tensor::arange(0u32, (batch * blocks_per_seq) as u32, &device)?
            .reshape((batch, blocks_per_seq))?;
        let context_lens =
            Tensor::new(lens.iter().map(|&l| l as u32).collect::<Vec<_>>(), &device)?;
        let actual = mistralrs_paged_attn::paged_attention(
            &q,
            None,
            None,
            &key_cache,
            &value_cache,
            &block_tables,
            &context_lens,
            None,
            max_len,
            scale,
            1.,
            None,
        )?;
        let expected = Tensor::cat(&expected, 0)?;
        let diff = (actual.to_dtype(DType::F32)? - expected.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(
            diff < 3e-2,
            "{q_heads}/{kv_heads}/{head_dim} {lens:?}: max abs diff {diff}"
        );
    }
    Ok(())
}

#[test]
fn sdpa_handles_token_major_kv_views() -> Result<()> {
    use crate::attention::{AttentionMask, Sdpa, SdpaParams};

    let device = Device::new_metal(0)?;
    let (q_heads, kv_heads, head_dim, kv_len) = (8, 2, 256, 90);
    let q = Tensor::randn(0f32, 1., (1, q_heads, 1, head_dim), &device)?.to_dtype(DType::BF16)?;
    let k_tokens =
        Tensor::randn(0f32, 1., (kv_len, kv_heads, head_dim), &device)?.to_dtype(DType::BF16)?;
    let v_tokens =
        Tensor::randn(0f32, 1., (kv_len, kv_heads, head_dim), &device)?.to_dtype(DType::BF16)?;
    let k = k_tokens.transpose(0, 1)?.unsqueeze(0)?;
    let v = v_tokens.transpose(0, 1)?.unsqueeze(0)?;
    let mask = Tensor::zeros((1, 1, 1, kv_len), DType::BF16, &device)?;
    let params = SdpaParams {
        n_kv_groups: q_heads / kv_heads,
        softcap: None,
        softmax_scale: 1. / (head_dim as f32).sqrt(),
        sliding_window: None,
        sinks: None,
    };
    for mask in [AttentionMask::None, AttentionMask::Custom(mask)] {
        let strided = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
        let packed =
            Sdpa.run_attention(&q, &k.contiguous()?, &v.contiguous()?, &mask, None, &params)?;
        let diff = (strided.to_dtype(DType::F32)? - packed.to_dtype(DType::F32)?)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        assert!(diff < 1e-3, "max abs diff {diff}");
    }
    Ok(())
}
