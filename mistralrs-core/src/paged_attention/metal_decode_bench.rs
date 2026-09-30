//! Metal attention-only benchmark: shared Q/K/V, including dispatch/allocation, excluding packing and model execution.
//! Run an ignored `metal_decode_bench*` test in release mode with `--nocapture`.

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::time::Instant;

use candle_core::{DType, Device, Result, Tensor};

const BLOCK_SIZE: usize = 32;
const CACHE_VECTOR_BYTES: usize = 16;
const BENCH_DTYPE: DType = DType::BF16;
const WARMUP_ITERS: usize = 25;
const TIMED_ITERS: usize = 50;
const QUICK_WARMUP_ITERS: usize = 3;
const QUICK_TIMED_ITERS: usize = 10;
const CONTEXT_LENS: [usize; 4] = [512, 2048, 8192, 32768];
const BATCHES: [usize; 4] = [1, 4, 8, 16];
const QUICK_CONTEXT_LENS: [usize; 2] = [512, 2048];
const QUICK_BATCHES: [usize; 1] = [1];
const SHORT_BENCH_CONTEXT_LENS: [usize; 7] = [128, 256, 288, 320, 512, 1024, 2048];
const SHORT_BENCH_BATCHES: [usize; 2] = [1, 4];
const SHORT_BENCH_SHAPES: [(usize, usize, usize); 2] = [(16, 8, 128), (16, 4, 256)];
const SHORT_CONTEXT_LENS: [usize; 23] = [
    1, 31, 32, 33, 127, 128, 129, 231, 255, 256, 257, 287, 288, 289, 319, 320, 321, 511, 512, 513,
    1023, 1024, 1025,
];
const SHORT_SHAPES: [(usize, usize, usize); 7] = [
    (4, 4, 64),
    (8, 4, 96),
    (12, 4, 128),
    (16, 4, 192),
    (32, 4, 256),
    (16, 8, 128),
    (16, 4, 256),
];
const QUERY_PADDING_HEADS: usize = 2;
const BF16_ABS_TOL: f32 = 3e-2;
const F16_ABS_TOL: f32 = 3e-3;
// (query heads, kv heads, head dim): Qwen3.8 full attention, then a common GQA-4 shape.
const SHAPES: [(usize, usize, usize); 2] = [(24, 4, 256), (32, 8, 128)];
const DECODE_CASES: [(usize, usize, usize, [usize; 4]); 7] = [
    (2, 2, 64, [1, 31, 32, 128]),
    (6, 2, 96, [1, 127, 128, 129]),
    (16, 2, 128, [33, 511, 512, 513]),
    // Eight KV heads and four sequences select 512-token partitions at this context length.
    (16, 8, 192, [1, 511, 512, 513]),
    (24, 4, 256, [1, 37, 700, 1500]),
    (4, 1, 256, [1, 512, 513, 2049]),
    (8, 2, 512, [1, 37, 90, 700]),
];

struct Case {
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    batch: usize,
    context: usize,
}

fn max_abs_diff(actual: &Tensor, expected: &Tensor) -> Result<f32> {
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let expected = expected
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    Ok(actual.iter().zip(&expected).fold(0.0f32, |max, (a, e)| {
        let diff = (a - e).abs();
        if diff.is_finite() {
            max.max(diff)
        } else {
            f32::INFINITY
        }
    }))
}

fn time_ms(
    device: &Device,
    warmup_iters: usize,
    timed_iters: usize,
    mut f: impl FnMut() -> Result<Tensor>,
) -> Result<f64> {
    for _ in 0..warmup_iters {
        f()?;
    }
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..timed_iters {
        f()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e3 / timed_iters as f64)
}

struct BenchInputs {
    q: Tensor,
    dense_q: Tensor,
    k: Tensor,
    v: Tensor,
    key_cache: Tensor,
    value_cache: Tensor,
    block_tables: Tensor,
    context_lens: Tensor,
}

impl BenchInputs {
    fn new(device: &Device, c: &Case) -> Result<Self> {
        let x = CACHE_VECTOR_BYTES / BENCH_DTYPE.size_in_bytes();
        let blocks_per_seq = c.context / BLOCK_SIZE;
        let num_blocks = c.batch * blocks_per_seq;
        let q = Tensor::randn(0f32, 1., (c.batch, c.q_heads, c.head_dim), device)?
            .to_dtype(BENCH_DTYPE)?;
        let kv = || {
            Tensor::randn(
                0f32,
                1.,
                (c.batch, c.kv_heads, c.context, c.head_dim),
                device,
            )?
            .to_dtype(BENCH_DTYPE)
        };
        let k = kv()?;
        let v = kv()?;
        let key_cache = k
            .reshape((
                c.batch,
                c.kv_heads,
                blocks_per_seq,
                BLOCK_SIZE,
                c.head_dim / x,
                x,
            ))?
            .permute((0, 2, 1, 4, 3, 5))?
            .contiguous()?
            .reshape((num_blocks, c.kv_heads, c.head_dim / x, BLOCK_SIZE, x))?;
        let value_cache = v
            .reshape((c.batch, c.kv_heads, blocks_per_seq, BLOCK_SIZE, c.head_dim))?
            .permute((0, 2, 1, 4, 3))?
            .contiguous()?
            .reshape((num_blocks, c.kv_heads, c.head_dim, BLOCK_SIZE))?;
        let block_tables =
            Tensor::arange(0u32, num_blocks as u32, device)?.reshape((c.batch, blocks_per_seq))?;
        let context_lens = Tensor::full(c.context as u32, c.batch, device)?;
        Ok(Self {
            dense_q: q.unsqueeze(2)?,
            q,
            k,
            v,
            key_cache,
            value_cache,
            block_tables,
            context_lens,
        })
    }

    fn paged(&self, c: &Case) -> Result<Tensor> {
        mistralrs_paged_attn::paged_attention(
            &self.q,
            None,
            None,
            &self.key_cache,
            &self.value_cache,
            &self.block_tables,
            &self.context_lens,
            None,
            c.context,
            1. / (c.head_dim as f32).sqrt(),
            1.,
            None,
        )
    }

    fn dense(&self, c: &Case) -> Result<Tensor> {
        candle_nn::ops::sdpa(
            &self.dense_q,
            &self.k,
            &self.v,
            None,
            false,
            1. / (c.head_dim as f32).sqrt(),
            1.,
        )
    }
}

fn run_benchmark(
    shapes: &[(usize, usize, usize)],
    batches: &[usize],
    context_lens: &[usize],
    warmup_iters: usize,
    timed_iters: usize,
) -> Result<()> {
    let device = Device::new_metal(0)?;
    println!(
        "Attention only; includes dispatch/allocation, excludes cache packing and model execution."
    );
    println!("GB/s counts logical K/V bytes once; it is not measured memory traffic.");
    println!("shape(qh/kvh/d) batch context | paged_ms  GB/s | dense_ms  GB/s | paged/dense");
    for &(q_heads, kv_heads, head_dim) in shapes {
        for &batch in batches {
            for &context in context_lens {
                let c = Case {
                    q_heads,
                    kv_heads,
                    head_dim,
                    batch,
                    context,
                };
                let inputs = BenchInputs::new(&device, &c)?;
                let diff = max_abs_diff(&inputs.paged(&c)?, &inputs.dense(&c)?)?;
                assert!(
                    diff < BF16_ABS_TOL,
                    "{q_heads}/{kv_heads}/{head_dim} batch={batch} context={context}: max abs diff {diff}"
                );
                let kv_gb =
                    (2 * batch * kv_heads * context * head_dim * BENCH_DTYPE.size_in_bytes())
                        as f64
                        / 1e9;
                let paged = time_ms(&device, warmup_iters, timed_iters, || inputs.paged(&c))?;
                let dense = time_ms(&device, warmup_iters, timed_iters, || inputs.dense(&c))?;
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

#[test]
#[ignore = "benchmark; needs a Metal GPU"]
fn metal_decode_bench() -> Result<()> {
    run_benchmark(&SHAPES, &BATCHES, &CONTEXT_LENS, WARMUP_ITERS, TIMED_ITERS)
}

#[test]
#[ignore = "quick benchmark; needs a Metal GPU"]
fn metal_decode_bench_quick() -> Result<()> {
    run_benchmark(
        &SHAPES,
        &QUICK_BATCHES,
        &QUICK_CONTEXT_LENS,
        QUICK_WARMUP_ITERS,
        QUICK_TIMED_ITERS,
    )
}

#[test]
#[ignore = "short-context benchmark; needs a Metal GPU"]
fn metal_decode_bench_short() -> Result<()> {
    run_benchmark(
        &SHORT_BENCH_SHAPES,
        &SHORT_BENCH_BATCHES,
        &SHORT_BENCH_CONTEXT_LENS,
        WARMUP_ITERS,
        TIMED_ITERS,
    )
}

/// Packs dense `[kv_heads, len, D]` K/V into one sequence's paged blocks, padding past `len` with NaN.
fn pack_seq(k: &Tensor, v: &Tensor, blocks: usize) -> Result<(Tensor, Tensor)> {
    let (kv_heads, len, d) = k.dims3()?;
    let x = CACHE_VECTOR_BYTES / k.dtype().size_in_bytes();
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

fn remap_cache_blocks(
    key_cache: &Tensor,
    value_cache: &Tensor,
    batch: usize,
    blocks_per_seq: usize,
) -> Result<(Tensor, Tensor, Tensor)> {
    let blocks = batch * blocks_per_seq;
    let device = key_cache.device();
    let order = (0..blocks)
        .rev()
        .flat_map(|block| [blocks as u32, block as u32])
        .chain(std::iter::once(blocks as u32))
        .collect::<Vec<_>>();
    let order = Tensor::new(order, device)?;
    let remap = |cache: &Tensor| -> Result<Tensor> {
        let poison = Tensor::full(f32::NAN, cache.narrow(0, 0, 1)?.shape(), device)?
            .to_dtype(cache.dtype())?;
        Tensor::cat(&[cache, &poison], 0)?.index_select(&order, 0)
    };
    let block_tables = (0..blocks)
        .map(|block| (2 * (blocks - block) - 1) as u32)
        .collect::<Vec<_>>();
    let block_tables = Tensor::new(block_tables, device)?.reshape((batch, blocks_per_seq))?;
    Ok((remap(key_cache)?, remap(value_cache)?, block_tables))
}

fn assert_decode_matches_dense(
    device: &Device,
    (q_heads, kv_heads, head_dim): (usize, usize, usize),
    lens: &[usize],
    dtype: DType,
    tolerance: f32,
) -> Result<()> {
    let batch = lens.len();
    let max_len = *lens.iter().max().unwrap();
    let blocks_per_seq = max_len.div_ceil(BLOCK_SIZE);
    let scale = 1. / (head_dim as f32).sqrt();
    let q = Tensor::randn(
        0f32,
        1.,
        (batch, q_heads + QUERY_PADDING_HEADS, head_dim),
        device,
    )?
    .to_dtype(dtype)?
    .narrow(1, QUERY_PADDING_HEADS / 2, q_heads)?;
    let (mut k_blocks, mut v_blocks, mut expected) = (vec![], vec![], vec![]);
    for (b, &len) in lens.iter().enumerate() {
        let k = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), device)?.to_dtype(dtype)?;
        let v = Tensor::randn(0f32, 1., (kv_heads, len, head_dim), device)?.to_dtype(dtype)?;
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
    let (key_cache, value_cache, block_tables) = remap_cache_blocks(
        &Tensor::cat(&k_blocks, 0)?,
        &Tensor::cat(&v_blocks, 0)?,
        batch,
        blocks_per_seq,
    )?;
    let context_lens = Tensor::new(lens.iter().map(|&l| l as u32).collect::<Vec<_>>(), device)?;
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
    let diff = max_abs_diff(&actual, &expected)?;
    assert!(
        diff < tolerance,
        "{dtype:?} {q_heads}/{kv_heads}/{head_dim} {lens:?}: max abs diff {diff}"
    );
    Ok(())
}

#[test]
fn paged_decode_matches_dense_sdpa() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
        for (q_heads, kv_heads, head_dim, lens) in DECODE_CASES {
            assert_decode_matches_dense(
                &device,
                (q_heads, kv_heads, head_dim),
                &lens,
                dtype,
                tolerance,
            )?;
        }
    }
    Ok(())
}

#[test]
fn paged_decode_short_contexts_match_dense() -> Result<()> {
    let device = Device::new_metal(0)?;
    for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
        for shape in SHORT_SHAPES {
            for len in SHORT_CONTEXT_LENS {
                assert_decode_matches_dense(&device, shape, &[len], dtype, tolerance)?;
            }
        }
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

#[test]
fn sdpa_honors_offset_mask_views() -> Result<()> {
    use crate::attention::{AttentionMask, Sdpa, SdpaParams};

    let device = Device::new_metal(0)?;
    let (q_heads, kv_heads, head_dim, q_len, full_kv, kv_len) = (8, 2, 256, 79, 1103, 591);
    let q =
        Tensor::randn(0f32, 1., (1, q_heads, q_len, head_dim), &device)?.to_dtype(DType::BF16)?;
    let k =
        Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?.to_dtype(DType::BF16)?;
    let v =
        Tensor::randn(0f32, 1., (1, kv_heads, kv_len, head_dim), &device)?.to_dtype(DType::BF16)?;
    let window = 512;
    let prefix = full_kv - q_len;
    let mut m = vec![0f32; q_len * full_kv];
    for i in 0..q_len {
        let qp = prefix + i;
        for j in 0..full_kv {
            if j > qp || j + window <= qp {
                m[i * full_kv + j] = f32::NEG_INFINITY;
            }
        }
    }
    let full_mask = Tensor::from_vec(m, (q_len, full_kv), &device)?.to_dtype(DType::BF16)?;
    let view = full_mask.narrow(1, full_kv - kv_len, kv_len)?;
    let params = SdpaParams {
        n_kv_groups: q_heads / kv_heads,
        softcap: None,
        softmax_scale: 1. / (head_dim as f32).sqrt(),
        sliding_window: Some(window),
        sinks: None,
    };
    let from_view = Sdpa.run_attention(
        &q,
        &k,
        &v,
        &AttentionMask::Custom(view.clone()),
        None,
        &params,
    )?;
    let from_copy = Sdpa.run_attention(
        &q,
        &k,
        &v,
        &AttentionMask::Custom(view.force_contiguous()?),
        None,
        &params,
    )?;
    let diff = (from_view.to_dtype(DType::F32)? - from_copy.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-3, "max abs diff {diff}");
    Ok(())
}
