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
const GEMMA_DECODE_SHAPE: (usize, usize, usize) = (8, 2, 256);
const GEMMA_GLOBAL_DECODE_SHAPE: (usize, usize, usize) = (8, 2, 512);
const GEMMA_SLIDING_WINDOW: usize = 512;
const MASKED_DECODE_CASES: [(usize, usize, usize); 2] = [(33, 0, 17), (543, 31, 543)];
const MASK_CONTROL_CONTEXT: usize = 256;
const MASK_QK_AMPLITUDE: f32 = 0.1;
const MASK_EXCLUDED_VALUE_BIAS: f32 = 4.0;
const CUSTOM_MASK_BATCH: usize = 2;
const CUSTOM_MASK_CONTEXT: usize = 33;
const CUSTOM_MASK_VISIBLE: usize = 9;
const CUSTOM_MASK_BIAS: f32 = 0.25;
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

fn cache_with_layout(k: &Tensor, v: &Tensor, hnd: bool) -> Result<(Tensor, Tensor)> {
    if !hnd {
        return Ok((k.clone(), v.clone()));
    }
    let (blocks, heads, head_chunks, block_size, x) = k.dims5()?;
    let k = k.permute((0, 1, 3, 2, 4))?.contiguous()?.reshape((
        blocks,
        heads,
        block_size,
        head_chunks * x,
    ))?;
    Ok((k, v.transpose(2, 3)?.contiguous()?))
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

fn decode_mask_errors(
    device: &Device,
    dtype: DType,
    (q_heads, kv_heads, head_dim): (usize, usize, usize),
    (context, visible_start, visible_end): (usize, usize, usize),
) -> Result<(f32, f32, f32)> {
    use crate::attention::{AttentionMask, Sdpa, SdpaParams};

    let cpu = Device::Cpu;
    let q = Tensor::from_vec(
        (0..q_heads * head_dim)
            .map(|i| (i as f32).sin() * MASK_QK_AMPLITUDE)
            .collect::<Vec<_>>(),
        (1, q_heads, 1, head_dim),
        &cpu,
    )?
    .to_dtype(dtype)?;
    let k = Tensor::from_vec(
        (0..context * kv_heads * head_dim)
            .map(|i| (i as f32).cos() * MASK_QK_AMPLITUDE)
            .collect::<Vec<_>>(),
        (context, kv_heads, head_dim),
        &cpu,
    )?
    .to_dtype(dtype)?;
    let v = Tensor::from_vec(
        (0..context * kv_heads * head_dim)
            .map(|i| {
                let token = i / (kv_heads * head_dim);
                let bias = if (visible_start..visible_end).contains(&token) {
                    0.0
                } else {
                    MASK_EXCLUDED_VALUE_BIAS
                };
                (i as f32).sin() + bias
            })
            .collect::<Vec<_>>(),
        (context, kv_heads, head_dim),
        &cpu,
    )?
    .to_dtype(dtype)?;
    let mask = Tensor::from_vec(
        (0..context)
            .map(|token| {
                if (visible_start..visible_end).contains(&token) {
                    0f32
                } else {
                    f32::NEG_INFINITY
                }
            })
            .collect::<Vec<_>>(),
        (1, 1, context),
        &cpu,
    )?;
    let q_cpu = q
        .to_dtype(DType::F32)?
        .reshape((kv_heads, q_heads / kv_heads, head_dim))?;
    let k_cpu = k.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
    let v_cpu = v.to_dtype(DType::F32)?.transpose(0, 1)?.contiguous()?;
    let logits = q_cpu.matmul(&k_cpu.t()?)?.broadcast_add(&mask)?;
    let expected = candle_nn::ops::softmax_last_dim(&logits)?
        .matmul(&v_cpu)?
        .reshape((1, q_heads, 1, head_dim))?;

    let q = q.to_device(device)?;
    let k = k.to_device(device)?.transpose(0, 1)?.unsqueeze(0)?;
    let v = v.to_device(device)?.transpose(0, 1)?.unsqueeze(0)?;
    let mask = AttentionMask::Custom(mask.unsqueeze(0)?.to_dtype(dtype)?.to_device(device)?);
    let params = SdpaParams {
        n_kv_groups: q_heads / kv_heads,
        softcap: None,
        softmax_scale: 1.0,
        sliding_window: (head_dim == GEMMA_DECODE_SHAPE.2).then_some(GEMMA_SLIDING_WINDOW),
        sinks: None,
    };
    let masked = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
    let unmasked = Sdpa.run_attention(&q, &k, &v, &AttentionMask::None, None, &params)?;
    let trimmed = Sdpa.run_attention(
        &q,
        &k.narrow(2, visible_start, visible_end - visible_start)?,
        &v.narrow(2, visible_start, visible_end - visible_start)?,
        &AttentionMask::None,
        None,
        &params,
    )?;
    let masked_error = max_abs_diff(&masked, &expected)?;
    let masked_vs_unmasked = max_abs_diff(&masked, &unmasked)?;
    let trimmed_error = max_abs_diff(&trimmed, &expected)?;

    let full_context = visible_start == 0 && visible_end == context;
    let generated_window = params.sliding_window.is_some_and(|window| {
        visible_start == context.saturating_sub(window) && visible_end == context
    });
    if full_context || generated_window {
        use crate::{
            paged_attention::PagedAttention,
            pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
        };
        use std::collections::HashMap;

        let mut cache_key = k.squeeze(0)?;
        let mut cache_value = v.squeeze(0)?;
        if generated_window && visible_start > 0 {
            let poison = Tensor::full(f32::NAN, (kv_heads, visible_start, head_dim), device)?
                .to_dtype(dtype)?;
            cache_key = Tensor::cat(
                &[
                    &poison,
                    &cache_key.narrow(1, visible_start, context - visible_start)?,
                ],
                1,
            )?;
            cache_value = Tensor::cat(
                &[
                    &poison,
                    &cache_value.narrow(1, visible_start, context - visible_start)?,
                ],
                1,
            )?;
        }
        let (key_cache, value_cache) =
            pack_seq(&cache_key, &cache_value, context.div_ceil(BLOCK_SIZE))?;
        for hnd in [false, true] {
            let (key_cache, value_cache) = cache_with_layout(&key_cache, &value_cache, hnd)?;
            let (key_cache, value_cache, block_tables) =
                remap_cache_blocks(&key_cache, &value_cache, 1, context.div_ceil(BLOCK_SIZE))?;
            let context_lens = Tensor::new(&[context as u32], device)?;
            let tolerance = if dtype == DType::BF16 {
                BF16_ABS_TOL
            } else {
                F16_ABS_TOL
            };
            if full_context {
                let paged = mistralrs_paged_attn::paged_attention(
                    &q.squeeze(2)?,
                    None,
                    None,
                    &key_cache,
                    &value_cache,
                    &block_tables,
                    &context_lens,
                    None,
                    context,
                    params.softmax_scale,
                    1.0,
                    None,
                )?;
                let paged_error = max_abs_diff(&paged, &expected)?;
                assert!(
                    paged_error < tolerance,
                    "{dtype:?} hnd={hnd}: paged control error {paged_error}"
                );
            }
            if generated_window {
                let mut metadata = PagedAttentionInputMetadata::dummy(device)?;
                metadata.is_first_prompt_chunk = false;
                metadata.block_size = Some(BLOCK_SIZE);
                metadata.block_tables = Some(HashMap::from([(device.location(), block_tables)]));
                metadata.context_lens = Some(HashMap::from([(device.location(), context_lens)]));
                metadata.paged_context_lens_cpu = Some(vec![context]);
                metadata.max_context_len = Some(context);
                let output = PagedAttention::new(head_dim, device, None)?.forward_donor_cache(
                    &q,
                    &key_cache,
                    &value_cache,
                    &AttentionMask::None,
                    &metadata,
                    &params,
                    None,
                )?;
                let error = max_abs_diff(&output, &expected)?;
                assert!(
                    error < tolerance,
                    "{dtype:?} context={context} hnd={hnd}: paged window error {error}"
                );
            }
        }
    }
    Ok((masked_error, masked_vs_unmasked, trimmed_error))
}

fn assert_decode_masks_match_cpu(
    dtype: DType,
    tolerance: f32,
    shape: (usize, usize, usize),
) -> Result<()> {
    let device = Device::new_metal(0)?;
    let mut failures = Vec::new();
    for case in MASKED_DECODE_CASES {
        let (masked, masked_vs_unmasked, trimmed) =
            decode_mask_errors(&device, dtype, shape, case)?;
        if masked >= tolerance || trimmed >= tolerance {
            failures.push(format!(
                "{dtype:?} {shape:?} {case:?}: masked error {masked}, masked/unmasked diff {masked_vs_unmasked}, trimmed error {trimmed}"
            ));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    Ok(())
}

#[test]
fn sdpa_decode_masks_match_cpu_bf16() -> Result<()> {
    assert_decode_masks_match_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_DECODE_SHAPE)
}

#[test]
fn sdpa_decode_masks_match_cpu_f16() -> Result<()> {
    assert_decode_masks_match_cpu(DType::F16, F16_ABS_TOL, GEMMA_DECODE_SHAPE)
}

#[test]
fn sdpa_decode_global_masks_match_cpu_bf16() -> Result<()> {
    assert_decode_masks_match_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_GLOBAL_DECODE_SHAPE)
}

fn assert_decode_noop_mask_matches_cpu(
    dtype: DType,
    tolerance: f32,
    shape: (usize, usize, usize),
) -> Result<()> {
    let device = Device::new_metal(0)?;
    let (masked, masked_vs_unmasked, trimmed) = decode_mask_errors(
        &device,
        dtype,
        shape,
        (MASK_CONTROL_CONTEXT, 0, MASK_CONTROL_CONTEXT),
    )?;
    assert!(
        masked < tolerance && trimmed < tolerance && masked_vs_unmasked < tolerance,
        "{dtype:?} {shape:?}: no-op mask error {masked}, masked/unmasked diff {masked_vs_unmasked}, unmasked/trimmed error {trimmed}"
    );
    Ok(())
}

#[test]
fn sdpa_decode_noop_sliding_mask_matches_cpu() -> Result<()> {
    for (dtype, tolerance) in [(DType::BF16, BF16_ABS_TOL), (DType::F16, F16_ABS_TOL)] {
        assert_decode_noop_mask_matches_cpu(dtype, tolerance, GEMMA_DECODE_SHAPE)?;
    }
    Ok(())
}

#[test]
fn sdpa_decode_global_noop_mask_matches_cpu_bf16() -> Result<()> {
    assert_decode_noop_mask_matches_cpu(DType::BF16, BF16_ABS_TOL, GEMMA_GLOBAL_DECODE_SHAPE)
}

#[test]
fn sdpa_decode_batch_head_masks_match_cpu() -> Result<()> {
    use crate::{
        attention::{AttentionMask, Sdpa, SdpaParams},
        paged_attention::PagedAttention,
        pipeline::text_models_inputs_processor::PagedAttentionInputMetadata,
    };
    use std::collections::HashMap;

    let device = Device::new_metal(0)?;
    let cpu = Device::Cpu;
    for ((q_heads, kv_heads, head_dim), dtype, tolerance) in [
        (GEMMA_DECODE_SHAPE, DType::BF16, BF16_ABS_TOL),
        (GEMMA_DECODE_SHAPE, DType::F16, F16_ABS_TOL),
        (GEMMA_GLOBAL_DECODE_SHAPE, DType::BF16, BF16_ABS_TOL),
    ] {
        let q = Tensor::from_vec(
            (0..CUSTOM_MASK_BATCH * q_heads * head_dim)
                .map(|i| (i as f32).sin() * MASK_QK_AMPLITUDE)
                .collect::<Vec<_>>(),
            (CUSTOM_MASK_BATCH, q_heads, 1, head_dim),
            &cpu,
        )?
        .to_dtype(dtype)?;
        let kv_shape = (CUSTOM_MASK_BATCH, kv_heads, CUSTOM_MASK_CONTEXT, head_dim);
        let kv_elements = CUSTOM_MASK_BATCH * kv_heads * CUSTOM_MASK_CONTEXT * head_dim;
        let k = Tensor::from_vec(
            (0..kv_elements)
                .map(|i| (i as f32).cos() * MASK_QK_AMPLITUDE)
                .collect::<Vec<_>>(),
            kv_shape,
            &cpu,
        )?
        .to_dtype(dtype)?;
        let v = Tensor::from_vec(
            (0..kv_elements)
                .map(|i| {
                    (i as f32).sin()
                        + ((i / head_dim) % CUSTOM_MASK_CONTEXT) as f32 / CUSTOM_MASK_CONTEXT as f32
                })
                .collect::<Vec<_>>(),
            kv_shape,
            &cpu,
        )?
        .to_dtype(dtype)?;
        let mask = Tensor::from_vec(
            (0..CUSTOM_MASK_BATCH * q_heads * CUSTOM_MASK_CONTEXT)
                .map(|i| {
                    let row = i / CUSTOM_MASK_CONTEXT;
                    let token = i % CUSTOM_MASK_CONTEXT;
                    let start = row % (CUSTOM_MASK_CONTEXT - CUSTOM_MASK_VISIBLE + 1);
                    if (start..start + CUSTOM_MASK_VISIBLE).contains(&token) {
                        (token % 2) as f32 * CUSTOM_MASK_BIAS
                    } else {
                        f32::NEG_INFINITY
                    }
                })
                .collect::<Vec<_>>(),
            (CUSTOM_MASK_BATCH, q_heads, 1, CUSTOM_MASK_CONTEXT),
            &cpu,
        )?
        .to_dtype(dtype)?;
        let mut expected = Vec::with_capacity(CUSTOM_MASK_BATCH);
        for batch in 0..CUSTOM_MASK_BATCH {
            let q = q.narrow(0, batch, 1)?.to_dtype(DType::F32)?.reshape((
                kv_heads,
                q_heads / kv_heads,
                head_dim,
            ))?;
            let k = k.narrow(0, batch, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
            let v = v.narrow(0, batch, 1)?.squeeze(0)?.to_dtype(DType::F32)?;
            let mask = mask.narrow(0, batch, 1)?.to_dtype(DType::F32)?.reshape((
                kv_heads,
                q_heads / kv_heads,
                CUSTOM_MASK_CONTEXT,
            ))?;
            let logits = q.matmul(&k.t()?)?.broadcast_add(&mask)?;
            expected.push(
                candle_nn::ops::softmax_last_dim(&logits)?
                    .matmul(&v)?
                    .reshape((1, q_heads, 1, head_dim))?,
            );
        }
        let params = SdpaParams {
            n_kv_groups: q_heads / kv_heads,
            softcap: None,
            softmax_scale: 1.0,
            sliding_window: None,
            sinks: None,
        };
        let q = q.to_device(&device)?;
        let k = k.to_device(&device)?;
        let v = v.to_device(&device)?;
        let mask = AttentionMask::Custom(mask.to_device(&device)?);
        let expected = Tensor::cat(&expected, 0)?;
        let actual = Sdpa.run_attention(&q, &k, &v, &mask, None, &params)?;
        let error = max_abs_diff(&actual, &expected)?;
        assert!(
            error < tolerance,
            "{dtype:?} head_dim={head_dim}: batch/head mask error {error}"
        );
        let blocks = CUSTOM_MASK_CONTEXT.div_ceil(BLOCK_SIZE);
        let mut key_blocks = Vec::with_capacity(CUSTOM_MASK_BATCH);
        let mut value_blocks = Vec::with_capacity(CUSTOM_MASK_BATCH);
        for batch in 0..CUSTOM_MASK_BATCH {
            let (key, value) = pack_seq(
                &k.narrow(0, batch, 1)?.squeeze(0)?,
                &v.narrow(0, batch, 1)?.squeeze(0)?,
                blocks,
            )?;
            key_blocks.push(key);
            value_blocks.push(value);
        }
        let key_cache = Tensor::cat(&key_blocks, 0)?;
        let value_cache = Tensor::cat(&value_blocks, 0)?;
        for hnd in [false, true] {
            let (key_cache, value_cache) = cache_with_layout(&key_cache, &value_cache, hnd)?;
            let (key_cache, value_cache, block_tables) =
                remap_cache_blocks(&key_cache, &value_cache, CUSTOM_MASK_BATCH, blocks)?;
            let context_lens =
                Tensor::full(CUSTOM_MASK_CONTEXT as u32, CUSTOM_MASK_BATCH, &device)?;
            let mut metadata = PagedAttentionInputMetadata::dummy(&device)?;
            metadata.is_first_prompt_chunk = false;
            metadata.block_size = Some(BLOCK_SIZE);
            metadata.block_tables = Some(HashMap::from([(device.location(), block_tables)]));
            metadata.context_lens = Some(HashMap::from([(device.location(), context_lens)]));
            metadata.paged_context_lens_cpu = Some(vec![CUSTOM_MASK_CONTEXT; CUSTOM_MASK_BATCH]);
            metadata.max_context_len = Some(CUSTOM_MASK_CONTEXT);
            metadata.query_lens = Some(vec![1; CUSTOM_MASK_BATCH]);
            let actual = PagedAttention::new(head_dim, &device, None)?.forward_donor_cache(
                &q,
                &key_cache,
                &value_cache,
                &mask,
                &metadata,
                &params,
                None,
            )?;
            let error = max_abs_diff(&actual, &expected)?;
            assert!(
                error < tolerance,
                "{dtype:?} head_dim={head_dim} hnd={hnd}: paged batch/head mask error {error}"
            );
        }
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
