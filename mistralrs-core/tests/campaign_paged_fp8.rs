#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::{paged_attention, reshape_and_cache};
use std::time::Instant;

const HEADS: usize = 32;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 64;
const BLOCK: usize = 32;
const VECTOR_BYTES: usize = 16;
const CONTEXTS: [usize; 3] = [128, 2048, 8192];
const K_SCALE: f32 = 0.5;
const V_SCALE: f32 = 2.0;
const WARMUP: usize = 4;
const SAMPLES: usize = 9;
const ITERATIONS: usize = 8;
const ATTENTION_TOLERANCE: f32 = 0.01;

struct Cache {
    keys: Tensor,
    values: Tensor,
    k_scale: Option<Tensor>,
    v_scale: Option<Tensor>,
}

struct Fixture {
    query: Tensor,
    keys: Tensor,
    values: Tensor,
    slots: Tensor,
    tables: Tensor,
    lengths: Tensor,
    context: usize,
}

impl Cache {
    fn new(dtype: DType, context: usize, device: &Device) -> Result<Self> {
        let blocks = context.div_ceil(BLOCK);
        let x = VECTOR_BYTES / dtype.size_in_bytes();
        let fp8 = dtype == DType::F8E4M3;
        Ok(Self {
            keys: Tensor::zeros((blocks, KV_HEADS, HEAD_DIM / x, BLOCK, x), dtype, device)?,
            values: Tensor::zeros((blocks, KV_HEADS, HEAD_DIM, BLOCK), dtype, device)?,
            k_scale: fp8.then(|| Tensor::new(K_SCALE, device)).transpose()?,
            v_scale: fp8.then(|| Tensor::new(V_SCALE, device)).transpose()?,
        })
    }

    fn write(&self, fixture: &Fixture) -> Result<()> {
        reshape_and_cache(
            &fixture.keys,
            &fixture.values,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            &self.keys,
            &self.values,
            &fixture.slots,
        )
    }

    fn decode(&self, fixture: &Fixture) -> Result<Tensor> {
        paged_attention(
            &fixture.query,
            self.k_scale.as_ref(),
            self.v_scale.as_ref(),
            &self.keys,
            &self.values,
            &fixture.tables,
            &fixture.lengths,
            None,
            fixture.context,
            1.0 / (HEAD_DIM as f32).sqrt(),
            1.0,
            None,
        )
    }

    fn bytes(&self) -> usize {
        (self.keys.elem_count() + self.values.elem_count()) * self.keys.dtype().size_in_bytes()
    }
}

fn key(index: usize) -> f32 {
    ((index * 17 % 31) as f32 - 15.0) / 32.0
}

fn value(index: usize) -> f32 {
    ((index * 7 % 17) as f32 - 8.0) / 8.0
}

fn query(index: usize) -> f32 {
    ((index * 3 % 9) as f32 - 4.0) / 8.0
}

fn reference(context: usize) -> Vec<f32> {
    let mut output = Vec::new();
    for head in 0..HEADS {
        let kv_head = head / (HEADS / KV_HEADS);
        let scores = (0..context)
            .map(|token| {
                let dot = (0..HEAD_DIM)
                    .map(|dim| {
                        f64::from(query(head * HEAD_DIM + dim))
                            * f64::from(key((token * KV_HEADS + kv_head) * HEAD_DIM + dim))
                    })
                    .sum::<f64>();
                dot / (HEAD_DIM as f64).sqrt()
            })
            .collect::<Vec<_>>();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights = scores
            .iter()
            .map(|score| (score - maximum).exp())
            .collect::<Vec<_>>();
        let total = weights.iter().sum::<f64>();
        for dim in 0..HEAD_DIM {
            output.push(
                (weights
                    .iter()
                    .enumerate()
                    .map(|(token, weight)| {
                        weight * f64::from(value((token * KV_HEADS + kv_head) * HEAD_DIM + dim))
                    })
                    .sum::<f64>()
                    / total) as f32,
            );
        }
    }
    output
}

fn max_error(actual: &Tensor, expected: &[f32]) -> Result<f32> {
    let actual = actual
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    assert!(actual.iter().all(|x| x.is_finite()));
    Ok(actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max))
}

fn elapsed_us(device: &Device, mut run: impl FnMut() -> Result<()>) -> Result<f64> {
    device.synchronize()?;
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        run()?;
    }
    device.synchronize()?;
    Ok(start.elapsed().as_secs_f64() * 1e6 / ITERATIONS as f64)
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

#[test]
fn campaign_cuda_fp8_kv_sm61_correctness_and_performance() -> Result<()> {
    assert!(mistralrs_paged_attn::USE_FP8);
    assert!(mistralrs_paged_attn::USE_FP8_LEGACY_SM61);
    let device = Device::new_cuda(0)?;
    let mut cases = Vec::new();
    for context in CONTEXTS {
        let blocks = context.div_ceil(BLOCK);
        let elements = context * KV_HEADS * HEAD_DIM;
        let fixture = Fixture {
            query: Tensor::from_vec(
                (0..HEADS * HEAD_DIM).map(query).collect(),
                (1, HEADS, HEAD_DIM),
                &device,
            )?
            .to_dtype(DType::F16)?,
            keys: Tensor::from_vec(
                (0..elements).map(key).collect(),
                (context, KV_HEADS, HEAD_DIM),
                &device,
            )?
            .to_dtype(DType::F16)?,
            values: Tensor::from_vec(
                (0..elements).map(value).collect(),
                (context, KV_HEADS, HEAD_DIM),
                &device,
            )?
            .to_dtype(DType::F16)?,
            slots: Tensor::from_vec(
                (0..context)
                    .map(|token| ((blocks - 1 - token / BLOCK) * BLOCK + token % BLOCK) as i64)
                    .collect(),
                context,
                &device,
            )?,
            tables: Tensor::from_vec((0..blocks as u32).rev().collect(), (1, blocks), &device)?,
            lengths: Tensor::new(&[context as u32], &device)?,
            context,
        };
        let caches = [
            Cache::new(DType::F16, context, &device)?,
            Cache::new(DType::F8E4M3, context, &device)?,
        ];
        for cache in &caches {
            cache.write(&fixture)?;
        }
        let fp8_keys = caches[1]
            .keys
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let fp8_values = caches[1]
            .values
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let x = VECTOR_BYTES;
        for token in 0..context {
            let block = blocks - 1 - token / BLOCK;
            for head in 0..KV_HEADS {
                for dim in 0..HEAD_DIM {
                    let logical = (token * KV_HEADS + head) * HEAD_DIM + dim;
                    let k = ((((block * KV_HEADS + head) * (HEAD_DIM / x) + dim / x) * BLOCK
                        + token % BLOCK)
                        * x)
                        + dim % x;
                    let v = ((block * KV_HEADS + head) * HEAD_DIM + dim) * BLOCK + token % BLOCK;
                    assert_eq!(fp8_keys[k] * K_SCALE, key(logical));
                    assert_eq!(fp8_values[v] * V_SCALE, value(logical));
                }
            }
        }
        let expected = reference(context);
        let fp16 = caches[0].decode(&fixture)?;
        let fp8 = caches[1].decode(&fixture)?;
        let fp16_error = max_error(&fp16, &expected)?;
        let fp8_error = max_error(&fp8, &expected)?;
        let fp16_values = fp16.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
        let pair_error = max_error(&fp8, &fp16_values)?;
        assert!(
            fp16_error < ATTENTION_TOLERANCE,
            "FP16 reference error {fp16_error}"
        );
        assert!(
            fp8_error < ATTENTION_TOLERANCE,
            "FP8 reference error {fp8_error}"
        );
        assert!(
            pair_error < ATTENTION_TOLERANCE,
            "FP8/FP16 error {pair_error}"
        );
        assert_eq!(caches[0].bytes(), 2 * caches[1].bytes());
        for _ in 0..WARMUP {
            for cache in &caches {
                cache.write(&fixture)?;
                cache.decode(&fixture)?;
            }
        }
        let mut write = [Vec::new(), Vec::new()];
        let mut decode = [Vec::new(), Vec::new()];
        for sample in 0..SAMPLES {
            for offset in 0..2 {
                let index = (sample + offset) % 2;
                write[index].push(elapsed_us(&device, || caches[index].write(&fixture))?);
                decode[index].push(elapsed_us(&device, || {
                    caches[index].decode(&fixture)?;
                    Ok(())
                })?);
            }
        }
        cases.push(serde_json::json!({
            "context": context, "query_heads": HEADS, "kv_heads": KV_HEADS, "head_dim": HEAD_DIM,
            "fp16_cache_bytes": caches[0].bytes(), "fp8_cache_bytes": caches[1].bytes(),
            "fp16_reference_max_abs_error": fp16_error, "fp8_reference_max_abs_error": fp8_error,
            "fp8_fp16_max_abs_error": pair_error, "exact_scaled_cache_roundtrip": true,
            "fp16_write_us": write[0], "fp8_write_us": write[1],
            "fp16_decode_us": decode[0], "fp8_decode_us": decode[1],
            "fp16_write_median_us": median(&write[0]), "fp8_write_median_us": median(&write[1]),
            "fp16_decode_median_us": median(&decode[0]), "fp8_decode_median_us": median(&decode[1]),
            "decode_speed_ratio_fp16_over_fp8": median(&decode[0]) / median(&decode[1]),
        }));
    }
    let report = serde_json::json!({
        "schema": 1, "scope": "synthetic-paged-kv-cache", "full_model_benchmark": false,
        "candidate": std::env::var("HARICOT_FP8_KV_CANDIDATE").unwrap(),
        "component": std::env::var("HARICOT_FP8_KV_COMPONENT").unwrap(),
        "run_id": std::env::var("GITHUB_RUN_ID").unwrap(),
        "run_attempt": std::env::var("GITHUB_RUN_ATTEMPT").unwrap(),
        "method": "synchronized-host-wall-time-including-api-overhead; alternating-order; allocations-outside-timed-write",
        "warmup": WARMUP, "samples": SAMPLES, "iterations_per_sample": ITERATIONS,
        "performance_threshold": null, "status": "PASS", "cases": cases,
    });
    let path = std::env::var("HARICOT_FP8_KV_REPORT").unwrap();
    std::fs::write(path, serde_json::to_vec_pretty(&report).unwrap())
        .map_err(candle_core::Error::msg)?;
    Ok(())
}
