#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::paged_attention_with_window;

const KV_HEADS: usize = 4;
const HEAD_DIM: usize = 64;
const BLOCK_SIZE: usize = 32;
const CACHE_VECTOR_BYTES: usize = 16;
const CONTEXTS: [usize; 17] = [
    0, 1, 31, 32, 33, 511, 512, 513, 514, 1023, 4097, 16385, 17, 61, 509, 777, 2047,
];
const MAX_CONTEXT: usize = 16385;
const WINDOWS: [Option<usize>; 5] = [None, Some(1), Some(31), Some(513), Some(1025)];

fn quantized(value: f32, dtype: DType) -> f32 {
    match dtype {
        DType::BF16 => half::bf16::from_f32(value).to_f32(),
        DType::F16 => half::f16::from_f32(value).to_f32(),
        _ => value,
    }
}

fn key_value(position: usize, head: usize, dim: usize) -> (f32, f32) {
    (
        (position % 29) as f32 * 0.01 - 0.14 + head as f32 * 0.02 + dim as f32 * 0.001,
        (position % 37) as f32 * 0.02 - 0.36 + head as f32 * 0.1 + dim as f32 * 0.002,
    )
}

#[test]
fn windowed_decode_matches_reference_across_pages_partitions_and_inactive_rows() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let blocks = MAX_CONTEXT.div_ceil(BLOCK_SIZE);
    let table_values = (0..CONTEXTS.len())
        .flat_map(|sequence| {
            (0..blocks).map(move |logical| ((blocks - 1 - logical + sequence) % blocks) as u32)
        })
        .collect::<Vec<_>>();
    let tables = Tensor::from_vec(table_values.clone(), (CONTEXTS.len(), blocks), &device)?;
    let contexts = Tensor::new(CONTEXTS.map(|length| length as u32).as_slice(), &device)?;

    for dtype in [DType::BF16, DType::F16, DType::F32] {
        let vector = CACHE_VECTOR_BYTES / dtype.size_in_bytes();
        let mut keys = vec![0f32; blocks * KV_HEADS * HEAD_DIM * BLOCK_SIZE];
        let mut values = keys.clone();
        for block in 0..blocks {
            for head in 0..KV_HEADS {
                for token in 0..BLOCK_SIZE {
                    for dim in 0..HEAD_DIM {
                        let (key, value) = key_value(block * BLOCK_SIZE + token, head, dim);
                        let key_index = ((((block * KV_HEADS + head) * (HEAD_DIM / vector)
                            + dim / vector)
                            * BLOCK_SIZE
                            + token)
                            * vector)
                            + dim % vector;
                        keys[key_index] = key;
                        values[((block * KV_HEADS + head) * HEAD_DIM + dim) * BLOCK_SIZE + token] =
                            value;
                    }
                }
            }
        }
        let keys = Tensor::from_vec(
            keys,
            (blocks, KV_HEADS, HEAD_DIM / vector, BLOCK_SIZE, vector),
            &device,
        )?
        .to_dtype(dtype)?;
        let values = Tensor::from_vec(values, (blocks, KV_HEADS, HEAD_DIM, BLOCK_SIZE), &device)?
            .to_dtype(dtype)?;
        for heads in [8, 32] {
            let queries = (0..CONTEXTS.len() * heads * HEAD_DIM)
                .map(|index| (index % 19) as f32 * 0.01 - 0.09)
                .collect::<Vec<_>>();
            let query =
                Tensor::from_vec(queries.clone(), (CONTEXTS.len(), heads, HEAD_DIM), &device)?
                    .to_dtype(dtype)?;
            let sinks = (0..heads)
                .map(|head| (head % 4) as f32 - 1.0)
                .collect::<Vec<_>>();
            let sink_tensor = Tensor::new(sinks.as_slice(), &device)?;
            let scale = 1.0 / (HEAD_DIM as f32).sqrt();
            for window in WINDOWS {
                let output = paged_attention_with_window(
                    &query,
                    None,
                    None,
                    &keys,
                    &values,
                    &tables,
                    &contexts,
                    None,
                    MAX_CONTEXT,
                    scale,
                    1.0,
                    Some(&sink_tensor),
                    window,
                )?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
                let tolerance = match dtype {
                    DType::BF16 => 0.005,
                    DType::F16 => 0.001,
                    _ => 0.00002,
                };
                for (sequence, &context) in CONTEXTS.iter().enumerate() {
                    let start = window.map_or(0, |window| context.saturating_sub(window));
                    for (head, &sink) in sinks.iter().enumerate() {
                        let mut denominator = f64::from(sink).exp();
                        let mut numerator = [0f64; HEAD_DIM];
                        let kv_head = head / (heads / KV_HEADS);
                        let query_start = (sequence * heads + head) * HEAD_DIM;
                        for token in start..context {
                            let block =
                                table_values[sequence * blocks + token / BLOCK_SIZE] as usize;
                            let position = block * BLOCK_SIZE + token % BLOCK_SIZE;
                            let mut dot = 0f64;
                            let mut reference_values = [0f32; HEAD_DIM];
                            for (dim, value) in reference_values.iter_mut().enumerate() {
                                let (key, v) = key_value(position, kv_head, dim);
                                dot += f64::from(quantized(queries[query_start + dim], dtype))
                                    * f64::from(quantized(key, dtype));
                                *value = quantized(v, dtype);
                            }
                            let weight = (dot * f64::from(scale)).exp();
                            denominator += weight;
                            for (sum, value) in numerator.iter_mut().zip(reference_values) {
                                *sum += weight * f64::from(value);
                            }
                        }
                        for (dim, sum) in numerator.into_iter().enumerate() {
                            let expected = (sum / denominator) as f32;
                            let actual = output[query_start + dim];
                            assert!(
                                (actual - expected).abs() < tolerance,
                                "{dtype:?}, heads {heads}, window {window:?}, context {context}, head {head}, dim {dim}: {actual} vs {expected}"
                            );
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

#[test]
fn windowed_decode_masks_nan_values_outside_the_exact_window() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let context = BLOCK_SIZE + 3;
    let window = 2;
    let keys = Tensor::zeros((2, 1, HEAD_DIM / 8, BLOCK_SIZE, 8), DType::BF16, &device)?;
    let mut values = vec![f32::NAN; 2 * HEAD_DIM * BLOCK_SIZE];
    for dim in 0..HEAD_DIM {
        for position in context - window..context {
            values[(position / BLOCK_SIZE * HEAD_DIM + dim) * BLOCK_SIZE + position % BLOCK_SIZE] =
                1.0;
        }
    }
    let values =
        Tensor::from_vec(values, (2, 1, HEAD_DIM, BLOCK_SIZE), &device)?.to_dtype(DType::BF16)?;
    let query = Tensor::zeros((1, 1, HEAD_DIM), DType::BF16, &device)?;
    let output = paged_attention_with_window(
        &query,
        None,
        None,
        &keys,
        &values,
        &Tensor::new(&[[0u32, 1]], &device)?,
        &Tensor::new(&[context as u32], &device)?,
        None,
        context,
        1.0,
        1.0,
        None,
        Some(window),
    )?
    .to_dtype(DType::F32)?
    .flatten_all()?
    .to_vec1::<f32>()?;
    assert!(output.iter().all(|value| (*value - 1.0).abs() < 0.01));
    Ok(())
}
