#![cfg(all(feature = "cuda", target_family = "unix"))]

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_paged_attn::{flash_attn_sinks_varlen, flash_attn_sinks_varlen_packed};

const HEADS: usize = 4;
const KV_HEADS: usize = 2;
const QUERY_LENGTHS: [usize; 4] = [1, 7, 9, 17];
const KV_LENGTHS: [usize; 4] = [5, 13, 19, 535];
const WINDOWS: [usize; 4] = [0, 1, 9, 513];
const SINKS: [f32; HEADS] = [-1.0, 0.5, 2.0, 8.0];

fn cumulative_lengths(lengths: &[usize]) -> Vec<u32> {
    std::iter::once(0)
        .chain(lengths.iter().scan(0, |total, length| {
            *total += u32::try_from(*length).unwrap();
            Some(*total)
        }))
        .collect()
}

#[test]
fn packed_and_padded_sinks_match_reference_for_cached_ragged_queries() -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cu_q = cumulative_lengths(&QUERY_LENGTHS);
    let cu_k = cumulative_lengths(&KV_LENGTHS);
    let total_q = QUERY_LENGTHS.iter().sum::<usize>();
    let total_k = KV_LENGTHS.iter().sum::<usize>();
    let max_q = *QUERY_LENGTHS.iter().max().unwrap();
    let cu_q_tensor = Tensor::new(cu_q.as_slice(), &device)?;
    let cu_k_tensor = Tensor::new(cu_k.as_slice(), &device)?;
    let sinks_tensor = Tensor::new(&SINKS, &device)?;

    for head_dim in [64, 128] {
        let scale = 1.0 / (head_dim as f32).sqrt();
        for dtype in [DType::BF16, DType::F16, DType::F32] {
            let query = Tensor::from_vec(
                (0..total_q * HEADS * head_dim)
                    .map(|i| (i as f32 * 0.017).sin() * 0.3)
                    .collect(),
                (total_q, HEADS, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let key = Tensor::from_vec(
                (0..total_k * KV_HEADS * head_dim)
                    .map(|i| (i as f32 * 0.013).cos() * 0.7)
                    .collect(),
                (total_k, KV_HEADS, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let value = Tensor::from_vec(
                (0..total_k * KV_HEADS * head_dim)
                    .map(|i| (i as f32 * 0.019).sin())
                    .collect(),
                (total_k, KV_HEADS, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;
            let q_values = query
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let k_values = key.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
            let v_values = value
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            let mut padded_values = vec![0f32; QUERY_LENGTHS.len() * HEADS * max_q * head_dim];
            for (batch, &q_len) in QUERY_LENGTHS.iter().enumerate() {
                for row in 0..q_len {
                    for head in 0..HEADS {
                        let src = ((cu_q[batch] as usize + row) * HEADS + head) * head_dim;
                        let dst = ((batch * HEADS + head) * max_q + row) * head_dim;
                        padded_values[dst..dst + head_dim]
                            .copy_from_slice(&q_values[src..src + head_dim]);
                    }
                }
            }
            let padded = Tensor::from_vec(
                padded_values,
                (QUERY_LENGTHS.len(), HEADS, max_q, head_dim),
                &device,
            )?
            .to_dtype(dtype)?;

            for window in WINDOWS {
                let mut expected = vec![0f32; q_values.len()];
                for (batch, (&q_len, &kv_len)) in QUERY_LENGTHS.iter().zip(&KV_LENGTHS).enumerate()
                {
                    for row in 0..q_len {
                        let last = kv_len - q_len + row;
                        let first = if window == 0 {
                            0
                        } else {
                            (last + 1).saturating_sub(window)
                        };
                        for (head, &sink) in SINKS.iter().enumerate() {
                            let q_start = ((cu_q[batch] as usize + row) * HEADS + head) * head_dim;
                            let kv_head = head / (HEADS / KV_HEADS);
                            let mut denominator = f64::from(sink).exp();
                            let mut numerator = vec![0f64; head_dim];
                            for position in first..=last {
                                let kv_start = ((cu_k[batch] as usize + position) * KV_HEADS
                                    + kv_head)
                                    * head_dim;
                                let dot = (0..head_dim)
                                    .map(|d| {
                                        f64::from(q_values[q_start + d])
                                            * f64::from(k_values[kv_start + d])
                                    })
                                    .sum::<f64>();
                                let weight = (dot * f64::from(scale)).exp();
                                denominator += weight;
                                for (d, sum) in numerator.iter_mut().enumerate() {
                                    *sum += weight * f64::from(v_values[kv_start + d]);
                                }
                            }
                            for (d, sum) in numerator.into_iter().enumerate() {
                                expected[q_start + d] = (sum / denominator) as f32;
                            }
                        }
                    }
                }

                let packed_output = flash_attn_sinks_varlen_packed(
                    &query,
                    &key,
                    &value,
                    Some(&sinks_tensor),
                    &cu_q_tensor,
                    &cu_k_tensor,
                    max_q,
                    scale,
                    window,
                )?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
                let padded_output = flash_attn_sinks_varlen(
                    &padded,
                    &key,
                    &value,
                    Some(&sinks_tensor),
                    &cu_q_tensor,
                    &cu_k_tensor,
                    scale,
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
                for (index, (&actual, &reference)) in
                    packed_output.iter().zip(&expected).enumerate()
                {
                    assert!(
                        (actual - reference).abs() < tolerance,
                        "{dtype:?}, D={head_dim}, window={window}, index={index}: {actual} vs {reference}"
                    );
                }
                for (batch, &q_len) in QUERY_LENGTHS.iter().enumerate() {
                    for head in 0..HEADS {
                        for row in 0..max_q {
                            for d in 0..head_dim {
                                let index = ((batch * HEADS + head) * max_q + row) * head_dim + d;
                                let reference = if row < q_len {
                                    expected[((cu_q[batch] as usize + row) * HEADS + head)
                                        * head_dim
                                        + d]
                                } else {
                                    0.0
                                };
                                assert!((padded_output[index] - reference).abs() < tolerance);
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
