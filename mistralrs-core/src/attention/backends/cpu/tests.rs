use super::*;
use crate::attention::{AttentionMask, Sdpa};
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::ops::softmax;

const EPS: f32 = 1e-4;
// f16 has ~11 bits of mantissa; outputs around 1 carry ~5e-4 of rounding.
const HALF_TOL: f32 = 2e-3;

fn sdpa(softcap: Option<f32>) -> SdpaParams {
    SdpaParams {
        softmax_scale: 1.0,
        softcap,
        n_kv_groups: 1,
        sliding_window: None,
        sinks: None,
    }
}

fn assert_close(lhs: &Tensor, rhs: &Tensor) -> CandleResult<()> {
    assert_close_within(lhs, rhs, EPS)
}

fn assert_close_within(lhs: &Tensor, rhs: &Tensor, tol: f32) -> CandleResult<()> {
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < tol, "{lhs} vs {rhs} beyond {tol}");
    }
    Ok(())
}

fn naive_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    softcap: Option<f32>,
) -> CandleResult<Tensor> {
    let (b, q_len, h, d) = q.dims4()?;
    let kv_len = k.dim(1)?;
    let q = q
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, q_len, d])?;
    let k = k
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, kv_len, d])?;
    let v = v
        .clone()
        .permute((0, 2, 1, 3))?
        .reshape(&[b * h, kv_len, d])?;

    let mut logits = q.matmul(&k.transpose(1, 2)?)?;
    if let Some(softcap) = softcap {
        logits = (logits / softcap as f64)?.tanh()?;
        logits = (logits * softcap as f64)?;
    }
    if let Some(mask) = mask {
        logits = logits.broadcast_add(mask)?;
    }
    let weights = softmax(&logits, D::Minus1)?;
    weights.matmul(&v)?.reshape(&[b, h, q_len, d])
}

#[test]
fn test_flash_attn_cpu_single_q() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 2);
    let q = Tensor::from_vec(vec![1.0f32; b * h * d], (b, 1, h, d), &Device::Cpu)?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_eq!(out.shape().dims(), &[b, h, 1, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_multiple_kv_chunks() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 4, 8, 1024);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 19) as f32 / 19.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 23) as f32 / 23.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_half_masks() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 3);
    let q = Tensor::from_vec(
        (0..b * h * d).map(|x| x as f32 / 17.0).collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| x as f32 / 19.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| x as f32 / 23.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    for values in [
        vec![0.0f32, f32::NEG_INFINITY, 0.0],
        vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0],
    ] {
        let mask_f32 = Tensor::from_vec(values, (1, kv_len), &Device::Cpu)?;
        let expected = naive_attention(&q, &k, &v, Some(&mask_f32), None)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let mask = mask_f32.to_dtype(dtype)?;
            let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
            assert_close(&out, &expected)?;
        }
    }

    Ok(())
}

#[test]
fn test_flash_attn_cpu_full_q() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 2);
    let q = Tensor::from_vec(
        vec![1.0f32; b * q_len * h * d],
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_finite_mask_values() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![0.0f32, f32::MIN, 0.0, 0.0, 0.0, f32::MIN, 0.0, 0.0],
        (1, q_len, kv_len),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_noncontiguous_mask() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![
            0.0f32,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            0.0,
            0.0,
        ],
        (1, q_len, kv_len),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_full_q_respects_head_specific_mask_classes() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 4);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::MIN,
            0.0,
            f32::NEG_INFINITY,
            f32::NEG_INFINITY,
            0.0,
            f32::NEG_INFINITY,
            0.0,
        ],
        (b, h, q_len, kv_len),
        &Device::Cpu,
    )?;
    let expected_mask = mask.reshape((b * h, q_len, kv_len))?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(
        &out,
        &naive_attention(&q, &k, &v, Some(&expected_mask), None)?,
    )
}

#[test]
fn test_flash_attn_cpu_full_q_with_more_queries_than_keys() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 4, 2, 4, 2);
    let q = Tensor::from_vec(
        (0..b * q_len * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::zeros((1, q_len, kv_len), DType::F32, &Device::Cpu)?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
    assert_close(&out, &naive_attention(&q, &k, &v, Some(&mask), None)?)
}

#[test]
fn test_flash_attn_cpu_single_q_softcap() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1, 2, 4, 2);
    let q = Tensor::from_vec(vec![1.0f32; b * h * d], (b, 1, h, d), &Device::Cpu)?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(Some(0.5)))?;
    assert_eq!(out.shape().dims(), &[b, h, 1, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, Some(0.5))?)
}

#[test]
fn test_flash_attn_cpu_full_q_softcap() -> CandleResult<()> {
    let (b, q_len, h, d, kv_len) = (1, 2, 2, 4, 2);
    let q = Tensor::from_vec(
        vec![1.0f32; b * q_len * h * d],
        (b, q_len, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        vec![1.0f32; b * kv_len * h * d],
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(Some(0.5)))?;
    assert_eq!(out.shape().dims(), &[b, h, q_len, d]);
    assert_close(&out, &naive_attention(&q, &k, &v, None, Some(0.5))?)
}

fn flat_f32(t: &Tensor) -> CandleResult<Vec<f32>> {
    t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()
}

/// Scalar reference in the kernel's layout: [b, q, h, d] with [b, kv, kvh, d] K/V. The sink is
/// one more logit in the denominator that carries no value.
fn naive_with_sinks(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    sinks: &[f32],
) -> CandleResult<Tensor> {
    let (b, q_len, h, d) = q.dims4()?;
    let kv_len = k.dim(1)?;
    let kvh = k.dim(2)?;
    let group = h / kvh;
    let qf = flat_f32(q)?;
    let kf = flat_f32(k)?;
    let vf = flat_f32(v)?;
    let mask_f = match mask {
        Some(mask) => Some((flat_f32(mask)?, mask.dims().to_vec())),
        None => None,
    };
    let kv_at =
        |bi: usize, j: usize, hi: usize, di: usize| ((bi * kv_len + j) * kvh + hi / group) * d + di;
    let mut out = vec![0f32; b * q_len * h * d];
    for bi in 0..b {
        for qi in 0..q_len {
            for hi in 0..h {
                let sink = sinks[hi];
                let mut scores = Vec::with_capacity(kv_len);
                for j in 0..kv_len {
                    let mut dot = 0f32;
                    for di in 0..d {
                        dot += qf[((bi * q_len + qi) * h + hi) * d + di] * kf[kv_at(bi, j, hi, di)];
                    }
                    let mut score = dot;
                    if let Some((m, dims)) = &mask_f {
                        score += match dims.as_slice() {
                            [1, _] | [_] => m[j],
                            [_, _, _, _] => m[((bi * h + hi) * q_len + qi) * kv_len + j],
                            other => candle_core::bail!("unsupported mask dims {other:?}"),
                        };
                    }
                    scores.push(score);
                }
                let m = scores.iter().copied().fold(sink, f32::max);
                let mut denom = (sink - m).exp();
                for score in &scores {
                    denom += (score - m).exp();
                }
                for di in 0..d {
                    let mut acc = 0f32;
                    for (j, score) in scores.iter().enumerate() {
                        acc += (score - m).exp() * vf[kv_at(bi, j, hi, di)];
                    }
                    out[((bi * q_len + qi) * h + hi) * d + di] = acc / denom;
                }
            }
        }
    }
    Tensor::from_vec(out, (b, q_len, h, d), &Device::Cpu)
}

fn sinks_of(heads: usize) -> CandleResult<Tensor> {
    let vals = (0..heads)
        .map(|h| (h as f32) * 0.5 - 1.0)
        .collect::<Vec<_>>();
    Tensor::from_vec(vals, (heads,), &Device::Cpu)
}

fn sdpa_with_sinks(sinks: &Tensor, n_kv_groups: usize) -> SdpaParams {
    SdpaParams {
        n_kv_groups,
        sinks: Some(sinks.clone()),
        ..sdpa(None)
    }
}

/// Both sink paths: the barrier merge (no mask, multi chunk) and the per-row online softmax
/// (mask), with group query attention so the sink is indexed by q head rather than kv head.
#[test]
fn test_flash_attn_cpu_single_q_with_sinks() -> CandleResult<()> {
    let (b, h, d) = (1usize, 4usize, 8usize);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let sinks = sinks_of(h)?;
    let sinks_vals = sinks.to_vec1::<f32>()?;
    for (kv_len, kvh) in [(2usize, 4usize), (1024, 4), (1024, 2)] {
        let k = Tensor::from_vec(
            (0..b * kv_len * kvh * d)
                .map(|x| (x % 19) as f32 / 19.0)
                .collect::<Vec<_>>(),
            (b, kv_len, kvh, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * kv_len * kvh * d)
                .map(|x| (x % 23) as f32 / 23.0)
                .collect::<Vec<_>>(),
            (b, kv_len, kvh, d),
            &Device::Cpu,
        )?;
        let params = sdpa_with_sinks(&sinks, h / kvh);
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &params)?;
        assert_close(&out, &naive_with_sinks(&q, &k, &v, None, &sinks_vals)?)?;

        let mask = Tensor::from_vec(vec![0.0f32; kv_len], (1, kv_len), &Device::Cpu)?;
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &params)?;
        assert_close(
            &out,
            &naive_with_sinks(&q, &k, &v, Some(&mask), &sinks_vals)?,
        )?;
    }
    Ok(())
}

/// The dispatcher the models call, with q in [b, h, q, d] and a custom decode mask.
#[test]
fn test_sdpa_run_attention_cpu_decode_with_sinks() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1usize, 4usize, 8usize, 5usize);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 11) as f32 / 11.0)
            .collect::<Vec<_>>(),
        (b, h, 1, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 17) as f32 / 17.0)
            .collect::<Vec<_>>(),
        (b, h, kv_len, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 29) as f32 / 29.0)
            .collect::<Vec<_>>(),
        (b, h, kv_len, d),
        &Device::Cpu,
    )?;
    let mask = Tensor::from_vec(
        vec![0.0f32, 0.0, f32::NEG_INFINITY, 0.0, 0.0],
        (1, kv_len),
        &Device::Cpu,
    )?;
    let sinks = sinks_of(h)?;
    let params = sdpa_with_sinks(&sinks, 1);

    let out = Sdpa.run_attention(
        &q,
        &k,
        &v,
        &AttentionMask::Custom(mask.clone()),
        None,
        &params,
    )?;

    let q_kernel = q.transpose(1, 2)?.contiguous()?;
    let k_kernel = k.transpose(1, 2)?.contiguous()?;
    let v_kernel = v.transpose(1, 2)?.contiguous()?;
    let want = naive_with_sinks(
        &q_kernel,
        &k_kernel,
        &v_kernel,
        Some(&mask),
        &sinks.to_vec1::<f32>()?,
    )?;
    assert_close(&out, &want)
}

/// A sink the scores cannot reach takes the whole denominator, so every output goes to zero; a
/// sink far below them is invisible. The first half catches a sink that is added to the
/// numerator, the second a sink that is dropped entirely.
#[test]
fn test_flash_attn_cpu_single_q_sink_weight() -> CandleResult<()> {
    let (b, h, d, kv_len) = (1usize, 2usize, 4usize, 3usize);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 7) as f32 / 7.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let k = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| (x % 5) as f32 / 5.0)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;
    let v = Tensor::from_vec(
        (0..b * kv_len * h * d)
            .map(|x| 1.0 + (x % 3) as f32)
            .collect::<Vec<_>>(),
        (b, kv_len, h, d),
        &Device::Cpu,
    )?;

    let dominated = Tensor::from_vec(vec![30.0f32; h], (h,), &Device::Cpu)?;
    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa_with_sinks(&dominated, 1))?;
    for value in out.flatten_all()?.to_vec1::<f32>()? {
        assert!(value.abs() < 1e-6, "sink-dominated output was {value}");
    }

    let negligible = Tensor::from_vec(vec![-30.0f32; h], (h,), &Device::Cpu)?;
    let out = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa_with_sinks(&negligible, 1))?;
    let without = run_flash_attn_cpu::<f32>(&q, &k, &v, None, &sdpa(None))?;
    assert_close(&out, &without)
}

/// The half dtypes take the barrier path, which merges chunk partials and folds the sink in at
/// the merge rather than in each chunk. A single chunk and many chunks are separate branches.
#[test]
fn test_flash_attn_cpu_single_q_with_sinks_half() -> CandleResult<()> {
    use half::{bf16, f16};

    let (b, h, d) = (1usize, 4usize, 8usize);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let sinks = sinks_of(h)?;
    let sinks_vals = sinks.to_vec1::<f32>()?;
    let params = sdpa_with_sinks(&sinks, 1);

    for kv_len in [2usize, 1024] {
        let k = Tensor::from_vec(
            (0..b * kv_len * h * d)
                .map(|x| (x % 19) as f32 / 19.0)
                .collect::<Vec<_>>(),
            (b, kv_len, h, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * kv_len * h * d)
                .map(|x| (x % 23) as f32 / 23.0)
                .collect::<Vec<_>>(),
            (b, kv_len, h, d),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(vec![0.0f32; kv_len], (1, kv_len), &Device::Cpu)?;

        for dtype in [DType::F16, DType::BF16] {
            let qd = q.to_dtype(dtype)?;
            let kd = k.to_dtype(dtype)?;
            let vd = v.to_dtype(dtype)?;
            let kf = kd.to_dtype(DType::F32)?;
            let vf = vd.to_dtype(DType::F32)?;
            let qf = qd.to_dtype(DType::F32)?;

            let want = naive_with_sinks(&qf, &kf, &vf, None, &sinks_vals)?;
            let out = match dtype {
                DType::F16 => run_flash_attn_cpu::<f16>(&qd, &kd, &vd, None, &params)?,
                _ => run_flash_attn_cpu::<bf16>(&qd, &kd, &vd, None, &params)?,
            };
            assert_close_within(&out.to_dtype(DType::F32)?, &want, HALF_TOL)?;

            let want = naive_with_sinks(&qf, &kf, &vf, Some(&mask), &sinks_vals)?;
            let out = match dtype {
                DType::F16 => run_flash_attn_cpu::<f16>(&qd, &kd, &vd, Some(&mask), &params)?,
                _ => run_flash_attn_cpu::<bf16>(&qd, &kd, &vd, Some(&mask), &params)?,
            };
            assert_close_within(&out.to_dtype(DType::F32)?, &want, HALF_TOL)?;
        }
    }
    Ok(())
}

/// Adding sinks must not move the numbers of models that have none: a -inf sink is filtered out,
/// so every branch has to reproduce the no-sink kernel bit for bit.
#[test]
fn test_flash_attn_cpu_single_q_sinks_off_is_bit_identical() -> CandleResult<()> {
    let (b, h, d) = (1usize, 4usize, 8usize);
    let q = Tensor::from_vec(
        (0..b * h * d)
            .map(|x| (x % 13) as f32 / 13.0)
            .collect::<Vec<_>>(),
        (b, 1, h, d),
        &Device::Cpu,
    )?;
    let off = Tensor::from_vec(vec![f32::NEG_INFINITY; h], (h,), &Device::Cpu)?;
    let bits = |t: &Tensor| -> CandleResult<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for kv_len in [2usize, 1024] {
        let k = Tensor::from_vec(
            (0..b * kv_len * h * d)
                .map(|x| (x % 19) as f32 / 19.0)
                .collect::<Vec<_>>(),
            (b, kv_len, h, d),
            &Device::Cpu,
        )?;
        let v = Tensor::from_vec(
            (0..b * kv_len * h * d)
                .map(|x| (x % 23) as f32 / 23.0)
                .collect::<Vec<_>>(),
            (b, kv_len, h, d),
            &Device::Cpu,
        )?;
        let mask = Tensor::from_vec(vec![0.0f32; kv_len], (1, kv_len), &Device::Cpu)?;
        for mask in [None, Some(&mask)] {
            let plain = run_flash_attn_cpu::<f32>(&q, &k, &v, mask, &sdpa(None))?;
            let zeroed = run_flash_attn_cpu::<f32>(&q, &k, &v, mask, &sdpa_with_sinks(&off, 1))?;
            assert_eq!(bits(&plain)?, bits(&zeroed)?);
        }
    }
    Ok(())
}
