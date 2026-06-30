use super::*;
use candle_core::{DType, Device, Result as CandleResult, Tensor, D};
use candle_nn::ops::softmax;

const EPS: f32 = 1e-4;

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
    let lhs = lhs.flatten_all()?.to_vec1::<f32>()?;
    let rhs = rhs.flatten_all()?.to_vec1::<f32>()?;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        assert!((lhs - rhs).abs() < EPS);
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
    let mask_f32 = Tensor::from_vec(
        vec![0.0f32, f32::NEG_INFINITY, 0.0],
        (1, kv_len),
        &Device::Cpu,
    )?;
    let expected = naive_attention(&q, &k, &v, Some(&mask_f32), None)?;

    for dtype in [DType::F16, DType::BF16] {
        let mask = mask_f32.to_dtype(dtype)?;
        let out = run_flash_attn_cpu::<f32>(&q, &k, &v, Some(&mask), &sdpa(None))?;
        assert_close(&out, &expected)?;
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
