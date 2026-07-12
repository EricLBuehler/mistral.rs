#![cfg(all(feature = "cuda", has_cutlass_moe_kernels))]

use candle_core::{DType, Device, Result, Tensor};

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

fn patterned(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let x = ((i.wrapping_mul(31) + salt.wrapping_mul(13)) % 211) as u8 as f32;
            ((x / 105.0) - 1.0) * scale
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_case(
    num_tokens: usize,
    hidden: usize,
    inter: usize,
    num_experts: usize,
    topk: usize,
    ids: Vec<u32>,
    weights: Vec<f32>,
    act: mistralrs_quant::moe::cuda::GatedAct,
    tol: f32,
) -> Result<()> {
    let device = Device::new_cuda(0)?;
    let cpu = Device::Cpu;
    let dev = device.as_cuda_device()?;

    let xs_f = patterned(num_tokens * hidden, 1, 1.0);
    let gu_f = patterned(num_experts * 2 * inter * hidden, 2, 0.08);
    let dn_f = patterned(num_experts * hidden * inter, 3, 0.08);

    let xs_cpu = Tensor::from_vec(xs_f, (num_tokens, hidden), &cpu)?;
    let gu_cpu = Tensor::from_vec(gu_f, (num_experts, 2 * inter, hidden), &cpu)?;
    let dn_cpu = Tensor::from_vec(dn_f, (num_experts, hidden, inter), &cpu)?;

    // bf16-quantized reference inputs so the comparison isolates kernel error from cast error.
    let xs_q = xs_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let gu_q = gu_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let dn_q = dn_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;

    let mut expect = vec![0f32; num_tokens * hidden];
    for e in 0..num_experts {
        let pre = xs_q.matmul(&gu_q.get(e)?.t()?.contiguous()?)?; // [T, 2I]
        let pre = pre.to_vec2::<f32>()?;
        let dn_e = dn_q.get(e)?.to_vec2::<f32>()?; // [H, I]
        for t in 0..num_tokens {
            for k in 0..topk {
                if ids[t * topk + k] as usize != e {
                    continue;
                }
                let w = weights[t * topk + k];
                let act_fn = match act {
                    mistralrs_quant::moe::cuda::GatedAct::GeluTanh => gelu_tanh as fn(f32) -> f32,
                    mistralrs_quant::moe::cuda::GatedAct::Silu => silu as fn(f32) -> f32,
                };
                let act: Vec<f32> = (0..inter)
                    .map(|i| act_fn(pre[t][i]) * pre[t][inter + i])
                    .collect();
                // bf16 round-trip after the activation, matching the kernel pipeline.
                let act: Vec<f32> = act
                    .into_iter()
                    .map(|v| f32::from(half::bf16::from_f32(v)))
                    .collect();
                for h in 0..hidden {
                    let mut acc = 0f32;
                    for i in 0..inter {
                        acc += dn_e[h][i] * act[i];
                    }
                    expect[t * hidden + h] += w * f32::from(half::bf16::from_f32(acc));
                }
            }
        }
    }

    let xs = xs_cpu.to_device(&device)?.to_dtype(DType::BF16)?;
    let gu = gu_cpu.to_device(&device)?.to_dtype(DType::BF16)?;
    let dn = dn_cpu.to_device(&device)?.to_dtype(DType::BF16)?;
    let ids_t = Tensor::from_vec(ids.clone(), (num_tokens, topk), &device)?;
    let w_t = Tensor::from_vec(weights.clone(), (num_tokens, topk), &device)?;

    let got = mistralrs_quant::moe::cutlass::cutlass_fused_moe(
        &xs,
        &gu,
        &dn,
        &ids_t,
        &w_t,
        num_experts,
        act,
        dev,
    )?
    .to_dtype(DType::F32)?
    .to_device(&cpu)?
    .flatten_all()?
    .to_vec1::<f32>()?;

    let mut max_diff = 0f32;
    let mut max_at = 0usize;
    for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
        let d = (g - e).abs();
        if d > max_diff {
            max_diff = d;
            max_at = i;
        }
    }
    assert!(
        max_diff < tol,
        "max diff {max_diff} at {max_at} (got {} expect {})",
        got[max_at],
        expect[max_at]
    );
    Ok(())
}

#[test]
fn cutlass_moe_imbalanced_routing() -> Result<()> {
    // 70% of primary slots routed to expert 0; experts 5 and 6 receive nothing.
    let (num_tokens, topk, num_experts) = (300usize, 2usize, 8usize);
    let allowed = [0u32, 1, 2, 3, 4, 7];
    let mut ids = Vec::with_capacity(num_tokens * topk);
    let mut weights = Vec::with_capacity(num_tokens * topk);
    for t in 0..num_tokens {
        let id0 = if t % 10 < 7 { 0 } else { allowed[t % 6] };
        let mut id1 = allowed[(t * 7 + 1) % 6];
        if id1 == id0 {
            id1 = allowed[(t * 7 + 2) % 6];
        }
        ids.extend([id0, id1]);
        weights.extend([0.7f32, 0.3f32]);
    }
    run_case(
        num_tokens,
        256,
        128,
        num_experts,
        topk,
        ids,
        weights,
        mistralrs_quant::moe::cuda::GatedAct::GeluTanh,
        0.1,
    )
}

#[test]
fn cutlass_moe_silu_imbalanced_routing() -> Result<()> {
    // Same imbalanced routing, SiLU activation (Mixtral/Qwen-style MoEs).
    let (num_tokens, topk, num_experts) = (300usize, 2usize, 8usize);
    let allowed = [0u32, 1, 2, 3, 4, 7];
    let mut ids = Vec::with_capacity(num_tokens * topk);
    let mut weights = Vec::with_capacity(num_tokens * topk);
    for t in 0..num_tokens {
        let id0 = if t % 10 < 7 { 0 } else { allowed[t % 6] };
        let mut id1 = allowed[(t * 7 + 1) % 6];
        if id1 == id0 {
            id1 = allowed[(t * 7 + 2) % 6];
        }
        ids.extend([id0, id1]);
        weights.extend([0.7f32, 0.3f32]);
    }
    run_case(
        num_tokens,
        256,
        128,
        num_experts,
        topk,
        ids,
        weights,
        mistralrs_quant::moe::cuda::GatedAct::Silu,
        0.1,
    )
}

#[test]
fn cutlass_moe_tiny_batch_many_empty_experts() -> Result<()> {
    // Decode-like shape: 4 tokens, 64 experts, almost all expert groups have M = 0.
    let (num_tokens, topk, num_experts) = (4usize, 4usize, 64usize);
    let mut ids = Vec::with_capacity(num_tokens * topk);
    let mut weights = Vec::with_capacity(num_tokens * topk);
    for t in 0..num_tokens {
        for k in 0..topk {
            ids.push(((t * 17 + k * 23) % num_experts) as u32);
            weights.push(0.25f32);
        }
    }
    run_case(
        num_tokens,
        64,
        32,
        num_experts,
        topk,
        ids,
        weights,
        mistralrs_quant::moe::cuda::GatedAct::GeluTanh,
        0.1,
    )
}

#[test]
fn cutlass_moe_large_avg_tokens() -> Result<()> {
    // avg tokens per expert = 100 -> exercises the large (128-row) tile config.
    let (num_tokens, topk, num_experts) = (400usize, 2usize, 8usize);
    let mut ids = Vec::with_capacity(num_tokens * topk);
    let mut weights = Vec::with_capacity(num_tokens * topk);
    for t in 0..num_tokens {
        let id0 = (t % num_experts) as u32;
        let id1 = ((t + 3) % num_experts) as u32;
        ids.extend([id0, id1]);
        weights.extend([0.6f32, 0.4f32]);
    }
    run_case(
        num_tokens,
        256,
        128,
        num_experts,
        topk,
        ids,
        weights,
        mistralrs_quant::moe::cuda::GatedAct::GeluTanh,
        0.1,
    )
}
