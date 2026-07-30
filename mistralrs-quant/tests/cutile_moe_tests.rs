#![cfg(all(feature = "cuda", feature = "cutile"))]

use candle_core::{DType, Device, Result, Storage, Tensor};

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

/// Mirrors CutileExpertsWeights::forward_impl: moe_align -> cuTile GEMM (gate_up) -> act ->
/// cuTile GEMM (down, routed weights) -> moe_sum.
#[allow(clippy::too_many_arguments)]
fn run_cutile_case(
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

    let xs_q = xs_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let gu_q = gu_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;
    let dn_q = dn_cpu.to_dtype(DType::BF16)?.to_dtype(DType::F32)?;

    let act_fn = match act {
        mistralrs_quant::moe::cuda::GatedAct::GeluTanh => gelu_tanh as fn(f32) -> f32,
        mistralrs_quant::moe::cuda::GatedAct::Silu => silu as fn(f32) -> f32,
    };
    let mut expect = vec![0f32; num_tokens * hidden];
    for e in 0..num_experts {
        let pre = xs_q.matmul(&gu_q.get(e)?.t()?.contiguous()?)?;
        let pre = pre.to_vec2::<f32>()?;
        let dn_e = dn_q.get(e)?.to_vec2::<f32>()?;
        for t in 0..num_tokens {
            for k in 0..topk {
                if ids[t * topk + k] as usize != e {
                    continue;
                }
                let w = weights[t * topk + k];
                let a: Vec<f32> = (0..inter)
                    .map(|i| f32::from(half::bf16::from_f32(act_fn(pre[t][i]) * pre[t][inter + i])))
                    .collect();
                for h in 0..hidden {
                    let mut acc = 0f32;
                    for i in 0..inter {
                        acc += dn_e[h][i] * a[i];
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
    let num_valid = num_tokens * topk;

    let cfg = mistralrs_quant::cutile::get_default_config(num_tokens, num_experts);

    let ti_flat = ids_t.flatten_all()?.contiguous()?;
    let (ti_storage, _l) = ti_flat.storage_and_layout();
    let ti_slice = match &*ti_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<u32>()?,
        _ => candle_core::bail!("topk_ids must be cuda"),
    };
    let (sids, eids, ntpp, em) = mistralrs_quant::moe::cuda::moe_align(
        ti_slice,
        num_tokens,
        num_experts,
        topk,
        cfg.bm,
        dev,
    )?;

    let ic1 = mistralrs_quant::cutile::cutile_grouped_gemm(
        &xs, &gu, &sids, &eids, &ntpp, None, em, num_valid, topk, false, cfg, dev,
    )?;
    let ic2 = mistralrs_quant::moe::cuda::act_and_mul(&ic1, inter, act, dev)?;

    let tw_flat = w_t.flatten_all()?.to_dtype(DType::F32)?.contiguous()?;
    let (tw_storage, _l) = tw_flat.storage_and_layout();
    let tw_slice = match &*tw_storage {
        Storage::Cuda(c) => c.as_cuda_slice::<f32>()?,
        _ => candle_core::bail!("topk_weights must be cuda"),
    };
    let ic3 = mistralrs_quant::cutile::cutile_grouped_gemm(
        &ic2,
        &dn,
        &sids,
        &eids,
        &ntpp,
        Some(tw_slice),
        em,
        num_valid,
        1,
        true,
        cfg,
        dev,
    )?;
    let got = mistralrs_quant::moe::cuda::moe_sum_bf16(&ic3, num_tokens, topk, dev)?
        .to_dtype(DType::F32)?
        .to_device(&cpu)?
        .flatten_all()?
        .to_vec1::<f32>()?;

    let mut max_diff = 0f32;
    for (&g, &e) in got.iter().zip(expect.iter()) {
        max_diff = max_diff.max((g - e).abs());
    }
    assert!(max_diff < tol, "max diff {max_diff}");
    Ok(())
}

#[test]
fn cutile_moe_silu_imbalanced_routing() -> Result<()> {
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
    run_cutile_case(
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
