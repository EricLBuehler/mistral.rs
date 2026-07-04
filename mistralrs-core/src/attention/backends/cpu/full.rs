use candle_core::{Device, Result, Tensor, WithDType};
use rayon::prelude::*;

use super::{elem::ElemOps, prefetch::prefetch, threading::FLASH_ATTN_POOL, CpuAttnCtx};

pub(super) fn run<T>(ctx: &CpuAttnCtx<'_, T>) -> Result<Tensor>
where
    T: WithDType + ElemOps + Send + Sync,
{
    let [b, q_len, h, d] = ctx.q.dims;
    let kv_len = ctx.k.dims[1];
    let k_h = ctx.k.dims[2];
    let v_h = ctx.v.dims[2];
    let rk2 = h / k_h;
    let rv2 = h / v_h;
    let dv = d;
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

    assert_eq!(ctx.q.stride[3], 1, "q must have contiguous rows");
    assert_eq!(ctx.k.stride[3], 1, "k must have contiguous rows");
    assert_eq!(ctx.v.stride[3], 1, "v must have contiguous rows");

    let mut out = vec![T::cast(0.0); b * q_len * h * dv];

    FLASH_ATTN_POOL.install(|| {
        out.par_chunks_mut(dv)
            .with_min_len(64)
            .enumerate()
            .for_each(|(row_idx, out_chunk)| {
                let rows_per_batch = h * q_len;
                let b_i = row_idx / rows_per_batch;
                let rem = row_idx % rows_per_batch;
                let h_i = rem / q_len;
                let q_pos = rem % q_len;

                let slope = if ctx.max_bias > 0.0 {
                    2.0f32.powf(-ctx.max_bias * ((h_i + 1) as f32) / n2 as f32)
                } else {
                    1.0
                };

                let k_head = h_i / rk2;
                let v_head = h_i / rv2;
                let q_base =
                    b_i * ctx.q.stride[0] + q_pos * ctx.q.stride[1] + h_i * ctx.q.stride[2];
                let q_row = &ctx.q.data[q_base..q_base + d];

                let mut vkq = vec![0f32; dv];
                let mut s = 0.0f32;
                let mut m = f32::NEG_INFINITY;

                for kv_pos in 0..kv_len {
                    let next_pos = kv_pos + 1;
                    if next_pos < kv_len {
                        let next_k_base = b_i * ctx.k.stride[0]
                            + next_pos * ctx.k.stride[1]
                            + k_head * ctx.k.stride[2];
                        let next_v_base = b_i * ctx.v.stride[0]
                            + next_pos * ctx.v.stride[1]
                            + v_head * ctx.v.stride[2];
                        unsafe {
                            prefetch(ctx.k.data.as_ptr().add(next_k_base) as *const u8);
                            prefetch(ctx.v.data.as_ptr().add(next_v_base) as *const u8);
                        }
                    }

                    let mv = ctx
                        .mask
                        .as_ref()
                        .map(|mask| slope * mask.value(b_i, h_i, q_pos, kv_pos))
                        .unwrap_or(0.0);
                    if mv == f32::NEG_INFINITY {
                        continue;
                    }

                    let k_base =
                        b_i * ctx.k.stride[0] + kv_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
                    let k_row = &ctx.k.data[k_base..k_base + d];

                    let mut scale = ctx.scale;
                    if ctx.logit_softcap != 0.0 {
                        scale /= ctx.logit_softcap;
                    }
                    let mut s_val = T::dot(q_row, k_row) * scale;
                    if ctx.logit_softcap != 0.0 {
                        s_val = ctx.logit_softcap * s_val.tanh();
                    }
                    s_val += mv;

                    let m_old = m;
                    let mut ms = 1.0f32;
                    let mut vs = 1.0f32;
                    if s_val > m {
                        m = s_val;
                        ms = (m_old - m).exp();
                        T::scale_acc(&mut vkq, ms);
                    } else {
                        vs = (s_val - m).exp();
                    }

                    let v_base =
                        b_i * ctx.v.stride[0] + kv_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
                    T::mad(&mut vkq, &ctx.v.data[v_base..v_base + dv], vs);

                    s = s * ms + vs;
                }

                let inv_s = 1.0 / s;
                for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
                    *o = T::cast(*v * inv_s);
                }
            });
    });

    Tensor::from_vec(out, (b, h, q_len, dv), &Device::Cpu)
}
