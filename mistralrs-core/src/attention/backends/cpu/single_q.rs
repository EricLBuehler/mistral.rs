use candle_core::{Device, Result, Tensor, WithDType};
use rayon::prelude::*;

use super::{
    elem::ElemOps, prefetch::prefetch, threading::FLASH_ATTN_POOL, CpuAttnCtx, SINGLE_Q_STACK_DV,
};

#[derive(Clone, Copy)]
struct SingleQMeta {
    h: usize,
    d: usize,
    dv: usize,
    kv_len: usize,
    rk2: usize,
    rv2: usize,
    n2: usize,
}

pub(super) fn run<T>(ctx: &CpuAttnCtx<'_, T>) -> Result<Tensor>
where
    T: WithDType + ElemOps + Send + Sync,
{
    let [b, _q_len, h, d] = ctx.q.dims;
    let kv_len = ctx.k.dims[1];
    let meta = SingleQMeta {
        h,
        d,
        dv: d,
        kv_len,
        rk2: h / ctx.k.dims[2],
        rv2: h / ctx.v.dims[2],
        n2: 2_usize.pow((h as f32).log2().ceil() as u32),
    };
    let mut out = vec![T::cast(0.0); b * h * meta.dv];

    if T::USE_BARRIER_POOL && ctx.mask.is_none() && ctx.max_bias == 0.0 && ctx.logit_softcap == 0.0
    {
        run_barrier(ctx, meta, &mut out);
    } else {
        FLASH_ATTN_POOL.install(|| {
            out.par_chunks_mut(meta.dv)
                .enumerate()
                .for_each(|(row_idx, out_chunk)| compute_row(ctx, meta, row_idx, out_chunk));
        });
    }

    Tensor::from_vec(out, (b, h, 1, meta.dv), &Device::Cpu)
}

fn run_barrier<T>(ctx: &CpuAttnCtx<'_, T>, meta: SingleQMeta, out: &mut [T])
where
    T: ElemOps + Send + Sync,
{
    let total_rows = ctx.q.dims[0] * meta.h;
    let pool = candle_core::utils::barrier_pool();
    let n_total = pool.n_workers() + 1;
    let rows_per_thread = total_rows.div_ceil(n_total);
    let out_ptr = out.as_mut_ptr() as usize;

    pool.execute(|tid| {
        let start = tid * rows_per_thread;
        if start >= total_rows {
            return;
        }
        let end = total_rows.min((tid + 1) * rows_per_thread);
        let out_ptr = out_ptr as *mut T;
        for row_idx in start..end {
            let out_chunk =
                unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_idx * meta.dv), meta.dv) };
            compute_row(ctx, meta, row_idx, out_chunk);
        }
    });
}

fn compute_row<T>(ctx: &CpuAttnCtx<'_, T>, meta: SingleQMeta, row_idx: usize, out_chunk: &mut [T])
where
    T: ElemOps,
{
    let b_i = row_idx / meta.h;
    let h_i = row_idx % meta.h;
    let slope = if ctx.max_bias > 0.0 {
        2.0f32.powf(-ctx.max_bias * ((h_i + 1) as f32) / meta.n2 as f32)
    } else {
        1.0
    };
    let k_head = h_i / meta.rk2;
    let v_head = h_i / meta.rv2;
    let q_base = b_i * ctx.q.stride[0] + h_i * ctx.q.stride[2];
    let q_row = &ctx.q.data[q_base..q_base + meta.d];

    let mut stack_vkq = [0f32; SINGLE_Q_STACK_DV];
    let mut heap_vkq;
    let vkq: &mut [f32] = if meta.dv <= SINGLE_Q_STACK_DV {
        &mut stack_vkq[..meta.dv]
    } else {
        heap_vkq = vec![0f32; meta.dv];
        heap_vkq.as_mut_slice()
    };
    let mut s = 0f32;
    let mut m = f32::NEG_INFINITY;

    for kv_pos in 0..meta.kv_len {
        let next_pos = kv_pos + 1;
        if next_pos < meta.kv_len {
            let next_k_base =
                b_i * ctx.k.stride[0] + next_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
            let next_v_base =
                b_i * ctx.v.stride[0] + next_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
            unsafe {
                prefetch(ctx.k.data.as_ptr().add(next_k_base) as *const u8);
                prefetch(ctx.v.data.as_ptr().add(next_v_base) as *const u8);
            }
        }

        let k_base = b_i * ctx.k.stride[0] + kv_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
        let k_row = &ctx.k.data[k_base..k_base + meta.d];

        let mut s_val = T::dot(q_row, k_row) * ctx.scale;
        if ctx.logit_softcap != 0.0 {
            s_val = ctx.logit_softcap * (s_val / ctx.logit_softcap).tanh();
        }
        let mask_delta = ctx
            .mask
            .as_ref()
            .map(|mask| slope * mask.value(b_i, h_i, 0, kv_pos))
            .unwrap_or(0.0);
        s_val += mask_delta;

        let m_old = m;
        let mut ms = 1.0;
        let mut vs = 1.0;
        if s_val > m {
            m = s_val;
            ms = (m_old - m).exp();
            T::scale_acc(vkq, ms);
        } else {
            vs = (s_val - m).exp();
        }

        let v_base = b_i * ctx.v.stride[0] + kv_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
        T::mad(vkq, &ctx.v.data[v_base..v_base + meta.dv], vs);
        s = s * ms + vs;
    }

    let inv_s = 1.0 / s;
    for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
        *o = T::cast(*v * inv_s);
    }
}
