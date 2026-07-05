use candle_core::{Device, Result, Tensor, WithDType};
use rayon::prelude::*;

use super::{
    elem::ElemOps, prefetch::prefetch, threading::FLASH_ATTN_POOL, CpuAttnCtx, SINGLE_Q_STACK_DV,
};

// q rows sharing K/V per pass; bounds the online-softmax state kept in registers/stack
const Q_BLOCK: usize = 8;

pub(super) fn run<T>(ctx: &CpuAttnCtx<'_, T>) -> Result<Tensor>
where
    T: WithDType + ElemOps + Send + Sync,
{
    let [b, q_len, h, d] = ctx.q.dims;
    let dv = d;

    assert_eq!(ctx.q.stride[3], 1, "q must have contiguous rows");
    assert_eq!(ctx.k.stride[3], 1, "k must have contiguous rows");
    assert_eq!(ctx.v.stride[3], 1, "v must have contiguous rows");

    let mut out = vec![T::cast(0.0); b * q_len * h * dv];

    if T::USE_BARRIER_POOL {
        // rayon here would fight the barrier workers spinning between the surrounding matmuls;
        // q rows of one head share K/V, so blocks of Q_BLOCK rows stream them once
        let out_ptr = out.as_mut_ptr() as usize;
        let n_q_blocks = q_len.div_ceil(Q_BLOCK);
        let total_units = b * h * n_q_blocks;
        candle_core::utils::barrier_pool().execute_chunked(total_units, |range| {
            let out_ptr = out_ptr as *mut T;
            for unit in range {
                let q_block_idx = unit % n_q_blocks;
                let bh = unit / n_q_blocks;
                let h_i = bh % h;
                let b_i = bh / h;
                let q_start = q_block_idx * Q_BLOCK;
                let q_end = q_len.min(q_start + Q_BLOCK);
                compute_full_qblock(ctx, b_i, h_i, q_start, q_end, out_ptr);
            }
        });
    } else {
        FLASH_ATTN_POOL.install(|| {
            out.par_chunks_mut(dv)
                .with_min_len(64)
                .enumerate()
                .for_each(|(row_idx, out_chunk)| compute_full_row(ctx, row_idx, out_chunk));
        });
    }

    Tensor::from_vec(out, (b, h, q_len, dv), &Device::Cpu)
}

// One K/V pass for q rows [q_start, q_end) of head h_i; per-row online softmax state.
fn compute_full_qblock<T>(
    ctx: &CpuAttnCtx<'_, T>,
    b_i: usize,
    h_i: usize,
    q_start: usize,
    q_end: usize,
    out_ptr: *mut T,
) where
    T: ElemOps,
{
    let [_b, q_len, h, d] = ctx.q.dims;
    let kv_len = ctx.k.dims[1];
    let rk2 = h / ctx.k.dims[2];
    let rv2 = h / ctx.v.dims[2];
    let dv = d;
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);
    let nq = q_end - q_start;

    let slope = if ctx.max_bias > 0.0 {
        2.0f32.powf(-ctx.max_bias * ((h_i + 1) as f32) / n2 as f32)
    } else {
        1.0
    };
    let k_head = h_i / rk2;
    let v_head = h_i / rv2;

    let mut vkq = vec![0f32; nq * dv];
    let mut m = [f32::NEG_INFINITY; Q_BLOCK];
    let mut s = [0f32; Q_BLOCK];
    let mut scale = ctx.scale;
    if ctx.logit_softcap != 0.0 {
        scale /= ctx.logit_softcap;
    }

    // Live kv range per row: everything outside [start, end) is -inf, so the kv loop
    // can skip those K/V loads and mask lookups entirely (most of the axis for causal).
    let mut row_start = [0usize; Q_BLOCK];
    let mut row_end = [kv_len; Q_BLOCK];
    if let Some(mask) = ctx.mask.as_ref() {
        // causal/window masks are a contiguous live block: -inf^a live^b -inf^c,
        // so both boundaries binary-search in ~log(kv) lookups per row
        for j in 0..nq {
            let q_pos = q_start + j;
            let is_inf = |kv: usize| mask.value(b_i, h_i, q_pos, kv) == f32::NEG_INFINITY;
            // the query's own position is always live, anchoring both searches
            let pivot = (kv_len - q_len + q_pos).min(kv_len - 1);
            if is_inf(pivot) {
                // unexpected mask shape: fall back to the full range
                continue;
            }
            let (mut lo, mut hi) = (0usize, pivot);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if is_inf(mid) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            row_start[j] = lo;
            let (mut lo, mut hi) = (pivot + 1, kv_len);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if !is_inf(mid) {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            row_end[j] = lo;
        }
    }
    let kv_lo = (0..nq).map(|j| row_start[j]).min().unwrap_or(0);
    let kv_hi = (0..nq).map(|j| row_end[j]).max().unwrap_or(kv_len);

    for kv_pos in kv_lo..kv_hi {
        let next_pos = kv_pos + 1;
        if next_pos < kv_len {
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
        let k_row = &ctx.k.data[k_base..k_base + d];
        let v_base = b_i * ctx.v.stride[0] + kv_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
        let v_row = &ctx.v.data[v_base..v_base + dv];

        for j in 0..nq {
            if kv_pos >= row_end[j] {
                continue;
            }
            let q_pos = q_start + j;
            let mv = ctx
                .mask
                .as_ref()
                .map(|mask| slope * mask.value(b_i, h_i, q_pos, kv_pos))
                .unwrap_or(0.0);
            if mv == f32::NEG_INFINITY {
                continue;
            }

            let q_base = b_i * ctx.q.stride[0] + q_pos * ctx.q.stride[1] + h_i * ctx.q.stride[2];
            let q_row = &ctx.q.data[q_base..q_base + d];

            let mut s_val = T::dot(q_row, k_row) * scale;
            if ctx.logit_softcap != 0.0 {
                s_val = ctx.logit_softcap * s_val.tanh();
            }
            s_val += mv;

            let acc = &mut vkq[j * dv..(j + 1) * dv];
            let m_old = m[j];
            let mut ms = 1.0f32;
            let mut vs = 1.0f32;
            if s_val > m[j] {
                m[j] = s_val;
                ms = super::elem::fast_exp(m_old - m[j]);
                T::scale_acc(acc, ms);
            } else {
                vs = super::elem::fast_exp(s_val - m[j]);
            }
            T::mad(acc, v_row, vs);
            s[j] = s[j] * ms + vs;
        }
    }

    for j in 0..nq {
        let q_pos = q_start + j;
        let row_idx = (b_i * h + h_i) * q_len + q_pos;
        let out_chunk = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_idx * dv), dv) };
        let inv_s = 1.0 / s[j];
        for (o, v) in out_chunk.iter_mut().zip(vkq[j * dv..(j + 1) * dv].iter()) {
            *o = T::cast(*v * inv_s);
        }
    }
}

fn compute_full_row<T>(ctx: &CpuAttnCtx<'_, T>, row_idx: usize, out_chunk: &mut [T])
where
    T: ElemOps,
{
    let [_b, q_len, h, d] = ctx.q.dims;
    let kv_len = ctx.k.dims[1];
    let rk2 = h / ctx.k.dims[2];
    let rv2 = h / ctx.v.dims[2];
    let dv = d;
    let n2 = 2_usize.pow((h as f32).log2().ceil() as u32);

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
    let q_base = b_i * ctx.q.stride[0] + q_pos * ctx.q.stride[1] + h_i * ctx.q.stride[2];
    let q_row = &ctx.q.data[q_base..q_base + d];

    let mut stack_vkq = [0f32; SINGLE_Q_STACK_DV];
    let mut heap_vkq;
    let vkq: &mut [f32] = if dv <= SINGLE_Q_STACK_DV {
        &mut stack_vkq[..dv]
    } else {
        heap_vkq = vec![0f32; dv];
        heap_vkq.as_mut_slice()
    };
    let mut s = 0.0f32;
    let mut m = f32::NEG_INFINITY;

    for kv_pos in 0..kv_len {
        let next_pos = kv_pos + 1;
        if next_pos < kv_len {
            let next_k_base =
                b_i * ctx.k.stride[0] + next_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
            let next_v_base =
                b_i * ctx.v.stride[0] + next_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
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

        let k_base = b_i * ctx.k.stride[0] + kv_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
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
            T::scale_acc(vkq, ms);
        } else {
            vs = (s_val - m).exp();
        }

        let v_base = b_i * ctx.v.stride[0] + kv_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2];
        T::mad(vkq, &ctx.v.data[v_base..v_base + dv], vs);

        s = s * ms + vs;
    }

    let inv_s = 1.0 / s;
    for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
        *o = T::cast(*v * inv_s);
    }
}
