use candle_core::{Device, Result, Tensor, WithDType};
use rayon::prelude::*;

use super::{
    elem::ElemOps, prefetch::prefetch, threading::FLASH_ATTN_POOL, CpuAttnCtx, SINGLE_Q_STACK_DV,
};

// Aim for a couple of units per thread so no worker idles and the tail chunk stays short.
const UNITS_PER_THREAD: usize = 2;
// Below this many kv positions per unit the partial-merge overhead outweighs the balance win.
const MIN_KV_SPAN: usize = 256;
// Cap on q rows streamed per K/V pass; bounds the per-unit accumulator footprint.
const MAX_GQA_GROUP: usize = 8;

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
    let out_ptr = out.as_mut_ptr() as usize;

    // Rows sharing a kv head stream K/V together so the cache traffic is paid once per group;
    // group must divide rk2 so a group never straddles kv heads.
    let mut group = 1;
    if meta.rk2 == meta.rv2 && meta.h.is_multiple_of(meta.rk2) {
        for g in 1..=meta.rk2.min(MAX_GQA_GROUP) {
            if meta.rk2.is_multiple_of(g) {
                group = g;
            }
        }
    }
    let n_groups = total_rows / group;

    let n_threads = pool.n_workers() + 1;
    let target_units = UNITS_PER_THREAD * n_threads;
    let max_chunks = meta.kv_len.div_ceil(MIN_KV_SPAN).max(1);
    let n_kv_chunks = target_units.div_ceil(n_groups.max(1)).clamp(1, max_chunks);
    let kv_chunk = meta.kv_len.div_ceil(n_kv_chunks).max(1);

    // Online-softmax partial per (kv chunk, row): vkq accumulator then running (max, sum).
    let stride = meta.dv + 2;
    let units = n_groups * n_kv_chunks;
    let mut partials = vec![0f32; total_rows * n_kv_chunks * stride];
    let p_ptr = partials.as_mut_ptr() as usize;

    pool.execute_chunked(units, |range| {
        let p_ptr = p_ptr as *mut f32;
        for unit in range {
            let group_idx = unit / n_kv_chunks;
            let chunk_idx = unit % n_kv_chunks;
            let kv_start = chunk_idx * kv_chunk;
            let kv_end = if n_kv_chunks == 1 {
                meta.kv_len
            } else {
                meta.kv_len.min(kv_start + kv_chunk)
            };
            let row0 = group_idx * group;
            let base = unsafe { p_ptr.add((chunk_idx * total_rows + row0) * stride) };
            let rows = unsafe { std::slice::from_raw_parts_mut(base, group * stride) };
            compute_group_range(ctx, meta, row0, group, kv_start, kv_end, rows);
        }
    });

    let out_ptr = out_ptr as *mut T;
    for row_idx in 0..total_rows {
        let mut m_all = f32::NEG_INFINITY;
        for c in 0..n_kv_chunks {
            let base = (c * total_rows + row_idx) * stride;
            m_all = m_all.max(partials[base + meta.dv]);
        }
        let out_chunk =
            unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_idx * meta.dv), meta.dv) };
        if n_kv_chunks == 1 {
            let base = row_idx * stride;
            let s = partials[base + meta.dv + 1];
            let inv_s = 1.0 / s;
            for (o, v) in out_chunk.iter_mut().zip(&partials[base..base + meta.dv]) {
                *o = T::cast(*v * inv_s);
            }
            continue;
        }
        let mut s_all = 0f32;
        let mut stack_acc = [0f32; SINGLE_Q_STACK_DV];
        let mut heap_acc;
        let acc: &mut [f32] = if meta.dv <= SINGLE_Q_STACK_DV {
            stack_acc[..meta.dv].fill(0.0);
            &mut stack_acc[..meta.dv]
        } else {
            heap_acc = vec![0f32; meta.dv];
            heap_acc.as_mut_slice()
        };
        for c in 0..n_kv_chunks {
            let base = (c * total_rows + row_idx) * stride;
            let m_c = partials[base + meta.dv];
            let s_c = partials[base + meta.dv + 1];
            if s_c == 0.0 || m_c == f32::NEG_INFINITY {
                continue;
            }
            let w = (m_c - m_all).exp();
            s_all += s_c * w;
            for (a, v) in acc.iter_mut().zip(&partials[base..base + meta.dv]) {
                *a += v * w;
            }
        }
        let inv_s = 1.0 / s_all;
        for (o, v) in out_chunk.iter_mut().zip(acc.iter()) {
            *o = T::cast(*v * inv_s);
        }
    }
}

// One pass over K/V for `group` consecutive q rows sharing the same kv head.
// `rows` spans one chunk's contiguous row slots: vkq in [..dv], then running max and sum.
#[allow(clippy::too_many_arguments)]
fn compute_group_range<T>(
    ctx: &CpuAttnCtx<'_, T>,
    meta: SingleQMeta,
    row0: usize,
    group: usize,
    kv_start: usize,
    kv_end: usize,
    rows: &mut [f32],
) where
    T: ElemOps,
{
    let stride = meta.dv + 2;
    let b_i = row0 / meta.h;
    let h0 = row0 % meta.h;
    let k_head = h0 / meta.rk2;
    let v_head = h0 / meta.rv2;

    let mut m = [f32::NEG_INFINITY; MAX_GQA_GROUP];
    let mut s = [0f32; MAX_GQA_GROUP];

    // Score a tile of kv positions per pass so the softmax correction runs once per tile
    // (vectorized max/exp) instead of a branchy scalar update per position.
    const TILE: usize = 128;
    let mut s_tile = [0f32; MAX_GQA_GROUP * TILE];
    let k_row = |kv_pos: usize| {
        let k_base = b_i * ctx.k.stride[0] + kv_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
        &ctx.k.data[k_base..k_base + meta.d]
    };
    let mut bs = kv_start;
    while bs < kv_end {
        let be = kv_end.min(bs + TILE);
        let bn = be - bs;

        for j in 0..group {
            let h_i = h0 + j;
            let q_base = b_i * ctx.q.stride[0] + h_i * ctx.q.stride[2];
            let q_row = &ctx.q.data[q_base..q_base + meta.d];
            let tile = &mut s_tile[j * TILE..j * TILE + bn];
            let mut kv_pos = bs;
            while kv_pos + 4 <= be {
                if kv_pos + 8 <= be {
                    unsafe {
                        prefetch(k_row(kv_pos + 4).as_ptr() as *const u8);
                        prefetch(k_row(kv_pos + 6).as_ptr() as *const u8);
                    }
                }
                let dots = T::dot4(
                    q_row,
                    k_row(kv_pos),
                    k_row(kv_pos + 1),
                    k_row(kv_pos + 2),
                    k_row(kv_pos + 3),
                );
                for (o, dot) in dots.iter().enumerate() {
                    tile[kv_pos + o - bs] = dot * ctx.scale;
                }
                kv_pos += 4;
            }
            while kv_pos < be {
                tile[kv_pos - bs] = T::dot(q_row, k_row(kv_pos)) * ctx.scale;
                kv_pos += 1;
            }

            let bmax = super::elem::simd_max_f32(tile);
            let chunk_base = j * stride;
            let vkq = &mut rows[chunk_base..chunk_base + meta.dv];
            if bmax > m[j] {
                if m[j] != f32::NEG_INFINITY {
                    let corr = super::elem::fast_exp(m[j] - bmax);
                    T::scale_acc(vkq, corr);
                    s[j] *= corr;
                }
                m[j] = bmax;
            }
            let local_sum = super::elem::simd_softmax_row_f32(tile, m[j]);
            s[j] += local_sum;
        }

        let v_row_of = |kv_pos: usize| {
            b_i * ctx.v.stride[0] + kv_pos * ctx.v.stride[1] + v_head * ctx.v.stride[2]
        };
        let handled = T::pv_tile(
            rows, stride, group, meta.dv, ctx.v.data, &v_row_of, bs, be, &s_tile, TILE,
        );
        if !handled {
            for kv_pos in bs..be {
                let v_base = v_row_of(kv_pos);
                if kv_pos + 2 < be {
                    unsafe { prefetch(ctx.v.data.as_ptr().add(v_row_of(kv_pos + 2)) as *const u8) };
                }
                let v_row = &ctx.v.data[v_base..v_base + meta.dv];
                for j in 0..group {
                    let p = s_tile[j * TILE + (kv_pos - bs)];
                    let chunk_base = j * stride;
                    let vkq = &mut rows[chunk_base..chunk_base + meta.dv];
                    T::mad(vkq, v_row, p);
                }
            }
        }

        bs = be;
    }

    for j in 0..group {
        let chunk_base = j * stride;
        rows[chunk_base + meta.dv] = m[j];
        rows[chunk_base + meta.dv + 1] = s[j];
    }
}

fn compute_row<T>(ctx: &CpuAttnCtx<'_, T>, meta: SingleQMeta, row_idx: usize, out_chunk: &mut [T])
where
    T: ElemOps,
{
    let mut stack_vkq = [0f32; SINGLE_Q_STACK_DV];
    let mut heap_vkq;
    let vkq: &mut [f32] = if meta.dv <= SINGLE_Q_STACK_DV {
        &mut stack_vkq[..meta.dv]
    } else {
        heap_vkq = vec![0f32; meta.dv];
        heap_vkq.as_mut_slice()
    };
    let (_m, s) = compute_row_range(ctx, meta, row_idx, 0, meta.kv_len, vkq);
    let inv_s = 1.0 / s;
    for (o, v) in out_chunk.iter_mut().zip(vkq.iter()) {
        *o = T::cast(*v * inv_s);
    }
}

// vkq must be zeroed by the caller; returns the running (max, sum) for partial merging.
fn compute_row_range<T>(
    ctx: &CpuAttnCtx<'_, T>,
    meta: SingleQMeta,
    row_idx: usize,
    kv_start: usize,
    kv_end: usize,
    vkq: &mut [f32],
) -> (f32, f32)
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

    let mut s = 0f32;
    let mut m = f32::NEG_INFINITY;

    for kv_pos in kv_start..kv_end {
        let next_pos = kv_pos + 1;
        if next_pos < kv_end {
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

    (m, s)
}
