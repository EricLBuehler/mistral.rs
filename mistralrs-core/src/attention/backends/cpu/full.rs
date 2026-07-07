use candle_core::{Device, Result, Tensor, WithDType};
use rayon::prelude::*;

use super::{
    elem::{fast_exp, ElemOps},
    prefetch::prefetch,
    threading::FLASH_ATTN_POOL,
    CpuAttnCtx, SINGLE_Q_STACK_DV,
};

// q rows sharing K/V per pass; bounds the online-softmax state kept in registers/stack
const Q_BLOCK: usize = 8;
// kv positions scored per tile; keeps the score tile and softmax pass in L1
const KV_BLOCK: usize = 128;

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
        let tiled = ctx.logit_softcap == 0.0;
        let out_ptr = out.as_mut_ptr() as usize;
        let n_q_blocks = q_len.div_ceil(Q_BLOCK);
        let total_units = b * h * n_q_blocks;
        candle_core::utils::barrier_pool().execute_chunked(total_units, |range| {
            let out_ptr = out_ptr as *mut T;
            let mut kscratch: Vec<f32> = Vec::new();
            let mut qscratch: Vec<f32> = Vec::new();
            for unit in range {
                let q_block_idx = unit % n_q_blocks;
                let bh = unit / n_q_blocks;
                let h_i = bh % h;
                let b_i = bh / h;
                let q_start = q_block_idx * Q_BLOCK;
                let q_end = q_len.min(q_start + Q_BLOCK);
                if !(tiled
                    && compute_tiled_qblock(
                        ctx,
                        b_i,
                        h_i,
                        q_start,
                        q_end,
                        out_ptr,
                        &mut kscratch,
                        &mut qscratch,
                    ))
                {
                    compute_full_qblock(ctx, b_i, h_i, q_start, q_end, out_ptr);
                }
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

// Blocked attention for one (head, q-block): scores a KV_BLOCK tile at a time, applies the
// online-softmax correction once per tile instead of per position, then accumulates P*V with
// each v row loaded once for the whole q block. Returns false when the mask layout prevents
// contiguous row access (caller falls back to the streaming path).
#[allow(clippy::too_many_arguments)]
fn compute_tiled_qblock<T>(
    ctx: &CpuAttnCtx<'_, T>,
    b_i: usize,
    h_i: usize,
    q_start: usize,
    q_end: usize,
    out_ptr: *mut T,
    kscratch: &mut Vec<f32>,
    qscratch: &mut Vec<f32>,
) -> bool
where
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

    let mut mask_rows: [Option<&[f32]>; Q_BLOCK] = [None; Q_BLOCK];
    let mut row_start = [0usize; Q_BLOCK];
    let mut row_end = [kv_len; Q_BLOCK];
    // causal/window masks are uniformly zero inside the live range; skipping their reads
    // removes an O(len^2) f32 stream that llama.cpp's mask-free kernels never pay
    let mut row_binary = [false; Q_BLOCK];
    if let Some(mask) = ctx.mask.as_ref() {
        for j in 0..nq {
            let q_pos = q_start + j;
            let Some(row) = mask.row(b_i, h_i, q_pos, kv_len) else {
                return false;
            };
            mask_rows[j] = Some(row);
            // contiguous live block: -inf^a live^b -inf^c, anchored at the self position
            let pivot = (kv_len - q_len + q_pos).min(kv_len - 1);
            if row[pivot] == f32::NEG_INFINITY {
                continue;
            }
            let (mut lo, mut hi) = (0usize, pivot);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if row[mid] == f32::NEG_INFINITY {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            row_start[j] = lo;
            let (mut lo, mut hi) = (pivot + 1, kv_len);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if row[mid] != f32::NEG_INFINITY {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            row_end[j] = lo;
            let (rs, re) = (row_start[j], row_end[j]);
            if re > rs {
                row_binary[j] = row[rs] == 0.0 && row[re - 1] == 0.0 && row[(rs + re) / 2] == 0.0;
            }
        }
    }
    let kv_lo = (0..nq).map(|j| row_start[j]).min().unwrap_or(0);
    let kv_hi = (0..nq).map(|j| row_end[j]).max().unwrap_or(kv_len);

    let mut acc = vec![0f32; nq * dv];
    let mut m = [f32::NEG_INFINITY; Q_BLOCK];
    let mut s = [0f32; Q_BLOCK];
    let mut s_tile = [0f32; Q_BLOCK * KV_BLOCK];

    if T::EXPAND_SCORE {
        qscratch.resize(nq * d, 0.0);
        for j in 0..nq {
            let q_pos = q_start + j;
            let q_base = b_i * ctx.q.stride[0] + q_pos * ctx.q.stride[1] + h_i * ctx.q.stride[2];
            T::expand_row(
                &mut qscratch[j * d..(j + 1) * d],
                &ctx.q.data[q_base..q_base + d],
            );
        }
        kscratch.resize((KV_BLOCK + 16) * d, 0.0);
    }

    let mut bs = kv_lo;
    while bs < kv_hi {
        let be = kv_hi.min(bs + KV_BLOCK);
        let bn = be - bs;

        // score the whole tile as a small gemm against transposed-K scratch when the
        // arch provides it: K streamed once for all q rows, 16 scores per fma
        let k_row_of =
            |kv: usize| b_i * ctx.k.stride[0] + kv * ctx.k.stride[1] + k_head * ctx.k.stride[2];
        let block_scored = T::EXPAND_SCORE
            && T::score_block(
                qscratch,
                nq,
                d,
                ctx.k.data,
                &k_row_of,
                bs,
                bn,
                kscratch,
                ctx.scale,
                &mut s_tile,
                KV_BLOCK,
            );

        // A: score the tile
        for j in 0..nq {
            let lo = row_start[j].max(bs);
            let hi = row_end[j].min(be);
            if lo >= hi {
                continue;
            }
            let q_pos = q_start + j;
            let q_base = b_i * ctx.q.stride[0] + q_pos * ctx.q.stride[1] + h_i * ctx.q.stride[2];
            let q_row = &ctx.q.data[q_base..q_base + d];
            let tile_row = &mut s_tile[j * KV_BLOCK..j * KV_BLOCK + bn];
            if !block_scored {
                score_rows(ctx, q_row, k_head, b_i, bs, lo, hi, ctx.scale, tile_row);
            }
            if !row_binary[j] {
                if let Some(mrow) = mask_rows[j] {
                    for kv in lo..hi {
                        tile_row[kv - bs] += slope * mrow[kv];
                    }
                }
            }
        }

        // B: one online-softmax correction per tile
        for j in 0..nq {
            let lo = row_start[j].max(bs);
            let hi = row_end[j].min(be);
            if lo >= hi {
                continue;
            }
            let tile_row = &mut s_tile[j * KV_BLOCK..j * KV_BLOCK + bn];
            let live = &mut tile_row[lo - bs..hi - bs];
            let bmax = super::elem::simd_max_f32(live);
            if bmax > m[j] {
                if m[j] != f32::NEG_INFINITY {
                    let corr = fast_exp(m[j] - bmax);
                    for a in &mut acc[j * dv..(j + 1) * dv] {
                        *a *= corr;
                    }
                    s[j] *= corr;
                }
                m[j] = bmax;
            }
            let mj = m[j];
            let local_sum = super::elem::simd_softmax_row_f32(live, mj);
            s[j] += local_sum;
        }

        // dead tile slots must be zero: the pv kernel below runs the full tile span
        for j in 0..nq {
            let lo = row_start[j].max(bs);
            let hi = row_end[j].min(be);
            let row = &mut s_tile[j * KV_BLOCK..j * KV_BLOCK + (be - bs)];
            if lo >= hi {
                row.fill(0.0);
            } else {
                row[..lo - bs].fill(0.0);
                row[hi - bs..].fill(0.0);
            }
        }

        // C: accumulate P*V with each v row shared across the q block; register-tiled
        // groups of 4 rows where the arch provides it
        let v_row_of =
            |kv: usize| b_i * ctx.v.stride[0] + kv * ctx.v.stride[1] + v_head * ctx.v.stride[2];
        let mut j0 = 0;
        while j0 < nq {
            let g = (nq - j0).min(4);
            let handled = T::pv_tile(
                &mut acc[j0 * dv..],
                dv,
                g,
                dv,
                ctx.v.data,
                &v_row_of,
                bs,
                be,
                &s_tile[j0 * KV_BLOCK..],
                KV_BLOCK,
            );
            if !handled {
                for kv_pos in bs..be {
                    let v_base = v_row_of(kv_pos);
                    let v_row = &ctx.v.data[v_base..v_base + dv];
                    for j in j0..j0 + g {
                        let p = s_tile[j * KV_BLOCK + (kv_pos - bs)];
                        if p != 0.0 {
                            T::mad(&mut acc[j * dv..(j + 1) * dv], v_row, p);
                        }
                    }
                }
            }
            j0 += g;
        }

        bs = be;
    }

    for j in 0..nq {
        let q_pos = q_start + j;
        let row_idx = (b_i * h + h_i) * q_len + q_pos;
        let out_chunk = unsafe { std::slice::from_raw_parts_mut(out_ptr.add(row_idx * dv), dv) };
        let inv_s = 1.0 / s[j];
        for (o, v) in out_chunk.iter_mut().zip(acc[j * dv..(j + 1) * dv].iter()) {
            *o = T::cast(*v * inv_s);
        }
    }
    true
}

// Dot q_row against k rows [lo, hi), writing scaled scores into tile[lo-bs..hi-bs].
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn score_rows<T: ElemOps>(
    ctx: &CpuAttnCtx<'_, T>,
    q_row: &[T],
    k_head: usize,
    b_i: usize,
    bs: usize,
    lo: usize,
    hi: usize,
    scale: f32,
    tile: &mut [f32],
) {
    let d = q_row.len();
    let k_row = |kv_pos: usize| {
        let k_base = b_i * ctx.k.stride[0] + kv_pos * ctx.k.stride[1] + k_head * ctx.k.stride[2];
        &ctx.k.data[k_base..k_base + d]
    };
    let mut kv_pos = lo;
    while kv_pos + 4 <= hi {
        if kv_pos + 8 <= hi {
            unsafe {
                super::prefetch::prefetch(k_row(kv_pos + 4).as_ptr() as *const u8);
                super::prefetch::prefetch(k_row(kv_pos + 6).as_ptr() as *const u8);
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
            tile[kv_pos + o - bs] = dot * scale;
        }
        kv_pos += 4;
    }
    while kv_pos < hi {
        tile[kv_pos - bs] = T::dot(q_row, k_row(kv_pos)) * scale;
        kv_pos += 1;
    }
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
