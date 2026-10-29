use std::arch::is_aarch64_feature_detected;

use candle_core::utils::barrier_pool;

// BlockQ6K field offsets: ql[128], qh[64], scales[16], d(f16)
pub(crate) const Q6K_BLOCK_BYTES: usize = 210;
const QL_OFF: usize = 0;
const QH_OFF: usize = 128;
const SC_OFF: usize = 192;
const D_OFF: usize = 208;
const SUPERBLOCK: usize = 256;

// Value chunk r*16 within a 128-value half: ql byte base, high-nibble flag, qh bit
// shift, scale index. The q6 layout pairs q1..q4 at l and l+32; scales run
// sequentially, chunk r of half h uses sc[8h + r].
const CHUNK_QL: [usize; 8] = [0, 16, 32, 48, 0, 16, 32, 48];
const CHUNK_HI: [bool; 8] = [false, false, false, false, true, true, true, true];
const CHUNK_SHIFT: [u32; 8] = [0, 0, 2, 2, 4, 4, 6, 6];

pub(crate) struct Q8Row {
    qs: Vec<i8>,
    bsums: Vec<i32>,
    d: Vec<f32>,
}

fn quantize_row_q8k(lhs: &[f32]) -> Q8Row {
    let blocks = lhs.len() / SUPERBLOCK;
    let mut qs = vec![0i8; lhs.len()];
    let mut bsums = vec![0i32; blocks * 16];
    let mut d = vec![0f32; blocks];
    for b in 0..blocks {
        let src = &lhs[b * SUPERBLOCK..(b + 1) * SUPERBLOCK];
        let max = src.iter().fold(0f32, |m, v| m.max(v.abs()));
        let id = if max == 0.0 { 0.0 } else { 127.0 / max };
        for (i, v) in src.iter().enumerate() {
            qs[b * SUPERBLOCK + i] = (v * id).round().clamp(-127.0, 127.0) as i8;
        }
        for c in 0..16 {
            let sum: i32 = qs[b * SUPERBLOCK + c * 16..b * SUPERBLOCK + (c + 1) * 16]
                .iter()
                .map(|v| *v as i32)
                .sum();
            bsums[b * 16 + c] = sum;
        }
        d[b] = max / 127.0;
    }
    Q8Row { qs, bsums, d }
}

fn row_dot_scalar(blocks: &[u8], k_blocks: usize, q8: &Q8Row) -> f32 {
    let mut out = 0f32;
    for b in 0..k_blocks {
        let blk = &blocks[b * Q6K_BLOCK_BYTES..(b + 1) * Q6K_BLOCK_BYTES];
        let d = f32::from(half::f16::from_bits(u16::from_le_bytes([
            blk[D_OFF],
            blk[D_OFF + 1],
        ])));
        let mut acc = 0i64;
        let mut bias = 0i64;
        for h in 0..2usize {
            for r in 0..8usize {
                let chunk = h * 8 + r;
                let ql_off = QL_OFF + h * 64 + CHUNK_QL[r];
                let qh_off = QH_OFF + h * 32 + (r % 2) * 16;
                let qs_off = b * SUPERBLOCK + chunk * 16;
                let mut isum = 0i64;
                for i in 0..16 {
                    let mut code = blk[ql_off + i] as i64;
                    if CHUNK_HI[r] {
                        code >>= 4;
                    } else {
                        code &= 0xF;
                    }
                    code += (((blk[qh_off + i] >> CHUNK_SHIFT[r]) & 0x3) as i64) << 4;
                    isum += code * q8.qs[qs_off + i] as i64;
                }
                let scale = blk[SC_OFF + 8 * h + r] as i8 as i64;
                acc += isum * scale;
                bias += q8.bsums[b * 16 + chunk] as i64 * scale;
            }
        }
        out += q8.d[b] * d * (acc - (bias << 5)) as f32;
    }
    out
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{
        CHUNK_HI, CHUNK_QL, CHUNK_SHIFT, Q6K_BLOCK_BYTES, QH_OFF, QL_OFF, SC_OFF, SUPERBLOCK,
    };
    use std::arch::aarch64::*;

    // Codes stay unsigned 0..63 through sdot (positive as i8), so the -32 offset per
    // value is one i32 subtraction of 32*bsum at block end, same as candle's bias.
    /// vdotq_s32 is newer than the workspace MSRV but is what the pinned candle
    /// itself compiles for dotprod targets.
    #[allow(clippy::incompatible_msrv)]
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn row_dot(
        blocks: &[u8],
        k_blocks: usize,
        q8_qs: &[i8],
        q8_bsums: &[i32],
        q8_d: &[f32],
    ) -> f32 {
        let lo4 = vdupq_n_u8(0x0F);
        let lo2 = vdupq_n_u8(0x03);
        let mut out = 0f32;
        for b in 0..k_blocks {
            let blk = blocks.as_ptr().add(b * Q6K_BLOCK_BYTES);
            let d = f32::from(half::f16::from_bits(u16::from_le_bytes([
                *blk.add(208),
                *blk.add(209),
            ])));
            let mut acc_a = vdupq_n_s32(0);
            let mut acc_b = vdupq_n_s32(0);
            let mut bias = 0i32;
            for h in 0..2usize {
                for r in 0..8usize {
                    let chunk = h * 8 + r;
                    let ql = blk.add(QL_OFF + h * 64 + CHUNK_QL[r]);
                    let qh = blk.add(QH_OFF + h * 32 + (r % 2) * 16);
                    let qlv = vld1q_u8(ql);
                    let qhv = vld1q_u8(qh);
                    let qh_bits = vandq_u8(
                        match CHUNK_SHIFT[r] {
                            0 => qhv,
                            2 => vshrq_n_u8(qhv, 2),
                            4 => vshrq_n_u8(qhv, 4),
                            _ => vshrq_n_u8(qhv, 6),
                        },
                        lo2,
                    );
                    let high = vshlq_n_u8(qh_bits, 4);
                    let code = if CHUNK_HI[r] {
                        vorrq_u8(vshrq_n_u8(qlv, 4), high)
                    } else {
                        vorrq_u8(vandq_u8(qlv, lo4), high)
                    };
                    let qs = vld1q_s8(q8_qs.as_ptr().add(b * SUPERBLOCK + chunk * 16));
                    let dot = vdotq_s32(vdupq_n_s32(0), vreinterpretq_s8_u8(code), qs);
                    let scale = vdupq_n_s32(*(blk.add(SC_OFF + 8 * h + r)) as i8 as i32);
                    if chunk.is_multiple_of(2) {
                        acc_a = vmlaq_s32(acc_a, dot, scale);
                    } else {
                        acc_b = vmlaq_s32(acc_b, dot, scale);
                    }
                    bias += q8_bsums[b * 16 + chunk] * vgetq_lane_s32::<0>(scale);
                }
            }
            let acc = vaddvq_s32(vaddq_s32(acc_a, acc_b));
            out += q8_d[b] * d * (acc - bias * 32) as f32;
        }
        out
    }

    pub(super) fn gemv_rows(
        bytes: &[u8],
        k_blocks: usize,
        q8: &super::Q8Row,
        dst: &mut [f32],
        rows: std::ops::Range<usize>,
    ) {
        for row in rows {
            let blocks =
                &bytes[row * k_blocks * Q6K_BLOCK_BYTES..(row + 1) * k_blocks * Q6K_BLOCK_BYTES];
            let v = unsafe { row_dot(blocks, k_blocks, &q8.qs, &q8.bsums, &q8.d) };
            dst[row] = v;
        }
    }
}

pub(crate) fn dotprod_supported() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        static SUPPORTED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *SUPPORTED.get_or_init(|| is_aarch64_feature_detected!("dotprod"))
    }
    #[cfg(not(target_arch = "aarch64"))]
    {
        false
    }
}

/// m==1 Q6_K matvec straight off the mapped block bytes. Returns false when the
/// layout is unsupported and the caller should fall back to candle.
pub(crate) fn gemv(n: usize, k: usize, bytes: &[u8], lhs: &[f32], dst: &mut [f32]) -> bool {
    if !k.is_multiple_of(SUPERBLOCK) || bytes.len() != n * k / SUPERBLOCK * Q6K_BLOCK_BYTES {
        return false;
    }
    let k_blocks = k / SUPERBLOCK;
    let q8 = quantize_row_q8k(lhs);
    let pool = barrier_pool();
    let dst_ptr = dst.as_mut_ptr() as usize;
    #[cfg(target_arch = "aarch64")]
    if dotprod_supported() {
        pool.execute_chunked(n, move |rows| {
            let dst =
                unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, rows.end.min(n)) };
            neon::gemv_rows(bytes, k_blocks, &q8, dst, rows);
        });
        return true;
    }
    pool.execute_chunked(n, move |rows| {
        for row in rows {
            let blocks =
                &bytes[row * k_blocks * Q6K_BLOCK_BYTES..(row + 1) * k_blocks * Q6K_BLOCK_BYTES];
            let dst = unsafe { std::slice::from_raw_parts_mut(dst_ptr as *mut f32, row + 1) };
            dst[row] = row_dot_scalar(blocks, k_blocks, &q8);
        }
    });
    true
}

#[cfg(test)]
pub(crate) fn test_quantize_row_q8k(lhs: &[f32]) -> Q8Row {
    quantize_row_q8k(lhs)
}

#[cfg(test)]
pub(crate) fn test_scalar_row_dot(blocks: &[u8], k_blocks: usize, q8: &Q8Row) -> f32 {
    row_dot_scalar(blocks, k_blocks, q8)
}

#[cfg(all(test, target_arch = "aarch64"))]
pub(crate) fn test_neon_row_dot(blocks: &[u8], k_blocks: usize, q8: &Q8Row) -> f32 {
    unsafe { neon::row_dot(blocks, k_blocks, &q8.qs, &q8.bsums, &q8.d) }
}
