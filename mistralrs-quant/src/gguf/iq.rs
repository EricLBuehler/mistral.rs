//! The IQ family: block layouts and dequantisation, ported from ggml's `ggml-quants.c`.
//!
//! Candle has no IQ block types, so these live here: the GGUF reader hands over the raw
//! block bytes, and this module turns them into f32. The arithmetic follows the C exactly -
//! same operation order, f32 throughout - so a block dequantises bit for bit the same.

use candle_core::{DType, Device, Result, Tensor};
use std::{
    ops::Range,
    sync::{
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
        Arc, OnceLock,
    },
};

use super::archive::{GgufArchive, GgufDType};
use super::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

const IQ1S_DELTA: f32 = 0.125;

#[inline]
fn f16_scale(block: &[u8]) -> f32 {
    f32::from(half::f16::from_bits(u16::from_le_bytes([
        block[0], block[1],
    ])))
}

/// `aux32` from four u16s, as ggml's `memcpy(aux32, qs + 4*ib32, 2*sizeof(uint32_t))`.
#[inline]
fn le_u32_pair(qs: &[u8]) -> [u32; 2] {
    let word = |b: &[u8]| u32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    [word(&qs[..4]), word(&qs[4..8])]
}

/// A grid row as eight bytes, little endian - what `(const uint8_t *)(grid + idx)` reads.
#[inline]
fn grid_bytes64(grid: &[u64], idx: usize) -> [u8; 8] {
    grid[idx].to_le_bytes()
}

/// A u32 grid row as four bytes.
#[inline]
fn grid_bytes32(grid: &[u32], idx: usize) -> [u8; 4] {
    grid[idx].to_le_bytes()
}

/// The sign bit ggml flips value `j` by.
#[inline]
fn sign_bit(signs: u8, j: usize) -> bool {
    signs & KMASK_IQ2XS[j] != 0
}

/// 2.0625 bpw. `d`, then 32 u16s: 4 grid indices and one packed scale-and-signs word per 32
/// values, with the signs unpacked through `ksigns_iq2xs`.
///
/// aarch64 runs the vector decoder; this is its oracle, and the fallback elsewhere.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dequantize_iq2_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<66>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..];
        for ib32 in 0..8 {
            let aux32 = le_u32_pair(&qs[8 * ib32..8 * ib32 + 8]);
            let db = d * (0.5f32 + (aux32[1] >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = grid_bytes64(&IQ2XXS_GRID, (aux32[0] >> (8 * l)) as usize & 0xff);
                let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                for (j, &g) in grid.iter().enumerate() {
                    *y.next().expect("iq2_xxs output length") =
                        db * g as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 2.3125 bpw. `d`, 32 u16s of 9-bit grid index plus 7-bit sign code, then 8 scale bytes.
fn dequantize_iq2_xs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<74>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..66];
        let scales = &block[66..74];
        for ib32 in 0..8 {
            let db = [
                d * (0.5f32 + (scales[ib32] & 0xf) as f32) * 0.25,
                d * (0.5f32 + (scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let index =
                    u16::from_le_bytes([qs[2 * (4 * ib32 + l)], qs[2 * (4 * ib32 + l) + 1]]);
                let grid = grid_bytes64(&IQ2XS_GRID, (index & 511) as usize);
                let signs = KSIGNS_IQ2XS[(index >> 9) as usize];
                for (j, &g) in grid.iter().enumerate() {
                    *y.next().expect("iq2_xs output length") =
                        db[l / 2] * g as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 2.5625 bpw. `d`, 64 bytes: 32 of grid indices then 32 of sign bytes, then `qh` and scales.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dequantize_iq2_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<82>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..34];
        // ggml's `signs` pointer is `qs + QK_K/8`: the signs live in the back half of the
        // qs array, not beside the scales.
        let signs = &block[34..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        for ib32 in 0..8 {
            let db = [
                d * (0.5f32 + (scales[ib32] & 0xf) as f32) * 0.25,
                d * (0.5f32 + (scales[ib32] >> 4) as f32) * 0.25,
            ];
            for l in 0..4 {
                let index =
                    qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                let grid = grid_bytes64(&IQ2S_GRID, index);
                let sign = signs[4 * ib32 + l];
                for (j, &g) in grid.iter().enumerate() {
                    *y.next().expect("iq2_s output length") =
                        db[l / 2] * g as f32 * if sign_bit(sign, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 3.0625 bpw. `d`, then 96 bytes: 64 of grid indices followed by 32 of packed scale-and-signs.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dequantize_iq3_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<98>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..98];
        let scales_and_signs = 64usize;
        for ib32 in 0..8 {
            let aux32 = u32::from_le_bytes([
                qs[scales_and_signs + 4 * ib32],
                qs[scales_and_signs + 4 * ib32 + 1],
                qs[scales_and_signs + 4 * ib32 + 2],
                qs[scales_and_signs + 4 * ib32 + 3],
            ]);
            let db = d * (0.5f32 + (aux32 >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                let grid1 = grid_bytes32(&IQ3XXS_GRID, qs[8 * ib32 + 2 * l] as usize);
                let grid2 = grid_bytes32(&IQ3XXS_GRID, qs[8 * ib32 + 2 * l + 1] as usize);
                // y[j] comes from grid1 and y[j+4] from grid2: the block is two halves.
                for (j, &g) in grid1.iter().enumerate() {
                    *y.next().expect("iq3_xxs output length") =
                        db * g as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
                for (j, &g) in grid2.iter().enumerate() {
                    *y.next().expect("iq3_xxs output length") =
                        db * g as f32 * if sign_bit(signs, j + 4) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 3.4375 bpw. `d`, 64 bytes of grid indices, 8 of high bits, 32 of signs, 4 of scales.
fn dequantize_iq3_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<110>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..66];
        let qh = &block[66..74];
        let signs = &block[74..106];
        let scales = &block[106..110];
        let mut qs_off = 0usize;
        let mut signs_off = 0usize;
        for outer in 0..4 {
            let scale = scales[outer];
            let dbs = [
                d * (1 + 2 * (scale & 0xf) as i32) as f32,
                d * (1 + 2 * (scale >> 4) as i32) as f32,
            ];
            for (half, db) in dbs.iter().enumerate() {
                let qh_byte = qh[2 * outer + half] as usize;
                for l in 0..4 {
                    let index1 = qs[qs_off + 2 * l] as usize | ((qh_byte << (8 - 2 * l)) & 256);
                    let index2 = qs[qs_off + 2 * l + 1] as usize | ((qh_byte << (7 - 2 * l)) & 256);
                    let grid1 = grid_bytes32(&IQ3S_GRID, index1);
                    let grid2 = grid_bytes32(&IQ3S_GRID, index2);
                    let sign = signs[signs_off + l];
                    // y[j] from grid1, y[j+4] from grid2: the block is two halves.
                    for (j, &g) in grid1.iter().enumerate() {
                        *y.next().expect("iq3_s output length") =
                            *db * g as f32 * if sign_bit(sign, j) { -1.0 } else { 1.0 };
                    }
                    for (j, &g) in grid2.iter().enumerate() {
                        *y.next().expect("iq3_s output length") =
                            *db * g as f32 * if sign_bit(sign, j + 4) { -1.0 } else { 1.0 };
                    }
                }
                qs_off += 8;
                signs_off += 4;
            }
        }
    }
}

/// 1.5625 bpw. `d`, 32 bytes of grid indices, then 8 u16s of per-32-value high bits and scale.
#[cfg_attr(target_arch = "aarch64", allow(dead_code))]
fn dequantize_iq1_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<50>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..34];
        let qh = &block[34..50];
        for ib in 0..8 {
            let qh_ib = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
            let dl = d * (2 * ((qh_ib >> 12) & 7) + 1) as f32;
            let delta = if qh_ib & 0x8000 != 0 {
                -IQ1S_DELTA
            } else {
                IQ1S_DELTA
            };
            for l in 0..4 {
                let index = qs[4 * ib + l] as usize | ((((qh_ib >> (3 * l)) & 7) as usize) << 8);
                let grid = IQ1S_GRID[index].to_le_bytes();
                for &g in grid.iter() {
                    *y.next().expect("iq1_s output length") = dl * (g as i8 as f32 + delta);
                }
            }
        }
    }
}

/// 1.75 bpw. No `d`: the scale is assembled from the top nibbles of the four scale words.
fn dequantize_iq1_m(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<56>().0.iter().take(n_blocks) {
        let qs = &block[0..32];
        let qh = &block[32..48];
        let scales = &block[48..56];
        let sc = [
            u16::from_le_bytes([scales[0], scales[1]]),
            u16::from_le_bytes([scales[2], scales[3]]),
            u16::from_le_bytes([scales[4], scales[5]]),
            u16::from_le_bytes([scales[6], scales[7]]),
        ];
        let scale_bits =
            (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
        let d = f32::from(half::f16::from_bits(scale_bits));

        for ib in 0..8 {
            let dl1 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) + 1) as f32;
            let dl2 = d * (2 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) + 1) as f32;
            let qh0 = qh[2 * ib] as usize;
            let qh1 = qh[2 * ib + 1] as usize;
            let idx = [
                qs[4 * ib] as usize | ((qh0 << 8) & 0x700),
                qs[4 * ib + 1] as usize | ((qh0 << 4) & 0x700),
                qs[4 * ib + 2] as usize | ((qh1 << 8) & 0x700),
                qs[4 * ib + 3] as usize | ((qh1 << 4) & 0x700),
            ];
            let deltas = [
                if qh0 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh0 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh1 & 0x08 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
                if qh1 & 0x80 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                },
            ];
            for l in 0..2 {
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for &g in grid.iter() {
                    *y.next().expect("iq1_m output length") = dl1 * (g as i8 as f32 + deltas[l]);
                }
            }
            for l in 2..4 {
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for &g in grid.iter() {
                    *y.next().expect("iq1_m output length") = dl2 * (g as i8 as f32 + deltas[l]);
                }
            }
        }
    }
}

/// 4.5 bpw. `d`, then 16 bytes of nibbles into `kvalues_iq4nl`.
fn dequantize_iq4_nl(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<18>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..18];
        // y[j] takes the low nibble and y[j+16] the high one: two halves, not interleaved.
        for j in 0..16 {
            *y.next().expect("iq4_nl output length") =
                d * KVALUES_IQ4NL[(qs[j] & 0xf) as usize] as f32;
        }
        for j in 0..16 {
            *y.next().expect("iq4_nl output length") =
                d * KVALUES_IQ4NL[(qs[j] >> 4) as usize] as f32;
        }
    }
}

/// 4.25 bpw. `d`, a 16-bit high-scale word, 4 low-scale bytes, then 128 bytes of nibbles.
fn dequantize_iq4_xs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.as_chunks::<136>().0.iter().take(n_blocks) {
        let d = f16_scale(block);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..8 {
            let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
                | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
            let dl = d * (ls - 32) as f32;
            let q = &qs[16 * ib..16 * ib + 16];
            // y[j] takes the low nibble and y[j+16] the high one: two halves.
            for j in 0..16 {
                *y.next().expect("iq4_xs output length") =
                    dl * KVALUES_IQ4NL[(q[j] & 0xf) as usize] as f32;
            }
            for j in 0..16 {
                *y.next().expect("iq4_xs output length") =
                    dl * KVALUES_IQ4NL[(q[j] >> 4) as usize] as f32;
            }
        }
    }
}

/// Eight values at a time: a grid row is eight i8s, so each 32-value group is four vectors of
/// eight, and the sign code becomes an xor mask. `db * g` and `g * db` are the same f32
/// product and a sign flip is the same negation as multiplying by -1.0, so this decodes bit
/// for bit what the scalar loop does. NEON is baseline on aarch64, so there is no feature
/// check to make.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::*;
    use std::{arch::aarch64::*, sync::OnceLock};

    /// For each 7-bit sign code, the lanes whose bit is set get 0x80000000 and the rest 0 -
    /// exactly the negation the scalar loop applies with `-1.0`.
    static SIGN_MASKS: OnceLock<Vec<[u32; 8]>> = OnceLock::new();

    fn sign_masks() -> &'static Vec<[u32; 8]> {
        SIGN_MASKS.get_or_init(|| {
            KSIGNS_IQ2XS
                .iter()
                .map(|&code| {
                    let mut mask = [0u32; 8];
                    for (j, lane) in mask.iter_mut().enumerate() {
                        if sign_bit(code, j) {
                            *lane = 0x8000_0000;
                        }
                    }
                    mask
                })
                .collect()
        })
    }

    /// IQ2_S stores the sign byte itself, not a 7-bit index into `ksigns_iq2xs`.
    static SIGN_BYTE_MASKS: OnceLock<Vec<[u32; 8]>> = OnceLock::new();

    fn sign_byte_masks() -> &'static Vec<[u32; 8]> {
        SIGN_BYTE_MASKS.get_or_init(|| {
            (0..256u16)
                .map(|sign| {
                    let mut mask = [0u32; 8];
                    for (j, lane) in mask.iter_mut().enumerate() {
                        if sign_bit(sign as u8, j) {
                            *lane = 0x8000_0000;
                        }
                    }
                    mask
                })
                .collect()
        })
    }

    /// Eight grid bytes as `g as f32`, times `db`, negated where `mask` has the sign bit.
    /// IQ2 and IQ3 grid bytes are <= 64, so the signed widen matches the scalar cast.
    unsafe fn scale_signed8(
        grid: *const u8,
        db: float32x4_t,
        mask: *const u32,
    ) -> (float32x4_t, float32x4_t) {
        let g = vld1_s8(grid as *const i8);
        let g16 = vmovl_s8(g);
        let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16)));
        let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16)));
        let flo = veorq_u32(vreinterpretq_u32_f32(vmulq_f32(lo, db)), vld1q_u32(mask));
        let fhi = veorq_u32(
            vreinterpretq_u32_f32(vmulq_f32(hi, db)),
            vld1q_u32(mask.add(4)),
        );
        (vreinterpretq_f32_u32(flo), vreinterpretq_f32_u32(fhi))
    }

    /// Four grid bytes, same scaling as [`scale_signed8`].
    unsafe fn scale_signed4(grid: *const u8, db: float32x4_t, mask: *const u32) -> float32x4_t {
        let packed = vreinterpret_s8_u32(vdup_n_u32(std::ptr::read_unaligned(grid as *const u32)));
        let g16 = vmovl_s8(packed);
        let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16)));
        vreinterpretq_f32_u32(veorq_u32(
            vreinterpretq_u32_f32(vmulq_f32(lo, db)),
            vld1q_u32(mask),
        ))
    }

    /// `dl * (g as i8 as f32 + delta)`: add, then multiply, which is the scalar order.
    unsafe fn scale_i8_delta(
        grid: *const u8,
        dl: float32x4_t,
        delta: f32,
    ) -> (float32x4_t, float32x4_t) {
        let g = vld1_s8(grid as *const i8);
        let g16 = vmovl_s8(g);
        let dlt = vdupq_n_f32(delta);
        let lo = vmulq_f32(
            vaddq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16))), dlt),
            dl,
        );
        let hi = vmulq_f32(
            vaddq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16))), dlt),
            dl,
        );
        (lo, hi)
    }

    struct Acc {
        v: [float32x4_t; 4],
        turn: usize,
    }

    impl Acc {
        fn new() -> Self {
            Self {
                v: [unsafe { vdupq_n_f32(0.0) }; 4],
                turn: 0,
            }
        }

        unsafe fn fma8(&mut self, lo: float32x4_t, hi: float32x4_t, x: *const f32) {
            self.v[self.turn] = vfmaq_f32(self.v[self.turn], lo, vld1q_f32(x));
            let next = (self.turn + 1) % 4;
            self.v[next] = vfmaq_f32(self.v[next], hi, vld1q_f32(x.add(4)));
            self.turn = (self.turn + 2) % 4;
        }

        fn finish(self) -> f32 {
            unsafe {
                let pairs = vaddq_f32(
                    vaddq_f32(self.v[0], self.v[1]),
                    vaddq_f32(self.v[2], self.v[3]),
                );
                vaddvq_f32(pairs)
            }
        }
    }

    /// `a . b`, four lanes of four sums: the fallback for types that do not have a fused
    /// decoder yet, where the row is decoded into a scratch row first.
    pub(super) fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut acc = [unsafe { vdupq_n_f32(0.0) }; 4];
        let chunks = a.len() / 4;
        for c in 0..chunks {
            unsafe {
                let x = vld1q_f32(a.as_ptr().add(c * 4));
                let y = vld1q_f32(b.as_ptr().add(c * 4));
                acc[c % 4] = vfmaq_f32(acc[c % 4], x, y);
            }
        }
        let mut tail = 0.0f32;
        for c in chunks * 4..a.len() {
            tail += a[c] * b[c];
        }
        unsafe {
            let pairs = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
            vaddvq_f32(pairs) + tail
        }
    }

    /// `row . x` for one IQ2_XXS row: the same decode, multiplied into four running sums
    /// instead of being stored. Because it never materialises the row, the fused gather reads
    /// the block bytes and nothing else.
    pub(super) fn iq2_xxs_row_dot(block: &[u8], n_blocks: usize, x: &[f32]) -> f32 {
        let masks = sign_masks();
        let mut acc = [unsafe { vdupq_n_f32(0.0) }; 4];
        let mut turn = 0usize;
        for b in 0..n_blocks {
            let blk = &block[b * 66..][..66];
            let d = f16_scale(blk);
            let qs = &blk[2..];
            for ib32 in 0..8 {
                let aux32 = le_u32_pair(&qs[8 * ib32..8 * ib32 + 8]);
                let db = d * (0.5f32 + (aux32[1] >> 28) as f32) * 0.25;
                let dbv = unsafe { vdupq_n_f32(db) };
                for l in 0..4 {
                    let grid = (aux32[0] >> (8 * l)) as usize & 0xff;
                    let code = ((aux32[1] >> (7 * l)) & 127) as usize;
                    let at = b * 256 + (ib32 * 4 + l) * 8;
                    unsafe {
                        let g = vld1_s8(IQ2XXS_GRID.as_ptr().add(grid) as *const i8);
                        let g16 = vmovl_s8(g);
                        let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16)));
                        let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16)));
                        let m = masks[code].as_ptr();
                        let flo =
                            veorq_u32(vreinterpretq_u32_f32(vmulq_f32(lo, dbv)), vld1q_u32(m));
                        let fhi = veorq_u32(
                            vreinterpretq_u32_f32(vmulq_f32(hi, dbv)),
                            vld1q_u32(m.add(4)),
                        );
                        let xp = x.as_ptr().add(at);
                        acc[turn] = vfmaq_f32(acc[turn], vreinterpretq_f32_u32(flo), vld1q_f32(xp));
                        acc[(turn + 1) % 4] = vfmaq_f32(
                            acc[(turn + 1) % 4],
                            vreinterpretq_f32_u32(fhi),
                            vld1q_f32(xp.add(4)),
                        );
                    }
                    turn = (turn + 2) % 4;
                }
            }
        }
        unsafe {
            let pairs = vaddq_f32(vaddq_f32(acc[0], acc[1]), vaddq_f32(acc[2], acc[3]));
            vaddvq_f32(pairs)
        }
    }

    pub(super) fn dequantize_iq2_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
        let masks = sign_masks();
        for (i, block) in blocks.as_chunks::<66>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..];
            let out = unsafe { y.as_mut_ptr().add(i * 256) };
            for ib32 in 0..8 {
                let aux32 = le_u32_pair(&qs[8 * ib32..8 * ib32 + 8]);
                let db = d * (0.5f32 + (aux32[1] >> 28) as f32) * 0.25;
                let dbv = unsafe { vdupq_n_f32(db) };
                for l in 0..4 {
                    let grid = (aux32[0] >> (8 * l)) as usize & 0xff;
                    let code = ((aux32[1] >> (7 * l)) & 127) as usize;
                    unsafe {
                        let g = vld1_s8(IQ2XXS_GRID.as_ptr().add(grid) as *const i8);
                        let g16 = vmovl_s8(g);
                        let lo = vcvtq_f32_s32(vmovl_s16(vget_low_s16(g16)));
                        let hi = vcvtq_f32_s32(vmovl_s16(vget_high_s16(g16)));
                        let m = masks[code].as_ptr();
                        let flo =
                            veorq_u32(vreinterpretq_u32_f32(vmulq_f32(lo, dbv)), vld1q_u32(m));
                        let fhi = veorq_u32(
                            vreinterpretq_u32_f32(vmulq_f32(hi, dbv)),
                            vld1q_u32(m.add(4)),
                        );
                        let dst = out.add((ib32 * 4 + l) * 8);
                        vst1q_f32(dst, vreinterpretq_f32_u32(flo));
                        vst1q_f32(dst.add(4), vreinterpretq_f32_u32(fhi));
                    }
                }
            }
        }
    }

    /// The two scales packed into one IQ2_S scale byte, in the scalar product order.
    fn iq2_s_db(d: f32, scale: u8) -> [f32; 2] {
        [
            d * (0.5f32 + (scale & 0xf) as f32) * 0.25,
            d * (0.5f32 + (scale >> 4) as f32) * 0.25,
        ]
    }

    pub(super) fn dequantize_iq2_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
        let masks = sign_byte_masks();
        let out = y.as_mut_ptr();
        for (i, block) in blocks.as_chunks::<82>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..34];
            let signs = &block[34..66];
            let qh = &block[66..74];
            let scales = &block[74..82];
            for ib32 in 0..8 {
                let db = iq2_s_db(d, scales[ib32]);
                for l in 0..4 {
                    let index =
                        qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let sign = signs[4 * ib32 + l];
                    unsafe {
                        let (lo, hi) = scale_signed8(
                            IQ2S_GRID.as_ptr().add(index) as *const u8,
                            vdupq_n_f32(db[l / 2]),
                            masks[sign as usize].as_ptr(),
                        );
                        let dst = out.add(i * 256 + (ib32 * 4 + l) * 8);
                        vst1q_f32(dst, lo);
                        vst1q_f32(dst.add(4), hi);
                    }
                }
            }
        }
    }

    pub(super) fn iq2_s_row_dot(blocks: &[u8], n_blocks: usize, x: &[f32]) -> f32 {
        let masks = sign_byte_masks();
        let mut acc = Acc::new();
        for (b, block) in blocks.as_chunks::<82>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..34];
            let signs = &block[34..66];
            let qh = &block[66..74];
            let scales = &block[74..82];
            for ib32 in 0..8 {
                let db = iq2_s_db(d, scales[ib32]);
                for l in 0..4 {
                    let index =
                        qs[4 * ib32 + l] as usize | (((qh[ib32] as usize) << (8 - 2 * l)) & 0x300);
                    let sign = signs[4 * ib32 + l];
                    let at = b * 256 + (ib32 * 4 + l) * 8;
                    unsafe {
                        let (lo, hi) = scale_signed8(
                            IQ2S_GRID.as_ptr().add(index) as *const u8,
                            vdupq_n_f32(db[l / 2]),
                            masks[sign as usize].as_ptr(),
                        );
                        acc.fma8(lo, hi, x.as_ptr().add(at));
                    }
                }
            }
        }
        acc.finish()
    }

    pub(super) fn dequantize_iq3_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
        let masks = sign_masks();
        let out = y.as_mut_ptr();
        for (i, block) in blocks.as_chunks::<98>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..98];
            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs[64 + 4 * ib32],
                    qs[64 + 4 * ib32 + 1],
                    qs[64 + 4 * ib32 + 2],
                    qs[64 + 4 * ib32 + 3],
                ]);
                let db = d * (0.5f32 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let code = ((aux32 >> (7 * l)) & 127) as usize;
                    let grid1 = qs[8 * ib32 + 2 * l] as usize;
                    let grid2 = qs[8 * ib32 + 2 * l + 1] as usize;
                    unsafe {
                        let db = vdupq_n_f32(db);
                        let m = masks[code].as_ptr();
                        let lo = scale_signed4(IQ3XXS_GRID.as_ptr().add(grid1) as *const u8, db, m);
                        let hi = scale_signed4(
                            IQ3XXS_GRID.as_ptr().add(grid2) as *const u8,
                            db,
                            m.add(4),
                        );
                        let dst = out.add(i * 256 + (ib32 * 4 + l) * 8);
                        vst1q_f32(dst, lo);
                        vst1q_f32(dst.add(4), hi);
                    }
                }
            }
        }
    }

    pub(super) fn iq3_xxs_row_dot(blocks: &[u8], n_blocks: usize, x: &[f32]) -> f32 {
        let masks = sign_masks();
        let mut acc = Acc::new();
        for (b, block) in blocks.as_chunks::<98>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..98];
            for ib32 in 0..8 {
                let aux32 = u32::from_le_bytes([
                    qs[64 + 4 * ib32],
                    qs[64 + 4 * ib32 + 1],
                    qs[64 + 4 * ib32 + 2],
                    qs[64 + 4 * ib32 + 3],
                ]);
                let db = d * (0.5f32 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let code = ((aux32 >> (7 * l)) & 127) as usize;
                    let grid1 = qs[8 * ib32 + 2 * l] as usize;
                    let grid2 = qs[8 * ib32 + 2 * l + 1] as usize;
                    let at = b * 256 + (ib32 * 4 + l) * 8;
                    unsafe {
                        let db = vdupq_n_f32(db);
                        let m = masks[code].as_ptr();
                        let lo = scale_signed4(IQ3XXS_GRID.as_ptr().add(grid1) as *const u8, db, m);
                        let hi = scale_signed4(
                            IQ3XXS_GRID.as_ptr().add(grid2) as *const u8,
                            db,
                            m.add(4),
                        );
                        acc.fma8(lo, hi, x.as_ptr().add(at));
                    }
                }
            }
        }
        acc.finish()
    }

    pub(super) fn dequantize_iq1_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
        let out = y.as_mut_ptr();
        for (i, block) in blocks.as_chunks::<50>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..34];
            let qh = &block[34..50];
            for ib in 0..8 {
                let qh_ib = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
                let dl = d * (2 * ((qh_ib >> 12) & 7) + 1) as f32;
                let delta = if qh_ib & 0x8000 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };
                for l in 0..4 {
                    let index =
                        qs[4 * ib + l] as usize | ((((qh_ib >> (3 * l)) & 7) as usize) << 8);
                    unsafe {
                        let (lo, hi) = scale_i8_delta(
                            IQ1S_GRID.as_ptr().add(index) as *const u8,
                            vdupq_n_f32(dl),
                            delta,
                        );
                        let dst = out.add(i * 256 + (ib * 4 + l) * 8);
                        vst1q_f32(dst, lo);
                        vst1q_f32(dst.add(4), hi);
                    }
                }
            }
        }
    }

    pub(super) fn iq1_s_row_dot(blocks: &[u8], n_blocks: usize, x: &[f32]) -> f32 {
        let mut acc = Acc::new();
        for (b, block) in blocks.as_chunks::<50>().0.iter().take(n_blocks).enumerate() {
            let d = f16_scale(block);
            let qs = &block[2..34];
            let qh = &block[34..50];
            for ib in 0..8 {
                let qh_ib = u16::from_le_bytes([qh[2 * ib], qh[2 * ib + 1]]);
                let dl = d * (2 * ((qh_ib >> 12) & 7) + 1) as f32;
                let delta = if qh_ib & 0x8000 != 0 {
                    -IQ1S_DELTA
                } else {
                    IQ1S_DELTA
                };
                for l in 0..4 {
                    let index =
                        qs[4 * ib + l] as usize | ((((qh_ib >> (3 * l)) & 7) as usize) << 8);
                    let at = b * 256 + (ib * 4 + l) * 8;
                    unsafe {
                        let (lo, hi) = scale_i8_delta(
                            IQ1S_GRID.as_ptr().add(index) as *const u8,
                            vdupq_n_f32(dl),
                            delta,
                        );
                        acc.fma8(lo, hi, x.as_ptr().add(at));
                    }
                }
            }
        }
        acc.finish()
    }

    /// One group's scale: 4 low bits plus 2 high bits, then the same `- 32` bias as the scalar loop.
    fn iq4_xs_dl(d: f32, scales_l: &[u8], scales_h: u16, ib: usize) -> f32 {
        let ls = (((scales_l[ib / 2] >> (4 * (ib % 2))) & 0xf) as i32)
            | (((scales_h >> (2 * ib)) & 3) as i32) << 4;
        d * (ls - 32) as f32
    }

    /// Low nibble is the first 16 values of the group and high nibble the next 16.
    unsafe fn fma_nibbles(
        acc: &mut Acc,
        bytes: *const u8,
        table: int8x16_t,
        dl: float32x4_t,
        x: *const f32,
        high: bool,
    ) {
        let q = vld1q_u8(bytes);
        let idx = if high {
            vshrq_n_u8(q, 4)
        } else {
            vandq_u8(q, vdupq_n_u8(0x0f))
        };
        let g = vqtbl1q_s8(table, idx);
        let lo16 = vmovl_s8(vget_low_s8(g));
        let hi16 = vmovl_s8(vget_high_s8(g));
        let a = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo16))), dl);
        let b = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(lo16))), dl);
        let c = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi16))), dl);
        let d = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(hi16))), dl);
        acc.fma8(a, b, x);
        acc.fma8(c, d, x.add(8));
    }

    pub(super) fn iq4_xs_row_dot(blocks: &[u8], n_blocks: usize, x: &[f32]) -> f32 {
        // The table is 16 i8s and is not 16-byte aligned, so the load is unaligned.
        let table =
            unsafe { core::ptr::read_unaligned(KVALUES_IQ4NL.as_ptr() as *const int8x16_t) };
        let mut acc = Acc::new();
        for (b, block) in blocks
            .as_chunks::<136>()
            .0
            .iter()
            .take(n_blocks)
            .enumerate()
        {
            let d = f16_scale(block);
            let scales_h = u16::from_le_bytes([block[2], block[3]]);
            let scales_l = &block[4..8];
            let qs = &block[8..136];
            for ib in 0..8 {
                let scale = iq4_xs_dl(d, scales_l, scales_h, ib);
                let q = qs[16 * ib..].as_ptr();
                let at = b * 256 + ib * 32;
                unsafe {
                    let dl = vdupq_n_f32(scale);
                    fma_nibbles(&mut acc, q, table, dl, x.as_ptr().add(at), false);
                    fma_nibbles(&mut acc, q, table, dl, x.as_ptr().add(at + 16), true);
                }
            }
        }
        acc.finish()
    }
}

/// aarch64 takes the vector decoder; everywhere else the scalar oracle.
macro_rules! dequant_arm {
    ($vector:path, $scalar:path, $blocks:expr, $n:expr, $y:expr) => {{
        #[cfg(target_arch = "aarch64")]
        {
            $vector($blocks, $n, $y);
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            $scalar($blocks, $n, $y);
        }
    }};
}

/// Dequantise `n_blocks` IQ blocks of `dtype` into the first `n_blocks * block_size` slots.
pub(crate) fn dequantize_blocks(
    dtype: GgufDType,
    blocks: &[u8],
    n_blocks: usize,
    y: &mut [f32],
) -> Result<()> {
    match dtype.raw() {
        16 => dequant_arm!(
            neon::dequantize_iq2_xxs,
            dequantize_iq2_xxs,
            blocks,
            n_blocks,
            y
        ),
        17 => dequantize_iq2_xs(blocks, n_blocks, y),
        18 => dequant_arm!(
            neon::dequantize_iq3_xxs,
            dequantize_iq3_xxs,
            blocks,
            n_blocks,
            y
        ),
        19 => dequant_arm!(
            neon::dequantize_iq1_s,
            dequantize_iq1_s,
            blocks,
            n_blocks,
            y
        ),
        20 => dequantize_iq4_nl(blocks, n_blocks, y),
        21 => dequantize_iq3_s(blocks, n_blocks, y),
        22 => dequant_arm!(
            neon::dequantize_iq2_s,
            dequantize_iq2_s,
            blocks,
            n_blocks,
            y
        ),
        23 => dequantize_iq4_xs(blocks, n_blocks, y),
        29 => dequantize_iq1_m(blocks, n_blocks, y),
        raw => candle_core::bail!("GGUF dtype {raw} is not an IQ type"),
    }
    Ok(())
}

/// Whether `raw` is one of the IQ types this module decodes.
pub(crate) const fn is_iq(raw: u32) -> bool {
    matches!(raw, 16..=23 | 29)
}

/// `row . x` where the row is `n_blocks` IQ blocks: the vector path where one exists, else
/// the row decoded into `scratch` and dotted. The two do not share a summation order, which is
/// why this serves the opt-in fused gather and not the default path.
fn row_dot(dtype: GgufDType, row_bytes: &[u8], x: &[f32], scratch: &mut [f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    match dtype.raw() {
        16 => return neon::iq2_xxs_row_dot(row_bytes, x.len() / 256, x),
        18 => return neon::iq3_xxs_row_dot(row_bytes, x.len() / 256, x),
        19 => return neon::iq1_s_row_dot(row_bytes, x.len() / 256, x),
        22 => return neon::iq2_s_row_dot(row_bytes, x.len() / 256, x),
        23 => return neon::iq4_xs_row_dot(row_bytes, x.len() / 256, x),
        _ => {}
    }
    let block_size = dtype.block_size().expect("iq block size");
    dequantize_blocks(dtype, row_bytes, x.len() / block_size, scratch)
        .expect("a row is a whole number of blocks of its own dtype");
    #[cfg(target_arch = "aarch64")]
    return neon::dot_f32(scratch, x);
    #[cfg(not(target_arch = "aarch64"))]
    scratch.iter().zip(x).map(|(w, a)| w * a).sum()
}

/// Blocks per parallel chunk: 32 chunks for a 1M-value expert slab, which is what an expert
/// projection is, so the decode fills a node rather than one thread.
const PAR_BLOCK_CHUNK: usize = 128;

/// Rows per chunk in the fused gather: a 512-row expert splits into 64 chunks, and the chunk
/// is small enough that its scratch row stays in L1.
const FUSED_ROWS_PER_CHUNK: usize = 8;

/// The fused gather decodes each block straight into the dot product instead of materialising
/// the f32 slab. It is on by default: the accumulation order differs from the two-step path's
/// last bit, both are f32 sums of the same products, and 400 greedy tokens of Flint came out
/// identical whichever arm ran. `MISTRALRS_IQ_FUSED_GEMV=0` turns it off.
static FUSED_IQ_GEMV: AtomicBool = AtomicBool::new(true);
static FUSED_IQ_GEMV_SEEDED: OnceLock<()> = OnceLock::new();

/// Three projections over 49 layers: the gathers one token of Flint makes, so the stat below
/// can report a per-token figure.
const GATHERS_PER_TOKEN: usize = 147;

static IQ_GATHER_STATS: OnceLock<bool> = OnceLock::new();
static GATHER_THREADS: OnceLock<
    std::sync::Mutex<std::collections::HashSet<std::thread::ThreadId>>,
> = OnceLock::new();
static GATHER_CALLS: AtomicUsize = AtomicUsize::new(0);
static GATHER_NANOS: AtomicU64 = AtomicU64::new(0);

/// `MISTRALRS_IQ_GATHER_STATS=1` prints the gather's time per call, which is how the model's
/// share was established: 3.44 ms a call against a 714 ms token is 71% of it.
fn iq_gather_stats() -> bool {
    *IQ_GATHER_STATS.get_or_init(|| std::env::var_os("MISTRALRS_IQ_GATHER_STATS").is_some())
}

fn fused_iq_gemv_enabled() -> bool {
    FUSED_IQ_GEMV_SEEDED.get_or_init(|| {
        let on = std::env::var("MISTRALRS_IQ_FUSED_GEMV")
            .map(|v| !matches!(v.trim(), "0" | "false" | "no" | "off"))
            .unwrap_or(true);
        FUSED_IQ_GEMV.store(on, Ordering::Relaxed);
    });
    FUSED_IQ_GEMV.load(Ordering::Relaxed)
}

/// The same decode, split over block ranges. A block reads only its own bytes and writes only
/// its own outputs, so how the range is cut cannot change a value - it changes where it is
/// computed, not what.
pub(crate) fn dequantize_blocks_par(
    dtype: GgufDType,
    bytes: &[u8],
    n_blocks: usize,
    y: &mut [f32],
) {
    let (block_size, type_size) = (
        dtype.block_size().expect("iq block size"),
        dtype.type_size().expect("iq type size"),
    );
    assert!(is_iq(dtype.raw()), "not an IQ dtype: {}", dtype.name());
    if n_blocks == 0 {
        return;
    }
    let bytes = &bytes[..n_blocks * type_size];
    crate::gemv_cpu::par_chunks(
        &mut y[..n_blocks * block_size],
        PAR_BLOCK_CHUNK * block_size,
        |chunk, out| {
            let n = out.len() / block_size;
            let start = chunk * PAR_BLOCK_CHUNK * type_size;
            dequantize_blocks(dtype, &bytes[start..start + n * type_size], n, out)
                .expect("the dtype was checked when the tensor was built");
        },
    );
}

/// Where a tensor's block bytes live: either a buffer this crate owns, or a slice of a mapped
/// GGUF shard. The mapped form holds the archive, so the mapping outlives the view and the
/// tensor never leaves the page cache.
#[derive(Debug)]
enum IqStorage {
    Owned(Box<[u8]>),
    Mapped {
        archive: Arc<GgufArchive>,
        shard: usize,
        range: Range<usize>,
    },
}

impl IqStorage {
    fn bytes(&self) -> &[u8] {
        match self {
            Self::Owned(bytes) => bytes,
            Self::Mapped {
                archive,
                shard,
                range,
            } => &archive.shard_bytes(*shard).expect("mapped GGUF shard")[range.clone()],
        }
    }
}

/// An IQ-quantised tensor: raw block bytes plus the shape they decode to.
#[derive(Debug)]
pub struct IqTensor {
    storage: IqStorage,
    dims: Vec<usize>,
    dtype: GgufDType,
}
impl IqTensor {
    pub fn new(bytes: Box<[u8]>, dims: Vec<usize>, dtype: GgufDType) -> Result<Self> {
        Self::check_bytes(bytes.len(), &dims, dtype)?;
        Ok(Self {
            storage: IqStorage::Owned(bytes),
            dims,
            dtype,
        })
    }

    /// A tensor whose blocks stay in the GGUF mapping, which is what the loader uses: the
    /// checkpoint's bytes are the weights, and copying them would turn page cache into
    /// anonymous memory.
    pub fn from_mapped(
        archive: Arc<GgufArchive>,
        shard: usize,
        range: Range<usize>,
        dims: Vec<usize>,
        dtype: GgufDType,
    ) -> Result<Self> {
        let mapped = archive
            .shard_bytes(shard)
            .ok_or_else(|| candle_core::Error::msg(format!("GGUF shard {shard} is not mapped")))?;
        let bytes = mapped.get(range.clone()).ok_or_else(|| {
            candle_core::Error::msg(format!("GGUF tensor range {range:?} is outside the shard"))
        })?;
        Self::check_bytes(bytes.len(), &dims, dtype)?;
        Ok(Self {
            storage: IqStorage::Mapped {
                archive,
                shard,
                range,
            },
            dims,
            dtype,
        })
    }

    fn check_bytes(len: usize, dims: &[usize], dtype: GgufDType) -> Result<()> {
        let (Some(block_size), Some(type_size)) = (dtype.block_size(), dtype.type_size()) else {
            candle_core::bail!("GGUF dtype {} has no block geometry", dtype.name());
        };
        if !is_iq(dtype.raw()) {
            candle_core::bail!("GGUF dtype {} is not an IQ type", dtype.name());
        }
        let elem_count: usize = dims.iter().product();
        let n_blocks = elem_count / block_size;
        if len != n_blocks * type_size {
            candle_core::bail!(
                "GGUF {} tensor needs {} bytes for {elem_count} elements, got {len}",
                dtype.name(),
                n_blocks * type_size,
            );
        }
        Ok(())
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// Where the bytes sit, so a test can assert a mapped tensor really is a view.
    #[cfg(test)]
    pub(crate) fn byte_ptr(&self) -> *const u8 {
        self.storage.bytes().as_ptr()
    }

    /// The whole tensor as f32.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let type_size = self.dtype.type_size().expect("iq type size");
        let block_size = self.dtype.block_size().expect("iq block size");
        let bytes = self.storage.bytes();
        let n_blocks = bytes.len() / type_size;
        let mut out = vec![0f32; n_blocks * block_size];
        dequantize_blocks_par(self.dtype, bytes, n_blocks, &mut out);
        Tensor::from_vec(out, self.dims.clone(), device)
    }

    /// The f32 slabs of the named experts of a `[n_experts, out, in]` stack, in ascending
    /// expert order, plus the same ids remapped onto that order.
    ///
    /// This is the IQ counterpart of dequantising a candle `QTensor` per named expert: the
    /// stack stays blocks, and only the experts a token routes to are decoded.
    pub fn dequantize_experts(
        &self,
        ids: &[u32],
        device: &Device,
    ) -> Result<(Vec<Tensor>, Vec<u32>)> {
        let n_experts = self.expert_geometry()?.0;
        if let Some(&bad) = ids.iter().find(|&&i| i as usize >= n_experts) {
            candle_core::bail!("IqTensor: expert id {bad} is out of range for {n_experts} experts");
        }
        let mut wanted: Vec<usize> = ids.iter().map(|&i| i as usize).collect();
        wanted.sort_unstable();
        wanted.dedup();
        if wanted.is_empty() {
            candle_core::bail!("IqTensor: no expert was routed to");
        }

        let bytes = self.storage.bytes();
        let total = bytes.len();
        if !total.is_multiple_of(n_experts) {
            candle_core::bail!("IqTensor: {total} bytes do not divide across {n_experts} experts");
        }
        let stride = total / n_experts;
        let type_size = self.dtype.type_size().expect("iq type size");
        let block_size = self.dtype.block_size().expect("iq block size");
        let (_, out_features, in_features) = self.expert_geometry()?;

        let mut slabs = Vec::with_capacity(wanted.len());
        for &e in &wanted {
            let lo = e * stride;
            let Some(slab) = bytes.get(lo..lo + stride) else {
                candle_core::bail!("IqTensor: expert {e} is out of range");
            };
            let n_blocks = slab.len() / type_size;
            let mut out = vec![0f32; n_blocks * block_size];
            dequantize_blocks_par(self.dtype, slab, n_blocks, &mut out);
            slabs.push(Tensor::from_vec(out, (out_features, in_features), device)?);
        }

        let mut dense = vec![u32::MAX; n_experts];
        for (row, &e) in wanted.iter().enumerate() {
            dense[e] = row as u32;
        }
        let remapped = ids.iter().map(|&i| dense[i as usize]).collect();
        Ok((slabs, remapped))
    }

    /// The `[n_experts, out_features, in_features]` stack geometry.
    pub fn expert_geometry(&self) -> Result<(usize, usize, usize)> {
        match self.dims.as_slice() {
            [n, out, k] => Ok((*n, *out, *k)),
            other => {
                candle_core::bail!("IqTensor: expected an [experts, out, in] stack, got {other:?}")
            }
        }
    }

    pub(crate) fn dtype(&self) -> GgufDType {
        self.dtype
    }

    pub(crate) fn block_geometry(&self) -> (usize, usize) {
        (
            self.dtype.block_size().expect("iq block size"),
            self.dtype.type_size().expect("iq type size"),
        )
    }

    /// One expert's slice of the stack: its rows are whole rows of blocks, so a row is a
    /// contiguous byte range and the fused gather can read it without the whole slab.
    pub(crate) fn expert_bytes(&self, expert: usize) -> Result<&[u8]> {
        let n_experts = self.expert_geometry()?.0;
        let bytes = self.storage.bytes();
        let stride = bytes.len() / n_experts;
        let start = expert * stride;
        bytes.get(start..start + stride).ok_or_else(|| {
            candle_core::Error::Msg(format!(
                "IqTensor: expert {expert} is out of range for {n_experts} experts"
            ))
        })
    }
}

/// A GGUF linear whose weights are IQ blocks.
///
/// A dense weight dequantises once, on the first forward, and stays cached: every token reads
/// it, so the block decode is paid once. A stacked expert `[n_experts, out, in]` tensor is
/// never materialised whole - the gather decodes only the experts a token routes to, which is
/// what keeps a 10 GB IQ checkpoint from becoming 79 GB of f32.
#[derive(Debug)]
pub struct IqGgufMatMul {
    w: Arc<IqTensor>,
    b: Option<Tensor>,
    dense: OnceLock<Tensor>,
}

impl IqGgufMatMul {
    pub fn new(w: Arc<IqTensor>, b: Option<Tensor>) -> Result<Self> {
        Ok(Self {
            w,
            b,
            dense: OnceLock::new(),
        })
    }

    fn matmul_f32(&self, a: &Tensor) -> Result<Tensor> {
        let device = a.device();
        // Decode reads the blocks. The f32 cache is four bytes a value, reread every token.
        if self.w.dims().len() == 2 && fused_iq_gemv_enabled() {
            if let Some(y) = self.dense_row_gemv(a)? {
                return Ok(y);
            }
        }
        // The same cache the fp8 dense path keeps: dequantise once, then read f32 forever.
        // A stacked expert weight is handled by the gather instead and is never cached.
        let cached = if self.w.dims().len() == 2 {
            match self.dense.get() {
                Some(cached) => cached.clone(),
                None => {
                    let dequantized = self.w.dequantize(device)?;
                    // A lost race just means the other copy is used.
                    let _ = self.dense.set(dequantized.clone());
                    dequantized
                }
            }
        } else {
            self.w.dequantize(device)?
        };
        // One row is a GEMV, and the dense f32 matmul behind it runs on one thread. Dequantised
        // once a tensor, these weights are read every token, so the row split is what pays.
        if crate::gemv_cpu::should_use_gemv(a, &cached) {
            return crate::gemv_cpu::gemv(a, &cached);
        }
        let w = match *a.dims() {
            [b1, b2, _, _] => cached.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => cached.broadcast_left(bsize)?.t()?,
            _ => cached.t()?,
        };
        a.matmul(&w)
    }

    /// One activation row against a rank-2 IQ weight. More than one row keeps the f32 cache.
    fn dense_row_gemv(&self, a: &Tensor) -> Result<Option<Tensor>> {
        if !a.device().is_cpu() || a.rank() < 2 {
            return Ok(None);
        }
        let dims = a.dims().to_vec();
        let k_in = dims[dims.len() - 1];
        let rows: usize = dims[..dims.len() - 1].iter().product();
        if rows != 1 {
            return Ok(None);
        }
        let &[n, k] = self.w.dims() else {
            return Ok(None);
        };
        if k_in != k {
            return Ok(None);
        }
        let (block_size, type_size) = self.w.block_geometry();
        // A row has to be a whole number of blocks, or the bytes of row r are not row r.
        if k == 0 || !k.is_multiple_of(block_size) {
            return Ok(None);
        }
        let row_bytes = (k / block_size) * type_size;
        let bytes = self.w.storage.bytes();
        if bytes.len() != n * row_bytes {
            return Ok(None);
        }
        let x = a.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        let dtype = self.w.dtype();
        let threads = crate::cpu_threads::resolve().max(1);
        let chunk = n.div_ceil(threads).max(1);
        let mut out = vec![0f32; n];
        crate::gemv_cpu::par_chunks(&mut out, chunk, |chunk_idx, dst| {
            let mut scratch = vec![0f32; k];
            let first = chunk_idx * chunk;
            for (i, slot) in dst.iter_mut().enumerate() {
                let row = first + i;
                let start = row * row_bytes;
                *slot = row_dot(dtype, &bytes[start..start + row_bytes], &x, &mut scratch);
            }
        });
        let last = dims.len() - 1;
        let mut shape = dims;
        shape[last] = n;
        Ok(Some(Tensor::from_vec(out, shape, a.device())?))
    }

    fn add_bias(&self, x: Tensor) -> Result<Tensor> {
        match &self.b {
            Some(b) => x.broadcast_add(b),
            None => Ok(x),
        }
    }
}

impl crate::QuantizedSerde for IqGgufMatMul {
    fn isq_serde_supported(&self) -> bool {
        false
    }

    fn name(&self) -> &'static str {
        "gguf-iq"
    }
}

impl crate::QuantMethod for IqGgufMatMul {
    fn new(_method: crate::QuantMethodConfig) -> Result<Self> {
        candle_core::bail!("IqGgufMatMul is constructed directly, not from a QuantMethodConfig")
    }

    fn dequantize_w(&self) -> Result<Tensor> {
        self.w.dequantize(&Device::Cpu)
    }

    /// The dense path: f32 matmul over the dequantised weight, which the first forward
    /// materialises and caches.
    fn forward_raw(&self, a: &Tensor) -> Result<Tensor> {
        let original = a.dtype();
        let a_f32 = if original == DType::F32 {
            a.clone()
        } else {
            a.to_dtype(DType::F32)?
        };
        let x = self.matmul_f32(&a_f32)?;
        let x = if original == DType::F32 {
            x
        } else {
            x.to_dtype(original)?
        };
        self.add_bias(x)
    }

    /// The expert path. Shapes are the ones the stacked CPU path hands a quantised expert
    /// projection: `(batch, seq, 1, 1, hidden)` with `(batch, seq, top_k)` indices for the
    /// gate and up projections, `(batch, seq, top_k, hidden)` for the down projection, and
    /// the flat `(tokens, 1|top_k, hidden)` / `(tokens, top_k)` pair.
    fn gather_forward_raw(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let (_, out_features, in_features) = self.w.expert_geometry()?;
        let gathered = match a.dims() {
            &[b, s, 1, 1, h] => {
                let (_ib, _is, k) = indices.dims3()?;
                let out =
                    self.gather_flat(&a.reshape((b * s, h))?, &indices.reshape((b * s, k))?, k)?;
                out.reshape((b, s, k, out_features))
            }
            &[b, s, k, h] => {
                let (_ib, _is, ik) = indices.dims3()?;
                if ik != k {
                    candle_core::bail!("IqGgufMatMul: input has {k} slots but indices have {ik}");
                }
                if h != in_features {
                    candle_core::bail!(
                        "IqGgufMatMul: input has {h} features, weights expect {in_features}"
                    );
                }
                let out = self.gather_flat(
                    &a.reshape((b * s * k, h))?,
                    &indices.reshape((b * s * k, 1))?,
                    crate::moe_gather_cpu::NO_REPEAT,
                )?;
                out.reshape((b, s, k, out_features))
            }
            &[t, xt, h] => {
                let (ib, ik) = indices.dims2()?;
                if ib != t {
                    candle_core::bail!(
                        "IqGgufMatMul: input is ({t}, {xt}, {h}) but indices are ({ib}, {ik})"
                    );
                }
                if h != in_features {
                    candle_core::bail!(
                        "IqGgufMatMul: input has {h} features, weights expect {in_features}"
                    );
                }
                // One row per token broadcasts over its top-k slots; anything else is already
                // per-slot.
                let (rows, repeat) = if xt == 1 {
                    (a.reshape((t, h))?, ik)
                } else {
                    if ik != xt {
                        candle_core::bail!(
                            "IqGgufMatMul: input has {xt} slots but indices have {ik}"
                        );
                    }
                    (a.reshape((t * ik, h))?, crate::moe_gather_cpu::NO_REPEAT)
                };
                let out = self.gather_flat(&rows, indices, repeat)?;
                out.reshape((t, ik, out_features))
            }
            dims => candle_core::bail!("IqGgufMatMul: unsupported input shape {dims:?}"),
        };

        if let Some(b) = &self.b {
            if b.rank() == 2 {
                let mut shape = indices.dims().to_vec();
                shape.push(b.dim(1)?);
                let bias = b
                    .index_select(&indices.flatten_all()?, 0)?
                    .reshape(candle_core::Shape::from(shape))?;
                gathered?.broadcast_add(&bias)
            } else {
                gathered?.broadcast_add(b)
            }
        } else {
            gathered
        }
    }

    fn quantized_act_type(&self) -> Option<DType> {
        None
    }

    fn dtype_and_device(&self) -> (DType, Device) {
        (DType::F32, Device::Cpu)
    }

    fn plan_isq(&self, _request: &crate::IsqRequest) -> Result<crate::IsqPlanParams> {
        candle_core::bail!("IqGgufMatMul does not support ISQ")
    }

    fn add_delta_w(&self, _delta: &Tensor) -> Result<std::sync::Arc<dyn crate::QuantMethod>> {
        candle_core::bail!("IqGgufMatMul does not support add_delta_w")
    }

    fn apply_isq(
        self: std::sync::Arc<Self>,
        _dtype: Option<crate::IsqType>,
        _device: Device,
        _n_quantized: &std::sync::atomic::AtomicUsize,
        _imatrix_weight: Option<Vec<f32>>,
        _guard: crate::QuantizeOntoGuard,
    ) -> Result<std::sync::Arc<dyn crate::QuantMethod>> {
        candle_core::bail!("IqGgufMatMul does not support ISQ")
    }
}

impl IqGgufMatMul {
    /// `(rows, hidden) @ gather(experts, ids).T`, one output row per slot.
    ///
    /// `repeat` is how many slots share one activation row: a token's top-k slots read the
    /// token's row, so the rows are passed once and the kernel addresses them by
    /// `slot / repeat`.
    fn gather_flat(&self, rows: &Tensor, ids: &Tensor, repeat: usize) -> Result<Tensor> {
        if !iq_gather_stats() {
            return self.gather_flat_inner(rows, ids, repeat);
        }
        let started = std::time::Instant::now();
        let out = self.gather_flat_inner(rows, ids, repeat);
        let n = GATHER_CALLS.fetch_add(1, Ordering::Relaxed) + 1;
        GATHER_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed);
        if n.is_multiple_of(GATHERS_PER_TOKEN) {
            let total = GATHER_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
            eprintln!(
                "iq gather: {n} calls, {:.2} ms each, {:.1} ms per token, {} threads",
                total / n as f64,
                total / (n / GATHERS_PER_TOKEN) as f64,
                GATHER_THREADS
                    .get()
                    .map(|m| m.lock().expect("thread set").len())
                    .unwrap_or(0)
            );
        }
        out
    }

    fn gather_flat_inner(&self, rows: &Tensor, ids: &Tensor, repeat: usize) -> Result<Tensor> {
        let (n_rows, in_features) = rows.dims2()?;
        let n_slots = ids.elem_count();
        if n_rows * repeat != n_slots {
            candle_core::bail!(
                "IqGgufMatMul: {n_slots} slots for {n_rows} rows at repeat {repeat}"
            );
        }
        let a: Vec<f32> = rows
            .to_dtype(DType::F32)?
            .contiguous()?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let ids_flat: Vec<u32> = ids
            .to_dtype(DType::U32)?
            .flatten_all()?
            .to_vec1::<u32>()?
            .to_vec();

        let (n_experts, out_features, _) = self.w.expert_geometry()?;
        let gather_shape = crate::moe_gather_cpu::should_use_gather_gemv_shape(
            n_experts,
            out_features,
            in_features,
            out_features,
            n_slots,
        );
        let out = if gather_shape && fused_iq_gemv_enabled() {
            self.fused_gather(&a, &ids_flat, repeat)?
        } else {
            let (slabs, dense) = self.w.dequantize_experts(&ids_flat, &Device::Cpu)?;
            if gather_shape {
                crate::moe_gather_cpu::gather_gemv_per_expert(
                    &a,
                    &slabs,
                    out_features,
                    in_features,
                    &dense,
                    repeat,
                )?
            } else {
                // The gather kernel refuses tiny shapes; the whole-stack path these replace would
                // fall back to a matmul, so dequantise the named slabs and matmul over those.
                let stack = Tensor::stack(&slabs, 0)?;
                let unquant = <crate::UnquantLinear as crate::QuantMethod>::new(
                    crate::QuantMethodConfig::Unquantized(candle_nn::Linear::new(stack, None)),
                )?;
                // The whole-stack gather takes the per-slot form, so every activation row is
                // expanded to the slots that read it.
                let slot_a: Vec<f32> = a
                    .chunks(in_features)
                    .flat_map(|row| std::iter::repeat_n(row, repeat).flatten().copied())
                    .collect();
                let x = Tensor::from_vec(slot_a, (1, n_slots, 1, in_features), &Device::Cpu)?;
                let slot_ids = Tensor::from_vec(dense.clone(), (1, n_slots, 1), &Device::Cpu)?;
                crate::QuantMethod::gather_forward(&unquant, &x, &slot_ids)?
                    .reshape((n_slots, out_features))?
                    .flatten_all()?
                    .to_vec1::<f32>()?
            }
        };
        Tensor::from_vec(out, (n_slots, out_features), &Device::Cpu)
    }

    /// The same gather, one row at a time and without the f32 slab: a row's bytes are decoded
    /// into a scratch row, dotted with the activation, and dropped. The traffic is the block
    /// bytes instead of the slab plus its read-back.
    fn fused_gather(&self, a: &[f32], ids: &[u32], repeat: usize) -> Result<Vec<f32>> {
        let (_, out_features, in_features) = self.w.expert_geometry()?;
        let (block_size, type_size) = self.w.block_geometry();
        if in_features % block_size != 0 {
            candle_core::bail!(
                "IqGgufMatMul: {in_features} features are not a whole number of {block_size}-value blocks"
            );
        }
        let n_blocks = in_features / block_size;
        let row_bytes = n_blocks * type_size;
        if !a.len().is_multiple_of(in_features) {
            candle_core::bail!(
                "IqGgufMatMul: {} activations are not a whole number of {in_features}-wide rows",
                a.len()
            );
        }
        let n_rows = a.len() / in_features;
        if n_rows * repeat != ids.len() {
            candle_core::bail!(
                "IqGgufMatMul: {} activations at repeat {repeat} for {} slots",
                a.len(),
                ids.len()
            );
        }

        let slabs: Vec<&[u8]> = ids
            .iter()
            .map(|&expert| self.w.expert_bytes(expert as usize))
            .collect::<Result<_>>()?;
        let mut out = vec![0f32; ids.len() * out_features];

        // One pass over every slot's rows rather than one pass a slot: the barrier is paid
        // once a gather instead of once an expert, and 147 of these run per token.
        let rows = FUSED_ROWS_PER_CHUNK;
        if !out_features.is_multiple_of(rows) {
            return self.fused_gather_per_slot(&slabs, a, repeat, out);
        }
        let chunks_per_slot = out_features / rows;
        let dtype = self.w.dtype();
        crate::gemv_cpu::par_chunks(&mut out, rows, |chunk, dst| {
            if iq_gather_stats() {
                GATHER_THREADS
                    .get_or_init(Default::default)
                    .lock()
                    .expect("thread set")
                    .insert(std::thread::current().id());
            }
            let slot = chunk / chunks_per_slot;
            let first = (chunk % chunks_per_slot) * rows;
            let slab = slabs[slot];
            let x = &a[(slot / repeat) * in_features..][..in_features];
            let mut scratch = vec![0f32; in_features];
            for (i, o) in dst.iter_mut().enumerate() {
                let row = first + i;
                *o = row_dot(
                    dtype,
                    &slab[row * row_bytes..][..row_bytes],
                    x,
                    &mut scratch,
                );
            }
        });
        Ok(out)
    }

    /// The same gather for a row count that does not fill whole chunks, where the chunk
    /// boundaries a barrier pool hands out would cross experts.
    fn fused_gather_per_slot(
        &self,
        slabs: &[&[u8]],
        a: &[f32],
        repeat: usize,
        mut out: Vec<f32>,
    ) -> Result<Vec<f32>> {
        let (_, out_features, in_features) = self.w.expert_geometry()?;
        let (block_size, type_size) = self.w.block_geometry();
        let row_bytes = (in_features / block_size) * type_size;
        let dtype = self.w.dtype();
        for (slot, slab) in slabs.iter().enumerate() {
            let x = &a[(slot / repeat) * in_features..][..in_features];
            let dst = &mut out[slot * out_features..][..out_features];
            crate::gemv_cpu::par_chunks(dst, FUSED_ROWS_PER_CHUNK, |chunk, rows| {
                let mut scratch = vec![0f32; in_features];
                for (i, o) in rows.iter_mut().enumerate() {
                    let row = chunk * FUSED_ROWS_PER_CHUNK + i;
                    *o = row_dot(
                        dtype,
                        &slab[row * row_bytes..][..row_bytes],
                        x,
                        &mut scratch,
                    );
                }
            });
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    include!("iq_tests.rs");
}
