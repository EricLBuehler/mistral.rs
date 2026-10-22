//! The IQ family: block layouts and dequantisation, ported from ggml's `ggml-quants.c`.
//!
//! Candle has no IQ block types, so these live here: the GGUF reader hands over the raw
//! block bytes, and this module turns them into f32. The arithmetic follows the C exactly -
//! same operation order, f32 throughout - so a block dequantises bit for bit the same.

use candle_core::{Device, Result, Tensor};

use super::archive::GgufDType;
use super::iq_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ2XXS_GRID, IQ3S_GRID, IQ3XXS_GRID, KMASK_IQ2XS,
    KSIGNS_IQ2XS, KVALUES_IQ4NL,
};

/// Blocks per super-block. Every IQ type except IQ4_NL quantises 256 values as one block.
pub(crate) const QK_K: usize = 256;
/// IQ4_NL's block is 32 values.
pub(crate) const QK4_NL: usize = 32;

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
fn dequantize_iq2_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(66).take(n_blocks) {
        let d = f16_scale(block);
        let qs = &block[2..];
        for ib32 in 0..8 {
            let aux32 = le_u32_pair(&qs[8 * ib32..8 * ib32 + 8]);
            let db = d * (0.5f32 + (aux32[1] >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid = grid_bytes64(&IQ2XXS_GRID, (aux32[0] >> (8 * l)) as usize & 0xff);
                let signs = KSIGNS_IQ2XS[((aux32[1] >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    *y.next().expect("iq2_xxs output length") =
                        db * grid[j] as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 2.3125 bpw. `d`, 32 u16s of 9-bit grid index plus 7-bit sign code, then 8 scale bytes.
fn dequantize_iq2_xs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(74).take(n_blocks) {
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
                for j in 0..8 {
                    *y.next().expect("iq2_xs output length") =
                        db[l / 2] * grid[j] as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 2.5625 bpw. `d`, 64 bytes: 32 of grid indices then 32 of sign bytes, then `qh` and scales.
fn dequantize_iq2_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(82).take(n_blocks) {
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
                for j in 0..8 {
                    *y.next().expect("iq2_s output length") =
                        db[l / 2] * grid[j] as f32 * if sign_bit(sign, j) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 3.0625 bpw. `d`, then 96 bytes: 64 of grid indices followed by 32 of packed scale-and-signs.
fn dequantize_iq3_xxs(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(98).take(n_blocks) {
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
                for j in 0..4 {
                    *y.next().expect("iq3_xxs output length") =
                        db * grid1[j] as f32 * if sign_bit(signs, j) { -1.0 } else { 1.0 };
                }
                for j in 0..4 {
                    *y.next().expect("iq3_xxs output length") =
                        db * grid2[j] as f32 * if sign_bit(signs, j + 4) { -1.0 } else { 1.0 };
                }
            }
        }
    }
}

/// 3.4375 bpw. `d`, 64 bytes of grid indices, 8 of high bits, 32 of signs, 4 of scales.
fn dequantize_iq3_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(110).take(n_blocks) {
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
                    for j in 0..4 {
                        *y.next().expect("iq3_s output length") =
                            *db * grid1[j] as f32 * if sign_bit(sign, j) { -1.0 } else { 1.0 };
                    }
                    for j in 0..4 {
                        *y.next().expect("iq3_s output length") =
                            *db * grid2[j] as f32 * if sign_bit(sign, j + 4) { -1.0 } else { 1.0 };
                    }
                }
                qs_off += 8;
                signs_off += 4;
            }
        }
    }
}

/// 1.5625 bpw. `d`, 32 bytes of grid indices, then 8 u16s of per-32-value high bits and scale.
fn dequantize_iq1_s(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(50).take(n_blocks) {
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
                for j in 0..8 {
                    *y.next().expect("iq1_s output length") = dl * (grid[j] as i8 as f32 + delta);
                }
            }
        }
    }
}

/// 1.75 bpw. No `d`: the scale is assembled from the top nibbles of the four scale words.
fn dequantize_iq1_m(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(56).take(n_blocks) {
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
                for j in 0..8 {
                    *y.next().expect("iq1_m output length") =
                        dl1 * (grid[j] as i8 as f32 + deltas[l]);
                }
            }
            for l in 2..4 {
                let grid = IQ1S_GRID[idx[l]].to_le_bytes();
                for j in 0..8 {
                    *y.next().expect("iq1_m output length") =
                        dl2 * (grid[j] as i8 as f32 + deltas[l]);
                }
            }
        }
    }
}

/// 4.5 bpw. `d`, then 16 bytes of nibbles into `kvalues_iq4nl`.
fn dequantize_iq4_nl(blocks: &[u8], n_blocks: usize, y: &mut [f32]) {
    let mut y = y.iter_mut();
    for block in blocks.chunks_exact(18).take(n_blocks) {
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
    for block in blocks.chunks_exact(136).take(n_blocks) {
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

/// Dequantise `n_blocks` IQ blocks of `dtype` into the first `n_blocks * block_size` slots.
pub(crate) fn dequantize_blocks(
    dtype: GgufDType,
    blocks: &[u8],
    n_blocks: usize,
    y: &mut [f32],
) -> Result<()> {
    match dtype.raw() {
        16 => dequantize_iq2_xxs(blocks, n_blocks, y),
        17 => dequantize_iq2_xs(blocks, n_blocks, y),
        18 => dequantize_iq3_xxs(blocks, n_blocks, y),
        19 => dequantize_iq1_s(blocks, n_blocks, y),
        20 => dequantize_iq4_nl(blocks, n_blocks, y),
        21 => dequantize_iq3_s(blocks, n_blocks, y),
        22 => dequantize_iq2_s(blocks, n_blocks, y),
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

/// An IQ-quantised tensor: raw block bytes plus the shape they decode to.
#[derive(Debug)]
pub struct IqTensor {
    bytes: Box<[u8]>,
    dims: Vec<usize>,
    dtype: GgufDType,
}

impl IqTensor {
    pub fn new(bytes: Box<[u8]>, dims: Vec<usize>, dtype: GgufDType) -> Result<Self> {
        let (Some(block_size), Some(type_size)) = (dtype.block_size(), dtype.type_size()) else {
            candle_core::bail!("GGUF dtype {} has no block geometry", dtype.name());
        };
        if !is_iq(dtype.raw()) {
            candle_core::bail!("GGUF dtype {} is not an IQ type", dtype.name());
        }
        let elem_count: usize = dims.iter().product();
        let n_blocks = elem_count / block_size;
        if bytes.len() != n_blocks * type_size {
            candle_core::bail!(
                "GGUF {} tensor needs {} bytes for {elem_count} elements, got {}",
                dtype.name(),
                n_blocks * type_size,
                bytes.len()
            );
        }
        Ok(Self { bytes, dims, dtype })
    }

    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    pub fn dtype(&self) -> GgufDType {
        self.dtype
    }

    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// The whole tensor as f32.
    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let type_size = self.dtype.type_size().expect("iq type size");
        let block_size = self.dtype.block_size().expect("iq block size");
        let n_blocks = self.bytes.len() / type_size;
        let mut out = vec![0f32; n_blocks * block_size];
        dequantize_blocks(self.dtype, &self.bytes, n_blocks, &mut out)?;
        Tensor::from_vec(out, self.dims.clone(), device)
    }
}

#[cfg(test)]
mod tests {
    include!("iq_tests.rs");
}
