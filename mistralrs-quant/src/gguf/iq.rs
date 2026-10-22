//! The IQ family: block layouts and dequantisation, ported from ggml's `ggml-quants.c`.
//!
//! Candle has no IQ block types, so these live here: the GGUF reader hands over the raw
//! block bytes, and this module turns them into f32. The arithmetic follows the C exactly -
//! same operation order, f32 throughout - so a block dequantises bit for bit the same.

use candle_core::{DType, Device, Result, Tensor};
use std::{
    ops::Range,
    sync::{Arc, OnceLock},
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
        dequantize_blocks(self.dtype, bytes, n_blocks, &mut out)?;
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
            dequantize_blocks(self.dtype, slab, n_blocks, &mut out)?;
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
        let w = match *a.dims() {
            [b1, b2, _, _] => cached.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => cached.broadcast_left(bsize)?.t()?,
            _ => cached.t()?,
        };
        a.matmul(&w)
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
        let (slabs, dense) = self.w.dequantize_experts(&ids_flat, &Device::Cpu)?;
        let out = if crate::moe_gather_cpu::should_use_gather_gemv_shape(
            n_experts,
            out_features,
            in_features,
            out_features,
            n_slots,
        ) {
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
        };
        Tensor::from_vec(out, (n_slots, out_features), &Device::Cpu)
    }
}

#[cfg(test)]
mod tests {
    include!("iq_tests.rs");
}
