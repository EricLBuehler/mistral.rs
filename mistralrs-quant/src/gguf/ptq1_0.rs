//! Prism ternary group-128 codec (ggml type 143): 5 trits/byte in `qs`, 4 trits/byte in `qh`, fp16 scale last.

use half::f16;

pub const PTQ1_0_GGUF_TYPE: u32 = 143;
pub const PTQ1_0_BLOCK_ELEMS: usize = 128;
pub const PTQ1_0_BLOCK_BYTES: usize = 28;
const QS_BYTES: usize = 24;
const QH_BYTES: usize = 2;
const SCALE_OFFSET: usize = QS_BYTES + QH_BYTES;
const QS_TRITS_PER_BYTE: usize = 5;
const QH_TRITS_PER_BYTE: usize = 4;
const QS_STAGES: [usize; 3] = [32, 16, 8];
const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];

#[inline(always)]
pub(super) const fn trit(byte: u8, n: usize) -> u8 {
    let q = byte.wrapping_mul(POW3[n]);
    ((q as u16 * 3) >> 8) as u8
}

/// Unpacks one block into trit codes 0..=2 (weight = code - 1), in element order.
pub fn unpack_block_trits(block: &[u8], out: &mut [u8; PTQ1_0_BLOCK_ELEMS]) {
    debug_assert_eq!(block.len(), PTQ1_0_BLOCK_BYTES);
    let mut j = 0;
    let mut o = 0;
    for chunk in QS_STAGES {
        while j + chunk <= QS_BYTES {
            for n in 0..QS_TRITS_PER_BYTE {
                for m in 0..chunk {
                    out[o] = trit(block[j + m], n);
                    o += 1;
                }
            }
            j += chunk;
        }
    }
    for n in 0..QH_TRITS_PER_BYTE {
        for h in 0..QH_BYTES {
            out[o] = trit(block[QS_BYTES + h], n);
            o += 1;
        }
    }
}

pub fn block_scale(block: &[u8]) -> f32 {
    f16::from_le_bytes([block[SCALE_OFFSET], block[SCALE_OFFSET + 1]]).to_f32()
}

/// Dequantizes whole blocks; `bytes.len()` must be `out.len() / 128 * 28`.
pub fn dequantize_row(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % PTQ1_0_BLOCK_ELEMS, 0);
    assert_eq!(
        bytes.len(),
        out.len() / PTQ1_0_BLOCK_ELEMS * PTQ1_0_BLOCK_BYTES
    );
    let mut trits = [0u8; PTQ1_0_BLOCK_ELEMS];
    for (block, dst) in bytes
        .as_chunks::<PTQ1_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<PTQ1_0_BLOCK_ELEMS>().0.iter_mut())
    {
        unpack_block_trits(block, &mut trits);
        let d = block_scale(block);
        for (y, t) in dst.iter_mut().zip(trits) {
            *y = (t as i32 - 1) as f32 * d;
        }
    }
}

#[cfg(test)]
pub(crate) fn encode_block(
    codes: &[u8; PTQ1_0_BLOCK_ELEMS],
    scale: f32,
) -> [u8; PTQ1_0_BLOCK_BYTES] {
    let ceil_pack = |q: u8| (q as u16 * 256).div_ceil(243) as u8;
    let mut block = [0u8; PTQ1_0_BLOCK_BYTES];
    let mut j = 0;
    let mut src = 0;
    for chunk in QS_STAGES {
        while j + chunk <= QS_BYTES {
            for m in 0..chunk {
                let mut q = 0u8;
                for n in 0..QS_TRITS_PER_BYTE {
                    q = q * 3 + codes[src + m + n * chunk];
                }
                block[j + m] = ceil_pack(q);
            }
            src += QS_TRITS_PER_BYTE * chunk;
            j += chunk;
        }
    }
    for h in 0..QH_BYTES {
        let mut q = 0u8;
        for m in 0..QH_TRITS_PER_BYTE {
            q = q * 3 + codes[src + h + m * QH_BYTES];
        }
        block[QS_BYTES + h] = ceil_pack(q * 3);
    }
    block[SCALE_OFFSET..].copy_from_slice(&f16::from_f32(scale).to_le_bytes());
    block
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(seed: usize) -> [u8; PTQ1_0_BLOCK_ELEMS] {
        let mut codes = [0u8; PTQ1_0_BLOCK_ELEMS];
        for (i, c) in codes.iter_mut().enumerate() {
            *c = ((i * 7 + i / 5 + seed * 13) % 3) as u8;
        }
        codes
    }

    #[test]
    fn block_layout_sizes() {
        assert_eq!(
            QS_BYTES * QS_TRITS_PER_BYTE + QH_BYTES * QH_TRITS_PER_BYTE,
            128
        );
        assert_eq!(SCALE_OFFSET + 2, PTQ1_0_BLOCK_BYTES);
    }

    #[test]
    fn roundtrips_every_element_position() {
        for seed in 0..16 {
            let codes = pattern(seed);
            let block = encode_block(&codes, 0.5);
            let mut out = [0u8; PTQ1_0_BLOCK_ELEMS];
            unpack_block_trits(&block, &mut out);
            assert_eq!(out, codes, "seed {seed}");
        }
    }

    #[test]
    fn single_nonzero_lands_at_its_own_index() {
        for idx in 0..PTQ1_0_BLOCK_ELEMS {
            let mut codes = [1u8; PTQ1_0_BLOCK_ELEMS];
            codes[idx] = 2;
            let block = encode_block(&codes, 1.0);
            let mut y = [0f32; PTQ1_0_BLOCK_ELEMS];
            dequantize_row(&block, &mut y);
            for (i, v) in y.iter().enumerate() {
                assert_eq!(*v, if i == idx { 1.0 } else { 0.0 }, "idx {idx} at {i}");
            }
        }
    }

    #[test]
    fn dequantizes_with_group_scale_across_blocks() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&encode_block(&[2; PTQ1_0_BLOCK_ELEMS], 0.25));
        bytes.extend_from_slice(&encode_block(&[0; PTQ1_0_BLOCK_ELEMS], 2.0));
        let mut y = vec![0f32; 2 * PTQ1_0_BLOCK_ELEMS];
        dequantize_row(&bytes, &mut y);
        assert!(y[..128].iter().all(|v| *v == 0.25));
        assert!(y[128..].iter().all(|v| *v == -2.0));
    }

    #[test]
    fn any_byte_decodes_to_a_valid_trit() {
        for byte in 0..=255u8 {
            for n in 0..QS_TRITS_PER_BYTE {
                assert!(trit(byte, n) <= 2);
            }
        }
    }
}
