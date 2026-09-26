//! Prism 2-bit group-128 codec (ggml type 142): fp16 scale first, then 4 codes per byte, code - 1 in -1..=2.

use half::f16;

pub const PQ2_0_GGUF_TYPE: u32 = 142;
pub const PQ2_0_BLOCK_ELEMS: usize = 128;
pub const PQ2_0_BLOCK_BYTES: usize = 34;
const SCALE_BYTES: usize = 2;
const CODES_PER_BYTE: usize = 4;
const CODE_BITS: usize = 2;
const CODE_MASK: u8 = 0x03;

/// Dequantizes whole blocks; `bytes.len()` must be `out.len() / 128 * 34`.
pub fn dequantize_row(bytes: &[u8], out: &mut [f32]) {
    assert_eq!(out.len() % PQ2_0_BLOCK_ELEMS, 0);
    assert_eq!(
        bytes.len(),
        out.len() / PQ2_0_BLOCK_ELEMS * PQ2_0_BLOCK_BYTES
    );
    for (block, dst) in bytes
        .as_chunks::<PQ2_0_BLOCK_BYTES>()
        .0
        .iter()
        .zip(out.as_chunks_mut::<PQ2_0_BLOCK_ELEMS>().0.iter_mut())
    {
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let qs = &block[SCALE_BYTES..];
        for (j, y) in dst.iter_mut().enumerate() {
            let code = (qs[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * CODE_BITS)) & CODE_MASK;
            *y = (code as i32 - 1) as f32 * d;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_codes_in_bit_order() {
        let mut block = [0u8; PQ2_0_BLOCK_BYTES];
        block[..2].copy_from_slice(&f16::from_f32(0.5).to_le_bytes());
        block[2] = 0b11_10_01_00;
        let mut y = [0f32; PQ2_0_BLOCK_ELEMS];
        dequantize_row(&block, &mut y);
        assert_eq!(&y[..4], &[-0.5, 0.0, 0.5, 1.0]);
        assert!(y[4..].iter().all(|v| *v == -0.5));
    }
}
