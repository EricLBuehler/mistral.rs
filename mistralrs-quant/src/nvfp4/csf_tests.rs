use super::*;

const ASCENDING_CODES: [u8; 8] = [0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe];
const DESCENDING_CODES: [u8; 8] = [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x01];
const TEST_SCALE_COLS: usize = 16;

#[test]
fn slab_layout_preserves_all_fp8_bits_and_exception_bytes() -> Result<()> {
    let mut fixed = (0..SLAB_ROWS)
        .map(|row| (row * 16) as u8)
        .collect::<Vec<_>>();
    fixed.extend(ASCENDING_CODES.repeat(SLAB_ROWS));
    fixed.extend((0..SLAB_ROWS).rev().map(|row| (row * 16) as u8));
    fixed.extend(DESCENDING_CODES.repeat(SLAB_ROWS));
    let mut expected = (0..=u8::MAX).chain((0..=u8::MAX).rev()).collect::<Vec<_>>();
    let exceptions = [0x8000_0000, 0x0000_00ff, 0xff00_0100, 0x7f00_01ff];
    for (index, byte) in [(0, 0x80), (255, 0x00), (256, 0xff), (511, 0x7f)] {
        expected[index] = byte;
    }
    let decoded = decode(&fixed, &exceptions, TEST_SCALE_COLS)?;
    assert_eq!(
        decoded
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        expected
    );
    Ok(())
}

#[test]
fn malformed_bases_and_exception_positions_are_rejected() {
    let mut fixed = vec![0; SLAB_ROWS * (1 + TEST_SCALE_COLS / CODES_PER_BYTE)];
    fixed[0] = MAX_BASE + 1;
    assert!(decode(&fixed, &[], TEST_SCALE_COLS)
        .unwrap_err()
        .to_string()
        .contains("row base"));
    fixed[0] = 0;
    assert!(decode(&fixed, &[256], TEST_SCALE_COLS)
        .unwrap_err()
        .to_string()
        .contains("position 256"));
    for exceptions in [[1, 1], [2, 1]] {
        assert!(decode(&fixed, &exceptions, TEST_SCALE_COLS)
            .unwrap_err()
            .to_string()
            .contains("strictly increasing"));
    }
}
