use std::borrow::Cow;

use mistralrs_quant::{GgufMatMul, HqqBits, HqqLayer, IsqType};

const UQFF_VERSION: u32 = 0x0002_00;
const UQFF_TYPE_GGUF: u8 = 0;
const UQFF_TYPE_HQQ: u8 = 2;

fn gguf_uqff_with_dtype(dtype: u32) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(&UQFF_VERSION.to_le_bytes());
    data.push(UQFF_TYPE_GGUF);
    data.extend(&0u32.to_le_bytes());
    data.push(0);
    data.extend(&dtype.to_le_bytes());
    data
}

fn append_empty_u8_tensor(data: &mut Vec<u8>) {
    data.extend(&0u32.to_le_bytes());
    data.extend(&0u32.to_le_bytes());
    data.extend(&0u32.to_le_bytes());
}

fn hqq_uqff_with_bits(bits: u8) -> Vec<u8> {
    let mut data = Vec::new();
    data.extend(&UQFF_VERSION.to_le_bytes());
    data.push(UQFF_TYPE_HQQ);
    data.push(0);
    append_empty_u8_tensor(&mut data);
    append_empty_u8_tensor(&mut data);
    append_empty_u8_tensor(&mut data);
    data.extend(&0u32.to_le_bytes());
    data.push(bits);
    data
}

#[test]
fn gguf_uqff_dtype_mapping_matches_isq_types() {
    let cases = [
        (2, IsqType::Q4_0),
        (3, IsqType::Q4_1),
        (6, IsqType::Q5_0),
        (7, IsqType::Q5_1),
        (8, IsqType::Q8_0),
        (9, IsqType::Q8_1),
        (10, IsqType::Q2K),
        (11, IsqType::Q3K),
        (12, IsqType::Q4K),
        (13, IsqType::Q5K),
        (14, IsqType::Q6K),
        (15, IsqType::Q8K),
    ];

    for (dtype, expected) in cases {
        let actual =
            GgufMatMul::get_isq_type_from_uqff(Cow::Owned(gguf_uqff_with_dtype(dtype))).unwrap();
        assert_eq!(actual, expected, "dtype {dtype}");
    }
}

#[test]
fn gguf_uqff_dtype_mapping_preserves_error_cases() {
    for dtype in [0, 1, 30] {
        let err = GgufMatMul::get_isq_type_from_uqff(Cow::Owned(gguf_uqff_with_dtype(dtype)))
            .unwrap_err();
        assert!(
            err.to_string().contains("Expected valid GGML ISQ type"),
            "dtype {dtype}: {err}"
        );
    }

    let err = GgufMatMul::get_isq_type_from_uqff(Cow::Owned(gguf_uqff_with_dtype(99))).unwrap_err();
    assert!(
        err.to_string()
            .contains("unknown dtype for quantized weight tensor 99"),
        "{err}"
    );
}

#[test]
fn hqq_bits_mapping_preserves_serialized_values() {
    let cases = [
        (8, HqqBits::Eight),
        (4, HqqBits::Four),
        (3, HqqBits::Three),
        (2, HqqBits::Two),
        (1, HqqBits::One),
    ];

    for (bits, expected) in cases {
        let actual = HqqBits::try_from(bits).unwrap();
        assert_eq!(actual as u8, expected as u8, "bits {bits}");
    }

    let err = HqqBits::try_from(5).unwrap_err();
    assert!(
        err.to_string().contains("Unexpected value for HQQ bits 5"),
        "{err}"
    );
}

#[test]
fn hqq_uqff_bits_mapping_matches_isq_types() {
    let cases = [(8, IsqType::HQQ8), (4, IsqType::HQQ4)];

    for (bits, expected) in cases {
        let actual =
            HqqLayer::get_isq_type_from_uqff(Cow::Owned(hqq_uqff_with_bits(bits))).unwrap();
        assert_eq!(actual, expected, "bits {bits}");
    }
}

#[test]
fn hqq_uqff_bits_mapping_preserves_error_cases() {
    for bits in [1, 2, 3] {
        let err =
            HqqLayer::get_isq_type_from_uqff(Cow::Owned(hqq_uqff_with_bits(bits))).unwrap_err();
        assert!(
            err.to_string()
                .contains("cannot convert hqq bits to isq type"),
            "bits {bits}: {err}"
        );
    }

    let err = HqqLayer::get_isq_type_from_uqff(Cow::Owned(hqq_uqff_with_bits(5))).unwrap_err();
    assert!(
        err.to_string().contains("Unexpected value for HQQ bits 5"),
        "{err}"
    );
}
