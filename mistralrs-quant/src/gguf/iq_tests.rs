// Golden vectors from the local GGUF models, extracted with gguf-py's reference
// dequantiser. The bytes are real model blocks, so this checks the port against a second
// implementation on data the models actually carry.

use crate::gguf::iq_golden::GOLDEN;
use super::{dequantize_blocks, IqTensor};
use crate::gguf::archive::GgufDType;

/// Each entry names the type, carries four real blocks, and expects 1024 values.
#[test]
fn every_iq_type_dequantises_the_golden_blocks_exactly() {
    for (name, hex_bytes, expected) in GOLDEN {
        let dtype_raw = match *name {
            "IQ2_XXS" => 16,
            "IQ2_XS" => 17,
            "IQ3_XXS" => 18,
            "IQ1_S" => 19,
            "IQ4_NL" => 20,
            "IQ3_S" => 21,
            "IQ2_S" => 22,
            "IQ4_XS" => 23,
            "IQ1_M" => 29,
            other => panic!("unknown golden type {other}"),
        };
        let dtype = GgufDType::new(dtype_raw);
        let type_size = dtype.type_size().expect("golden type size");
        let block_size = dtype.block_size().expect("golden block size");

        let bytes = hex_to_bytes(hex_bytes);
        assert_eq!(bytes.len() % type_size, 0, "{name}: partial block");
        let n_blocks = bytes.len() / type_size;

        let mut got = vec![0f32; n_blocks * block_size];
        dequantize_blocks(dtype, &bytes, n_blocks, &mut got)
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(
            got.len(),
            expected.len(),
            "{name}: {} blocks decode to {} values, expected {}",
            n_blocks,
            got.len(),
            expected.len()
        );
        for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{name}: value {i} is {g}, expected {w}"
            );
        }
    }
}

/// The tensor wrapper rejects a byte count that does not match its shape.
#[test]
fn iq_tensor_checks_the_byte_count() {
    let dtype = GgufDType::new(23); // IQ4_XS: 136 bytes per 256 values
    let ok = IqTensor::new(vec![0u8; 136 * 4].into_boxed_slice(), vec![1024], dtype);
    assert!(ok.is_ok());
    let short = IqTensor::new(vec![0u8; 136 * 4 - 1].into_boxed_slice(), vec![1024], dtype);
    assert!(short.is_err());
    let not_iq = IqTensor::new(
        vec![0u8; 144 * 4].into_boxed_slice(),
        vec![1024],
        GgufDType::new(12), // Q4_K
    );
    assert!(not_iq.is_err());
}

fn hex_to_bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).expect("golden hex"))
        .collect()
}
