// Golden test for the qwen35 YaRN partial rotary: values at position 262145
// (the first position beyond the native 262144 window) computed from the exact
// llama.cpp formulas (ggml.c corr_dims, rope.cu rope_yarn, llama-context.cpp
// factor resolution). Reference values generated independently in Python.
use candle_core::{DType, Device, Tensor};
use mistralrs_core::layers::RotaryEmbedding;

const BASE: f32 = 1e7;
const ROT_DIM: usize = 64; // rotary dim; head dim is 256 with 64 rotated
const HEAD_DIM: usize = 256;
const NATIVE_CTX: usize = 262144;
const TARGET_CTX: usize = 320000;
const FACTOR: f32 = 320000.0 / 262144.0;

fn build_yarn() -> RotaryEmbedding {
    RotaryEmbedding::new_partial_yarn(
        BASE,
        ROT_DIM,
        TARGET_CTX,
        NATIVE_CTX,
        FACTOR,
        1.0,
        1.0,
        32.0,
        1.0,
        &Device::Cpu,
        DType::F32,
    )
    .unwrap()
}

// For a neox partial rotary, pair p rotates dims (p, p + ROT_DIM/2); with
// q[p] = 1, q[p + ROT_DIM/2] = 0, the rotated output is (cos[p], sin[p]).
fn rotate_pair(rope: &RotaryEmbedding, position: usize, pair: usize) -> (f32, f32) {
    let mut qv = vec![0f32; HEAD_DIM];
    qv[pair] = 1.0;
    let q = Tensor::from_vec(qv, (1, 1, 1, HEAD_DIM), &Device::Cpu).unwrap();
    let k = Tensor::zeros((1, 1, 1, HEAD_DIM), DType::F32, &Device::Cpu).unwrap();
    let positions = Tensor::new(&[position as u32], &Device::Cpu).unwrap();
    let (q_out, _) = rope.forward(&q, &k, &positions).unwrap();
    let qv = q_out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    (qv[pair + ROT_DIM / 2], qv[pair])
}

#[test]
fn yarn_matches_llama_cpp_reference_at_position_262145() {
    let yarn = build_yarn();
    let position = NATIVE_CTX + 1;

    // (pair, expected_sin, expected_cos) from the reference computation.
    let cases = [
        (0usize, -0.901560650f32, -0.476939735f32),
        (15usize, 0.862911408f32, -0.543752718f32),
        (30usize, 0.059945550f32, 1.018179575f32),
    ];
    for (pair, expected_sin, expected_cos) in cases {
        let (sin, cos) = rotate_pair(&yarn, position, pair);
        assert!(
            (sin - expected_sin).abs() < 1e-4,
            "pair {pair} sin: got {sin}, expected {expected_sin}"
        );
        assert!(
            (cos - expected_cos).abs() < 1e-4,
            "pair {pair} cos: got {cos}, expected {expected_cos}"
        );
    }
}

#[test]
fn yarn_matches_llama_cpp_reference_at_boundary_positions() {
    // llama.cpp cache values (net mscale 1.0199427) at the last native
    // position, the first beyond-native position, and the last scaled
    // position, from the exact ggml.c / ops.cpp / llama-context.cpp chain.
    let yarn = build_yarn();
    let cases = [
        (262144usize, 30usize, 0.0599453214f32, 1.0181795887f32),
        (262145usize, 0usize, -0.9015606498f32, -0.4769397353f32),
        (262145usize, 15usize, 0.8629114077f32, -0.5437527184f32),
        (262145usize, 30usize, 0.0599455498f32, 1.0181795752f32),
        (319999usize, 13usize, -0.1119046827f32, 1.0137852131f32),
        (319999usize, 30usize, 0.0731545384f32, 1.0173158457f32),
    ];
    for (position, pair, expected_sin, expected_cos) in cases {
        let (sin, cos) = rotate_pair(&yarn, position, pair);
        assert!(
            (sin - expected_sin).abs() < 1e-4,
            "pos {position} pair {pair} sin: got {sin}, expected {expected_sin}"
        );
        assert!(
            (cos - expected_cos).abs() < 1e-4,
            "pos {position} pair {pair} cos: got {cos}, expected {expected_cos}"
        );
    }
}

#[test]
fn yarn_high_frequency_pairs_are_pure_extrapolation_scaled_by_mscale() {
    // Pairs below the correction range (low = 14) keep the native frequency and
    // are scaled by mscale = 1 + 0.1*ln(factor); the plain rope is the reference.
    let yarn = build_yarn();
    let plain = RotaryEmbedding::new_partial(
        BASE,
        ROT_DIM,
        TARGET_CTX,
        &Device::Cpu,
        true,
        DType::F32,
    )
    .unwrap();
    let mscale = 1.0 + 0.1 * FACTOR.ln();
    let position = NATIVE_CTX + 1;
    for pair in [0usize, 5, 13] {
        let (yarn_sin, yarn_cos) = rotate_pair(&yarn, position, pair);
        let (plain_sin, plain_cos) = rotate_pair(&plain, position, pair);
        assert!(
            (yarn_sin - plain_sin * mscale).abs() < 1e-5,
            "pair {pair} sin not extrapolation-scaled: {yarn_sin} vs {plain_sin} * {mscale}"
        );
        assert!(
            (yarn_cos - plain_cos * mscale).abs() < 1e-5,
            "pair {pair} cos not extrapolation-scaled: {yarn_cos} vs {plain_cos} * {mscale}"
        );
    }
}

#[test]
fn yarn_table_extends_to_the_scaled_context() {
    let yarn = build_yarn();
    // The last position of the scaled window must rotate without error.
    let (sin, cos) = rotate_pair(&yarn, TARGET_CTX - 1, 7);
    assert!(sin.is_finite() && cos.is_finite());
}
