use super::{
    automatic_decode_kernel, automatic_prefill_kernel, deferred_decode_batch_supported,
    parse_decode_kernel, parse_prefill_kernel, prefill_kernel_supported, select_decode_kernel,
    select_prefill_kernel, GdnDecodeKernel, GdnDecodePolicy, GdnPrefillKernel, GdnPrefillPolicy,
};
use crate::kv_cache::RecurrentStateLayout;

const TEST_SM_COUNT: usize = 132;

fn sm90_policy(state_blocks: usize) -> GdnDecodePolicy {
    GdnDecodePolicy {
        compute_major: 9,
        multiprocessor_count: TEST_SM_COUNT,
        state_blocks,
        head_k_dim: 128,
        head_v_dim: 128,
        vector_aligned: true,
        bf16: true,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
    }
}

fn sm90_value_major_policy(batch_size: usize) -> GdnDecodePolicy {
    GdnDecodePolicy {
        state_blocks: batch_size * 48,
        state_layout: RecurrentStateLayout::GdnValueMajor,
        ..sm90_policy(0)
    }
}

fn sm90_prefill_policy(state_blocks: usize, seq_len: usize) -> GdnPrefillPolicy {
    GdnPrefillPolicy {
        compute_major: 9,
        multiprocessor_count: TEST_SM_COUNT,
        state_blocks,
        seq_len,
        head_k_dim: 128,
        head_v_dim: 128,
        bf16: true,
        state_layout: RecurrentStateLayout::GdnValueMajor,
    }
}

#[test]
fn decode_dispatch_uses_sm90_state_waves() {
    assert_eq!(
        automatic_decode_kernel(sm90_policy(TEST_SM_COUNT)),
        GdnDecodeKernel::Cooperative
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(2 * TEST_SM_COUNT - 1)),
        GdnDecodeKernel::Cooperative
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(2 * TEST_SM_COUNT)),
        GdnDecodeKernel::Pipelined
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(3 * TEST_SM_COUNT)),
        GdnDecodeKernel::Pipelined
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(4 * TEST_SM_COUNT)),
        GdnDecodeKernel::Baseline
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(6 * TEST_SM_COUNT)),
        GdnDecodeKernel::Baseline
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(8 * TEST_SM_COUNT - 1)),
        GdnDecodeKernel::Baseline
    );
    assert_eq!(
        automatic_decode_kernel(sm90_policy(8 * TEST_SM_COUNT)),
        GdnDecodeKernel::Pipelined
    );
}

#[test]
fn automatic_decode_dispatch_requires_a_supported_architecture_and_shape() {
    let mut unsupported = sm90_policy(12 * TEST_SM_COUNT);
    unsupported.compute_major = 7;
    assert_eq!(
        automatic_decode_kernel(unsupported),
        GdnDecodeKernel::Baseline
    );
    for compute_major in [8, 10, 12] {
        let mut supported = sm90_policy(12 * TEST_SM_COUNT);
        supported.compute_major = compute_major;
        assert_eq!(
            automatic_decode_kernel(supported),
            GdnDecodeKernel::Pipelined
        );
    }
    unsupported = sm90_policy(12 * TEST_SM_COUNT);
    unsupported.head_k_dim = 64;
    assert_eq!(
        automatic_decode_kernel(unsupported),
        GdnDecodeKernel::Baseline
    );
    unsupported = sm90_policy(12 * TEST_SM_COUNT);
    unsupported.head_v_dim = 127;
    assert_eq!(
        automatic_decode_kernel(unsupported),
        GdnDecodeKernel::Baseline
    );
    unsupported = sm90_policy(12 * TEST_SM_COUNT);
    unsupported.head_v_dim = 32;
    assert_eq!(
        automatic_decode_kernel(unsupported),
        GdnDecodeKernel::Baseline
    );
    unsupported = sm90_policy(12 * TEST_SM_COUNT);
    unsupported.vector_aligned = false;
    assert_eq!(
        automatic_decode_kernel(unsupported),
        GdnDecodeKernel::Baseline
    );
}

#[test]
fn value_major_dispatch_uses_the_measured_sm90_crossover() {
    for batch_size in 1..=5 {
        assert_eq!(
            automatic_decode_kernel(sm90_value_major_policy(batch_size)),
            GdnDecodeKernel::ValueMajor4
        );
    }
    for batch_size in [6, 8, 16] {
        assert_eq!(
            automatic_decode_kernel(sm90_value_major_policy(batch_size)),
            GdnDecodeKernel::ValueMajor32
        );
    }

    let mut unsupported = sm90_value_major_policy(8);
    unsupported.bf16 = false;
    assert!(!super::decode_kernel_supported(
        GdnDecodeKernel::ValueMajor32,
        unsupported
    ));
    unsupported = sm90_value_major_policy(8);
    unsupported.compute_major = 7;
    assert!(!super::decode_kernel_supported(
        GdnDecodeKernel::ValueMajor32,
        unsupported
    ));
    for compute_major in [8, 10, 12] {
        let mut supported = sm90_value_major_policy(8);
        supported.compute_major = compute_major;
        assert_eq!(
            automatic_decode_kernel(supported),
            GdnDecodeKernel::ValueMajor32
        );
    }
}

#[test]
fn deferred_decode_dispatch_preserves_single_token_and_large_batches() {
    for value_heads in [32, 48, 64] {
        for multiprocessors in [48, 80, 132] {
            assert!(!deferred_decode_batch_supported(
                1,
                value_heads,
                multiprocessors
            ));
            assert!(deferred_decode_batch_supported(
                8,
                value_heads,
                multiprocessors
            ));
            assert!(deferred_decode_batch_supported(
                32,
                value_heads,
                multiprocessors
            ));
        }
    }
}

#[test]
fn deferred_decode_dispatch_uses_two_waves_for_intermediate_batches() {
    for (batch, heads, multiprocessors, expected) in [
        (2, 32, 48, false),
        (2, 48, 48, true),
        (2, 64, 48, true),
        (3, 32, 48, true),
        (3, 48, 80, false),
        (3, 64, 80, true),
        (4, 32, 48, true),
        (4, 32, 80, false),
        (4, 48, 80, true),
        (4, 48, 132, false),
        (5, 32, 80, true),
        (6, 48, 132, true),
        (7, 32, 132, false),
        (7, 48, 132, true),
    ] {
        assert_eq!(
            deferred_decode_batch_supported(batch, heads, multiprocessors),
            expected
        );
    }
}

#[test]
fn decode_kernel_override_is_forceable_and_validated() {
    assert_eq!(
        parse_prefill_kernel("cutile").unwrap(),
        Some(GdnPrefillKernel::Cutile)
    );
    assert_eq!(parse_decode_kernel("auto").unwrap(), None);
    assert_eq!(
        parse_decode_kernel("BASELINE").unwrap(),
        Some(GdnDecodeKernel::Baseline)
    );
    assert_eq!(
        parse_decode_kernel("cooperative").unwrap(),
        Some(GdnDecodeKernel::Cooperative)
    );
    assert_eq!(
        parse_decode_kernel("pipelined").unwrap(),
        Some(GdnDecodeKernel::Pipelined)
    );
    assert_eq!(
        parse_decode_kernel("vmajor4").unwrap(),
        Some(GdnDecodeKernel::ValueMajor4)
    );
    assert_eq!(
        parse_decode_kernel("vmajor32").unwrap(),
        Some(GdnDecodeKernel::ValueMajor32)
    );
    assert!(parse_decode_kernel("unknown").is_err());

    let policy = sm90_policy(3 * TEST_SM_COUNT);
    assert_eq!(
        select_decode_kernel(policy, Some(GdnDecodeKernel::Baseline)).unwrap(),
        GdnDecodeKernel::Baseline
    );
    let mut unsupported = policy;
    unsupported.compute_major = 7;
    assert_eq!(
        select_decode_kernel(unsupported, Some(GdnDecodeKernel::Pipelined)),
        Err(GdnDecodeKernel::Pipelined)
    );
    let mut blackwell = policy;
    blackwell.compute_major = 12;
    assert_eq!(
        select_decode_kernel(blackwell, Some(GdnDecodeKernel::Pipelined)).unwrap(),
        GdnDecodeKernel::Pipelined
    );
}

#[test]
fn prefill_auto_dispatch_tracks_available_parallelism() {
    for (state_blocks, seq_len, expected) in [
        (1, 2, GdnPrefillKernel::ValueMajor2),
        (48, 64, GdnPrefillKernel::ValueMajor2),
        (48, 8_192, GdnPrefillKernel::ValueMajor2),
        (8 * 48, 2_048, GdnPrefillKernel::ValueMajor4),
        (16 * 48, 2_048, GdnPrefillKernel::ValueMajor8),
    ] {
        assert_eq!(
            automatic_prefill_kernel(sm90_prefill_policy(state_blocks, seq_len)),
            expected
        );
    }
}

#[test]
fn prefill_kernel_override_is_forceable_and_validated() {
    assert_eq!(parse_prefill_kernel("auto").unwrap(), None);
    assert_eq!(
        parse_prefill_kernel("mma").unwrap(),
        Some(GdnPrefillKernel::Mma)
    );
    assert_eq!(
        parse_prefill_kernel("mma-fused").unwrap(),
        Some(GdnPrefillKernel::MmaFused)
    );
    assert_eq!(
        parse_prefill_kernel("flashinfer-sm90").unwrap(),
        Some(GdnPrefillKernel::FlashInferSm90)
    );
    assert_eq!(
        parse_prefill_kernel("VMAJOR1").unwrap(),
        Some(GdnPrefillKernel::ValueMajor1)
    );
    assert_eq!(
        parse_prefill_kernel("vmajor2").unwrap(),
        Some(GdnPrefillKernel::ValueMajor2)
    );
    assert_eq!(
        parse_prefill_kernel("vmajor4").unwrap(),
        Some(GdnPrefillKernel::ValueMajor4)
    );
    assert_eq!(
        parse_prefill_kernel("vmajor8").unwrap(),
        Some(GdnPrefillKernel::ValueMajor8)
    );
    assert_eq!(
        parse_prefill_kernel("legacy-chunked").unwrap(),
        Some(GdnPrefillKernel::LegacyChunked)
    );
    assert!(parse_prefill_kernel("unknown").is_err());

    let policy = sm90_prefill_policy(48, 129);
    assert!(!prefill_kernel_supported(
        GdnPrefillKernel::FlashInferSm90,
        policy
    ));
    for kernel in [
        GdnPrefillKernel::ValueMajor1,
        GdnPrefillKernel::ValueMajor2,
        GdnPrefillKernel::ValueMajor4,
        GdnPrefillKernel::ValueMajor8,
        GdnPrefillKernel::LegacyChunked,
    ] {
        assert!(prefill_kernel_supported(kernel, policy));
        assert_eq!(select_prefill_kernel(policy, Some(kernel)), Ok(kernel));
    }

    let mut unsupported = policy;
    unsupported.compute_major = 7;
    assert_eq!(
        select_prefill_kernel(unsupported, Some(GdnPrefillKernel::ValueMajor4)),
        Err(GdnPrefillKernel::ValueMajor4)
    );
    let mut blackwell = policy;
    blackwell.compute_major = 12;
    assert_eq!(
        select_prefill_kernel(blackwell, Some(GdnPrefillKernel::ValueMajor4)),
        Ok(GdnPrefillKernel::ValueMajor4)
    );
    unsupported = policy;
    unsupported.bf16 = false;
    assert!(!prefill_kernel_supported(
        GdnPrefillKernel::ValueMajor4,
        unsupported
    ));
    unsupported = policy;
    unsupported.head_v_dim = 64;
    assert!(!prefill_kernel_supported(
        GdnPrefillKernel::ValueMajor4,
        unsupported
    ));
    unsupported = policy;
    unsupported.seq_len = 0;
    assert!(!prefill_kernel_supported(
        GdnPrefillKernel::ValueMajor4,
        unsupported
    ));
}

fn state_index(
    layout: RecurrentStateLayout,
    key: usize,
    value: usize,
    key_dim: usize,
    value_dim: usize,
) -> usize {
    match layout {
        RecurrentStateLayout::GdnKeyMajor => key * value_dim + value,
        RecurrentStateLayout::GdnValueMajor => value * key_dim + key,
        RecurrentStateLayout::Opaque => unreachable!(),
    }
}

struct ReferenceRecurrence<'a> {
    q: &'a [f32],
    k: &'a [f32],
    v: &'a [f32],
    g: &'a [f32],
    beta: &'a [f32],
    key_dim: usize,
    value_dim: usize,
}

fn reference_recurrence(
    layout: RecurrentStateLayout,
    state: &mut [f32],
    inputs: &ReferenceRecurrence<'_>,
) -> Vec<f32> {
    let seq_len = inputs.g.len();
    let mut output = vec![0.0f32; seq_len * inputs.value_dim];
    for token in 0..seq_len {
        let decay = inputs.g[token].exp();
        for value in 0..inputs.value_dim {
            let mut state_dot_k = 0.0f32;
            for key in 0..inputs.key_dim {
                let index = state_index(layout, key, value, inputs.key_dim, inputs.value_dim);
                state[index] *= decay;
                state_dot_k += state[index] * inputs.k[token * inputs.key_dim + key];
            }
            let delta =
                (inputs.v[token * inputs.value_dim + value] - state_dot_k) * inputs.beta[token];
            let mut state_dot_q = 0.0f32;
            for key in 0..inputs.key_dim {
                let index = state_index(layout, key, value, inputs.key_dim, inputs.value_dim);
                state[index] += inputs.k[token * inputs.key_dim + key] * delta;
                state_dot_q += state[index] * inputs.q[token * inputs.key_dim + key];
            }
            output[token * inputs.value_dim + value] = state_dot_q;
        }
    }
    output
}

#[test]
fn recurrence_reference_is_layout_invariant() {
    const KEY_DIM: usize = 3;
    const VALUE_DIM: usize = 2;
    const SEQ_LEN: usize = 5;

    let key_major = vec![0.1, -0.2, 0.3, 0.4, -0.5, 0.6];
    let mut value_major = vec![0.0f32; KEY_DIM * VALUE_DIM];
    for key in 0..KEY_DIM {
        for value in 0..VALUE_DIM {
            value_major[state_index(
                RecurrentStateLayout::GdnValueMajor,
                key,
                value,
                KEY_DIM,
                VALUE_DIM,
            )] = key_major[state_index(
                RecurrentStateLayout::GdnKeyMajor,
                key,
                value,
                KEY_DIM,
                VALUE_DIM,
            )];
        }
    }

    let q = (0..SEQ_LEN * KEY_DIM)
        .map(|index| (f32::from(u16::try_from(index).unwrap()) - 5.0) * 0.03)
        .collect::<Vec<_>>();
    let k = (0..SEQ_LEN * KEY_DIM)
        .map(|index| (7.0 - f32::from(u16::try_from(index).unwrap())) * 0.02)
        .collect::<Vec<_>>();
    let v = (0..SEQ_LEN * VALUE_DIM)
        .map(|index| (f32::from(u16::try_from(index).unwrap()) - 3.0) * 0.04)
        .collect::<Vec<_>>();
    let g = vec![-0.01, -0.03, -0.02, -0.04, -0.05];
    let beta = vec![0.2, 0.7, 0.4, 0.9, 0.5];
    let inputs = ReferenceRecurrence {
        q: &q,
        k: &k,
        v: &v,
        g: &g,
        beta: &beta,
        key_dim: KEY_DIM,
        value_dim: VALUE_DIM,
    };

    let mut key_major_result = key_major;
    let key_major_output = reference_recurrence(
        RecurrentStateLayout::GdnKeyMajor,
        &mut key_major_result,
        &inputs,
    );
    let value_major_output = reference_recurrence(
        RecurrentStateLayout::GdnValueMajor,
        &mut value_major,
        &inputs,
    );

    for (left, right) in key_major_output.iter().zip(&value_major_output) {
        assert!((left - right).abs() <= 1.0e-6);
    }
    for key in 0..KEY_DIM {
        for value in 0..VALUE_DIM {
            let key_major_index = state_index(
                RecurrentStateLayout::GdnKeyMajor,
                key,
                value,
                KEY_DIM,
                VALUE_DIM,
            );
            let value_major_index = state_index(
                RecurrentStateLayout::GdnValueMajor,
                key,
                value,
                KEY_DIM,
                VALUE_DIM,
            );
            assert!(
                (key_major_result[key_major_index] - value_major[value_major_index]).abs()
                    <= 1.0e-6
            );
        }
    }
}
