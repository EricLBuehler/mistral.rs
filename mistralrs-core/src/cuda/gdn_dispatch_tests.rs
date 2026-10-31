use super::{
    automatic_decode_kernel, deferred_decode_batch_supported, parse_decode_kernel,
    parse_prefill_kernel, select_decode_kernel, select_prefill_kernel, GdnDecodeKernel,
    GdnDecodePolicy, GdnPrefillKernel, GdnPrefillPolicy,
};
use crate::kv_cache::RecurrentStateLayout;

const TEST_SM_COUNT: usize = 132;

#[test]
fn dispatch_rejects_unsupported_shapes_and_architectures() {
    let decode = GdnDecodePolicy {
        compute_major: 9,
        multiprocessor_count: TEST_SM_COUNT,
        state_blocks: 12 * TEST_SM_COUNT,
        head_k_dim: 128,
        head_v_dim: 128,
        vector_aligned: true,
        bf16: true,
        state_layout: RecurrentStateLayout::GdnKeyMajor,
    };
    assert_eq!(automatic_decode_kernel(decode), GdnDecodeKernel::Pipelined);
    for (compute_major, head_k_dim, head_v_dim, vector_aligned) in [
        (7, 128, 128, true),
        (9, 64, 128, true),
        (9, 128, 127, true),
        (9, 128, 128, false),
    ] {
        let unsupported = GdnDecodePolicy {
            compute_major,
            head_k_dim,
            head_v_dim,
            vector_aligned,
            ..decode
        };
        assert_eq!(
            automatic_decode_kernel(unsupported),
            GdnDecodeKernel::Baseline
        );
        assert!(select_decode_kernel(unsupported, Some(GdnDecodeKernel::Pipelined)).is_err());
    }
    let prefill = GdnPrefillPolicy {
        compute_major: 9,
        multiprocessor_count: TEST_SM_COUNT,
        state_blocks: 48,
        seq_len: 65,
        head_k_dim: 128,
        head_v_dim: 128,
        bf16: true,
        state_layout: RecurrentStateLayout::GdnValueMajor,
    };
    let forced = Some(GdnPrefillKernel::ValueMajor4);
    assert_eq!(
        select_prefill_kernel(prefill, forced),
        Ok(GdnPrefillKernel::ValueMajor4)
    );
    for (compute_major, seq_len, head_v_dim, bf16) in [
        (7, 65, 128, true),
        (9, 0, 128, true),
        (9, 65, 64, true),
        (9, 65, 128, false),
    ] {
        assert!(select_prefill_kernel(
            GdnPrefillPolicy {
                compute_major,
                seq_len,
                head_v_dim,
                bf16,
                ..prefill
            },
            forced
        )
        .is_err());
    }
    assert_eq!(parse_decode_kernel("auto").unwrap(), None);
    assert_eq!(parse_prefill_kernel("auto").unwrap(), None);
    assert!(parse_decode_kernel("unknown").is_err());
    assert!(parse_prefill_kernel("unknown").is_err());
}

#[test]
fn deferred_dispatch_preserves_small_batch_and_occupancy_boundaries() {
    for (batch, heads, multiprocessors, expected) in [
        (1, 48, 48, false),
        (8, 32, 132, true),
        (2, 32, 48, false),
        (2, 48, 48, true),
        (4, 48, 132, false),
        (6, 48, 132, true),
    ] {
        assert_eq!(
            deferred_decode_batch_supported(batch, heads, multiprocessors),
            expected
        );
    }
}
