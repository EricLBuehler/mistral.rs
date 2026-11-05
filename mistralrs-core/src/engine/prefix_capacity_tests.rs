use super::effective_prefix_cache_capacity;

#[test]
fn paged_recurrent_prefix_retention_uses_the_selected_capacity() {
    for (requested, planned, expected) in [
        (16, Some(8), 8),
        (16, Some(16), 16),
        (4, Some(8), 4),
        (0, Some(8), 0),
        (16, Some(0), 0),
        (16, None, 16),
    ] {
        assert_eq!(
            effective_prefix_cache_capacity(requested, true, true, planned),
            expected,
        );
    }
}

#[test]
fn dense_and_nonpaged_prefix_retention_ignore_recurrent_limits() {
    for (paged, hybrid) in [(false, false), (false, true), (true, false)] {
        assert_eq!(
            effective_prefix_cache_capacity(16, paged, hybrid, Some(0)),
            16
        );
    }
}
