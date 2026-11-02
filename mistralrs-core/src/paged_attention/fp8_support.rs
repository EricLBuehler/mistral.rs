pub(super) fn supports_arch(major: i32, minor: i32, fp8: bool, legacy_sm61: bool) -> bool {
    fp8 && (major >= 8 || (legacy_sm61 && (major, minor) == (6, 1)))
}

#[cfg(test)]
mod tests {
    use super::supports_arch;

    #[test]
    fn campaign_cpu_fp8_kv_capability_matrix() {
        for (major, minor, native, legacy) in [
            (6, 0, false, false),
            (6, 1, false, true),
            (7, 0, false, false),
            (7, 5, false, false),
            (8, 0, true, true),
            (8, 9, true, true),
            (9, 0, true, true),
        ] {
            assert_eq!(supports_arch(major, minor, true, false), native);
            assert_eq!(supports_arch(major, minor, true, true), legacy);
            for opt_in in [false, true] {
                assert!(!supports_arch(major, minor, false, opt_in));
            }
        }
        assert!(!supports_arch(-1, -1, true, true));
    }
}
