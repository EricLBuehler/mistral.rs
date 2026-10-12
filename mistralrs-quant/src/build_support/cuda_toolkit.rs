pub const SUPPORTED_CUDA_TOOLKIT_VERSIONS: &[(usize, usize)] = &[
    (13, 4),
    (13, 3),
    (13, 2),
    (13, 1),
    (13, 0),
    (12, 9),
    (12, 8),
    (12, 6),
    (12, 5),
    (12, 4),
    (12, 3),
    (12, 2),
    (12, 1),
    (12, 0),
    (11, 8),
    (11, 7),
    (11, 6),
    (11, 5),
    (11, 4),
];
const NVFP4_MIN_CUDA_VERSION: (usize, usize) = (13, 3);
const NVFP4_COMPUTE_CAP: usize = 121;

pub fn parse_cuda_version_from_nvcc(stdout: &str) -> Option<(usize, usize)> {
    let release = stdout.split("release ").nth(1)?;
    let version = release
        .split(|c: char| c == ',' || c.is_whitespace())
        .next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

pub fn nvfp4_cutlass_supported_for_build(
    cuda_version: (usize, usize),
    compute_cap: usize,
    cutile: bool,
    target: &str,
) -> bool {
    cuda_version >= NVFP4_MIN_CUDA_VERSION
        && compute_cap == NVFP4_COMPUTE_CAP
        && cutile
        && target.contains("linux")
}

#[cfg(test)]
mod tests {
    use super::*;

    const LINUX_TARGET: &str = "aarch64-unknown-linux-gnu";

    #[test]
    fn accepts_cuda_13_4_toolkit() {
        let version = parse_cuda_version_from_nvcc(
            "nvcc: NVIDIA (R) Cuda compiler driver\nCuda compilation tools, release 13.4, V13.4.0\n",
        )
        .unwrap();
        assert_eq!(version, (13, 4));
        assert!(SUPPORTED_CUDA_TOOLKIT_VERSIONS.contains(&version));
    }

    #[test]
    fn nvfp4_accepts_cuda_13_3_and_newer() {
        for version in [(13, 3), (13, 4)] {
            assert!(nvfp4_cutlass_supported_for_build(
                version,
                121,
                true,
                LINUX_TARGET
            ));
        }
        assert!(!nvfp4_cutlass_supported_for_build(
            (13, 2),
            121,
            true,
            LINUX_TARGET
        ));
    }

    #[test]
    fn nvfp4_preserves_target_and_feature_requirements() {
        for compute_cap in [80, 90, 100, 120] {
            assert!(!nvfp4_cutlass_supported_for_build(
                (13, 4),
                compute_cap,
                true,
                LINUX_TARGET
            ));
        }
        assert!(!nvfp4_cutlass_supported_for_build(
            (13, 4),
            121,
            false,
            LINUX_TARGET
        ));
        assert!(!nvfp4_cutlass_supported_for_build(
            (13, 4),
            121,
            true,
            "x86_64-pc-windows-msvc"
        ));
    }
}
