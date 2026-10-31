use crate::cuda_device::CudaDeviceInfo;

pub fn prefers_weight_prefetch(device: &CudaDeviceInfo) -> bool {
    device.integrated && device.has_cp_async()
}

pub(crate) fn fuse_underfilled_grids(
    device: &CudaDeviceInfo,
    blocks: &[usize],
    threads_per_block: usize,
) -> bool {
    let resident_blocks = device.sms * (device.threads_per_sm / threads_per_block);
    // Separate full waves amortize launch overhead; combining underfilled grids shares that cost.
    blocks.iter().all(|&grid| grid <= resident_blocks)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn device(sms: usize, threads_per_sm: usize, integrated: bool) -> CudaDeviceInfo {
        CudaDeviceInfo {
            major: 8,
            minor: 0,
            sms,
            threads_per_sm,
            l2_bytes: 0,
            smem_optin: 0,
            warp_size: 32,
            integrated,
        }
    }

    #[test]
    fn fusion_scales_with_parallel_capacity_and_launch_geometry() {
        let small = device(48, 1536, true);
        let large = device(132, 2048, false);
        assert!(fuse_underfilled_grids(&small, &[256, 256], 256));
        assert!(!fuse_underfilled_grids(&small, &[320, 320], 256));
        assert!(fuse_underfilled_grids(&large, &[320, 320], 256));
        assert!(fuse_underfilled_grids(&small, &[320, 320], 128));
        assert!(!fuse_underfilled_grids(&small, &[64, 1024], 256));
    }

    #[test]
    fn prefetch_policy_requires_memory_topology_and_instruction_support() {
        assert!(prefers_weight_prefetch(&device(48, 1536, true)));
        assert!(!prefers_weight_prefetch(&device(48, 1536, false)));
        assert!(!prefers_weight_prefetch(&CudaDeviceInfo {
            major: 7,
            minor: 5,
            ..device(48, 1536, true)
        }));
    }
}
