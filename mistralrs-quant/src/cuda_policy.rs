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
