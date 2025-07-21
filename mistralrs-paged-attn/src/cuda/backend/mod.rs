mod cache;
mod paged_attention;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, DevicePtr, DeviceRepr},
};
pub use paged_attention::{paged_attention, reshape_and_cache};

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}
