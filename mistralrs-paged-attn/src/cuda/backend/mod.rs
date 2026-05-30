mod cache;
mod context_attention_mla;
#[cfg(feature = "cutile")]
mod cutile_attention;
mod flash_attn_sinks;
mod flashinfer;
mod gather_kv;
mod mla;
mod paged_attention;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr},
};
pub use context_attention_mla::context_attention_fwd_mla;
#[cfg(feature = "cutile")]
pub use cutile_attention::{
    cutile_paged_attention_decode, cutile_paged_attention_prefill,
    cutile_paged_attention_supported, register_cutile_attention_q_group,
    warmup_cutile_attention_kernels,
};
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
pub use flashinfer::{
    flashinfer_decode, flashinfer_prefill, gather_kv_cache_flashinfer, is_flashinfer_cache,
    reshape_and_cache_flashinfer,
};
pub use gather_kv::gather_kv_cache;
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    slice_ptr_on_stream(v, lo, v.stream())
}

pub fn slice_ptr_on_stream<'a, T: DeviceRepr>(
    v: &'a CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}

pub fn slice_ptr_mut_on_stream<'a, T: DeviceRepr>(
    v: &'a mut CudaSlice<T>,
    lo: usize,
    stream: &'a CudaStream,
) -> (u64, cudarc::driver::SyncOnDrop<'a>) {
    let (ptr, guard) = v.device_ptr_mut(stream);
    (ptr + (lo * std::mem::size_of::<T>()) as u64, guard)
}
