mod cache;
mod context_attention_mla;
mod flash_attn_sinks;
mod gather_kv;
mod mla;
mod paged_attention;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, DevicePtr, DeviceRepr},
};
pub use context_attention_mla::context_attention_fwd_mla;
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
pub use gather_kv::gather_kv_cache;
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;

pub fn slice_ptr<T: DeviceRepr>(
    v: &CudaSlice<T>,
    lo: usize,
) -> (u64, cudarc::driver::SyncOnDrop<'_>) {
    let (_, guard) = v.device_ptr(v.stream());
    let (ptr, _) = v.slice(lo..).device_ptr(v.stream());
    (ptr, guard)
}
