#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const COPY_BLOCKS_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/copy_blocks_kernel.ptx"));
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const PAGEDATTENTION: &str = include_str!(concat!(env!("OUT_DIR"), "/pagedattention.ptx"));
#[cfg(all(feature = "cuda", target_family = "unix"))]
pub const RESHAPE_AND_CACHE_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/reshape_and_cache_kernel.ptx"));

#[cfg(all(feature = "cuda", target_family = "unix"))]
mod backend;
#[cfg(all(feature = "cuda", target_family = "unix"))]
mod ffi;

#[cfg(all(feature = "cuda", target_family = "unix"))]
pub use backend::{copy_blocks, paged_attention, reshape_and_cache, swap_blocks};
