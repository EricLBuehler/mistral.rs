pub const COPY_BLOCKS_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/copy_blocks_kernel.ptx"));
pub const PAGEDATTENTION: &str = include_str!(concat!(env!("OUT_DIR"), "/pagedattention.ptx"));
pub const RESHAPE_AND_CACHE_KERNEL: &str =
    include_str!(concat!(env!("OUT_DIR"), "/reshape_and_cache_kernel.ptx"));

mod backend;
mod ffi;

pub use backend::{copy_blocks, paged_attention, reshape_and_cache, swap_blocks};
