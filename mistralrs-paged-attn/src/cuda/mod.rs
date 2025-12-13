pub const USE_FP8: bool = true;

mod backend;
mod ffi;

pub use backend::{copy_blocks, kv_scale_update, paged_attention, reshape_and_cache, swap_blocks};
