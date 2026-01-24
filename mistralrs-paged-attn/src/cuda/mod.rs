pub const USE_FP8: bool = true;

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache, copy_blocks, kv_scale_update,
    paged_attention, reshape_and_cache, swap_blocks,
};
