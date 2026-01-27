pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, copy_blocks, flashinfer_mla_decode, gather_mla_cache, kv_scale_update,
    paged_attention, reshape_and_cache, swap_blocks,
};
