pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, copy_blocks, flash_attn_sinks,
    flash_attn_sinks_varlen, flashinfer_mla_decode, gather_kv_cache, gather_mla_cache,
    kv_scale_update, paged_attention, reshape_and_cache, swap_blocks,
};
