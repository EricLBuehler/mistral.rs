pub const USE_FP8: bool = cfg!(has_fp8);

mod backend;
mod ffi;

pub use backend::{
    concat_and_cache_mla, context_attention_fwd_mla, copy_blocks, flash_attn_sinks,
    flash_attn_sinks_varlen, flashinfer_decode, flashinfer_mla_decode, flashinfer_prefill,
    gather_kv_cache, gather_kv_cache_flashinfer, gather_mla_cache, is_flashinfer_cache,
    kv_scale_update, paged_attention, reshape_and_cache, reshape_and_cache_flashinfer, swap_blocks,
};

#[cfg(feature = "cutile")]
pub use backend::{
    cutile_paged_attention_decode, cutile_paged_attention_prefill,
    cutile_paged_attention_supported, register_cutile_attention_q_group,
    warmup_cutile_attention_kernels,
};
