mod backend;
mod kernels;

pub use backend::{
    copy_blocks, gather_kv_cache, kv_scale_update, metal_paged_prefill, paged_attention,
    paged_attention_with_window, prepare_metal_paged_attention, reshape_and_cache, swap_blocks,
    MetalPagedAttentionWarmup, MetalPagedAttentionWorkspace, MetalPagedPrefillRequest,
    MetalPagedPrefillShape, METAL_PAGED_CACHE_VECTOR_BYTES,
};

pub fn warmup_metal_paged_library(device: &candle_core::MetalDevice) -> candle_core::Result<()> {
    kernels::Kernels::new()
        .load_library(device.device())
        .map_err(candle_core::Error::wrap)?;
    Ok(())
}
