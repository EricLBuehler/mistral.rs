mod cache;
mod context_attention_mla;
mod fa3;
mod flash_attn_sinks;
mod flashinfer;
mod gather_kv;
mod mla;
mod paged_attention;
mod scale_update;
pub use cache::{copy_blocks, swap_blocks};
use candle_core::cuda::cudarc::{
    self,
    driver::{CudaSlice, CudaStream, DevicePtr, DeviceRepr},
};
use candle_core::{Layout, Result};
pub use context_attention_mla::context_attention_fwd_mla;
pub use fa3::{
    fa3_fp8_decode, fa3_prepare_decode_metadata, fa3_prepare_paged_metadata, Fa3DecodeMetadata,
    Fa3DecodeParams, Fa3DecodeSchedule, Fa3PagedMetadataLayout, FA3_DECODE_MAX_QUERY_LEN,
    USE_FA3_FP8_PAGED,
};
pub use flash_attn_sinks::{flash_attn_sinks, flash_attn_sinks_varlen};
pub use flashinfer::{
    flashinfer_decode, gather_kv_cache_flashinfer, is_flashinfer_cache,
    reshape_and_cache_flashinfer, FlashInferDecodeScratch,
};
pub use gather_kv::gather_kv_cache;
pub use mla::{concat_and_cache_mla, flashinfer_mla_decode, gather_mla_cache};
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;

fn cache_input_layout(
    layout: &Layout,
    name: &str,
    op: &str,
) -> Result<(usize, usize, usize, usize)> {
    let (num_tokens, num_heads, head_size, row_stride) = match *layout.dims() {
        [num_tokens, num_heads, head_size] => {
            (num_tokens, num_heads, head_size, layout.stride()[0])
        }
        [batch, seq_len, num_heads, head_size] => {
            let num_tokens = batch
                .checked_mul(seq_len)
                .ok_or_else(|| candle_core::Error::msg("cache input token count overflow"))?;
            let row_stride = if seq_len == 1 {
                layout.stride()[0]
            } else {
                layout.stride()[1]
            };
            if batch > 1 && seq_len > 1 && layout.stride()[0] != seq_len.saturating_mul(row_stride)
            {
                candle_core::bail!("{op} cannot flatten {name} batch/sequence strides: {layout:?}");
            }
            (num_tokens, num_heads, head_size, row_stride)
        }
        _ => candle_core::bail!("{op} expects rank-3 or rank-4 {name} input, got {layout:?}"),
    };
    if layout.stride()[layout.stride().len() - 1] != 1
        || layout.stride()[layout.stride().len() - 2] != head_size
        || row_stride < num_heads.saturating_mul(head_size)
    {
        candle_core::bail!("{op} expects dense {name} heads, got {layout:?}");
    }
    Ok((num_tokens, num_heads, head_size, row_stride))
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_input_layout_flattens_uniform_rank_four_rows() -> Result<()> {
        let prefill = Layout::new((2, 3, 2, 4).into(), vec![48, 16, 4, 1], 7);
        assert_eq!(cache_input_layout(&prefill, "key", "test")?, (6, 2, 4, 16));

        let decode = Layout::new((8, 1, 2, 4).into(), vec![24, 8, 4, 1], 5);
        assert_eq!(cache_input_layout(&decode, "value", "test")?, (8, 2, 4, 24));
        Ok(())
    }
}
