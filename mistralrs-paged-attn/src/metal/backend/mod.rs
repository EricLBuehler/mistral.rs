mod cache;
mod gather_kv;
mod paged_attention;
mod scale_update;

use candle_core::{DType, Result, Tensor};

fn validate_kv_cache_scales(
    cache_dtype: DType,
    k_scale: Option<&Tensor>,
    v_scale: Option<&Tensor>,
    op: &str,
) -> Result<()> {
    match (cache_dtype, k_scale, v_scale) {
        (DType::F8E4M3, Some(k_scale), Some(v_scale)) => {
            if k_scale.dtype() != DType::F32
                || v_scale.dtype() != DType::F32
                || k_scale.elem_count() != 1
                || v_scale.elem_count() != 1
            {
                candle_core::bail!("{op} requires scalar f32 K/V scales for an f8e4m3 cache");
            }
        }
        (DType::F8E4M3, _, _) => {
            candle_core::bail!("{op} requires explicit K/V scales for an f8e4m3 cache");
        }
        (_, None, None) => {}
        (_, _, _) => candle_core::bail!("{op} only accepts K/V scales for an f8e4m3 cache"),
    }
    Ok(())
}

pub use cache::{copy_blocks, swap_blocks};
pub use gather_kv::gather_kv_cache;
pub use paged_attention::{paged_attention, reshape_and_cache};
pub use scale_update::kv_scale_update;
