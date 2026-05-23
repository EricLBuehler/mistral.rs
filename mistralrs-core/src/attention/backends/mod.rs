pub(super) mod cpu;
mod flash;
#[cfg(feature = "metal")]
pub(crate) mod metal_flash_attn;
mod naive;
mod sinks;

pub(crate) use flash::flash_attn;
pub(crate) use naive::{maybe_synchronize, naive_sdpa};
pub(crate) use sinks::sinks_attn;

#[cfg(not(feature = "metal"))]
pub(crate) mod metal_flash_attn {
    use candle_core::{Result, Tensor};
    pub fn try_flash_attn_ext_bf16_dk512(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _mask: &Tensor,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
    pub fn try_flash_attn_ext_vec_bf16_dk512(
        _q: &Tensor,
        _k: &Tensor,
        _v: &Tensor,
        _mask: Option<&Tensor>,
        _scale: f32,
    ) -> Result<Option<Tensor>> {
        Ok(None)
    }
}
