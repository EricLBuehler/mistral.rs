pub(super) mod cpu;
mod flash;
mod naive;
mod sinks;
#[cfg(feature = "mlx")]
pub(crate) mod mlx_sdpa;

pub(crate) use flash::flash_attn;
pub(crate) use naive::{maybe_synchronize, naive_sdpa};
pub(crate) use sinks::sinks_attn;
