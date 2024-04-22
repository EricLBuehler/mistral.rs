#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod layer_norm;
mod rope;
use candle_core::WithDType;
pub use layer_norm::{rms_norm_non_quant, RmsNorm, RmsNormNonQuantized, RmsNormQuantized};
pub use rope::RotaryEmbedding;

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}
