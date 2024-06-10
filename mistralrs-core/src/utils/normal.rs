use anyhow::Result;
use candle_core::{DType, Device};
use tracing::info;

#[derive(Clone, Copy, Default)]
/// DType for the model.
///
/// If the model is quantized, this is ignored so it is reasonable to use the [`Default`] impl.
///
/// ## `Auto` rules
/// - If CUDA device or CPU, use BF16
/// - Fallback to F16
pub enum ModelDType {
    #[default]
    Auto,
    BF16,
    F16,
    F32,
}

/// Type which can be converted to a DType
pub trait TryIntoDType {
    fn try_into_dtype(&self, device: &Device) -> Result<DType>;
}

impl TryIntoDType for DType {
    fn try_into_dtype(&self, _: &Device) -> Result<DType> {
        info!("DType selected is {self:?}.");
        if !matches!(self, DType::BF16 | DType::F32 | DType::F64 | DType::F16) {
            anyhow::bail!("DType must be one of BF16, F16, F32, F64");
        }
        Ok(*self)
    }
}

impl TryIntoDType for ModelDType {
    fn try_into_dtype(&self, device: &Device) -> Result<DType> {
        match self {
            Self::Auto => {
                if device.is_cuda() || device.is_cpu() {
                    Ok(DType::BF16)
                } else {
                    Ok(DType::F32)
                }
            }
            Self::BF16 => Ok(DType::BF16),
            Self::F16 => Ok(DType::F16),
            Self::F32 => Ok(DType::F32),
        }
    }
}
