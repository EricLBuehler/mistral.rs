use std::{fmt::Display, str::FromStr};

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use serde::Deserialize;
use tracing::info;

#[derive(Clone, Copy, Default, Debug, Deserialize)]
/// DType for the model.
///
/// If the model is quantized, this is ignored so it is reasonable to use the [`Default`] impl.
///
/// Note: When using `Auto`, fallback pattern is: BF16 -> F16 -> 32
pub enum ModelDType {
    #[default]
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "bf16")]
    BF16,
    #[serde(rename = "f16")]
    F16,
    #[serde(rename = "f32")]
    F32,
}

impl Display for ModelDType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Auto => write!(f, "auto"),
            Self::BF16 => write!(f, "bf16"),
            Self::F16 => write!(f, "f16"),
            Self::F32 => write!(f, "f32"),
        }
    }
}

impl FromStr for ModelDType {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "auto" => Ok(Self::Auto),
            "bf16" => Ok(Self::BF16),
            "f16" => Ok(Self::F16),
            "f32" => Ok(Self::F32),
            other => Err(format!("Model DType `{other}` is not supported.")),
        }
    }
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

fn determine_auto_dtype(device: &Device) -> candle_core::Result<DType> {
    for dtype in [DType::BF16, DType::F16] {
        // Try a matmul
        let x = Tensor::zeros((2, 2), dtype, device)?;
        let y = x.matmul(&x);
        match y {
            Ok(_) => return Ok(dtype),
            Err(e) => match e {
                candle_core::Error::UnsupportedDTypeForOp(_, _) => continue,
                other => return Err(other),
            },
        }
    }
    Ok(DType::F32)
}

impl TryIntoDType for ModelDType {
    fn try_into_dtype(&self, device: &Device) -> Result<DType> {
        let dtype = match self {
            Self::Auto => Ok(determine_auto_dtype(device).map_err(anyhow::Error::msg)?),
            Self::BF16 => Ok(DType::BF16),
            Self::F16 => Ok(DType::F16),
            Self::F32 => Ok(DType::F32),
        };
        info!("DType selected is {:?}.", dtype.as_ref().unwrap());
        dtype
    }
}
