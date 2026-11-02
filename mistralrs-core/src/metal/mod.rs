pub mod gdn;
#[cfg(feature = "metal")]
pub(crate) mod kernels;
pub mod ssm;

use candle_core::{Device, Result};

#[cfg(feature = "metal")]
use candle_metal_kernels::source::Source;

pub fn warmup_metal_kernels(device: &Device) -> Result<()> {
    warmup_common_kernels(device)?;
    #[cfg(feature = "metal")]
    if let Device::Metal(device) = device {
        kernels::warmup_libraries(device.device())?;
        mistralrs_paged_attn::warmup_metal_paged_library(device)?;
    }
    Ok(())
}

pub(crate) fn warmup_common_kernels(device: &Device) -> Result<()> {
    #[cfg(not(feature = "metal"))]
    let _ = device;

    #[cfg(feature = "metal")]
    if let Device::Metal(device) = device {
        let sources = [
            Source::Affine,
            Source::Binary,
            Source::Cast,
            Source::Conv,
            Source::Fill,
            Source::Gemm,
            Source::Gemv,
            Source::Indexing,
            Source::MlxSort,
            Source::Quantized,
            Source::Random,
            Source::Reduce,
            Source::Sort,
            Source::Ternary,
            Source::Unary,
            Source::Sdpa,
        ];
        for source in sources {
            device
                .kernels()
                .load_library(device.device(), source)
                .map_err(candle_core::Error::wrap)?;
        }
        mistralrs_quant::metal_kernels::Kernels::global()
            .load_library(device.device())
            .map_err(candle_core::Error::wrap)?;
    }
    Ok(())
}
