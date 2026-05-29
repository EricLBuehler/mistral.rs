use candle_core::{DType, Device};
use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

/// Configuration for [`super::MoEExperts`].
pub struct MoEExpertsConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
}

/// Which expert kernel runs the forward.
#[derive(Clone, Copy)]
pub(super) enum MoEExpertsBackend {
    /// Fused CUDA kernels over raw ENK tensors.
    Fused,
    /// cuTile JIT grouped GEMM over raw ENK tensors (CUDA bf16, GeLU).
    #[cfg(feature = "cutile")]
    Cutile,
    /// Gather-based (Metal, ISQ, pre-quantized).
    Fast,
    /// Loop-based fallback (quantized / CPU).
    Slow,
}

/// Everything backend selection depends on, gathered once per layer load.
pub(super) struct BackendChoice {
    pub device: Device,
    #[cfg_attr(not(feature = "cutile"), allow(dead_code))]
    pub dtype: DType,
    pub loading_isq: bool,
    pub quantized: bool,
    pub immediate_isq: bool,
    #[cfg_attr(not(feature = "cutile"), allow(dead_code))]
    pub act: Activation,
}

impl BackendChoice {
    pub(super) fn new(
        device: Device,
        dtype: DType,
        loading_isq: bool,
        quantization_config: &Option<QuantizedConfig>,
        act: Activation,
    ) -> Self {
        Self {
            device,
            dtype,
            loading_isq,
            quantized: quantization_config.is_some(),
            immediate_isq: mistralrs_quant::get_immediate_isq().is_some(),
            act,
        }
    }
}

impl MoEExpertsBackend {
    fn from_env() -> Option<Self> {
        let force = std::env::var("MISTRALRS_MOE_BACKEND").ok()?;
        Some(match force.as_str() {
            "fused" | "native" | "legacy" | "wmma" => Self::Fused,
            "fast" => Self::Fast,
            "slow" => Self::Slow,
            #[cfg(feature = "cutile")]
            "cutile" => Self::Cutile,
            _ => return None,
        })
    }

    /// Single source of truth for the backend: env override, then device/quant/ISQ, then cuTile when eligible, else Fused.
    pub(super) fn resolve(c: &BackendChoice) -> Self {
        if let Some(forced) = Self::from_env() {
            return forced;
        }
        if c.device.is_metal()
            || (c.device.is_cuda() && (c.loading_isq || c.quantized || c.immediate_isq))
        {
            return Self::Fast;
        }
        if c.device.is_cuda() && !c.quantized && !c.loading_isq && !c.immediate_isq {
            #[cfg(feature = "cutile")]
            if c.dtype == DType::BF16
                && matches!(c.act, Activation::NewGelu | Activation::GeluPytorchTanh)
                && cutile_arch_supported(&c.device)
            {
                return Self::Cutile;
            }
            return Self::Fused;
        }
        Self::Slow
    }
}

/// cuTile only on archs its JIT supports (Ampere or Blackwell+, not Hopper); otherwise fall back to Fused.
#[cfg(feature = "cutile")]
fn cutile_arch_supported(device: &Device) -> bool {
    match device {
        Device::Cuda(dev) => mistralrs_quant::cutile::device_supported(dev),
        _ => false,
    }
}
