use candle_core::{DType, Device, Result};
#[cfg(feature = "cuda")]
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

/// Configuration for [`super::MoEExperts`].
pub struct MoEExpertsConfig {
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub hidden_size: usize,
    pub moe_intermediate_size: usize,
    pub expert_proj_names: ExpertProjNames,
}

/// One of the three expert projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    /// Inverse of `UqffExpertKeys::new`: canonical tracked key -> (experts prefix, projection).
    pub(crate) fn split_canonical_key(key: &str) -> Option<(&str, Self)> {
        if let Some(prefix) = key.strip_suffix(".gate_proj") {
            Some((prefix, Self::Gate))
        } else if let Some(prefix) = key.strip_suffix(".up_proj") {
            Some((prefix, Self::Up))
        } else if let Some(prefix) = key.strip_suffix(".down_proj") {
            Some((prefix, Self::Down))
        } else {
            None
        }
    }

    pub(crate) fn name_in(self, names: &ExpertProjNames) -> &'static str {
        match self {
            Self::Gate => names.gate,
            Self::Up => names.up,
            Self::Down => names.down,
        }
    }
}

/// Per-expert projection tensor names; mixtral-style checkpoints use `w1`/`w3`/`w2`.
#[derive(Clone, Copy)]
pub struct ExpertProjNames {
    pub gate: &'static str,
    pub up: &'static str,
    pub down: &'static str,
}

impl ExpertProjNames {
    pub const MIXTRAL: Self = Self {
        gate: "w1",
        up: "w3",
        down: "w2",
    };
    pub const DEFAULT: Self = Self {
        gate: "gate_proj",
        up: "up_proj",
        down: "down_proj",
    };
    /// Every naming family any model uses; source-weight probing tries each.
    pub const KNOWN: [Self; 2] = [Self::DEFAULT, Self::MIXTRAL];
}

/// Which expert kernel runs the forward.
#[derive(Clone, Copy)]
pub(super) enum MoEExpertsBackend {
    /// Fused CUDA kernels over raw ENK tensors.
    Fused,
    /// cuTile JIT grouped GEMM over raw ENK tensors (CUDA bf16, GeLU).
    #[cfg(feature = "cutile")]
    Cutile,
    /// CUTLASS grouped GEMM over raw ENK tensors (CUDA bf16, GeLU); universal sm_80+ fallback.
    #[cfg(feature = "cuda")]
    Cutlass,
    /// Gather-based (Metal, CPU, ISQ, pre-quantized).
    Fast,
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

/// Resolve and log the MoE backend choice up front so the one-time INFO line lands before
/// any loading progress bar starts instead of splitting it.
pub fn prelog_moe_backend(
    device: Device,
    dtype: DType,
    loading_isq: bool,
    quantization_config: &Option<QuantizedConfig>,
    act: Activation,
) -> Result<()> {
    MoEExpertsBackend::resolve(&BackendChoice::new(
        device,
        dtype,
        loading_isq,
        quantization_config,
        act,
    ))?;
    Ok(())
}

#[cfg(feature = "cuda")]
pub(super) fn gated_act(act: Activation) -> Result<mistralrs_quant::moe::cuda::GatedAct> {
    match act {
        Activation::Silu => Ok(mistralrs_quant::moe::cuda::GatedAct::Silu),
        Activation::NewGelu | Activation::GeluPytorchTanh => {
            Ok(mistralrs_quant::moe::cuda::GatedAct::GeluTanh)
        }
        _ => candle_core::bail!("activation {act:?} is not supported by grouped MoE kernels"),
    }
}

#[cfg(feature = "cuda")]
fn validate_raw_weights(c: &BackendChoice, backend: &str) -> Result<()> {
    if c.quantized || c.loading_isq || c.immediate_isq {
        candle_core::bail!(
            "MISTRALRS_MOE_BACKEND={backend} requires raw, unquantized expert weights"
        );
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_fused(c: &BackendChoice) -> Result<()> {
    validate_raw_weights(c, "fused")?;
    if !matches!(c.dtype, DType::F16 | DType::BF16) {
        candle_core::bail!("MISTRALRS_MOE_BACKEND=fused requires F16 or BF16 weights");
    }
    if !c.device.is_cuda() {
        candle_core::bail!("MISTRALRS_MOE_BACKEND=fused requires a CUDA device");
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_grouped(c: &BackendChoice, backend: &str) -> Result<()> {
    validate_raw_weights(c, backend)?;
    if c.dtype != DType::BF16 {
        candle_core::bail!("MISTRALRS_MOE_BACKEND={backend} requires BF16 weights");
    }
    gated_act(c.act)?;
    if !c.device.is_cuda() {
        candle_core::bail!("MISTRALRS_MOE_BACKEND={backend} requires a CUDA device");
    }
    Ok(())
}

impl MoEExpertsBackend {
    fn from_env() -> Option<Self> {
        let force = std::env::var("MISTRALRS_MOE_BACKEND").ok()?;
        Some(match force.as_str() {
            "fused" | "native" | "legacy" | "wmma" => Self::Fused,
            "fast" => Self::Fast,
            #[cfg(feature = "cutile")]
            "cutile" => Self::Cutile,
            #[cfg(feature = "cuda")]
            "cutlass" => Self::Cutlass,
            _ => return None,
        })
    }

    /// Single source of truth for the backend: env override, then raw CUDA kernels (cuTile when
    /// eligible, else Fused) for unquantized bf16, else gather-based Fast.
    pub(super) fn resolve(c: &BackendChoice) -> Result<Self> {
        if let Some(forced) = Self::from_env() {
            match forced {
                Self::Fast => return Ok(Self::Fast),
                Self::Fused => {
                    #[cfg(feature = "cuda")]
                    {
                        validate_fused(c)?;
                        return Ok(Self::Fused);
                    }
                    #[cfg(not(feature = "cuda"))]
                    candle_core::bail!("MISTRALRS_MOE_BACKEND=fused requires a CUDA build");
                }
                #[cfg(feature = "cutile")]
                Self::Cutile => {
                    validate_grouped(c, "cutile")?;
                    if !cutile_arch_supported(&c.device) {
                        candle_core::bail!(
                            "MISTRALRS_MOE_BACKEND=cutile is unsupported by this CUDA/GPU pair"
                        );
                    }
                    if !cutile_jit_available(&c.device) {
                        candle_core::bail!(
                            "MISTRALRS_MOE_BACKEND=cutile requires tileiras support for this GPU"
                        );
                    }
                    return Ok(Self::Cutile);
                }
                #[cfg(feature = "cuda")]
                Self::Cutlass => {
                    validate_grouped(c, "cutlass")?;
                    if !cutlass_moe_supported(&c.device) {
                        candle_core::bail!(
                            "MISTRALRS_MOE_BACKEND=cutlass is unsupported by this CUDA/GPU pair"
                        );
                    }
                    return Ok(Self::Cutlass);
                }
            }
        }
        if c.device.is_cuda()
            && !c.quantized
            && !c.loading_isq
            && !c.immediate_isq
            && matches!(c.dtype, DType::F16 | DType::BF16)
        {
            #[cfg(feature = "cuda")]
            let bf16_gated = c.dtype == DType::BF16 && gated_act(c.act).is_ok();
            #[cfg(feature = "cutile")]
            if bf16_gated && cutile_arch_supported(&c.device) {
                if cutile_jit_available(&c.device) {
                    return Ok(Self::Cutile);
                }
                once_log_info(
                    "cuTile JIT assembler is unavailable for this GPU; using CUTLASS MoE kernels",
                );
            }
            #[cfg(feature = "cuda")]
            if bf16_gated && cutlass_moe_supported(&c.device) {
                once_log_info("MoE experts backend: CUTLASS grouped GEMM");
                return Ok(Self::Cutlass);
            }
            return Ok(Self::Fused);
        }
        Ok(Self::Fast)
    }
}

/// cuTile only on build CUDA and GPU pairs supported by the pinned compiler.
#[cfg(feature = "cutile")]
fn cutile_arch_supported(device: &Device) -> bool {
    match device {
        Device::Cuda(dev) => mistralrs_quant::cutile::device_supported(dev),
        _ => false,
    }
}

#[cfg(feature = "cutile")]
fn cutile_jit_available(device: &Device) -> bool {
    match device {
        Device::Cuda(dev) => mistralrs_quant::cutile::jit_available(dev),
        _ => false,
    }
}

#[cfg(feature = "cuda")]
fn cutlass_moe_supported(device: &Device) -> bool {
    match device {
        Device::Cuda(dev) => mistralrs_quant::moe::cutlass_moe_available(dev),
        _ => false,
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    fn choice(dtype: DType, act: Activation) -> BackendChoice {
        BackendChoice {
            device: Device::Cpu,
            dtype,
            loading_isq: false,
            quantized: false,
            immediate_isq: false,
            act,
        }
    }

    #[test]
    fn grouped_activation_mapping_is_exhaustive() {
        assert_eq!(
            gated_act(Activation::Silu).unwrap(),
            mistralrs_quant::moe::cuda::GatedAct::Silu
        );
        assert_eq!(
            gated_act(Activation::NewGelu).unwrap(),
            mistralrs_quant::moe::cuda::GatedAct::GeluTanh
        );
        assert_eq!(
            gated_act(Activation::GeluPytorchTanh).unwrap(),
            mistralrs_quant::moe::cuda::GatedAct::GeluTanh
        );
        assert!(gated_act(Activation::Gelu).is_err());
        assert!(gated_act(Activation::Relu).is_err());
    }

    #[test]
    fn forced_grouped_validation_rejects_ineligible_inputs() {
        let error = validate_grouped(&choice(DType::BF16, Activation::Silu), "cutlass")
            .unwrap_err()
            .to_string();
        assert!(error.contains("CUDA device"));

        let error = validate_grouped(&choice(DType::F16, Activation::Silu), "cutlass")
            .unwrap_err()
            .to_string();
        assert!(error.contains("BF16"));

        let mut quantized = choice(DType::BF16, Activation::Silu);
        quantized.quantized = true;
        let error = validate_grouped(&quantized, "cutlass")
            .unwrap_err()
            .to_string();
        assert!(error.contains("raw, unquantized"));
    }
}
