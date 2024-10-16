use std::{
    borrow::Cow,
    fmt::{Debug, Display},
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Tensor,
};

mod cublaslt;
mod dummy;
mod fp8;
mod gguf;
mod gptq;
mod hqq;
mod unquantized;
mod utils;

pub use dummy::DummyLayer;
pub use fp8::FP8Linear;
pub use gguf::GgufMatMul;
use gptq::gptq_linear;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use unquantized::UnquantLinear;

use candle_nn::{Linear, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum QuantMethodType {
    #[default]
    #[serde(rename = "gptq")]
    Gptq,
}

impl Display for QuantMethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gptq => write!(f, "GPTQ"),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct QuantizedConfig {
    pub bits: usize,
    pub quant_method: QuantMethodType,
    pub group_size: usize,
    pub checkpoint_format: Option<String>,
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    Gptq {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        gptq_qzeros: Option<Tensor>,
        gptq_scales: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        workspace: Option<Tensor>,
        is_marlin: bool,
    },
    Gguf {
        q_weight: Arc<QTensor>,
        b: Option<Tensor>,
    },
    Unquantized(Linear),
    Hqq {
        tensor: Tensor,
        bits: HqqBits,
        group_size: NonZeroUsize,
        axis: HqqAxis,
        optimization_steps: Option<usize>,
        round_zeros: Option<bool>,
        channel_wise: Option<bool>,
        bias: Option<Tensor>,
    },
    Dummy,
    FP8 {
        lin: Linear,
        dtype: DType,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum IsqType {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    HQQ8,
    HQQ4,
    // HQQ3,
    // HQQ2,
    // HQQ1,
    F8E4M3,
}

impl TryFrom<IsqType> for GgmlDType {
    type Error = candle_core::Error;

    fn try_from(value: IsqType) -> Result<Self> {
        let tp = match value {
            IsqType::Q2K => Self::Q2K,
            IsqType::Q3K => Self::Q3K,
            IsqType::Q4K => Self::Q4K,
            IsqType::Q4_0 => Self::Q4_0,
            IsqType::Q4_1 => Self::Q4_1,
            IsqType::Q5K => Self::Q5K,
            IsqType::Q5_0 => Self::Q5_0,
            IsqType::Q5_1 => Self::Q5_1,
            IsqType::Q6K => Self::Q6K,
            IsqType::Q8K => Self::Q8K,
            IsqType::Q8_0 => Self::Q8_0,
            IsqType::Q8_1 => Self::Q8_1,
            _ => candle_core::bail!("Expected valid GGML ISQ type."),
        };
        #[cfg(feature = "cuda")]
        {
            if !matches!(
                tp,
                GgmlDType::Q4_0
                    | GgmlDType::Q4_1
                    | GgmlDType::Q5_0
                    | GgmlDType::Q5_1
                    | GgmlDType::Q8_0
                    | GgmlDType::Q2K
                    | GgmlDType::Q3K
                    | GgmlDType::Q4K
                    | GgmlDType::Q5K
                    | GgmlDType::Q6K
            ) {
                candle_core::bail!("GGML ISQ type on CUDA must be one of `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q2K`, `Q3K`, `Q4K`, `Q5K`, `Q6K`, `HQQ8`, `HQQ4`")
            }
        }
        Ok(tp)
    }
}

pub enum QuantizedSerdeType {
    Gguf = 0,
    Unquant = 1,
    Hqq = 2,
    Fp8 = 3,
}

impl TryFrom<usize> for QuantizedSerdeType {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gguf),
            1 => Ok(Self::Unquant),
            2 => Ok(Self::Hqq),
            3 => Ok(Self::Fp8),
            other => candle_core::bail!("QuantizedSerdeType {other} is invalid."),
        }
    }
}

pub trait QuantizedSerde {
    fn name(&self) -> &'static str;
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn serialize(&self) -> Result<Cow<[u8]>> {
        candle_core::bail!("`QuantizedSerde::serialize` is not supported.")
    }
    fn deserialize(_data: Cow<[u8]>, _device: &Device) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!("`QuantizedSerde::deserialize` is not supported.")
    }
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    fn forward(&self, a: &Tensor) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// This may go via half precision if it is supported.
    fn forward_via_half(&self, a: &Tensor) -> Result<Tensor> {
        self.forward(a)
    }

    /// If a quantized method, return the activation dtype.
    fn quantized_act_type(&self) -> Option<DType>;

    /// Weight dtype and device
    fn dtype_and_device(&self) -> (DType, Device);

    /// Add a delta weight from LoRA to the weights. This should be prescaled with alpha.
    fn add_delta_w(&self, delta: &Tensor) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn apply_isq(
        self: Arc<Self>,
        dtype: Option<IsqType>,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn get_bias_mut(&mut self) -> Option<&mut Tensor>;

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize>;

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        None
    }
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let layer = if let Some(quant_conf) = &config {
        match quant_conf.quant_method {
            QuantMethodType::Gptq => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let layer = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    Ok(layer)
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let layer = if let Some(quant_conf) = &config {
        match quant_conf.quant_method {
            QuantMethodType::Gptq => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !(vb.contains_tensor("weight") && vb.contains_tensor("bias")) {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let layer = candle_nn::linear(in_dim, out_dim, vb)?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    Ok(layer)
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: &Option<QuantizedConfig>,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}
