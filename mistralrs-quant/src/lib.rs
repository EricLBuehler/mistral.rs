use std::{
    fmt::{Debug, Display},
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc},
};

use candle_core::{
    quantized::{GgmlDType, QTensor},
    DType, Device, Result, Tensor,
};

mod gguf;
mod gptq;
mod hqq;
mod unquantized;
mod utils;

pub use gguf::GgufMatMul;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use unquantized::UnquantLinear;

use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
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

#[derive(Debug, Clone, Deserialize, Default)]
pub struct QuantizedConfig {
    pub bits: usize,
    pub quant_method: QuantMethodType,
    pub group_size: usize,
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    Gptq {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        gptq_qzeros: Tensor,
        gptq_scales: Tensor,
        g_idx: Tensor,
        bias: Tensor,
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
}

#[derive(Clone, Copy, Debug, PartialEq)]
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
}

impl TryFrom<IsqType> for GgmlDType {
    type Error = candle_core::Error;

    fn try_from(value: IsqType) -> Result<Self> {
        match value {
            IsqType::Q2K => Ok(Self::Q2K),
            IsqType::Q3K => Ok(Self::Q3K),
            IsqType::Q4K => Ok(Self::Q4K),
            IsqType::Q4_0 => Ok(Self::Q4_0),
            IsqType::Q4_1 => Ok(Self::Q4_1),
            IsqType::Q5K => Ok(Self::Q5K),
            IsqType::Q5_0 => Ok(Self::Q5_0),
            IsqType::Q5_1 => Ok(Self::Q5_1),
            IsqType::Q6K => Ok(Self::Q6K),
            IsqType::Q8K => Ok(Self::Q8K),
            IsqType::Q8_0 => Ok(Self::Q8_0),
            IsqType::Q8_1 => Ok(Self::Q8_1),
            _ => candle_core::bail!("Expected valid GGML ISQ type."),
        }
    }
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug {
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
        dtype: IsqType,
        device: Device,
        n_quantized: &AtomicUsize,
    ) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn get_bias_mut(&mut self) -> Option<&mut Tensor>;

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize>;
}

macro_rules! pack_factor {
    ($bits:expr) => {
        32 / $bits
    };
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
        let layer = candle_nn::linear_no_bias(in_dim, out_dim, vb)?;

        let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
        Arc::new(layer) as Arc<dyn QuantMethod>
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
        let layer = candle_nn::linear(in_dim, out_dim, vb)?;

        let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(layer))?;
        Arc::new(layer) as Arc<dyn QuantMethod>
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

pub fn gptq_linear(
    in_dim: usize,
    out_dim: usize,
    config: &QuantizedConfig,
    vb: VarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let qweight = vb.get_with_hints_dtype(
        (in_dim / pack_factor!(config.bits), out_dim),
        "qweight",
        Default::default(),
        DType::I32,
    )?;
    let scale_and_zero_size = in_dim / config.group_size;
    let qzeros = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim / pack_factor!(config.bits)),
        "qzeros",
        Default::default(),
        DType::I32,
    )?;
    let g_idx = vb.get_with_hints_dtype((in_dim,), "g_idx", Default::default(), DType::I32)?;
    let scales = vb.get_with_hints_dtype(
        (scale_and_zero_size, out_dim),
        "scales",
        Default::default(),
        DType::F16,
    )?;
    let bias = vb.get_with_hints_dtype((out_dim,), "bias", Default::default(), DType::F16)?;

    let config = QuantMethodConfig::Gptq {
        bits: config.bits as i32,
        use_exllama: false,
        q_weight: qweight,
        gptq_qzeros: qzeros,
        gptq_scales: scales,
        g_idx,
        bias,
    };
    Ok(Arc::new(GptqLayer::new(config)?))
}
