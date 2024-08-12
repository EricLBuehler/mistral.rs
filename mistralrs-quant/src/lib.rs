use std::{fmt::Display, sync::Arc};

use candle_core::{
    quantized::{QMatMul, QTensor},
    DType, Device, Result, Tensor,
};

mod gguf;
mod gptq;
mod hqq;
mod unquantized;
mod utils;

pub use gguf::GgufMatMul;
pub use gptq::GptqLayer;
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
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync {
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
    fn get_qmatmul(&mut self) -> Option<&mut QMatMul>;

    /// If the quant is backed by a qmatmul.
    fn get_bias_mut(&mut self) -> Option<&mut Tensor>;

    /// Convert this layer to an ISQ-able layer if possible.
    fn convert_to_isq(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>>;
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
