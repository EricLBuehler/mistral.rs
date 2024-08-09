use std::sync::Arc;

use candle_core::{quantized::QTensor, DType, Result, Tensor};

mod gguf;
mod gptq;
mod unquantized;

pub use gguf::GgufMatMul;
pub use gptq::GptqMatMul;
pub use unquantized::UnquantLinear;

use candle_nn::{Linear, VarBuilder};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize, Default)]
pub enum QuantMethodType {
    #[default]
    #[serde(rename = "gptq")]
    Gptq,
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
}

macro_rules! pack_factor {
    ($bits:expr) => {
        32 / $bits
    };
}

pub fn gptq_linear_no_bias(
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
    Ok(Arc::new(GptqMatMul::new(config)?))
}
