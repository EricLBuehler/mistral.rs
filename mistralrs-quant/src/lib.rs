use std::{
    borrow::Cow,
    fmt::{Debug, Display},
    num::NonZeroUsize,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
};

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};

#[cfg(feature = "metal")]
mod metal_kernels;

mod bitsandbytes;
mod cublaslt;
mod dummy;
mod fp8;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
mod unquantized;
mod utils;

pub use bitsandbytes::{BnbLinear, BnbQuantParmas, BnbQuantType};
pub use dummy::DummyLayer;
use fp8::fp8_linear_b;
pub use fp8::FP8Linear;
pub use gguf::GgufMatMul;
use gptq::gptq_linear;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::ImatrixLayerStats;
pub use unquantized::UnquantLinear;
pub use utils::UQFF_QUANT_TYPE_OFFSET;

use candle_nn::{Linear, Module, VarBuilder};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub enum QuantMethodType {
    #[serde(rename = "fp8")]
    Fp8,
    #[serde(rename = "gptq")]
    Gptq,
    #[serde(rename = "unreachable")]
    Unreachable,
    #[default]
    #[serde(rename = "bitsandbytes")]
    Bitsandbytes,
}

impl Display for QuantMethodType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gptq => write!(f, "gptq"),
            Self::Fp8 => write!(f, "fp8"),
            Self::Bitsandbytes => write!(f, "bnb"),
            Self::Unreachable => write!(f, "unreachable",),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct QuantizedConfig {
    // GPTQ
    pub bits: Option<usize>,
    pub group_size: Option<usize>,
    pub checkpoint_format: Option<String>,

    // BNB
    pub bnb_4bit_quant_type: Option<String>,

    // FP8
    pub weight_block_size: Option<Vec<usize>>,

    pub quant_method: QuantMethodType,
}

impl QuantizedConfig {
    pub fn get_bits_name(&self, _vb: &VarBuilder) -> String {
        match self.bits {
            Some(bits) => format!("{bits} bits"),
            None => {
                // Assume bnb
                self.bnb_4bit_quant_type
                    .clone()
                    .unwrap_or("int8".to_string())
            }
        }
    }
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
    Bnb {
        weight: Tensor,
        bias: Option<Tensor>,
        params: BnbQuantParmas,
        quant_ty: BnbQuantType,
    },
}

/// Device/configurable intelligent matrix multiplication
/// - Configurable to be via f16 (to use the faster GEMM kernels) optionally.
/// - Handles limitation of `accelerate` which requires f32
pub struct MatMul;

pub static INHIBIT_GEMM_F16: AtomicBool = AtomicBool::new(false);

/// Set the matmuls to go via f16
pub(crate) static USE_MATMUL_VIA_F16: AtomicBool = AtomicBool::new(false);

pub fn set_use_matmul_via_f16(via_f16: bool) {
    if !INHIBIT_GEMM_F16.load(Ordering::Relaxed) {
        USE_MATMUL_VIA_F16.store(via_f16, Ordering::Relaxed)
    }
}
pub fn get_use_matmul_via_f16() -> bool {
    USE_MATMUL_VIA_F16.load(Ordering::Relaxed)
}

impl MatMul {
    /// Compute matrix-matrix product, optionally casting to f16 to use specialized GEMM kernels.
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        #[cfg(feature = "accelerate")]
        {
            let original_dtype = a.dtype();
            a.to_dtype(DType::F32)?
                .matmul(&b.to_dtype(DType::F32)?)?
                .to_dtype(original_dtype)
        }
        #[cfg(not(feature = "accelerate"))]
        {
            if a.device().is_cpu() {
                let original_dtype = a.dtype();
                return a
                    .to_dtype(DType::F16)?
                    .matmul(&b.to_dtype(DType::F16)?)?
                    .to_dtype(original_dtype);
            } else if !get_use_matmul_via_f16() {
                return a.matmul(b);
            }
            let original_dtype = a.dtype();
            a.to_dtype(DType::F16)?
                .matmul(&b.to_dtype(DType::F16)?)?
                .to_dtype(original_dtype)
        }
    }

    /// Compute matrix-matrix product, optionally casting to f16 to use specialized GEMM kernels.
    /// The result will be divided by the `scale` parameter in an affine division.
    pub fn matmul_affine_div(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? / scale
    }

    /// Compute matrix-matrix product, optionally casting to f16 to use specialized GEMM kernels.
    /// The result will be divided by the `scale` parameter in an affine multiplication.
    pub fn matmul_affine_mul(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? * scale
    }

    /// Compute quantized matrix-matrix product, optionally casting to f16 to use specialized GEMM kernels.
    pub fn qmatmul(&self, x: &Tensor, matmul: &QMatMul) -> Result<Tensor> {
        if get_use_matmul_via_f16() {
            matmul.forward_via_f16(x)
        } else {
            matmul.forward(x)
        }
    }

    /// Compute quantized matrix-matrix product, optionally casting to f16 to use specialized GEMM kernels.
    pub fn qmethod_matmul(&self, x: &Tensor, matmul: &dyn QuantMethod) -> Result<Tensor> {
        if get_use_matmul_via_f16() {
            matmul.forward_via_half(x)
        } else {
            matmul.forward(x)
        }
    }
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

impl IsqType {
    /// Factor by which the weight size is reduced over the given dtype.
    /// original size / pack factor = quantized size
    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::Q4_0 => {
                (dtype.size_in_bytes() * GgmlDType::Q4_0.block_size()) / GgmlDType::Q4_0.type_size()
            }
            Self::Q4_1 => {
                (dtype.size_in_bytes() * GgmlDType::Q4_1.block_size()) / GgmlDType::Q4_1.type_size()
            }
            Self::Q5_0 => {
                (dtype.size_in_bytes() * GgmlDType::Q5_0.block_size()) / GgmlDType::Q5_0.type_size()
            }
            Self::Q5_1 => {
                (dtype.size_in_bytes() * GgmlDType::Q5_1.block_size()) / GgmlDType::Q5_1.type_size()
            }
            Self::Q8_0 => {
                (dtype.size_in_bytes() * GgmlDType::Q8_0.block_size()) / GgmlDType::Q8_0.type_size()
            }
            Self::Q8_1 => {
                (dtype.size_in_bytes() * GgmlDType::Q8_1.block_size()) / GgmlDType::Q8_1.type_size()
            }
            Self::Q2K => {
                (dtype.size_in_bytes() * GgmlDType::Q2K.block_size()) / GgmlDType::Q2K.type_size()
            }
            Self::Q3K => {
                (dtype.size_in_bytes() * GgmlDType::Q3K.block_size()) / GgmlDType::Q3K.type_size()
            }
            Self::Q4K => {
                (dtype.size_in_bytes() * GgmlDType::Q4K.block_size()) / GgmlDType::Q4K.type_size()
            }
            Self::Q5K => {
                (dtype.size_in_bytes() * GgmlDType::Q5K.block_size()) / GgmlDType::Q5K.type_size()
            }
            Self::Q6K => {
                (dtype.size_in_bytes() * GgmlDType::Q6K.block_size()) / GgmlDType::Q6K.type_size()
            }
            Self::Q8K => {
                (dtype.size_in_bytes() * GgmlDType::Q8K.block_size()) / GgmlDType::Q8K.type_size()
            }
            // Estimates
            Self::HQQ4 => 4,
            Self::HQQ8 => 2,
            Self::F8E4M3 => 2,
        }
    }
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

impl TryFrom<GgmlDType> for IsqType {
    type Error = candle_core::Error;

    fn try_from(value: GgmlDType) -> Result<Self> {
        match value {
            GgmlDType::Q2K => Ok(Self::Q2K),
            GgmlDType::Q3K => Ok(Self::Q3K),
            GgmlDType::Q4K => Ok(Self::Q4K),
            GgmlDType::Q5K => Ok(Self::Q5K),
            GgmlDType::Q6K => Ok(Self::Q6K),
            GgmlDType::Q4_0 => Ok(Self::Q4_0),
            GgmlDType::Q4_1 => Ok(Self::Q4_1),
            GgmlDType::Q5_0 => Ok(Self::Q5_0),
            GgmlDType::Q5_1 => Ok(Self::Q5_1),
            GgmlDType::Q8_0 => Ok(Self::Q8_0),
            GgmlDType::Q8_1 => Ok(Self::Q8_1),
            GgmlDType::Q8K => Ok(Self::Q8K),
            GgmlDType::BF16 | GgmlDType::F32 | GgmlDType::F16 => {
                candle_core::bail!("Expected valid GGML ISQ type.")
            }
        }
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

    fn dequantize_w(&self) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically cast to required quantization actiation type and back
    fn forward_autocast(&self, a: &Tensor) -> Result<Tensor> {
        let original_ty = a.dtype();
        let a = if let Some(t) = self.quantized_act_type() {
            a.to_dtype(t)?
        } else {
            a.clone()
        };
        self.forward(&a)?.to_dtype(original_ty)
    }

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
        imatrix_weight: Option<Vec<f32>>,
    ) -> Result<Arc<dyn QuantMethod>>;

    /// Convert to an equivalent gguf quantization, if applicable.
    fn maybe_to_gguf_quant(self: Arc<Self>) -> Result<Arc<dyn QuantMethod>>;

    /// If the quant is backed by a qmatmul.
    fn get_bias_mut(&mut self) -> Option<&mut Tensor>;

    fn get_max_isq_cpu_threads(&self, dtype: IsqType) -> Option<NonZeroUsize>;

    fn unquant_weight_bias(&self) -> Option<(Tensor, Option<Tensor>)> {
        None
    }

    /// Begin tracking stats into an ImatrixLayerStats
    fn begin_track_stats(&mut self) -> Result<()> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// End tracking stats into an ImatrixLayerStats. Returns the computed imatrix.
    fn end_track_stats(&self) -> Result<Tensor> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }
}

impl Module for dyn QuantMethod {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        Self::forward(self, xs)
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
            QuantMethodType::Fp8 => fp8_linear_b(in_dim, out_dim, quant_conf, false, vb)?,
            QuantMethodType::Bitsandbytes => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
            }
            QuantMethodType::Unreachable => unreachable!(),
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
            QuantMethodType::Fp8 => fp8_linear_b(in_dim, out_dim, quant_conf, true, vb)?,
            QuantMethodType::Bitsandbytes => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
            }
            QuantMethodType::Unreachable => unreachable!(),
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
