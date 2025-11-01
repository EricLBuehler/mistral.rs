use std::{
    borrow::Cow,
    fmt::Debug,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard},
};

use blockwise_fp8::blockwise_fp8_linear_b;
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};

#[cfg(feature = "metal")]
mod metal_kernels;

mod afq;
mod bitsandbytes;
mod blockwise_fp8;
pub mod cublaslt;
pub mod distributed;
mod dummy;
mod fp8;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
mod lora;
mod mxfp4;
pub mod rotary;
pub mod safetensors;
mod scalar_fp8;
mod unquantized;
mod utils;
mod vector_fp8;

use gptq::gptq_linear;
use lora::merge_lora_weights;
use regex::Regex;
pub use safetensors::{Shard, ShardedSafeTensors, ShardedVarBuilder};

pub use afq::{AfqBits, AfqGroupSize, AfqLayer};
pub use bitsandbytes::{BnbLinear, BnbQuantParams, BnbQuantType};
pub use blockwise_fp8::{fp8_blockwise_dequantize, fp8_blockwise_quantize};
pub use distributed::{
    layers::{
        compute_kv_shard, compute_n_kv_groups, ColumnParallelLayer, FusedExperts, PackedExperts,
        ReplicatedLayer, RowParallelLayer,
    },
    socket::{Client, Server},
    BarrierLike, Comm, Id, RingConfig, SumAllReduce,
};
pub use dummy::DummyLayer;
pub use fp8::FP8Linear;
pub use gguf::GgufMatMul;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::{CollectedImatrixData, ImatrixLayerStats};
pub use lora::{
    clear_applied_loras, get_applied_loras, linear_no_bias_static_lora, push_applied_lora,
    LoraAdapter, LoraConfig, StaticLoraConfig, MULTI_LORA_DELIMITER,
};
pub use mxfp4::MXFP4Layer;
pub use unquantized::UnquantLinear;
pub use utils::isq::apply_immediate_isq;
pub use utils::{log, BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp, UQFF_QUANT_TYPE_OFFSET};
pub use vector_fp8::{fp8_vector_dequantize, fp8_vector_quantize};

use candle_nn::{Conv1d, Conv2d, Linear, Module};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug)]
pub struct ImmediateIsqParams {
    pub guard: QuantizeOntoGuard,
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqOverride {
    pub predicate: Regex,
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqMatch {
    pub ty: IsqType,
    pub device: Option<Device>,
}

thread_local! {
    static ENGINE_IMMEDIATE_ISQ: std::cell::RefCell<Option<ImmediateIsqParams>> = const { std::cell::RefCell::new(None) } ;
}

pub fn set_immediate_isq(isq: Option<IsqType>, predicates: Vec<Regex>) {
    set_immediate_isq_with_overrides(isq, predicates, Vec::new());
}

pub fn set_immediate_isq_with_overrides(
    isq: Option<IsqType>,
    predicates: Vec<Regex>,
    overrides: Vec<ImmediateIsqOverride>,
) {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = Some(ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty: isq,
            predicates,
            overrides,
        });
    });
}

pub fn get_immediate_isq() -> Option<ImmediateIsqParams> {
    ENGINE_IMMEDIATE_ISQ.with(|cell| cell.borrow().clone())
}

pub fn clear_immediate_isq() {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

pub fn should_apply_immediate_isq(vb: &ShardedVarBuilder) -> bool {
    immediate_isq_match(vb).is_some()
}

pub fn immediate_isq_match(vb: &ShardedVarBuilder) -> Option<ImmediateIsqMatch> {
    let immediate_isq = get_immediate_isq()?;
    // Add a .weight to match the ISQ regexes!
    let prefix = format!("{}.weight", vb.prefix());
    resolve_immediate_isq(&immediate_isq, &prefix)
}

fn resolve_immediate_isq(params: &ImmediateIsqParams, prefix: &str) -> Option<ImmediateIsqMatch> {
    if let Some(override_hit) = params
        .overrides
        .iter()
        .find(|override_pred| override_pred.predicate.is_match(prefix))
    {
        if let Some(ty) = override_hit.ty.or(params.ty) {
            return Some(ImmediateIsqMatch {
                ty,
                device: override_hit.device.clone(),
            });
        }
        return None;
    }

    if let Some(ty) = params.ty {
        if params
            .predicates
            .iter()
            .any(|predicate| predicate.is_match(prefix))
        {
            return Some(ImmediateIsqMatch { ty, device: None });
        }
    }

    None
}

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "quant_method", rename_all = "lowercase")]
pub enum QuantizedConfig {
    GptqAwq {
        bits: usize,
        group_size: usize,
        checkpoint_format: Option<String>,
        is_awq: bool,
    },
    Fp8 {
        weight_block_size: Vec<usize>,
    },
    Bitsandbytes {
        bnb_4bit_quant_type: Option<String>,
    },
    Afq {
        bits: usize,
        group_size: usize,
    },
    MXFP4 {},
}

// Common fields for all variants
#[derive(Deserialize)]
struct RawConfig {
    quant_method: Option<String>,
    bits: Option<usize>,
    group_size: Option<usize>,
    checkpoint_format: Option<String>,
    weight_block_size: Option<Vec<usize>>,
    bnb_4bit_quant_type: Option<String>,
}

// Custom deserializer implementation
impl<'de> Deserialize<'de> for QuantizedConfig {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = RawConfig::deserialize(deserializer)?;

        match &raw.quant_method {
            Some(m) if m == "gptq" || m == "awq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::GptqAwq {
                    bits,
                    group_size,
                    checkpoint_format: raw.checkpoint_format,
                    is_awq: m == "awq",
                })
            }
            Some(m) if m == "fp8" => {
                let weight_block_size = raw
                    .weight_block_size
                    .ok_or_else(|| serde::de::Error::missing_field("weight_block_size"))?;
                Ok(QuantizedConfig::Fp8 { weight_block_size })
            }
            Some(m) if m == "bitsandbytes" => Ok(QuantizedConfig::Bitsandbytes {
                bnb_4bit_quant_type: raw.bnb_4bit_quant_type,
            }),
            Some(m) if m == "afq" => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(m) if m == "mxfp4" => {
                Ok(QuantizedConfig::MXFP4 {  })
            }
            None => {
                let bits = raw
                    .bits
                    .ok_or_else(|| serde::de::Error::missing_field("bits"))?;
                let group_size = raw
                    .group_size
                    .ok_or_else(|| serde::de::Error::missing_field("group_size"))?;
                Ok(QuantizedConfig::Afq { bits, group_size })
            }
            Some(unknown_method) => {
                Err(serde::de::Error::custom(format!(
                    "Unknown quantization method: {unknown_method}. Expected one of: gptq, fp8, bitsandbytes, afq, or not specified"
                )))
            },
        }
    }
}

impl QuantizedConfig {
    pub fn name(&self) -> &'static str {
        match self {
            Self::GptqAwq { .. } => "gptq",
            Self::Fp8 { .. } => "fp8",
            Self::Bitsandbytes { .. } => "bitsandbytes",
            Self::Afq { .. } => "afq",
            Self::MXFP4 { .. } => "mxfp4",
        }
    }

    pub fn get_bits_name(&self, _vb: &ShardedVarBuilder) -> String {
        match self {
            Self::GptqAwq { bits, .. } => format!("{bits} bits"),
            Self::Fp8 { .. } => "8 bits".to_string(),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            } => "4 bits".to_string(),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => "8 bits".to_string(),
            Self::Afq { bits, .. } => format!("{bits} bits"),
            Self::MXFP4 {} => format!("{} bits", mxfp4::N_BITS),
        }
    }

    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::GptqAwq { bits, .. } | Self::Afq { bits, .. } => match bits {
                2 => IsqType::Q2K.pack_factor(dtype),
                3 => IsqType::Q3K.pack_factor(dtype),
                4 => IsqType::Q4K.pack_factor(dtype),
                5 => IsqType::Q5K.pack_factor(dtype),
                6 => IsqType::Q6K.pack_factor(dtype),
                8 => IsqType::Q8_0.pack_factor(dtype),
                40 => 4, // mxfp4: 2 FP4 values per byte = factor of 4
                other => panic!("Unexpected bits in `pack_factor` {other}"),
            },
            Self::Fp8 { .. } => IsqType::Q8_0.pack_factor(dtype),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            }
            | Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => IsqType::Q4K.pack_factor(dtype),
            Self::MXFP4 {} => IsqType::Q4_0.pack_factor(dtype),
        }
    }
}

#[derive(Debug, Clone)]
pub enum QuantMethodConfig {
    GptqAwq {
        bits: i32,
        use_exllama: bool,
        q_weight: Tensor,
        qzeros: Option<Tensor>,
        scales: Tensor,
        g_idx: Option<Tensor>,
        bias: Option<Tensor>,
        workspace: Option<Tensor>,
        is_marlin: bool,
        is_awq: bool,
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
        params: BnbQuantParams,
        quant_ty: BnbQuantType,
    },
    BlockwiseFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        bias: Option<Tensor>,
        dequant_dtype: DType,
        weight_block_size: Vec<usize>,
    },
    Afq {
        weight: Tensor,
        bias: Option<Tensor>,
        bits: AfqBits,
        group_size: AfqGroupSize,
    },
    MXFP4 {
        blocks: Tensor,
        scales: Tensor,
        bias: Option<Tensor>,
    },
}

/// Device/configurable intelligent matrix multiplication
/// - Handles limitation of `accelerate` which requires f32
pub struct MatMul;

impl MatMul {
    /// Compute matrix-matrix product.
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
                a.to_dtype(DType::F16)?
                    .matmul(&b.to_dtype(DType::F16)?)?
                    .to_dtype(original_dtype)
            } else {
                a.matmul(b)
            }
        }
    }

    /// Compute matrix-matrix product.
    /// The result will be divided by the `scale` parameter in an affine division.
    pub fn matmul_affine_div(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? / scale
    }

    /// Compute matrix-matrix product.
    /// The result will be divided by the `scale` parameter in an affine multiplication.
    pub fn matmul_affine_mul(&self, a: &Tensor, b: &Tensor, scale: f64) -> Result<Tensor> {
        // TODO(EricLBuehler): Optimize this by using the gemm parameter?
        self.matmul(a, b)? * scale
    }

    /// Compute quantized matrix-matrix product.
    pub fn qmatmul(&self, x: &Tensor, matmul: &QMatMul) -> Result<Tensor> {
        matmul.forward(x)
    }

    /// Compute quantized matrix-matrix product.
    pub fn qmethod_matmul(&self, x: &Tensor, matmul: &dyn QuantMethod) -> Result<Tensor> {
        matmul.forward(x)
    }
}

/// Device/configurable intelligent convolution
/// - Handles limitation of cpu which requires f32
pub struct Convolution;

impl Convolution {
    pub fn forward_1d(&self, layer: &Conv1d, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            let original_dtype = x.dtype();
            Conv1d::new(
                layer.weight().to_dtype(DType::F32)?,
                layer.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                *layer.config(),
            )
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(original_dtype)
        } else {
            layer.forward(x)
        }
    }

    pub fn forward_2d(&self, layer: &Conv2d, x: &Tensor) -> Result<Tensor> {
        if x.device().is_cpu() {
            let original_dtype = x.dtype();
            Conv2d::new(
                layer.weight().to_dtype(DType::F32)?,
                layer.bias().map(|b| b.to_dtype(DType::F32)).transpose()?,
                *layer.config(),
            )
            .forward(&x.to_dtype(DType::F32)?)?
            .to_dtype(original_dtype)
        } else {
            layer.forward(x)
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq, Serialize, Deserialize)]
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
    AFQ8,
    AFQ6,
    AFQ4,
    AFQ3,
    AFQ2,
}

impl IsqType {
    /// Factor by which the weight size is reduced over the given dtype.
    /// original size / pack factor = quantized size
    pub fn pack_factor(&self, dtype: DType) -> usize {
        match self {
            Self::Q4_0 | Self::AFQ4 => (dtype.size_in_bytes() * GgmlDType::Q4_0.block_size())
                .div_ceil(GgmlDType::Q4_0.type_size()),
            Self::Q4_1 => (dtype.size_in_bytes() * GgmlDType::Q4_1.block_size())
                .div_ceil(GgmlDType::Q4_1.type_size()),
            Self::Q5_0 => (dtype.size_in_bytes() * GgmlDType::Q5_0.block_size())
                .div_ceil(GgmlDType::Q5_0.type_size()),
            Self::Q5_1 => (dtype.size_in_bytes() * GgmlDType::Q5_1.block_size())
                .div_ceil(GgmlDType::Q5_1.type_size()),
            Self::Q8_0 | Self::AFQ8 => (dtype.size_in_bytes() * GgmlDType::Q8_0.block_size())
                .div_ceil(GgmlDType::Q8_0.type_size()),
            Self::Q8_1 => (dtype.size_in_bytes() * GgmlDType::Q8_1.block_size())
                .div_ceil(GgmlDType::Q8_1.type_size()),
            Self::Q2K | Self::AFQ2 => (dtype.size_in_bytes() * GgmlDType::Q2K.block_size())
                .div_ceil(GgmlDType::Q2K.type_size()),
            Self::Q3K | Self::AFQ3 => (dtype.size_in_bytes() * GgmlDType::Q3K.block_size())
                .div_ceil(GgmlDType::Q3K.type_size()),
            Self::Q4K => (dtype.size_in_bytes() * GgmlDType::Q4K.block_size())
                .div_ceil(GgmlDType::Q4K.type_size()),
            Self::Q5K => (dtype.size_in_bytes() * GgmlDType::Q5K.block_size())
                .div_ceil(GgmlDType::Q5K.type_size()),
            Self::Q6K | Self::AFQ6 => (dtype.size_in_bytes() * GgmlDType::Q6K.block_size())
                .div_ceil(GgmlDType::Q6K.type_size()),
            Self::Q8K => (dtype.size_in_bytes() * GgmlDType::Q8K.block_size())
                .div_ceil(GgmlDType::Q8K.type_size()),
            // Estimates
            Self::HQQ4 => 4,
            Self::HQQ8 => 2,
            Self::F8E4M3 => 2,
        }
    }

    pub fn get_max_isq_cpu_threads(&self) -> Option<NonZeroUsize> {
        match self {
            /*IsqType::HQQ1 | IsqType::HQQ2 | IsqType::HQQ3 | */
            IsqType::HQQ4
            | IsqType::HQQ8
            | IsqType::AFQ2
            | IsqType::AFQ3
            | IsqType::AFQ4
            | IsqType::AFQ6
            | IsqType::AFQ8 => {
                // Use 1 because our HQQ quantizes on the GPU
                Some(1.try_into().unwrap())
            }
            IsqType::F8E4M3 => None,
            IsqType::Q2K
            | IsqType::Q3K
            | IsqType::Q4K
            | IsqType::Q4_0
            | IsqType::Q4_1
            | IsqType::Q5K
            | IsqType::Q5_0
            | IsqType::Q5_1
            | IsqType::Q6K
            | IsqType::Q8K
            | IsqType::Q8_0
            | IsqType::Q8_1 => None,
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

#[derive(Debug, Clone, Copy)]
pub enum QuantizedSerdeType {
    Gguf = 0,
    Unquant = 1,
    Hqq = 2,
    Fp8 = 3,
    Afq = 4,
}

impl TryFrom<usize> for QuantizedSerdeType {
    type Error = candle_core::Error;
    fn try_from(value: usize) -> std::result::Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Gguf),
            1 => Ok(Self::Unquant),
            2 => Ok(Self::Hqq),
            3 => Ok(Self::Fp8),
            4 => Ok(Self::Afq),
            other => candle_core::bail!("QuantizedSerdeType {other} is invalid."),
        }
    }
}

pub trait QuantizedSerde {
    fn name(&self) -> &'static str;
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn serialize(&self) -> Result<Cow<'_, [u8]>> {
        candle_core::bail!("`QuantizedSerde::serialize` is not supported.")
    }
    fn deserialize(
        _data: Cow<[u8]>,
        _device: &Device,
        _comm: &Arc<crate::Comm>,
        _guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!("`QuantizedSerde::deserialize` is not supported.")
    }
    fn deserialize_ext_bias(
        _data: Cow<[u8]>,
        _device: &Device,
        _guard: QuantizeOntoGuard,
    ) -> Result<(Arc<dyn QuantMethod>, Option<Tensor>)>
    where
        Self: Sized,
    {
        candle_core::bail!("`QuantizedSerde::deserialize_ext_bias` is not supported.")
    }
    /// NOT meant for external calling
    fn serialize_with_bias(&self, _bias: Option<Tensor>) -> Result<Cow<'_, [u8]>> {
        candle_core::bail!("`QuantizedSerde::serialize_with_bias` is not supported.")
    }
}

/// Used to gate access to quantizing onto the host device
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct QuantizeOntoGuard {
    pub inner: Arc<Mutex<()>>,
}

/// Real (for Metal) and Fake (for CUDA)
pub enum QuantizeOntoDropGuard<'a> {
    Real(MutexGuard<'a, ()>),
    Fake,
}

impl Default for QuantizeOntoGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizeOntoGuard {
    pub fn new() -> Self {
        QuantizeOntoGuard {
            inner: Arc::new(Mutex::new(())),
        }
    }

    /// Acquire the quantize drop guard to protect the critical section.
    ///
    /// On metal, this flushes the command buffer to avoid "A command encoder is already encoding to this command buffer"
    pub fn acquire(&self, device: &Device) -> QuantizeOntoDropGuard<'_> {
        #[cfg(feature = "cuda")]
        {
            let _ = device;
            QuantizeOntoDropGuard::Fake
        }

        #[cfg(not(feature = "cuda"))]
        {
            #[cfg(feature = "metal")]
            if let Device::Metal(dev) = device {
                // This is necessary to avoid the errors of "A command encoder is already encoding to this command buffer"
                dev.flush_command_buffer()
                    .expect("Failed to flush command buffer.");
            }
            #[cfg(not(feature = "metal"))]
            let _ = device;

            QuantizeOntoDropGuard::Real(self.inner.lock().expect("QuantizeOntoGuard was poisoned!"))
        }
    }
}

pub enum DistributedKind {
    ColumnParallel,
    RowParallel,
    Replicated,
}

/// Quantized method for a quantized matmul.
pub trait QuantMethod: Send + Sync + Debug + QuantizedSerde {
    fn new(method: QuantMethodConfig) -> Result<Self>
    where
        Self: Sized;

    fn dequantize_w(&self) -> Result<Tensor>;

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically cast to required quantization activation type and back
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
    /// Automatically cast to required quantization activation type and back.
    ///
    /// If `a` is (n_tokens, n_experts, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts).
    fn gather_forward_autocast(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        let original_ty = a.dtype();
        let a = if let Some(t) = self.quantized_act_type() {
            a.to_dtype(t)?
        } else {
            a.clone()
        };
        self.gather_forward(&a, indices)?.to_dtype(original_ty)
    }

    /// Compute matmul of `self` and `a`. `self` should contain the weights.
    ///
    /// If `a` is (n_tokens, n_experts, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts).
    fn gather_forward(&self, _a: &Tensor, _indices: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support `gather_forward`. Please raise an issue.",
            self.name()
        )
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
        guard: QuantizeOntoGuard,
    ) -> Result<Arc<dyn QuantMethod>>;

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

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
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
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { .. } => {
                blockwise_fp8_linear_b(in_dim, out_dim, quant_conf, false, Default::default(), vb)?
            }
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, false, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, false, vb)?
            }
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !vb.contains_tensor("weight") {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    apply_immediate_isq(layer, base_vb)
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    let vb = if should_apply_immediate_isq(&vb) {
        vb.set_device(Device::Cpu)
    } else {
        vb
    };

    let layer = if let Some(quant_conf) = &config {
        match quant_conf {
            QuantizedConfig::GptqAwq { .. } => gptq_linear(in_dim, out_dim, quant_conf, vb)?,
            QuantizedConfig::Fp8 { .. } => {
                blockwise_fp8_linear_b(in_dim, out_dim, quant_conf, true, Default::default(), vb)?
            }
            QuantizedConfig::Bitsandbytes { .. } => {
                Arc::new(BnbLinear::linear_b(in_dim, out_dim, true, vb)?) as Arc<_>
            }
            QuantizedConfig::Afq { .. } => {
                AfqLayer::afq_linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
            QuantizedConfig::MXFP4 {} => {
                MXFP4Layer::linear_b(in_dim, out_dim, quant_conf, true, vb)?
            }
        }
    } else {
        // Handle the case where the layer is dummy (no tensors)
        if !(vb.contains_tensor("weight") && vb.contains_tensor("bias")) {
            let layer = <DummyLayer as QuantMethod>::new(QuantMethodConfig::Dummy)?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let weight = merge_lora_weights(&vb, weight, in_dim, out_dim, Default::default())?;
            let bias = vb.get_with_hints((out_dim,), "bias", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, Some(bias)),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    apply_immediate_isq(layer, base_vb)
}

pub fn linear_b(
    in_dim: usize,
    out_dim: usize,
    bias: bool,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    if bias {
        linear(in_dim, out_dim, config, vb)
    } else {
        linear_no_bias(in_dim, out_dim, config, vb)
    }
}

/// Execute a grouped GEMM where each token is routed to a subset of experts.
///
/// - `xs`: activations shaped `(batch_size, seq_len, hidden)`
/// - `ids`: expert identifiers shaped `(batch_size, seq_len, topk)`
/// - `experts`: collection of expert projections. Each expert must implement [`QuantMethod`].
///
/// The returned tensor has shape `(batch_size, seq_len, topk, out_dim)`, where `out_dim` is the
/// output dimension of the provided experts.
pub fn grouped_gemm(xs: &Tensor, ids: &Tensor, experts: &[Arc<dyn QuantMethod>]) -> Result<Tensor> {
    if experts.is_empty() {
        candle_core::bail!("grouped_gemm requires at least one expert.");
    }
    if xs.rank() != 3 {
        candle_core::bail!(
            "grouped_gemm expects `xs` to have rank 3, got shape {:?}.",
            xs.dims()
        );
    }
    if ids.rank() != 3 {
        candle_core::bail!(
            "grouped_gemm expects `ids` to have rank 3, got shape {:?}.",
            ids.dims()
        );
    }

    let (bs, seq_len, hidden_size) = xs.dims3()?;
    let ids_shape = ids.dims();
    if ids_shape[0] != bs || ids_shape[1] != seq_len {
        candle_core::bail!(
            "grouped_gemm expects `ids` to match the first two dimensions of `xs` \
             (got ids shape {:?}, xs shape {:?}).",
            ids_shape,
            xs.dims()
        );
    }
    let topk = ids_shape[2];
    if topk == 0 {
        candle_core::bail!("grouped_gemm received `ids` with top-k dimension 0.");
    }

    let xs_flat = xs.reshape((bs * seq_len, hidden_size))?;
    let ids_u32 = if ids.dtype() == DType::U32 {
        ids.clone()
    } else {
        ids.to_dtype(DType::U32)?
    };
    let ids_flat = ids_u32.reshape((bs * seq_len, topk))?;
    let ids_host = ids_flat.to_vec2::<u32>()?;

    let mut routing: Vec<Vec<(usize, usize)>> = vec![Vec::new(); experts.len()];
    for (token_idx, routed) in ids_host.iter().enumerate() {
        for (slot_idx, expert_id) in routed.iter().enumerate() {
            let expert_idx = *expert_id as usize;
            if expert_idx >= experts.len() {
                candle_core::bail!(
                    "grouped_gemm received expert id {expert_idx} but only {} experts provided.",
                    experts.len()
                );
            }
            routing[expert_idx].push((token_idx, slot_idx));
        }
    }

    let mut output: Option<Tensor> = None;
    let mut output_dim: Option<usize> = None;
    let mut output_dtype: Option<DType> = None;
    let total_slots = bs * seq_len * topk;

    for (expert_idx, assignments) in routing.into_iter().enumerate() {
        if assignments.is_empty() {
            continue;
        }

        let token_indices: Vec<i64> = assignments
            .iter()
            .map(|(token_idx, _)| *token_idx as i64)
            .collect();
        let token_indices = Tensor::new(&token_indices[..], xs.device())?;
        let expert_in = xs_flat.index_select(&token_indices, 0)?;

        let mut expert_out = experts[expert_idx].forward_autocast(&expert_in)?;
        let expert_shape = expert_out.dims();
        if expert_shape.len() != 2 {
            candle_core::bail!(
                "grouped_gemm expects expert outputs with rank 2, got {:?} for expert {expert_idx}.",
                expert_shape
            );
        }
        let current_out_dim = expert_shape[1];
        if expert_shape[0] != assignments.len() {
            candle_core::bail!(
                "grouped_gemm expected expert {expert_idx} to produce {} rows but received {}.",
                assignments.len(),
                expert_shape[0]
            );
        }

        if output.is_none() {
            output_dim = Some(current_out_dim);
            output_dtype = Some(expert_out.dtype());
            output = Some(Tensor::zeros(
                (total_slots, current_out_dim),
                expert_out.dtype(),
                xs.device(),
            )?);
        } else {
            let expected_dim = output_dim.expect("output_dim should be set");
            if current_out_dim != expected_dim {
                candle_core::bail!(
                    "grouped_gemm expects all experts to share the same output dimension \
                     ({expected_dim}), but expert {expert_idx} produced {current_out_dim}."
                );
            }
            let expected_dtype = output_dtype.expect("output_dtype should be set");
            if expert_out.dtype() != expected_dtype {
                expert_out = expert_out.to_dtype(expected_dtype)?;
            }
        }

        expert_out = expert_out.reshape((assignments.len(), current_out_dim))?;

        let flat_indices: Vec<i64> = assignments
            .iter()
            .map(|(token_idx, slot_idx)| (token_idx * topk + slot_idx) as i64)
            .collect();
        let flat_indices = Tensor::new(&flat_indices[..], xs.device())?;

        // Safety: the Option has been initialized in the branch above.
        let current_out = output.take().unwrap();
        let updated = current_out.index_add(&flat_indices, &expert_out, 0)?;
        output = Some(updated);
    }

    let mut output = output.ok_or_else(|| {
        candle_core::Error::Msg(
            "grouped_gemm produced no outputs; ensure at least one token is routed to an expert."
                .into(),
        )
    })?;
    let output_dim = output_dim.ok_or_else(|| {
        candle_core::Error::Msg("grouped_gemm could not infer expert output dimension.".into())
    })?;

    output = output.reshape((bs, seq_len, topk, output_dim))?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn grouped_gemm_matches_naive_reference() -> Result<()> {
        let device = Device::Cpu;
        let batch_size = 2;
        let seq_len = 3;
        let hidden = 5;
        let topk = 2;
        let out_dim = 4;
        let num_experts = 4;
        let total_tokens = batch_size * seq_len;

        let xs_data: Vec<f32> = (0..(total_tokens * hidden))
            .map(|v| v as f32 * 0.1 + 0.5)
            .collect();
        let xs = Tensor::from_vec(xs_data, (batch_size, seq_len, hidden), &device)?
            .to_dtype(DType::F32)?;

        let ids_data: Vec<f32> = (0..total_tokens)
            .flat_map(|token_idx| {
                let base = token_idx % num_experts;
                [base as f32, ((base + 1) % num_experts) as f32]
            })
            .collect();
        let ids = Tensor::from_vec(ids_data, (batch_size, seq_len, topk), &device)?
            .to_dtype(DType::U32)?;

        let mut experts: Vec<Arc<dyn QuantMethod>> = Vec::with_capacity(num_experts);
        for expert_idx in 0..num_experts {
            let weight_data: Vec<f32> = (0..(out_dim * hidden))
                .map(|v| (expert_idx * out_dim * hidden + v) as f32 * 0.01 + 0.2)
                .collect();
            let weight = Tensor::from_vec(weight_data, (out_dim, hidden), &device)?;
            let linear = Linear::new(weight, None);
            let expert = UnquantLinear::new(QuantMethodConfig::Unquantized(linear))?;
            experts.push(Arc::new(expert));
        }

        let ids_host = ids
            .reshape((total_tokens, topk))?
            .to_dtype(DType::U32)?
            .to_vec2::<u32>()?;
        let xs_flat = xs.reshape((total_tokens, hidden))?;

        let mut expected_chunks = Vec::with_capacity(total_tokens * topk);
        for (token_idx, assignments) in ids_host.iter().enumerate() {
            let token_x = xs_flat.i(token_idx)?.reshape((1, hidden))?;
            for expert_id in assignments {
                let y = experts[*expert_id as usize].forward_autocast(&token_x)?;
                expected_chunks.push(y);
            }
        }
        let expected =
            Tensor::cat(&expected_chunks, 0)?.reshape((batch_size, seq_len, topk, out_dim))?;

        let actual = grouped_gemm(&xs, &ids, &experts)?;

        let actual_flat = actual.flatten_all()?.to_vec1::<f32>()?;
        let expected_flat = expected.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(actual_flat.len(), expected_flat.len());
        for (a, e) in actual_flat.iter().zip(expected_flat.iter()) {
            assert!((a - e).abs() < 1e-5, "mismatch: {a} vs {e}");
        }

        Ok(())
    }
}
