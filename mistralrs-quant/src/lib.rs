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
use pertensor_fp8::pertensor_fp8_linear_b;

#[cfg(feature = "metal")]
mod metal_kernels;

mod afq;
mod bitsandbytes;
mod blockwise_fp8;
pub mod cublaslt;
pub mod distributed;
mod dummy;
pub mod f8q8;
mod fp8;
pub mod gemv;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
mod lora;
mod mxfp4;
mod pending_layer;
mod pertensor_fp8;
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
pub use blockwise_fp8::{
    blockwise_fp8_moe, fp8_blockwise_dequantize, fp8_blockwise_quantize, BlockwiseFP8Linear,
};
pub use distributed::{
    layers::{
        compute_kv_shard, compute_n_kv_groups, ColumnParallelLayer, FusedExperts, PackedExperts,
        ReplicatedLayer, RowParallelLayer,
    },
    socket::{Client, Server},
    BarrierLike, Comm, Id, RingConfig, SumAllReduce,
};
pub use dummy::DummyLayer;
pub use f8q8::F8Q8Linear;
pub use fp8::FP8Linear;
#[cfg(feature = "cuda")]
pub use gemv::gemv;
pub use gemv::{should_use_gemv, GEMV_CONTROLLER};
pub use gguf::GgufMatMul;
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::{CollectedImatrixData, ImatrixLayerStats};
pub use lora::{
    clear_applied_loras, get_applied_loras, linear_no_bias_static_lora, push_applied_lora,
    LoraAdapter, LoraConfig, StaticLoraConfig, MULTI_LORA_DELIMITER,
};
pub use mxfp4::MXFP4Layer;
pub use pending_layer::PendingIsqLayer;
pub use pertensor_fp8::PerTensorFP8Linear;
pub use unquantized::UnquantLinear;
pub use utils::flash_attn_sinks_metal;
pub use utils::flash_attn_sinks_varlen_metal;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_fused;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_interleaved;
pub use utils::isq::apply_immediate_isq;
pub use utils::softmax_with_sinks;
pub use utils::{fused_glu, GluActivationType};
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
    /// Thread pool for parallel immediate ISQ on discrete GPUs.
    /// When `Some`, `apply_immediate_isq` will spawn quantization tasks
    /// on this pool and return `PendingIsqLayer` wrappers.
    pub pool: Option<Arc<rayon::ThreadPool>>,
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
    let (pool, _) = create_isq_thread_pool(isq);
    set_immediate_isq_with_pool(isq, predicates, Vec::new(), pool);
}

pub fn set_immediate_isq_with_pool(
    isq: Option<IsqType>,
    predicates: Vec<Regex>,
    overrides: Vec<ImmediateIsqOverride>,
    pool: rayon::ThreadPool,
) {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = Some(ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty: isq,
            predicates,
            overrides,
            pool: Some(Arc::new(pool)),
        });
    });
}

/// Create a rayon thread pool for parallel immediate ISQ.
/// Returns `(pool, num_threads)` so callers can log the thread count.
///
/// Thread count is based on the quantization type:
/// - GGML types (Q2K-Q8K) and F8E4M3: `rayon::current_num_threads()` (CPU quantization)
/// - HQQ/AFQ: 1 thread (GPU quantization, serialized by `QuantizeOntoGuard`)
pub fn create_isq_thread_pool(ty: Option<IsqType>) -> (rayon::ThreadPool, usize) {
    let num_threads = if std::env::var("MISTRALRS_ISQ_SINGLETHREAD").is_ok() {
        1
    } else if let Some(ty) = ty {
        ty.get_max_isq_cpu_threads()
            .map(usize::from)
            .unwrap_or_else(rayon::current_num_threads)
    } else {
        rayon::current_num_threads()
    };

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build()
        .expect("Failed to create ISQ thread pool");
    (pool, num_threads)
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
        weight_block_size: Option<Vec<usize>>,
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
                // weight_block_size is optional - None means per-tensor quantization
                Ok(QuantizedConfig::Fp8 {
                    weight_block_size: raw.weight_block_size,
                })
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
    PerTensorFP8 {
        weight: Tensor,
        weight_scale_inv: Tensor,
        activation_scale: Option<Tensor>,
        bias: Option<Tensor>,
        dequant_dtype: DType,
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

/// In-situ quantization type specifying the format to apply to model weights.
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
    F8Q8,
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
            // F8Q8: 33 bytes per 32 values -> similar to Q8_0
            Self::F8Q8 => (dtype.size_in_bytes() * 32).div_ceil(33),
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
            IsqType::F8E4M3 | IsqType::F8Q8 => None,
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
    F8Q8 = 5,
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
            5 => Ok(Self::F8Q8),
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
    /// On metal, this waits for outstanding work to finish to avoid "A command encoder is already encoding to this command buffer"
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
                dev.wait_until_completed()
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
            QuantizedConfig::Fp8 { weight_block_size } => {
                if weight_block_size.is_some() {
                    blockwise_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        false,
                        Default::default(),
                        vb,
                    )?
                } else {
                    pertensor_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        false,
                        Default::default(),
                        vb,
                    )?
                }
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
            QuantizedConfig::Fp8 { weight_block_size } => {
                if weight_block_size.is_some() {
                    blockwise_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        true,
                        Default::default(),
                        vb,
                    )?
                } else {
                    pertensor_fp8_linear_b(
                        in_dim,
                        out_dim,
                        quant_conf,
                        true,
                        Default::default(),
                        vb,
                    )?
                }
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
