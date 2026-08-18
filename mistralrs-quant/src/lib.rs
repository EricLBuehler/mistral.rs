use std::{
    fmt::Debug,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard},
};

use blockwise_fp8::blockwise_fp8_linear_b;
#[cfg(feature = "metal")]
use candle_core::D;
use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};
use pertensor_fp8::pertensor_fp8_linear_b;

#[cfg(feature = "metal")]
pub mod metal_kernels;

mod afq;
mod bitsandbytes;
mod blockwise_fp8;
pub mod cublaslt;
#[cfg(test)]
#[path = "build_support/cuda_headers.rs"]
mod cuda_headers;
#[cfg(all(feature = "cuda", feature = "cutile"))]
pub mod cutile;
pub mod distributed;
mod dummy;
pub mod f8q8;
mod fp8;
pub mod gemv;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
mod isq_executor;
mod lora;
#[cfg(feature = "cuda")]
pub mod moe;
mod mxfp4;
mod pending_layer;
mod pertensor_fp8;
pub mod rotary;
pub mod safetensors;
mod scalar_fp8;
mod unquantized;
mod uqff;
mod utils;
mod vector_fp8;

use gptq::gptq_linear;
use regex::Regex;
pub use safetensors::{Shard, ShardedSafeTensors, TensorShapes};
pub use uqff::{
    bias_shard, build_output_report_from_layers, build_uqff_report,
    build_uqff_report_from_artifacts, inspect_uqff_artifacts, inspect_uqff_path, shard_range,
    slice_blocked_data, stored_type_from_tensors, uqff_version_tensors, verify_uqff_artifacts,
    verify_uqff_path, write_uqff_report, BiasShard, QuantizationIssue, QuantizationReport,
    QuantizedExpertKeys, ShardedVarBuilder, TrackedModule, Tracker, UqffArtifactFile,
    UqffArtifactGroup, UqffArtifacts, UqffExpertKeys, UqffFallbackReport, UqffGeneratedBy,
    UqffInspection, UqffLayerReport, UqffMetadataSummary, UqffOutputReport, UqffReader, UqffReport,
    UqffReportOptions, UqffTensor, UqffTensorSummary, UqffVerifyOptions, UqffVerifyResult,
    UQFF_REPORT_JSON, UQFF_VERSION_MAJOR, UQFF_VERSION_MINOR, UQFF_VERSION_PATCH,
};

pub trait QuantizedWeightSource: Send + Sync {
    fn contains(&self, name: &str) -> bool;

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>>;

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>>;

    fn shard_alignment(&self, key: &str) -> Result<usize>;

    fn pack_factor(&self, dtype: DType) -> Result<usize>;

    fn pack_factor_for(&self, key: &str, dtype: DType) -> Result<Option<usize>>;
}

impl<T: QuantizedWeightSource + ?Sized> QuantizedWeightSource for Arc<T> {
    fn contains(&self, name: &str) -> bool {
        (**self).contains(name)
    }

    fn load_linear(
        &self,
        key: &str,
        device: &Device,
        shard: Shard,
    ) -> Result<Option<Arc<dyn QuantMethod>>> {
        (**self).load_linear(key, device, shard)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>> {
        (**self).load_optional_tensor(name, device)
    }

    fn shard_alignment(&self, key: &str) -> Result<usize> {
        (**self).shard_alignment(key)
    }

    fn pack_factor(&self, dtype: DType) -> Result<usize> {
        (**self).pack_factor(dtype)
    }

    fn pack_factor_for(&self, key: &str, dtype: DType) -> Result<Option<usize>> {
        (**self).pack_factor_for(key, dtype)
    }
}

/// Bytes the packed-affine (Marlin layout) GGUF weights will occupy on this device.
///
/// Subtract from the KV cache budget so the repack is planned for rather than corrected after.
/// Returns 0 when the feature is off, which is the default.
pub fn gguf_affine_budget_bytes(device: &Device, dtype: DType) -> usize {
    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    {
        gguf::gguf_affine_budget_bytes(device, dtype)
    }
    #[cfg(not(all(feature = "cuda", has_marlin_kernels)))]
    {
        let _ = (device, dtype);
        0
    }
}

#[cfg(feature = "metal")]
pub use afq::ops::{
    afq_gather_qmm_rhs_sorted, afq_gather_qmm_rhs_sorted_gate_up, metal_arg_sort_u32_1d,
    metal_moe_weighted_reduce_flat,
};
pub use afq::{AfqBits, AfqGroupSize, AfqInner, AfqLayer};
pub use bitsandbytes::{BnbLinear, BnbQuantParams, BnbQuantType};
pub use blockwise_fp8::{
    blockwise_fp8_moe, fp8_blockwise_dequantize, fp8_blockwise_quantize, BlockwiseFP8Linear,
};
pub use distributed::{
    layers::{
        compute_kv_shard, compute_n_kv_groups, validate_tp_head_layout, ColumnParallelLayer,
        PreQuantizedExperts, ReplicatedLayer, RowParallelLayer,
    },
    socket::{Client, Server},
    BarrierLike, Comm, Id, RingConfig, SumAllReduce,
};
pub use dummy::{DummyLayer, DummyLayerInfo};
pub use f8q8::F8Q8Linear;
pub use fp8::FP8Linear;
#[cfg(feature = "cuda")]
pub use gemv::gemv;
pub use gemv::{should_use_gemv, GEMV_CONTROLLER};
pub use gguf::archive::{
    GgufArchive, GgufDType, GgufEndian, GgufShardInfo, GgufTensorData, GgufTensorInfo, GgufVersion,
};
pub use gguf::cpu::cpu_indexed_moe_forward;
#[cfg(feature = "cuda")]
pub use gguf::cuda::{
    grouped_moe_gemm_prequantized, indexed_moe_fused_decode, moe_dispatch_build,
    moe_weighted_reduce_flat, moe_weighted_reduce_flat_bf16, moe_weighted_reduce_flat_same_dtype,
    quantize_input_q8_1, IndexedMoeLoraDecode, IndexedMoeLoraWeights, IndexedMoeRouting,
    ACT_GELU_PYTORCH_TANH, ACT_SILU,
};
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use gguf::fast_mmq::grouped_from_glu_sorted_pair as grouped_moe_mmq_from_glu_sorted_pair;
#[cfg(feature = "cuda")]
pub use gguf::fast_mmq::{
    grouped as grouped_moe_mmq, grouped_from_glu_packed as grouped_moe_mmq_from_glu_packed,
    grouped_from_glu_pair as grouped_moe_mmq_from_glu_pair, grouped_pair as grouped_moe_mmq_pair,
    grouped_pair_packed as grouped_moe_mmq_pair_packed, supports as supports_mmq,
};
pub use gguf::GgufMatMul;
pub use gguf::{
    GgufBindingMap, GgufBindingResolver, GgufTensorBackend, GgufTensorBinding, GgufWeightSource,
};
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::{CollectedImatrixData, ImatrixLayerStats};
pub use isq_executor::{
    conservative_plan, elem_count, estimate_output_bytes, ggml_output_bytes, plan_weight_isq,
    tensor_bytes, IsqConsumer, IsqExecutor, IsqExecutorConfig, IsqJobOutput, IsqKernelKind,
    IsqPlanParams, IsqRequest, IsqResourceEstimate,
};
pub use lora::{
    add_expert_delta_reference, apply_dynamic_lora_delta, has_active_lora_execution,
    is_dynamic_lora_site_active, linear_no_bias_static_lora, load_dynamic_lora_weights,
    maybe_wrap_dynamic_lora, plan_dynamic_lora_weights, register_dynamic_lora_site,
    with_lora_execution, with_lora_execution_repeated_row, with_lora_execution_row_range,
    DynamicLoraLoadPlan, DynamicLoraWeights, LoraAdapterWeights, LoraConfig, LoraExecution,
    LoraExecutionArena, LoraExecutionArenaStats, LoraExpertDelta, LoraExpertExecution,
    LoraExpertInputMode, LoraExpertProjection, LoraExpertProjectionNames,
    LoraExpertProjectionWeights, LoraExpertSiteHandle, LoraExpertSiteSpec, LoraExpertWeights,
    LoraGateUpOrder, LoraLayerRegistry, LoraLinearSpec, LoraRuntimeId, LoraSiteHandle, LoraSiteKey,
    LoraSiteSlice, LoraSlotId, LoraTargetModules, LoraWeights, RoutedLoraAdapterWeight,
    RoutedLoraInputMode, RoutedLoraMetadataLayout, RoutedLoraProjectionLayout, StaticLoraConfig,
    ROUTED_LORA_BASE_SLOT, ROUTED_LORA_BLOCK_SIZE, ROUTED_LORA_MAX_RANK, ROUTED_LORA_WMMA_RANK_CAP,
};
#[cfg(feature = "cuda")]
pub use lora::{
    launch_routed_lora_direct, launch_routed_lora_grouped, RoutedLoraCudaMetadata,
    RoutedLoraCudaWeightTable, RoutedLoraDirectLaunch, RoutedLoraGroupedLaunch,
};
pub use mxfp4::MXFP4Layer;
pub use pending_layer::{pending_isq_channel, PendingIsqLayer};
pub use pertensor_fp8::PerTensorFP8Linear;
pub use unquantized::UnquantLinear;
pub use utils::flash_attn_sinks_metal;
pub use utils::flash_attn_sinks_varlen_metal;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_fused;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_interleaved;
pub use utils::isq::{
    apply_immediate_isq, apply_immediate_isq_sharded, apply_immediate_isq_with_key,
    quantize_expert_stack, quantize_expert_stack_with_bias, requantize_tracked, RequantizeHandles,
};
pub use utils::softcap;
pub use utils::softmax_with_sinks;
pub use utils::{fused_glu, fused_split_glu, GluActivationType};
pub use utils::{log, BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp};
pub use vector_fp8::{fp8_vector_dequantize, fp8_vector_quantize};

use candle_nn::{Conv1d, Conv2d, Linear, Module};
use serde::{Deserialize, Deserializer, Serialize};

#[derive(Clone, Debug)]
pub struct ImmediateIsqParams {
    pub guard: QuantizeOntoGuard,
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub promoted_predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
    pub executor: IsqExecutor,
    pub capture: IsqCaptureMode,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqConfig {
    pub ty: Option<IsqType>,
    pub predicates: Vec<Regex>,
    pub promoted_predicates: Vec<Regex>,
    pub overrides: Vec<ImmediateIsqOverride>,
    pub capture: IsqCaptureMode,
}

impl ImmediateIsqConfig {
    pub fn new(ty: Option<IsqType>, predicates: Vec<Regex>, capture: IsqCaptureMode) -> Self {
        Self {
            ty,
            predicates,
            promoted_predicates: Vec::new(),
            overrides: Vec::new(),
            capture,
        }
    }

    pub fn with_promoted_predicates(mut self, promoted_predicates: Vec<Regex>) -> Self {
        self.promoted_predicates = promoted_predicates;
        self
    }

    pub fn with_overrides(mut self, overrides: Vec<ImmediateIsqOverride>) -> Self {
        self.overrides = overrides;
        self
    }
}

/// Whether load-time ISQ quantizes layers or captures them unquantized for later quantization.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum IsqCaptureMode {
    /// Quantize matching layers as they load.
    #[default]
    Immediate,
    /// Capture every layer unquantized (UQFF serialization needs all of them).
    CaptureAll,
    /// Capture matching layers unquantized for deferred quantization (e.g. calibration).
    CaptureMatches,
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqOverride {
    pub predicate: Option<Regex>,
    /// Decoder layer index range, matched via the `layers.N` segment of the weight prefix.
    pub layer_range: Option<std::ops::Range<usize>>,
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
}

impl ImmediateIsqOverride {
    fn matches(&self, prefix: &str) -> bool {
        if let Some(predicate) = &self.predicate {
            if predicate.is_match(prefix) {
                return true;
            }
        }
        if let Some(range) = &self.layer_range {
            if let Some(index) = layer_index_from_prefix(prefix) {
                return range.contains(&index);
            }
        }
        false
    }
}

static LAYER_INDEX_REGEX: std::sync::OnceLock<Regex> = std::sync::OnceLock::new();

/// Extract the decoder layer index from a weight prefix like `model.layers.12.self_attn.q_proj`.
pub fn layer_index_from_prefix(prefix: &str) -> Option<usize> {
    let regex = LAYER_INDEX_REGEX
        .get_or_init(|| Regex::new(r"(?:^|\.)(?:layers|h)\.(\d+)(?:\.|$)").expect("valid regex"));
    regex.captures(prefix)?.get(1)?.as_str().parse().ok()
}

#[derive(Clone, Debug)]
pub struct ImmediateIsqMatch {
    pub ty: Option<IsqType>,
    pub device: Option<Device>,
    pub promote_default: bool,
}

thread_local! {
    static ENGINE_IMMEDIATE_ISQ: std::cell::RefCell<Option<ImmediateIsqParams>> = const { std::cell::RefCell::new(None) } ;
}

pub fn set_immediate_isq(isq: Option<IsqType>, predicates: Vec<Regex>, capture: IsqCaptureMode) {
    let (executor, _) = create_isq_executor(IsqExecutorConfig::new(isq));
    set_immediate_isq_with_executor(isq, predicates, Vec::new(), capture, executor);
}

pub fn set_immediate_isq_with_executor(
    isq: Option<IsqType>,
    predicates: Vec<Regex>,
    overrides: Vec<ImmediateIsqOverride>,
    capture: IsqCaptureMode,
    executor: IsqExecutor,
) {
    set_immediate_isq_config(
        ImmediateIsqConfig::new(isq, predicates, capture).with_overrides(overrides),
        executor,
    );
}

pub fn set_immediate_isq_config(config: ImmediateIsqConfig, executor: IsqExecutor) {
    ENGINE_IMMEDIATE_ISQ.with(|cell| {
        *cell.borrow_mut() = Some(ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty: config.ty,
            predicates: config.predicates,
            promoted_predicates: config.promoted_predicates,
            overrides: config.overrides,
            executor,
            capture: config.capture,
        });
    });
}

#[cfg(target_os = "macos")]
unsafe fn set_isq_thread_affinity() {
    use libc::{pthread_set_qos_class_self_np, qos_class_t::QOS_CLASS_USER_INTERACTIVE};
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
}

#[cfg(not(target_os = "macos"))]
unsafe fn set_isq_thread_affinity() {}

/// Legacy Rayon pool helper for callers that still need raw pool semantics.
/// New ISQ scheduling should use `create_isq_executor`.
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
        .start_handler(|_| unsafe {
            set_isq_thread_affinity();
        })
        .build()
        .expect("Failed to create ISQ thread pool");
    (pool, num_threads)
}

pub fn create_isq_executor(config: IsqExecutorConfig) -> (IsqExecutor, usize) {
    let executor = IsqExecutor::new(config);
    let num_threads = executor.worker_threads();
    (executor, num_threads)
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

pub fn weight_source_load_device(vb: &ShardedVarBuilder) -> Device {
    if should_apply_immediate_isq(vb) {
        Device::Cpu
    } else {
        vb.device().clone()
    }
}

pub fn immediate_isq_match(vb: &ShardedVarBuilder) -> Option<ImmediateIsqMatch> {
    let immediate_isq = get_immediate_isq()?;
    // Add a .weight to match the ISQ regexes!
    let prefix = format!("{}.weight", vb.prefix());
    resolve_immediate_isq(&immediate_isq, &prefix)
}

fn resolve_immediate_isq(params: &ImmediateIsqParams, prefix: &str) -> Option<ImmediateIsqMatch> {
    let promote_default = params
        .promoted_predicates
        .iter()
        .any(|predicate| predicate.is_match(prefix));
    let default_ty = params.ty.map(|ty| {
        if promote_default {
            ty.promote_for_sensitive_tensor()
        } else {
            ty
        }
    });
    if params.capture == IsqCaptureMode::CaptureAll {
        // Capture everything; topology overrides still pin per-layer ty/device.
        if let Some(override_hit) = params
            .overrides
            .iter()
            .find(|override_entry| override_entry.matches(prefix))
        {
            return Some(ImmediateIsqMatch {
                ty: override_hit.ty.or(default_ty),
                device: override_hit.device.clone(),
                promote_default,
            });
        }
        return Some(ImmediateIsqMatch {
            ty: None,
            device: None,
            promote_default,
        });
    }

    if let Some(override_hit) = params
        .overrides
        .iter()
        .find(|override_entry| override_entry.matches(prefix))
    {
        let ty = override_hit.ty.or(default_ty);
        // Device-only overrides still need a match so the layer gets relocated
        if ty.is_some() || override_hit.device.is_some() {
            return Some(ImmediateIsqMatch {
                ty,
                device: override_hit.device.clone(),
                promote_default,
            });
        }
        return None;
    }

    if let Some(ty) = default_ty {
        if params
            .predicates
            .iter()
            .any(|predicate| predicate.is_match(prefix))
        {
            return Some(ImmediateIsqMatch {
                ty: Some(ty),
                device: None,
                promote_default,
            });
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
            Self::GptqAwq { bits, .. } => match bits {
                2 => IsqType::Q2K.pack_factor(dtype),
                3 => IsqType::Q3K.pack_factor(dtype),
                4 => IsqType::Q4K.pack_factor(dtype),
                5 => IsqType::Q5K.pack_factor(dtype),
                6 => IsqType::Q6K.pack_factor(dtype),
                8 => IsqType::Q8_0.pack_factor(dtype),
                40 => IsqType::MXFP4.pack_factor(dtype),
                other => panic!("Unexpected bits in `pack_factor` {other}"),
            },
            Self::Fp8 { .. } => IsqType::F8E4M3.pack_factor(dtype),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: Some(_),
            } => IsqType::Q4K.pack_factor(dtype),
            Self::Bitsandbytes {
                bnb_4bit_quant_type: None,
            } => 1,
            Self::Afq { bits, group_size } => afq_pack_factor(dtype, *bits, *group_size),
            Self::MXFP4 {} => IsqType::MXFP4.pack_factor(dtype),
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
    MXFP4,
}

/// Target bit width for automatic ISQ quantization.
///
/// On Metal, these select AFQ variants; on CUDA/CPU, they select Q*K variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IsqBits {
    /// 2-bit quantization (AFQ2 on Metal, Q2K otherwise).
    Two,
    /// 3-bit quantization (AFQ3 on Metal, Q3K otherwise).
    Three,
    /// 4-bit quantization (AFQ4 on Metal, Q4K otherwise).
    Four,
    /// 5-bit quantization (Q5K on all platforms).
    Five,
    /// 6-bit quantization (AFQ6 on Metal, Q6K otherwise).
    Six,
    /// 8-bit quantization (AFQ8 on Metal, Q8_0 otherwise).
    Eight,
}

impl IsqBits {
    /// Resolve to the platform-appropriate `IsqType` for the given device.
    pub fn resolve(self, device: &Device) -> IsqType {
        match (self, device.is_metal()) {
            (Self::Two, true) => IsqType::AFQ2,
            (Self::Two, false) => IsqType::Q2K,
            (Self::Three, true) => IsqType::AFQ3,
            (Self::Three, false) => IsqType::Q3K,
            (Self::Four, true) => IsqType::AFQ4,
            (Self::Four, false) => IsqType::Q4K,
            (Self::Five, _) => IsqType::Q5K,
            (Self::Six, true) => IsqType::AFQ6,
            (Self::Six, false) => IsqType::Q6K,
            (Self::Eight, true) => IsqType::AFQ8,
            (Self::Eight, false) => IsqType::Q8_0,
        }
    }

    /// Return all platform variants, with the current platform's preferred variant first.
    /// On Metal, AFQ variants come first; on other platforms, GGUF/Q variants come first.
    pub fn expand(self) -> Vec<IsqType> {
        #[cfg(feature = "metal")]
        match self {
            Self::Two => vec![IsqType::AFQ2, IsqType::Q2K],
            Self::Three => vec![IsqType::AFQ3, IsqType::Q3K],
            Self::Four => vec![IsqType::AFQ4, IsqType::Q4K],
            Self::Five => vec![IsqType::Q5K],
            Self::Six => vec![IsqType::AFQ6, IsqType::Q6K],
            Self::Eight => vec![IsqType::AFQ8, IsqType::Q8_0],
        }
        #[cfg(not(feature = "metal"))]
        match self {
            Self::Two => vec![IsqType::Q2K, IsqType::AFQ2],
            Self::Three => vec![IsqType::Q3K, IsqType::AFQ3],
            Self::Four => vec![IsqType::Q4K, IsqType::AFQ4],
            Self::Five => vec![IsqType::Q5K],
            Self::Six => vec![IsqType::Q6K, IsqType::AFQ6],
            Self::Eight => vec![IsqType::Q8_0, IsqType::AFQ8],
        }
    }
}

impl TryFrom<&str> for IsqBits {
    type Error = ();
    fn try_from(s: &str) -> std::result::Result<Self, ()> {
        match s {
            "2" => Ok(Self::Two),
            "3" => Ok(Self::Three),
            "4" => Ok(Self::Four),
            "5" => Ok(Self::Five),
            "6" => Ok(Self::Six),
            "8" => Ok(Self::Eight),
            _ => Err(()),
        }
    }
}

impl std::fmt::Display for IsqType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Q4_1 => write!(f, "q4_1"),
            Self::Q5_0 => write!(f, "q5_0"),
            Self::Q5_1 => write!(f, "q5_1"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q8_1 => write!(f, "q8_1"),
            Self::Q2K => write!(f, "q2k"),
            Self::Q3K => write!(f, "q3k"),
            Self::Q4K => write!(f, "q4k"),
            Self::Q5K => write!(f, "q5k"),
            Self::Q6K => write!(f, "q6k"),
            Self::Q8K => write!(f, "q8k"),
            Self::HQQ8 => write!(f, "hqq8"),
            Self::HQQ4 => write!(f, "hqq4"),
            Self::F8E4M3 => write!(f, "fp8"),
            Self::AFQ8 => write!(f, "afq8"),
            Self::AFQ6 => write!(f, "afq6"),
            Self::AFQ4 => write!(f, "afq4"),
            Self::AFQ3 => write!(f, "afq3"),
            Self::AFQ2 => write!(f, "afq2"),
            Self::F8Q8 => write!(f, "f8q8"),
            Self::MXFP4 => write!(f, "mxfp4"),
        }
    }
}

const AFFINE_METADATA_TENSOR_COUNT: usize = 2;

fn integer_pack_factor(dense_bytes: usize, packed_bytes: usize) -> usize {
    (dense_bytes / packed_bytes.max(1)).max(1)
}

pub(crate) fn block_pack_factor(block_elements: usize, dtype: DType, packed_bytes: usize) -> usize {
    let dtype_bytes = dtype.size_in_bytes();
    let mut factor = integer_pack_factor(block_elements * dtype_bytes, packed_bytes);
    while factor > 1 && block_elements / factor * dtype_bytes < packed_bytes {
        factor -= 1;
    }
    factor
}

pub(crate) fn afq_pack_factor(dtype: DType, bits: usize, group_size: usize) -> usize {
    affine_pack_factor(dtype, bits, group_size, dtype.size_in_bytes())
}

pub(crate) fn hqq_pack_factor(dtype: DType, bits: usize, group_size: usize) -> usize {
    affine_pack_factor(dtype, bits, group_size, DType::F32.size_in_bytes())
}

fn affine_pack_factor(
    dtype: DType,
    bits: usize,
    group_size: usize,
    metadata_element_bytes: usize,
) -> usize {
    let packed_bytes =
        (group_size * bits).div_ceil(8) + AFFINE_METADATA_TENSOR_COUNT * metadata_element_bytes;
    block_pack_factor(group_size, dtype, packed_bytes)
}

impl IsqType {
    pub fn promote_for_sensitive_tensor(self) -> Self {
        match self {
            Self::AFQ2 | Self::AFQ3 | Self::AFQ4 => Self::AFQ6,
            Self::AFQ6 | Self::AFQ8 => Self::AFQ8,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q4_0 | Self::Q4_1 => Self::Q6K,
            Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_0
            | Self::Q8_1 => Self::Q8_0,
            ty => ty,
        }
    }

    /// Integer factor used to estimate the weight-size reduction over the given dtype.
    pub fn pack_factor(&self, dtype: DType) -> usize {
        let ggml = |quantized_dtype: GgmlDType| {
            block_pack_factor(
                quantized_dtype.block_size(),
                dtype,
                quantized_dtype.type_size(),
            )
        };
        match self {
            Self::Q4_0 => ggml(GgmlDType::Q4_0),
            Self::Q4_1 => ggml(GgmlDType::Q4_1),
            Self::Q5_0 => ggml(GgmlDType::Q5_0),
            Self::Q5_1 => ggml(GgmlDType::Q5_1),
            Self::Q8_0 => ggml(GgmlDType::Q8_0),
            Self::Q8_1 => ggml(GgmlDType::Q8_1),
            Self::Q2K => ggml(GgmlDType::Q2K),
            Self::Q3K => ggml(GgmlDType::Q3K),
            Self::Q4K => ggml(GgmlDType::Q4K),
            Self::Q5K => ggml(GgmlDType::Q5K),
            Self::Q6K => ggml(GgmlDType::Q6K),
            Self::Q8K => ggml(GgmlDType::Q8K),
            Self::AFQ2 => afq_pack_factor(dtype, 2, AfqGroupSize::Low as usize),
            Self::AFQ3 => afq_pack_factor(dtype, 3, AfqGroupSize::Low as usize),
            Self::AFQ4 => afq_pack_factor(dtype, 4, AfqGroupSize::Low as usize),
            Self::AFQ6 => afq_pack_factor(dtype, 6, AfqGroupSize::Low as usize),
            Self::AFQ8 => afq_pack_factor(dtype, 8, AfqGroupSize::Low as usize),
            Self::F8Q8 => {
                block_pack_factor(f8q8::QK8_0, dtype, std::mem::size_of::<f8q8::BlockF8Q8>())
            }
            Self::HQQ4 => hqq_pack_factor(dtype, 4, hqq::ISQ_HQQ_GROUP_SIZE),
            Self::HQQ8 => hqq_pack_factor(dtype, 8, hqq::ISQ_HQQ_GROUP_SIZE),
            Self::F8E4M3 => 1,
            Self::MXFP4 => block_pack_factor(
                mxfp4::MXFP4_BLOCK_SIZE,
                dtype,
                mxfp4::MXFP4_BLOCK_SIZE * mxfp4::N_BITS / u8::BITS as usize
                    + DType::U8.size_in_bytes(),
            ),
        }
    }

    /// Only the K-quant formats consume importance weights; the rest quantize without them.
    pub fn supports_imatrix(self) -> bool {
        matches!(
            self,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K
        )
    }

    pub fn supports_uqff(self) -> bool {
        matches!(
            self,
            Self::Q2K
                | Self::Q3K
                | Self::Q4K
                | Self::Q4_0
                | Self::Q4_1
                | Self::Q5K
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q6K
                | Self::Q8K
                | Self::Q8_0
                | Self::Q8_1
                | Self::HQQ4
                | Self::HQQ8
                | Self::F8E4M3
                | Self::AFQ2
                | Self::AFQ3
                | Self::AFQ4
                | Self::AFQ6
                | Self::AFQ8
                | Self::F8Q8
                | Self::MXFP4
        )
    }

    pub fn supports_stacked_gather(self) -> bool {
        GgmlDType::try_from(self).is_ok()
            || matches!(
                self,
                Self::AFQ2 | Self::AFQ3 | Self::AFQ4 | Self::AFQ6 | Self::AFQ8
            )
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
            | IsqType::AFQ8
            | IsqType::MXFP4 => {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum QuantizedSerdeType {
    Gguf = 0,
    Unquant = 1,
    Hqq = 2,
    Fp8 = 3,
    Afq = 4,
    F8Q8 = 5,
    Mxfp4 = 6,
}

impl QuantizedSerdeType {
    pub const ALL: [Self; 7] = [
        Self::Gguf,
        Self::Unquant,
        Self::Hqq,
        Self::Fp8,
        Self::Afq,
        Self::F8Q8,
        Self::Mxfp4,
    ];

    pub fn stored_label(self, quant_group: &str) -> String {
        match self {
            Self::Gguf => {
                if quant_group.starts_with('q') {
                    quant_group.to_string()
                } else {
                    "gguf".to_string()
                }
            }
            Self::Unquant => "unquant".to_string(),
            Self::Hqq => {
                if quant_group.starts_with("hqq") {
                    quant_group.to_string()
                } else {
                    "hqq".to_string()
                }
            }
            Self::Fp8 => "fp8".to_string(),
            Self::Afq => {
                if quant_group.starts_with("afq") {
                    quant_group.to_string()
                } else {
                    "afq".to_string()
                }
            }
            Self::F8Q8 => "f8q8".to_string(),
            Self::Mxfp4 => "mxfp4".to_string(),
        }
    }

    pub(crate) fn inspect_uqff_header(
        self,
        layer: &uqff::UqffLayerHeaderView<'_>,
    ) -> Option<uqff::UqffHeaderMatch> {
        match self {
            Self::Gguf => GgufMatMul::inspect_uqff_header(layer),
            Self::Unquant => UnquantLinear::inspect_uqff_header(layer),
            Self::Hqq => HqqLayer::inspect_uqff_header(layer),
            Self::Fp8 => FP8Linear::inspect_uqff_header(layer),
            Self::Afq => AfqLayer::inspect_uqff_header(layer),
            Self::F8Q8 => F8Q8Linear::inspect_uqff_header(layer),
            Self::Mxfp4 => MXFP4Layer::inspect_uqff_header(layer),
        }
    }

    pub(crate) fn stored_label_from_uqff_tensors(
        self,
        tensors: &[uqff::UqffTensor],
        prefix: &str,
    ) -> Result<String> {
        match self {
            Self::Gguf => GgufMatMul::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Unquant => UnquantLinear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Hqq => HqqLayer::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Fp8 => FP8Linear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Afq => AfqLayer::stored_label_from_uqff_tensors(tensors, prefix),
            Self::F8Q8 => F8Q8Linear::stored_label_from_uqff_tensors(tensors, prefix),
            Self::Mxfp4 => MXFP4Layer::stored_label_from_uqff_tensors(tensors, prefix),
        }
    }
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
            6 => Ok(Self::Mxfp4),
            other => candle_core::bail!("QuantizedSerdeType {other} is invalid."),
        }
    }
}

pub trait QuantizedSerde {
    fn name(&self) -> &'static str;
    fn isq_serde_supported(&self) -> bool {
        false
    }
    fn uqff_type(&self) -> Option<IsqType> {
        None
    }
    fn serialize_uqff(&self, _prefix: &str, ty: IsqType) -> Result<Vec<UqffTensor>> {
        candle_core::bail!(
            "`{}` does not support UQFF serialization for {ty}.",
            self.name()
        )
    }
    fn deserialize_uqff(
        _reader: &UqffReader,
        _prefix: &str,
        _device: &Device,
        _shard: Shard,
    ) -> Result<Arc<dyn QuantMethod>>
    where
        Self: Sized,
    {
        candle_core::bail!(
            "`{}` does not support UQFF deserialization.",
            std::any::type_name::<Self>()
        )
    }
    fn isq_type_from_uqff(_reader: &UqffReader, _prefix: &str) -> Result<IsqType>
    where
        Self: Sized,
    {
        candle_core::bail!(
            "`{}` does not support UQFF type detection.",
            std::any::type_name::<Self>()
        )
    }
}

/// Used to gate access to quantizing onto the host device
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct QuantizeOntoGuard {
    pub inner: Arc<Mutex<()>>,
    module_key: Option<Arc<str>>,
    report: Option<QuantizationReport>,
    requested: Option<Arc<str>>,
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
            module_key: None,
            report: None,
            requested: None,
        }
    }

    pub fn with_module_key(mut self, module_key: impl Into<String>) -> Self {
        self.module_key = Some(Arc::<str>::from(module_key.into()));
        self
    }

    pub fn module_key(&self) -> Option<&str> {
        self.module_key.as_deref()
    }

    pub fn with_report(mut self, report: QuantizationReport) -> Self {
        self.report = Some(report);
        self
    }

    pub fn with_requested(mut self, requested: impl Into<String>) -> Self {
        self.requested = Some(Arc::<str>::from(requested.into()));
        self
    }

    pub fn report(&self) -> Option<&QuantizationReport> {
        self.report.as_ref()
    }

    pub fn requested(&self) -> Option<&str> {
        self.requested.as_deref()
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
    /// Automatically casts to the required quantization activation type and back.
    fn forward(&self, a: &Tensor) -> Result<Tensor> {
        if let Some(t) = self.quantized_act_type() {
            let original_ty = a.dtype();
            self.forward_raw(&a.to_dtype(t)?)?.to_dtype(original_ty)
        } else {
            self.forward_raw(a)
        }
    }

    /// Raw matmul without dtype casting. Implementors override this.
    /// Callers should use `forward` instead.
    fn forward_raw(&self, a: &Tensor) -> Result<Tensor>;

    /// Compute gather matmul of `self` and `a`. `self` should contain the weights.
    /// Automatically casts to the required quantization activation type and back.
    ///
    /// If `a` is (n_tokens, n_experts, cols), `self` weights are (n_experts, rows, cols),
    /// then the indices are (n_tokens, n_experts).
    fn gather_forward(&self, a: &Tensor, indices: &Tensor) -> Result<Tensor> {
        if let Some(t) = self.quantized_act_type() {
            let original_ty = a.dtype();
            self.gather_forward_raw(&a.to_dtype(t)?, indices)?
                .to_dtype(original_ty)
        } else {
            self.gather_forward_raw(a, indices)
        }
    }

    /// Raw gather matmul without dtype casting. Implementors override this.
    /// Callers should use `gather_forward` instead.
    fn gather_forward_raw(&self, _a: &Tensor, _indices: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support `gather_forward`. Please raise an issue.",
            self.name()
        )
    }

    fn embedding_forward(&self, ids: &Tensor, output_dtype: DType) -> Result<Tensor> {
        self.embedding_forward_raw(ids)?.to_dtype(output_dtype)
    }

    fn embedding_forward_raw(&self, _ids: &Tensor) -> Result<Tensor> {
        candle_core::bail!(
            "{} does not support `embedding_forward`. Please raise an issue.",
            self.name()
        )
    }

    /// Get the underlying QTensor if this is a GGUF quantized layer.
    /// Used for direct kernel access in grouped MoE prefill and CPU fused GEMV paths.
    fn get_qtensor(&self) -> Option<Arc<candle_core::quantized::QTensor>> {
        None
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    #[doc(hidden)]
    fn prepare_gguf_affine_raw(
        &self,
        _flat_batch: usize,
        _dtype: DType,
        _device: &Device,
    ) -> Result<bool> {
        Ok(false)
    }

    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    #[doc(hidden)]
    fn try_gguf_affine_forward_raw(&self, _a: &Tensor) -> Result<Option<Tensor>> {
        Ok(None)
    }

    /// If this is an AFQ layer, return its (w_q, scales, biases, bits, group_size).
    /// Used by Metal fused QKV / gate-up paths.
    fn afq_inner(&self) -> Option<crate::afq::AfqInner> {
        None
    }

    /// If a quantized method, return the activation dtype.
    fn quantized_act_type(&self) -> Option<DType>;

    /// Weight dtype and device
    fn dtype_and_device(&self) -> (DType, Device);

    fn plan_isq(&self, request: &IsqRequest) -> Result<IsqPlanParams>;

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

    fn is_dynamic_lora_active(&self) -> bool {
        false
    }

    fn preserve_dynamic_lora(&self, replacement: Arc<dyn QuantMethod>) -> Arc<dyn QuantMethod> {
        replacement
    }

    fn has_bias(&self) -> bool {
        false
    }

    /// Begin tracking stats into an ImatrixLayerStats
    fn begin_track_stats(&self) -> Result<()> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// End tracking stats into an ImatrixLayerStats. Returns the computed imatrix.
    fn end_track_stats(&self) -> Result<Tensor> {
        candle_core::bail!("`{}` does not support tracking stats.", self.name())
    }

    /// (forward calls, token rows) accumulated by stats tracking, if enabled.
    fn stats_snapshot(&self) -> Option<(usize, usize)> {
        None
    }

    /// Feed routed activations for per-expert stats; called by the owning MoE block, which alone
    /// knows the token-to-expert pairing. No-op unless routed tracking is enabled.
    fn process_routed_stats(&self, _x: &Tensor, _ids: &Tensor) -> Result<()> {
        Ok(())
    }

    fn is_distributed(&self) -> Option<DistributedKind> {
        None
    }

    fn dummy_info(&self) -> Option<DummyLayerInfo> {
        None
    }
}

impl Module for dyn QuantMethod {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        QuantMethod::forward(self, xs)
    }
}

#[cfg(feature = "cuda")]
pub fn try_fused_quantized_ffn(
    xs: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    down: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    if !xs.device().is_cuda() {
        return Ok(None);
    }
    if gate.stats_snapshot().is_some()
        || up.stats_snapshot().is_some()
        || down.stats_snapshot().is_some()
    {
        return Ok(None);
    }
    if gate.is_dynamic_lora_active() || up.is_dynamic_lora_active() || down.is_dynamic_lora_active()
    {
        return Ok(None);
    }
    if gate.has_bias() || up.has_bias() || down.has_bias() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let (flat_batch, k) = match xs.dims() {
        [batch, k] => (*batch, *k),
        [batch, rows, k] => (*batch * *rows, *k),
        _ => return Ok(None),
    };
    if flat_batch < gguf::GGUF_AFFINE_MIN_BATCH {
        return Ok(None);
    }

    let Some(gate_q) = gate.get_qtensor() else {
        return Ok(None);
    };
    let Some(up_q) = up.get_qtensor() else {
        return Ok(None);
    };
    let Some(down_q) = down.get_qtensor() else {
        return Ok(None);
    };
    if gate_q.shape() != up_q.shape() {
        return Ok(None);
    }
    let (intermediate, gate_cols) = gate_q.shape().dims2()?;
    let (_, down_cols) = down_q.shape().dims2()?;
    if gate_cols != k || down_cols != intermediate {
        return Ok(None);
    }
    if !gate_q.device().same_device(xs.device())
        || !up_q.device().same_device(xs.device())
        || !down_q.device().same_device(xs.device())
    {
        return Ok(None);
    }

    #[cfg(has_marlin_kernels)]
    {
        let gate_ready = gate.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        let up_ready = up.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        let down_ready = down.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        if gate_ready && up_ready && down_ready {
            let gate_out = gate
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine gate projection");
            let up_out = up
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine up projection");
            let intermediate = fused_glu(&gate_out, &up_out, activation)?;
            drop(gate_out);
            drop(up_out);
            let out = down
                .try_gguf_affine_forward_raw(&intermediate)?
                .expect("prepared GGUF affine down projection");
            return Ok(Some(out));
        }
    }

    if flat_batch <= gguf::fast_mmvq::MMVQ_MAX_BATCH {
        return Ok(None);
    }
    if gate_q.dtype() != up_q.dtype()
        || !gguf::fast_mmq::supports(gate_q.dtype())
        || !gguf::fast_mmq::supports(down_q.dtype())
    {
        return Ok(None);
    }

    Ok(Some(gguf::fast_mmq::fused_ffn(
        &gate_q, &up_q, &down_q, xs, activation,
    )?))
}

#[cfg(feature = "cuda")]
pub fn try_fused_quantized_gate_up(
    xs: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    if !xs.device().is_cuda() {
        return Ok(None);
    }
    if gate.stats_snapshot().is_some() || up.stats_snapshot().is_some() {
        return Ok(None);
    }
    if gate.is_dynamic_lora_active() || up.is_dynamic_lora_active() {
        return Ok(None);
    }
    if gate.has_bias() || up.has_bias() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let Some(gate_q) = gate.get_qtensor() else {
        return Ok(None);
    };
    let Some(up_q) = up.get_qtensor() else {
        return Ok(None);
    };
    if gate_q.shape() != up_q.shape() {
        return Ok(None);
    }

    let Some((&k, batch_dims)) = xs.dims().split_last() else {
        return Ok(None);
    };
    let flat_batch = batch_dims.iter().product::<usize>();
    if flat_batch == 0 {
        return Ok(None);
    }
    let (_, ncols) = gate_q.shape().dims2()?;
    if k != ncols {
        return Ok(None);
    }

    #[cfg(has_marlin_kernels)]
    if flat_batch >= gguf::GGUF_AFFINE_MIN_BATCH {
        let gate_ready = gate.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        let up_ready = up.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        if gate_ready && up_ready {
            let gate_out = gate
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine gate projection");
            let up_out = up
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine up projection");
            return Ok(Some(fused_glu(&gate_out, &up_out, activation)?));
        }
    }

    if gate_q.dtype() != up_q.dtype() {
        return Ok(None);
    }
    if flat_batch <= gguf::fast_mmvq::MMVQ_MAX_BATCH {
        if !gguf::fast_mmvq::supports_fused_glu(xs.dtype(), gate_q.dtype()) {
            return Ok(None);
        }
        Ok(Some(gguf::fast_mmvq::fused_glu(
            &gate_q, &up_q, xs, activation,
        )?))
    } else {
        if !gguf::fast_mmq::supports(gate_q.dtype()) {
            return Ok(None);
        }
        Ok(Some(gguf::fast_mmq::fused_glu(
            &gate_q, &up_q, xs, activation,
        )?))
    }
}

/// CPU fused m==1 matmuls sharing one lhs: one input quantization + one parallel region
/// (aarch64 repacked GEMV). Returns None when any weight is unsupported.
pub fn try_fused_gemv_shared_lhs_cpu(
    xs: &Tensor,
    ws: &[&dyn QuantMethod],
) -> Result<Option<Vec<Tensor>>> {
    if !xs.device().is_cpu() || xs.dtype() != DType::F32 {
        return Ok(None);
    }
    if ws.iter().any(|w| w.has_bias()) {
        return Ok(None);
    }
    let mut qs = Vec::with_capacity(ws.len());
    for w in ws {
        let Some(q) = w.get_qtensor() else {
            return Ok(None);
        };
        qs.push(q);
    }
    let refs: Vec<&candle_core::quantized::QTensor> = qs.iter().map(|a| a.as_ref()).collect();
    candle_core::quantized::QTensor::gemv_fused_shared_lhs(&refs, xs)
}

#[cfg(feature = "cuda")]
pub fn try_fused_quantized_qkv(
    xs: &Tensor,
    q: &dyn QuantMethod,
    k: &dyn QuantMethod,
    v: &dyn QuantMethod,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    if !xs.device().is_cuda() {
        return Ok(None);
    }
    if q.stats_snapshot().is_some() || k.stats_snapshot().is_some() || v.stats_snapshot().is_some()
    {
        return Ok(None);
    }
    if q.is_dynamic_lora_active() || k.is_dynamic_lora_active() || v.is_dynamic_lora_active() {
        return Ok(None);
    }
    if q.has_bias() || k.has_bias() || v.has_bias() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }

    let Some(q_q) = q.get_qtensor() else {
        return Ok(None);
    };
    let Some(k_q) = k.get_qtensor() else {
        return Ok(None);
    };
    let Some(v_q) = v.get_qtensor() else {
        return Ok(None);
    };

    let Some((&input_cols, batch_dims)) = xs.dims().split_last() else {
        return Ok(None);
    };
    let flat_batch = batch_dims.iter().product::<usize>();
    if flat_batch == 0 {
        return Ok(None);
    }
    let (_, q_cols) = q_q.shape().dims2()?;
    let (_, k_cols) = k_q.shape().dims2()?;
    let (_, v_cols) = v_q.shape().dims2()?;
    if input_cols != q_cols || input_cols != k_cols || input_cols != v_cols {
        return Ok(None);
    }

    #[cfg(has_marlin_kernels)]
    if flat_batch >= gguf::GGUF_AFFINE_MIN_BATCH {
        let q_ready = q.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        let k_ready = k.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        let v_ready = v.prepare_gguf_affine_raw(flat_batch, xs.dtype(), xs.device())?;
        if q_ready && k_ready && v_ready {
            let q_out = q
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine query projection");
            let k_out = k
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine key projection");
            let v_out = v
                .try_gguf_affine_forward_raw(xs)?
                .expect("prepared GGUF affine value projection");
            return Ok(Some((q_out, k_out, v_out)));
        }
    }

    let dtype = q_q.dtype();
    if dtype != k_q.dtype() || dtype != v_q.dtype() {
        return Ok(None);
    }
    if flat_batch <= gguf::fast_mmvq::MMVQ_MAX_BATCH {
        if !gguf::fast_mmvq::supports(dtype) {
            return Ok(None);
        }
        Ok(Some(gguf::fast_mmvq::fused_qkv(&q_q, &k_q, &v_q, xs)?))
    } else {
        if !gguf::fast_mmq::supports(dtype) {
            return Ok(None);
        }
        Ok(Some(gguf::fast_mmq::fused_qkv(&q_q, &k_q, &v_q, xs)?))
    }
}

/// Metal fused gate+up: single Metal kernel that does both matmuls with shared
/// x reads and applies the GLU activation in-register before writing one output.
#[cfg(feature = "metal")]
pub fn try_fused_gate_up_metal(
    xs: &Tensor,
    gate: &dyn QuantMethod,
    up: &dyn QuantMethod,
    activation: GluActivationType,
) -> Result<Option<Tensor>> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if gate.has_bias() || up.has_bias() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }
    if !xs.device().is_metal() {
        return Ok(None);
    }

    let Some(gi) = gate.afq_inner() else {
        return Ok(None);
    };
    let Some(ui) = up.afq_inner() else {
        return Ok(None);
    };
    if gi.bits != ui.bits || gi.group_size != ui.group_size {
        return Ok(None);
    }
    if gi.scales.dtype() != ui.scales.dtype() {
        return Ok(None);
    }
    if gi.w_q.rank() != 2 || ui.w_q.rank() != 2 {
        return Ok(None);
    }
    let k = xs.dim(D::Minus1)?;
    let n_gate = gi.w_q.dim(0)?;
    let n_up = ui.w_q.dim(0)?;
    if n_gate != n_up {
        return Ok(None);
    }
    let n = n_gate;
    // qmm_t kernel uses BM=32 tiles; for small M (decode) it wastes most of the
    // tile. Let the caller fall back to separate qmv-based forwards.
    let probe_m = xs.elem_count() / k;
    if probe_m < 16 {
        return Ok(None);
    }
    if k * gi.bits as usize / 8 / 4 != gi.w_q.dim(1)? {
        // unexpected pack factor; let the generic path handle it
        return Ok(None);
    }

    let act_code: u32 = match activation {
        GluActivationType::Silu => 0,
        GluActivationType::Gelu => 1,
        GluActivationType::GeluErf => 2,
        GluActivationType::Relu => 3,
    };

    let xs = xs.contiguous()?;
    let m = xs.elem_count() / k;
    if m == 0 {
        return Ok(None);
    }

    let (xs_storage, xs_layout) = xs.storage_and_layout();
    let Storage::Metal(xs_storage) = &*xs_storage else {
        return Ok(None);
    };
    let (g_w_s, _) = gi.w_q.storage_and_layout();
    let Storage::Metal(g_w_s) = &*g_w_s else {
        return Ok(None);
    };
    let (g_s_s, _) = gi.scales.storage_and_layout();
    let Storage::Metal(g_s_s) = &*g_s_s else {
        return Ok(None);
    };
    let (g_b_s, _) = gi.biases.storage_and_layout();
    let Storage::Metal(g_b_s) = &*g_b_s else {
        return Ok(None);
    };
    let (u_w_s, _) = ui.w_q.storage_and_layout();
    let Storage::Metal(u_w_s) = &*u_w_s else {
        return Ok(None);
    };
    let (u_s_s, _) = ui.scales.storage_and_layout();
    let Storage::Metal(u_s_s) = &*u_s_s else {
        return Ok(None);
    };
    let (u_b_s, _) = ui.biases.storage_and_layout();
    let Storage::Metal(u_b_s) = &*u_b_s else {
        return Ok(None);
    };

    let device = xs_storage.device().clone();
    let dtype = xs.dtype();
    let mut out_shape = xs.dims().to_vec();
    *out_shape.last_mut().unwrap() = n;
    let out = device.new_buffer(out_shape.iter().product(), dtype, "afq-gate-up-out")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("afq-gate-up");

    metal_kernels::call_afq_qmm_gate_up(
        device.device(),
        &encoder,
        metal_kernels::Kernels::global(),
        dtype,
        (
            xs_storage.buffer(),
            xs_layout.start_offset() * dtype.size_in_bytes(),
        ),
        g_w_s.buffer(),
        g_s_s.buffer(),
        g_b_s.buffer(),
        u_w_s.buffer(),
        u_s_s.buffer(),
        u_b_s.buffer(),
        &out,
        m,
        n,
        k,
        gi.bits as usize,
        gi.group_size as usize,
        act_code,
    )
    .map_err(candle_core::Error::wrap)?;

    let out_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            out,
            device.clone(),
            out_shape.iter().product(),
            dtype,
        )),
        Shape::from(out_shape),
    ));
    Ok(Some(out_t))
}

/// Metal fused QKV: single Metal kernel that handles all three projections,
/// routing per-tile to the right weight matrix.
#[cfg(feature = "metal")]
pub fn try_fused_qkv_metal(
    xs: &Tensor,
    q: &dyn QuantMethod,
    k: &dyn QuantMethod,
    v: &dyn QuantMethod,
) -> Result<Option<(Tensor, Tensor, Tensor)>> {
    use candle_core::{backend::BackendStorage, MetalStorage, Shape, Storage};

    if q.has_bias() || k.has_bias() || v.has_bias() {
        return Ok(None);
    }
    if !matches!(xs.dtype(), DType::BF16 | DType::F16 | DType::F32) {
        return Ok(None);
    }
    if !xs.device().is_metal() {
        return Ok(None);
    }

    let Some(qi) = q.afq_inner() else {
        return Ok(None);
    };
    let Some(ki) = k.afq_inner() else {
        return Ok(None);
    };
    let Some(vi) = v.afq_inner() else {
        return Ok(None);
    };
    if qi.bits != ki.bits || qi.bits != vi.bits {
        return Ok(None);
    }
    if qi.group_size != ki.group_size || qi.group_size != vi.group_size {
        return Ok(None);
    }
    if qi.scales.dtype() != ki.scales.dtype() || qi.scales.dtype() != vi.scales.dtype() {
        return Ok(None);
    }
    if qi.w_q.rank() != 2 || ki.w_q.rank() != 2 || vi.w_q.rank() != 2 {
        return Ok(None);
    }
    let n_q = qi.w_q.dim(0)?;
    let n_k = ki.w_q.dim(0)?;
    let n_v = vi.w_q.dim(0)?;
    // The kernel routes by tile-aligned column boundaries; require N_q and
    // N_k to be multiples of the tile width (32). For Gemma-style models
    // those are already 32-multiples; fall back when they're not.
    if n_q % 32 != 0 || n_k % 32 != 0 || n_v % 32 != 0 {
        return Ok(None);
    }
    let k_dim = xs.dim(D::Minus1)?;
    // qmm_t kernel uses BM=32; for small M (decode) the tile is mostly empty.
    // Fall back to separate qmv calls.
    let probe_m = xs.elem_count() / k_dim;
    if probe_m < 16 {
        return Ok(None);
    }

    let xs = xs.contiguous()?;
    let m = xs.elem_count() / k_dim;
    if m == 0 {
        return Ok(None);
    }

    let (xs_s, xs_l) = xs.storage_and_layout();
    let Storage::Metal(xs_s) = &*xs_s else {
        return Ok(None);
    };
    let qws = qi.w_q.storage_and_layout().0;
    let qss = qi.scales.storage_and_layout().0;
    let qbs = qi.biases.storage_and_layout().0;
    let kws = ki.w_q.storage_and_layout().0;
    let kss = ki.scales.storage_and_layout().0;
    let kbs = ki.biases.storage_and_layout().0;
    let vws = vi.w_q.storage_and_layout().0;
    let vss = vi.scales.storage_and_layout().0;
    let vbs = vi.biases.storage_and_layout().0;
    let (Storage::Metal(qw_m), Storage::Metal(qs_m), Storage::Metal(qb_m)) = (&*qws, &*qss, &*qbs)
    else {
        return Ok(None);
    };
    let (Storage::Metal(kw_m), Storage::Metal(ks_m), Storage::Metal(kb_m)) = (&*kws, &*kss, &*kbs)
    else {
        return Ok(None);
    };
    let (Storage::Metal(vw_m), Storage::Metal(vs_m), Storage::Metal(vb_m)) = (&*vws, &*vss, &*vbs)
    else {
        return Ok(None);
    };

    let device = xs_s.device().clone();
    let dtype = xs.dtype();
    let mut q_shape = xs.dims().to_vec();
    let mut k_shape = q_shape.clone();
    let mut v_shape = q_shape.clone();
    *q_shape.last_mut().unwrap() = n_q;
    *k_shape.last_mut().unwrap() = n_k;
    *v_shape.last_mut().unwrap() = n_v;
    let q_out = device.new_buffer(q_shape.iter().product(), dtype, "afq-qkv-q")?;
    let k_out = device.new_buffer(k_shape.iter().product(), dtype, "afq-qkv-k")?;
    let v_out = device.new_buffer(v_shape.iter().product(), dtype, "afq-qkv-v")?;

    let encoder = device.command_encoder()?;
    encoder.set_label("afq-qkv");

    metal_kernels::call_afq_qmm_qkv(
        device.device(),
        &encoder,
        metal_kernels::Kernels::global(),
        dtype,
        (xs_s.buffer(), xs_l.start_offset() * dtype.size_in_bytes()),
        qw_m.buffer(),
        qs_m.buffer(),
        qb_m.buffer(),
        kw_m.buffer(),
        ks_m.buffer(),
        kb_m.buffer(),
        vw_m.buffer(),
        vs_m.buffer(),
        vb_m.buffer(),
        &q_out,
        &k_out,
        &v_out,
        m,
        n_q,
        n_k,
        n_v,
        k_dim,
        qi.bits as usize,
        qi.group_size as usize,
    )
    .map_err(candle_core::Error::wrap)?;

    let q_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            q_out,
            device.clone(),
            q_shape.iter().product(),
            dtype,
        )),
        Shape::from(q_shape),
    ));
    let k_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            k_out,
            device.clone(),
            k_shape.iter().product(),
            dtype,
        )),
        Shape::from(k_shape),
    ));
    let v_t = Tensor::from((
        Storage::Metal(MetalStorage::new(
            v_out,
            device.clone(),
            v_shape.iter().product(),
            dtype,
        )),
        Shape::from(v_shape),
    ));
    Ok(Some((q_t, k_t, v_t)))
}

fn tensor_prefix(vb: &ShardedVarBuilder) -> String {
    let prefix = vb.prefix();
    if prefix.is_empty() {
        "<root>".to_string()
    } else {
        prefix
    }
}

fn missing_required_tensors(vb: &ShardedVarBuilder, required: &[&str]) -> Vec<String> {
    required
        .iter()
        .copied()
        .filter(|name| !vb.contains_tensor(name))
        .map(|name| safetensors::full_tensor_name(vb, name))
        .collect()
}

pub(crate) fn has_missing_required_tensors(vb: &ShardedVarBuilder, required: &[&str]) -> bool {
    required.iter().any(|name| !vb.contains_tensor(name))
}

pub(crate) fn make_dummy_or_error(
    context: &str,
    vb: &ShardedVarBuilder,
    required: &[&str],
) -> Result<Arc<dyn QuantMethod>> {
    let missing = missing_required_tensors(vb, required);
    if missing.is_empty() {
        candle_core::bail!(
            "Internal error: requested DummyLayer for {context} without missing tensors"
        );
    }

    let has_uqff_placeholder = required
        .iter()
        .any(|name| safetensors::is_uqff_dummy_tensor(vb, name));
    if !has_uqff_placeholder {
        candle_core::bail!(
            "Missing required tensor(s) for {context} at prefix `{}`: {}. Dummy layers are only allowed for tensors intentionally omitted while loading UQFF artifacts.",
            tensor_prefix(vb),
            missing.join(", ")
        );
    }

    Ok(Arc::new(DummyLayer::placeholder(DummyLayerInfo {
        context: context.to_string(),
        prefix: tensor_prefix(vb),
        missing_tensors: missing,
    })))
}

pub fn linear_no_bias(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    if let Some(source) = base_vb.weight_source() {
        let load_device = weight_source_load_device(&base_vb);
        if let Some(layer) =
            source.load_linear(&base_vb.prefix(), &load_device, Shard::default())?
        {
            let layer = maybe_wrap_dynamic_lora(
                &base_vb,
                layer,
                LoraLinearSpec::replicated(in_dim, out_dim),
            )?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()));
        }
    }
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
        if !vb.contains_tensor("weight") {
            make_dummy_or_error("linear_no_bias", &vb, &["weight"])?
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, None),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    let layer =
        maybe_wrap_dynamic_lora(&base_vb, layer, LoraLinearSpec::replicated(in_dim, out_dim))?;
    apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()))
}

pub fn linear(
    in_dim: usize,
    out_dim: usize,
    config: &Option<QuantizedConfig>,
    vb: ShardedVarBuilder,
) -> Result<Arc<dyn QuantMethod>> {
    let base_vb = vb.clone();
    if let Some(source) = base_vb.weight_source() {
        let load_device = weight_source_load_device(&base_vb);
        if let Some(layer) =
            source.load_linear(&base_vb.prefix(), &load_device, Shard::default())?
        {
            let layer = maybe_wrap_dynamic_lora(
                &base_vb,
                layer,
                LoraLinearSpec::replicated(in_dim, out_dim),
            )?;
            return apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()));
        }
    }
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
        if has_missing_required_tensors(&vb, &["weight", "bias"]) {
            make_dummy_or_error("linear", &vb, &["weight", "bias"])?
        } else {
            let weight = vb.get_with_hints((out_dim, in_dim), "weight", Default::default())?;
            let bias = vb.get_with_hints((out_dim,), "bias", Default::default())?;

            let layer = <UnquantLinear as QuantMethod>::new(QuantMethodConfig::Unquantized(
                Linear::new(weight, Some(bias)),
            ))?;
            Arc::new(layer) as Arc<dyn QuantMethod>
        }
    };
    let layer =
        maybe_wrap_dynamic_lora(&base_vb, layer, LoraLinearSpec::replicated(in_dim, out_dim))?;
    apply_immediate_isq_sharded(layer, base_vb, Some(Shard::default()))
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

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    struct DenseWeightSource;

    impl QuantizedWeightSource for DenseWeightSource {
        fn contains(&self, name: &str) -> bool {
            name == "foo.weight"
        }

        fn load_linear(
            &self,
            key: &str,
            device: &Device,
            _shard: Shard,
        ) -> Result<Option<Arc<dyn QuantMethod>>> {
            if key != "foo" {
                return Ok(None);
            }
            let weight = Tensor::zeros((3, 2), DType::F32, device)?;
            Ok(Some(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(weight, None)),
            )?)))
        }

        fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
            Ok(None)
        }

        fn shard_alignment(&self, _key: &str) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    #[cfg(feature = "cuda")]
    struct RecordingWeightSource {
        load_devices: Arc<std::sync::Mutex<Vec<bool>>>,
    }

    #[cfg(feature = "cuda")]
    impl QuantizedWeightSource for RecordingWeightSource {
        fn contains(&self, name: &str) -> bool {
            name == "foo.weight"
        }

        fn load_linear(
            &self,
            key: &str,
            device: &Device,
            _shard: Shard,
        ) -> Result<Option<Arc<dyn QuantMethod>>> {
            if key != "foo" {
                return Ok(None);
            }
            self.load_devices.lock().unwrap().push(device.is_cpu());
            let weight = Tensor::zeros((3, 2), DType::F32, device)?;
            Ok(Some(Arc::new(UnquantLinear::new(
                QuantMethodConfig::Unquantized(Linear::new(weight, None)),
            )?)))
        }

        fn load_optional_tensor(&self, _name: &str, _device: &Device) -> Result<Option<Tensor>> {
            Ok(None)
        }

        fn shard_alignment(&self, _key: &str) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> Result<usize> {
            Ok(1)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> Result<Option<usize>> {
            Ok(Some(1))
        }
    }

    fn empty_vb(make_dummy_regexes: Option<Vec<&str>>) -> ShardedVarBuilder {
        let backend: HashMap<String, Tensor> = HashMap::new();
        let make_dummy_regexes = make_dummy_regexes.map(|regexes| {
            Arc::new(
                regexes
                    .into_iter()
                    .map(Regex::new)
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .unwrap(),
            )
        });
        ShardedSafeTensors::wrap_with_dummy_regexes(
            backend,
            DType::F32,
            Device::Cpu,
            make_dummy_regexes,
        )
    }

    fn immediate_params(
        ty: Option<IsqType>,
        overrides: Vec<ImmediateIsqOverride>,
    ) -> ImmediateIsqParams {
        let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
        ImmediateIsqParams {
            guard: QuantizeOntoGuard::new(),
            ty,
            predicates: vec![Regex::new(r"\.weight$").unwrap()],
            promoted_predicates: vec![
                Regex::new(r"^model\.embed_tokens\.(?:weight|bias)$").unwrap()
            ],
            overrides,
            executor,
            capture: IsqCaptureMode::Immediate,
        }
    }

    #[test]
    fn sensitive_tensor_policy_promotes_expected_types() {
        for ty in [IsqType::AFQ2, IsqType::AFQ3, IsqType::AFQ4] {
            assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::AFQ6);
        }
        for ty in [IsqType::AFQ6, IsqType::AFQ8] {
            assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::AFQ8);
        }
        for ty in [
            IsqType::Q2K,
            IsqType::Q3K,
            IsqType::Q4K,
            IsqType::Q4_0,
            IsqType::Q4_1,
        ] {
            assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::Q6K);
        }
        for ty in [
            IsqType::Q5K,
            IsqType::Q6K,
            IsqType::Q8K,
            IsqType::Q5_0,
            IsqType::Q5_1,
            IsqType::Q8_0,
            IsqType::Q8_1,
        ] {
            assert_eq!(ty.promote_for_sensitive_tensor(), IsqType::Q8_0);
        }
        assert_eq!(IsqType::HQQ4.promote_for_sensitive_tensor(), IsqType::HQQ4);
    }

    #[test]
    fn ggml_pack_factor_uses_a_safe_integer_lower_bound() {
        let dtype = DType::BF16;
        let cases = [
            (IsqType::Q4K, GgmlDType::Q4K, 512, 144, 4, 128, 3, 170),
            (IsqType::Q6K, GgmlDType::Q6K, 512, 210, 3, 170, 2, 256),
            (IsqType::Q8_0, GgmlDType::Q8_0, 64, 34, 2, 32, 1, 64),
        ];

        for (
            ty,
            quantized_dtype,
            dense_bytes,
            packed_bytes,
            ceil_factor,
            ceil_bytes,
            factor,
            bytes,
        ) in cases
        {
            let actual_dense_bytes = dtype.size_in_bytes() * quantized_dtype.block_size();
            let actual_packed_bytes = quantized_dtype.type_size();
            assert_eq!(actual_dense_bytes, dense_bytes);
            assert_eq!(actual_packed_bytes, packed_bytes);
            assert_eq!(
                actual_dense_bytes.div_ceil(actual_packed_bytes),
                ceil_factor
            );
            assert_eq!(actual_dense_bytes / ceil_factor, ceil_bytes);
            assert!(ceil_bytes < actual_packed_bytes);
            assert_eq!(ty.pack_factor(dtype), factor);
            assert_eq!(actual_dense_bytes / factor, bytes);
            assert!(bytes >= actual_packed_bytes);
        }
    }

    #[test]
    fn ggml_and_f8q8_pack_factors_never_undercount_block_storage() {
        let ggml_types = [
            (IsqType::Q4_0, GgmlDType::Q4_0),
            (IsqType::Q4_1, GgmlDType::Q4_1),
            (IsqType::Q5_0, GgmlDType::Q5_0),
            (IsqType::Q5_1, GgmlDType::Q5_1),
            (IsqType::Q8_0, GgmlDType::Q8_0),
            (IsqType::Q8_1, GgmlDType::Q8_1),
            (IsqType::Q2K, GgmlDType::Q2K),
            (IsqType::Q3K, GgmlDType::Q3K),
            (IsqType::Q4K, GgmlDType::Q4K),
            (IsqType::Q5K, GgmlDType::Q5K),
            (IsqType::Q6K, GgmlDType::Q6K),
            (IsqType::Q8K, GgmlDType::Q8K),
        ];

        for dtype in [DType::F16, DType::BF16, DType::F32] {
            for (ty, quantized_dtype) in ggml_types {
                let estimated_bytes =
                    quantized_dtype.block_size() / ty.pack_factor(dtype) * dtype.size_in_bytes();
                assert!(estimated_bytes >= quantized_dtype.type_size());
            }

            let f8q8_bytes = f8q8::QK8_0 / IsqType::F8Q8.pack_factor(dtype) * dtype.size_in_bytes();
            assert!(f8q8_bytes >= std::mem::size_of::<f8q8::BlockF8Q8>());

            let mxfp4_bytes =
                mxfp4::MXFP4_BLOCK_SIZE / IsqType::MXFP4.pack_factor(dtype) * dtype.size_in_bytes();
            let packed_mxfp4_bytes = mxfp4::MXFP4_BLOCK_SIZE * mxfp4::N_BITS / u8::BITS as usize
                + DType::U8.size_in_bytes();
            assert!(mxfp4_bytes >= packed_mxfp4_bytes);
        }
    }

    #[test]
    fn affine_pack_factors_include_group_metadata() {
        let afq_types = [
            (IsqType::AFQ2, 2),
            (IsqType::AFQ3, 3),
            (IsqType::AFQ4, 4),
            (IsqType::AFQ6, 6),
            (IsqType::AFQ8, 8),
        ];
        for (dtype, expected) in [
            (DType::F16, [5, 4, 3, 2, 1]),
            (DType::BF16, [5, 4, 3, 2, 1]),
            (DType::F32, [8, 6, 5, 4, 3]),
        ] {
            for ((ty, bits), expected_factor) in afq_types.into_iter().zip(expected) {
                let group_size = AfqGroupSize::Low as usize;
                let packed_bytes = (group_size * bits).div_ceil(u8::BITS as usize)
                    + AFFINE_METADATA_TENSOR_COUNT * dtype.size_in_bytes();
                assert_eq!(ty.pack_factor(dtype), expected_factor);
                assert!(group_size / ty.pack_factor(dtype) * dtype.size_in_bytes() >= packed_bytes);
            }
        }

        for (dtype, hqq4, hqq8) in [(DType::F16, 3, 1), (DType::BF16, 3, 1), (DType::F32, 6, 3)] {
            for (ty, bits, expected_factor) in [(IsqType::HQQ4, 4, hqq4), (IsqType::HQQ8, 8, hqq8)]
            {
                let packed_bytes = (hqq::ISQ_HQQ_GROUP_SIZE * bits).div_ceil(u8::BITS as usize)
                    + AFFINE_METADATA_TENSOR_COUNT * DType::F32.size_in_bytes();
                assert_eq!(ty.pack_factor(dtype), expected_factor);
                assert!(
                    hqq::ISQ_HQQ_GROUP_SIZE / ty.pack_factor(dtype) * dtype.size_in_bytes()
                        >= packed_bytes
                );
            }
        }
    }

    #[test]
    fn configured_pack_factors_use_their_storage_contracts() {
        let afq = QuantizedConfig::Afq {
            bits: 4,
            group_size: 64,
        };
        assert_eq!(afq.pack_factor(DType::F32), 6);

        let legacy_mxfp4 = QuantizedConfig::GptqAwq {
            bits: 40,
            group_size: 32,
            checkpoint_format: None,
            is_awq: false,
        };
        assert_eq!(legacy_mxfp4.pack_factor(DType::BF16), 3);
        assert_eq!(legacy_mxfp4.pack_factor(DType::F32), 6);

        let bnb8 = QuantizedConfig::Bitsandbytes {
            bnb_4bit_quant_type: None,
        };
        assert_eq!(bnb8.pack_factor(DType::BF16), 1);
    }

    #[test]
    fn immediate_isq_promotion_requires_an_explicit_predicate() {
        let params = immediate_params(Some(IsqType::AFQ4), Vec::new());
        let promoted = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
        assert_eq!(promoted.ty, Some(IsqType::AFQ6));
        assert!(promoted.promote_default);

        let lookalike = resolve_immediate_isq(&params, "vision.embed_tokens.weight").unwrap();
        assert_eq!(lookalike.ty, Some(IsqType::AFQ4));
        assert!(!lookalike.promote_default);
    }

    #[test]
    fn immediate_isq_explicit_type_wins_over_sensitive_default() {
        let params = immediate_params(
            Some(IsqType::AFQ4),
            vec![ImmediateIsqOverride {
                predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
                layer_range: None,
                ty: Some(IsqType::AFQ2),
                device: None,
            }],
        );

        let matched = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
        assert_eq!(matched.ty, Some(IsqType::AFQ2));
        assert!(matched.promote_default);
    }

    #[test]
    fn immediate_isq_uses_sensitive_default_only_for_sensitive_tensors() {
        for (base, sensitive) in [
            (IsqType::AFQ6, IsqType::AFQ8),
            (IsqType::Q4K, IsqType::Q6K),
            (IsqType::Q6K, IsqType::Q8_0),
        ] {
            let params = immediate_params(Some(base), Vec::new());
            let embedding = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
            assert_eq!(embedding.ty, Some(sensitive));
            let projection =
                resolve_immediate_isq(&params, "model.layers.0.mlp.down_proj.weight").unwrap();
            assert_eq!(projection.ty, Some(base));
        }
    }

    #[test]
    fn immediate_isq_device_only_override_keeps_sensitive_default() {
        let params = immediate_params(
            Some(IsqType::AFQ4),
            vec![ImmediateIsqOverride {
                predicate: Some(Regex::new(r"^model\.embed_tokens\.weight$").unwrap()),
                layer_range: None,
                ty: None,
                device: Some(Device::Cpu),
            }],
        );

        let matched = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
        assert_eq!(matched.ty, Some(IsqType::AFQ6));
    }

    #[test]
    fn capture_all_records_promotion_without_pinning_a_type() {
        let mut params = immediate_params(Some(IsqType::AFQ4), Vec::new());
        params.capture = IsqCaptureMode::CaptureAll;

        let promoted = resolve_immediate_isq(&params, "model.embed_tokens.weight").unwrap();
        assert_eq!(promoted.ty, None);
        assert!(promoted.promote_default);

        let regular =
            resolve_immediate_isq(&params, "model.layers.0.mlp.down_proj.weight").unwrap();
        assert_eq!(regular.ty, None);
        assert!(!regular.promote_default);
    }

    #[test]
    fn missing_linear_weight_outside_uqff_errors() {
        let err = linear_no_bias(2, 3, &None, empty_vb(None).pp("foo")).unwrap_err();
        let msg = err.to_string();

        assert!(msg.contains("Missing required tensor(s)"));
        assert!(msg.contains("foo.weight"));
        assert!(msg.contains("UQFF"));
    }

    #[test]
    fn missing_uqff_placeholder_creates_contextual_dummy() -> Result<()> {
        let layer = linear_no_bias(
            2,
            3,
            &None,
            empty_vb(Some(vec![r"^foo\.weight$"])).pp("foo"),
        )?;

        let info = layer.dummy_info().unwrap();
        assert_eq!(layer.name(), "dummy");
        assert_eq!(info.context, "linear_no_bias");
        assert_eq!(info.prefix, "foo");
        assert_eq!(info.missing_tensors, vec!["foo.weight"]);

        let input = Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?;
        let err = layer.forward_raw(&input).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("forward pass"));
        assert!(msg.contains("foo.weight"));
        assert!(msg.contains("temporary UQFF placeholders"));

        Ok(())
    }

    #[test]
    fn weight_source_linear_participates_in_immediate_isq_tracking() -> Result<()> {
        let ty = Some(IsqType::Q8_0);
        let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
        set_immediate_isq_config(
            ImmediateIsqConfig::new(
                ty,
                vec![Regex::new(r"^foo\.weight$").unwrap()],
                IsqCaptureMode::CaptureMatches,
            ),
            executor,
        );

        let vb = empty_vb(None)
            .with_weight_source(Arc::new(DenseWeightSource))
            .pp("foo");
        let tracker = vb.tracker().clone();
        let layer = linear_no_bias(2, 3, &None, vb);
        clear_immediate_isq();

        layer?.forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?;
        assert_eq!(tracker.get().len(), 1);
        assert_eq!(tracker.get()[0].key, "foo");
        assert_eq!(tracker.get()[0].ty, ty);
        Ok(())
    }

    #[test]
    fn weight_source_precedes_checkpoint_quantization_for_plain_linears() -> Result<()> {
        let config = Some(QuantizedConfig::GptqAwq {
            bits: 4,
            group_size: 128,
            checkpoint_format: None,
            is_awq: true,
        });
        let vb = || {
            empty_vb(None)
                .with_weight_source(Arc::new(DenseWeightSource))
                .pp("foo")
        };
        for layer in [
            linear_no_bias(2, 3, &config, vb())?,
            linear(2, 3, &config, vb())?,
        ] {
            assert_eq!(layer.name(), "unquant-linear");
            assert_eq!(
                layer
                    .forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?
                    .dims(),
                &[1, 3]
            );
        }
        Ok(())
    }

    #[test]
    fn uqff_source_precedes_checkpoint_quantization_for_plain_linears() -> Result<()> {
        let dir = tempfile::tempdir().map_err(candle_core::Error::wrap)?;
        let path = dir.path().join("source-first.uqff");
        let mut tensors = uqff_version_tensors();
        for (prefix, bias) in [("no_bias", false), ("with_bias", true)] {
            let bias = if bias {
                Some(Tensor::zeros(3, DType::F32, &Device::Cpu)?)
            } else {
                None
            };
            let layer = UnquantLinear::new(QuantMethodConfig::Unquantized(Linear::new(
                Tensor::zeros((3, 2), DType::F32, &Device::Cpu)?,
                bias,
            )))?;
            tensors.extend(layer.serialize_uqff(prefix, IsqType::Q4K)?);
        }
        ::safetensors::serialize_to_file(
            tensors.iter().map(|tensor| (tensor.name(), tensor)),
            None,
            &path,
        )
        .map_err(candle_core::Error::wrap)?;

        let reader = Arc::new(UqffReader::open(&[path])?);
        let config = Some(QuantizedConfig::GptqAwq {
            bits: 4,
            group_size: 128,
            checkpoint_format: None,
            is_awq: true,
        });
        let vb = |prefix| empty_vb(None).with_uqff_reader(reader.clone()).pp(prefix);
        for layer in [
            linear_no_bias(2, 3, &config, vb("no_bias"))?,
            linear(2, 3, &config, vb("with_bias"))?,
        ] {
            assert_eq!(layer.name(), "unquant-linear");
            assert_eq!(
                layer
                    .forward_raw(&Tensor::zeros((1, 2), DType::F32, &Device::Cpu)?)?
                    .dims(),
                &[1, 3]
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn weight_source_stages_immediate_isq_on_cpu() -> Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let load_devices = Arc::new(std::sync::Mutex::new(Vec::new()));
        let source = Arc::new(RecordingWeightSource {
            load_devices: load_devices.clone(),
        });
        let vb = ShardedSafeTensors::wrap(HashMap::new(), DType::F32, device)
            .with_weight_source(source)
            .pp("foo");

        let ty = Some(IsqType::Q8_0);
        let (executor, _) = create_isq_executor(IsqExecutorConfig::new(ty));
        set_immediate_isq_config(
            ImmediateIsqConfig::new(
                ty,
                vec![Regex::new(r"^foo\.weight$").unwrap()],
                IsqCaptureMode::CaptureMatches,
            ),
            executor,
        );
        let tracker = vb.tracker().clone();
        linear_no_bias(2, 3, &None, vb.clone())?;
        tracker.get()[0].ct.resolve()?;
        clear_immediate_isq();

        linear_no_bias(2, 3, &None, vb)?;
        assert_eq!(&*load_devices.lock().unwrap(), &[true, false]);
        Ok(())
    }
}
