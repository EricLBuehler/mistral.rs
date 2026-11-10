use std::{
    fmt::Debug,
    num::NonZeroUsize,
    sync::{atomic::AtomicUsize, Arc, Mutex, MutexGuard},
};

use candle_core::{
    quantized::{GgmlDType, QMatMul, QTensor},
    DType, Device, Result, Tensor,
};

#[cfg(feature = "metal")]
mod metal_fp8;
#[cfg(feature = "metal")]
pub mod metal_kernels;

mod afq;
mod bitsandbytes;
mod blockwise_fp8;
mod cpu_rms_norm;
pub mod cublaslt;
#[cfg(feature = "cuda")]
pub mod cuda_device;
#[cfg(test)]
#[path = "build_support/cuda_headers.rs"]
mod cuda_headers;
#[cfg(feature = "cuda")]
pub mod cuda_memory;
#[cfg(feature = "cuda")]
pub mod cuda_policy;
#[cfg(test)]
#[path = "build_support/cuda_toolkit.rs"]
mod cuda_toolkit;
#[cfg(cuda_cutile)]
pub mod cutile;
pub mod distributed;
mod dummy;
pub mod elementwise;
pub mod embedding;
pub mod f8q8;
mod fp8;
mod fp8_config;
pub mod gemv;
mod gguf;
mod gptq;
mod hqq;
mod imatrix;
mod isq_executor;
mod lfm3_conv;
pub mod lfm3_spatial_conv;
mod lora;
#[cfg(feature = "cuda")]
pub mod moe;
mod mxfp4;
mod mxfp8;
pub mod nvfp4;
mod pending_layer;
mod pertensor_fp8;
pub mod rotary;
mod routed_gemv;
pub mod routed_moe;
pub mod safetensors;
mod scalar_fp8;
mod unquantized;
mod uqff;
mod utils;

pub use cpu_rms_norm::cpu_rms_norm;
use gptq::gptq_linear;
pub use lfm3_conv::lfm3_causal_conv1d;
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

    /// Whether this source loads an ordinary bias-free dense linear for this key.
    fn supports_dense_packing(&self, _key: &str) -> Result<bool> {
        Ok(false)
    }

    fn load_optional_tensor(&self, name: &str, device: &Device) -> Result<Option<Tensor>>;

    fn shard_alignment(&self, key: &str) -> Result<usize>;

    fn pack_factor(&self, dtype: DType) -> Result<usize>;

    fn pack_factor_for(&self, key: &str, dtype: DType) -> Result<Option<usize>>;

    /// Reclaim copied source pages after uploads and shared weight bindings finish loading.
    fn release_source_cache(&self) -> Result<()> {
        Ok(())
    }

    /// Exact resident bytes of the bound weights, when the source knows them; device mapping
    /// falls back to the pack-factor estimate otherwise.
    fn resident_layer_bytes(&self, _dtype: DType) -> Result<Option<ResidentLayerBytes>> {
        Ok(None)
    }
}

/// Resident weight bytes per text layer, plus everything outside the repeating layers.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ResidentLayerBytes {
    pub layers: std::collections::BTreeMap<usize, usize>,
    pub outside: usize,
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

    fn supports_dense_packing(&self, key: &str) -> Result<bool> {
        (**self).supports_dense_packing(key)
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

    fn release_source_cache(&self) -> Result<()> {
        (**self).release_source_cache()
    }

    fn resident_layer_bytes(&self, dtype: DType) -> Result<Option<ResidentLayerBytes>> {
        (**self).resident_layer_bytes(dtype)
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

pub fn gguf_affine_reserved_bytes(device: &Device, dtype: DType) -> usize {
    #[cfg(all(feature = "cuda", has_marlin_kernels))]
    {
        gguf::gguf_affine_reserved_bytes(device, dtype)
    }
    #[cfg(not(all(feature = "cuda", has_marlin_kernels)))]
    {
        let _ = (device, dtype);
        0
    }
}

#[cfg(feature = "metal")]
pub use afq::ops::{
    afq_gather_qmm_rhs_sorted, afq_gather_qmm_rhs_sorted_gate_up, afq_sorted_gather_supported,
    metal_arg_sort_u32_1d, metal_moe_weighted_reduce_flat,
};
pub use afq::{AfqBits, AfqGroupSize, AfqInner, AfqLayer};
pub use bitsandbytes::{BnbLinear, BnbQuantParams, BnbQuantType};
pub use blockwise_fp8::{
    blockwise_fp8_moe, fp8_blockwise_dequantize, fp8_blockwise_quantize,
    fused_add_rms_norm_quantized, fused_add_rms_norm_quantized_with_normalized, BlockwiseFP8Linear,
};
pub use distributed::{
    layers::{
        compute_kv_shard, compute_n_kv_groups, validate_tp_head_layout, ColumnParallelLayer,
        PackedColumnParallel, PackedLinear, PackedOutputLayout, PreQuantizedExperts,
        ReplicatedLayer, RowParallelLayer,
    },
    socket::{Client, Server},
    BarrierLike, Comm, Id, RingConfig, SumAllReduce,
};
pub use dummy::{DummyLayer, DummyLayerInfo};
pub use embedding::{EmbeddingOutput, EmbeddingTable};
pub use f8q8::F8Q8Linear;
pub use fp8::FP8Linear;
pub use fp8_config::{
    CheckpointDialect, CheckpointLinearSpec, CheckpointQuantConfig, Fp8ActivationMode,
    Fp8LinearSpec, Fp8ScaleNames, Fp8WeightScaleLayout, Mxfp8LinearSpec, Nvfp4ActivationMode,
    Nvfp4LinearSpec, Nvfp4ScaleEncoding, Nvfp4ScaleNames, ScaleConvention, NVFP4_BLOCK_SIZE,
};
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
pub use gguf::embedding::GgufEmbedding;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use gguf::fast_mmq::grouped_from_glu_sorted_pair as grouped_moe_mmq_from_glu_sorted_pair;
#[cfg(feature = "cuda")]
pub use gguf::fast_mmq::{
    grouped as grouped_moe_mmq, grouped_from_glu_packed as grouped_moe_mmq_from_glu_packed,
    grouped_from_glu_pair as grouped_moe_mmq_from_glu_pair, grouped_pair as grouped_moe_mmq_pair,
    grouped_pair_packed as grouped_moe_mmq_pair_packed, supports as supports_mmq,
};
#[cfg(feature = "cuda")]
pub use gguf::gated_mix::{try_fused_quantized_gated_mix, QuantizedInputTransform};
pub use gguf::GgufMatMul;
pub use gguf::{
    GgufBindingMap, GgufBindingResolver, GgufTensorBackend, GgufTensorBinding, GgufWeightSource,
};
pub use gptq::GptqLayer;
pub use hqq::{HqqAxis, HqqBits, HqqConfig, HqqLayer};
pub use imatrix::{CollectedImatrixData, ImatrixLayerStats};
pub use isq_executor::{
    conservative_plan, elem_count, estimate_isq_weight_bytes, estimate_output_bytes,
    ggml_output_bytes, plan_weight_isq, tensor_bytes, IsqConsumer, IsqExecutor, IsqExecutorConfig,
    IsqJobOutput, IsqKernelKind, IsqPlanParams, IsqRequest, IsqResourceEstimate,
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
pub use nvfp4::{Nvfp4Layer, Nvfp4LayerParts};
pub use pending_layer::{pending_isq_channel, PendingIsqLayer};
pub use pertensor_fp8::{fp8_w8a16_linear, fp8_w8a8_linear, Fp8W8A8LinearArgs, PerTensorFP8Linear};
pub use routed_gemv::{
    try_routed_ffn, try_routed_gather, try_routed_glu, RoutedCodec, RoutedWeights,
};
pub use unquantized::UnquantLinear;
pub use utils::flash_attn_sinks_metal;
pub use utils::flash_attn_sinks_varlen_metal;
#[cfg(feature = "cuda")]
pub use utils::gptoss_swiglu_fused;
pub use utils::isq::{
    apply_immediate_isq, apply_immediate_isq_sharded, apply_immediate_isq_with_key,
    quantize_expert_stack, quantize_expert_stack_with_bias, requantize_tracked, RequantizeHandles,
};
pub use utils::softcap;
pub use utils::softmax_with_sinks;
pub use utils::{fused_glu, fused_split_glu, GluActivationType};
pub use utils::{log, BitWiseOp, CumSumOp, LeftshiftOp, NonZeroOp, SortOp};

use candle_nn::{Conv1d, Conv2d, Linear, Module};
use serde::{Deserialize, Deserializer, Serialize};

mod activation;
mod config;
mod immediate_isq;
mod isq_type;
mod linear;
mod method;
mod quantized_serde;

pub use activation::*;
pub use config::*;
pub use immediate_isq::*;
pub use isq_type::*;
pub use linear::*;
pub use method::*;
pub use quantized_serde::*;

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

/// Used to gate access to quantizing onto the host device
#[derive(Clone, Debug)]
#[allow(unused)]
pub struct QuantizeOntoGuard {
    pub inner: Arc<Mutex<()>>,
    module_key: Option<Arc<str>>,
    report: Option<QuantizationReport>,
    requested: Option<Arc<str>>,
    consumer: Option<IsqConsumer>,
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
            consumer: None,
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

    pub fn with_consumer(mut self, consumer: IsqConsumer) -> Self {
        self.consumer = Some(consumer);
        self
    }

    pub fn consumer(&self) -> Option<IsqConsumer> {
        self.consumer
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

#[cfg(test)]
#[path = "lib_tests.rs"]
mod tests;

#[cfg(all(test, feature = "metal"))]
mod fused_projection_metal_tests {
    use candle_core::{DType, Device, Result, Tensor};

    use crate::{
        try_fused_gate_up, AfqBits, AfqGroupSize, AfqLayer, GluActivationType, QuantMethod,
        QuantMethodConfig,
    };

    fn afq(device: &Device, out: usize, inp: usize, seed: f64) -> Result<AfqLayer> {
        let weight =
            (Tensor::randn(0f32, 1., (out, inp), device)? * seed)?.to_dtype(DType::BF16)?;
        AfqLayer::new(QuantMethodConfig::Afq {
            weight,
            bias: None,
            bits: AfqBits::Four,
            group_size: AfqGroupSize::Med,
        })
    }

    #[test]
    fn fused_gate_up_matches_separate_and_steps_aside_while_collecting_stats() -> Result<()> {
        let device = Device::new_metal(0)?;
        let (rows, hidden, inter) = (32, 512, 1024);
        let gate = afq(&device, inter, hidden, 0.05)?;
        let up = afq(&device, inter, hidden, 0.07)?;
        let xs = Tensor::randn(0f32, 1., (1, rows, hidden), &device)?.to_dtype(DType::BF16)?;

        let fused = try_fused_gate_up(&xs, &gate, &up, GluActivationType::Silu)?
            .expect("AFQ gate/up at 32 rows fuses on Metal");
        let reference = (candle_nn::ops::silu(&gate.forward(&xs)?)? * up.forward(&xs)?)?;
        let reference = reference.to_dtype(DType::F32)?;
        let rel = ((fused.to_dtype(DType::F32)? - &reference)?.abs()?
            / reference.abs()?.maximum(0.05)?)?
        .max_all()?
        .to_scalar::<f32>()?;
        assert!(rel < 3e-2, "fused gate/up relative error {rel}");

        gate.begin_track_stats()?;
        assert!(
            try_fused_gate_up(&xs, &gate, &up, GluActivationType::Silu)?.is_none(),
            "fusion must step aside so imatrix stats see every input"
        );
        Ok(())
    }
}
