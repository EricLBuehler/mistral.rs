mod amoe;
mod auto;
mod causal;
pub mod chat_template;
#[cfg(feature = "cuda")]
pub(crate) mod cuda_graph;
#[cfg(feature = "cuda")]
#[doc(hidden)]
pub use cuda_graph::CudaDecodeGraphLaunch;
mod diffusion;
mod embedding;
pub(crate) mod execution;
mod ggml;
mod gguf;
pub(crate) mod hf;
mod hf_config;
mod inputs_processor;
mod isq;
mod isq_flow;
#[cfg(feature = "metal")]
mod metal_warmup;
pub use isq_flow::CalibrationStatus;
#[cfg(any(feature = "cuda", test))]
mod cuda_graph_budget;
pub(crate) mod llg;
mod loaders;
mod macros;
mod multimodal;
mod normal;
mod paths;
mod plan_log;
mod processing;
pub(crate) mod prompt_chunks;
mod response;
pub(crate) mod sampling;
mod speech;
mod step;

pub use super::diffusion_models::DiffusionGenerationParams;
use crate::amoe::{AnyMoeConfig, AnyMoeExpertType, AnyMoeTrainingInputs, AnyMoeTrainingResult};
use crate::device_map::DeviceMapper;
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{
    AttentionBackendKind, CacheConfig, CacheEngine, CacheMemoryReservations, MemoryGoal,
    ModelConfigLike,
};
use crate::prefix_cacher::{PagedAuxiliaryPrefixState, PrefixCacheManagerV2};
use crate::IntervalLogger;
use crate::PagedAttentionConfig;
pub use amoe::{AnyMoeLoader, AnyMoePipeline};
pub use auto::{AutoLoader, AutoLoaderBuilder};
use chat_template::ChatTemplate;
pub use diffusion::{DiffusionLoader, DiffusionLoaderBuilder};
pub(crate) use embedding::EmbeddingLoadContext;
pub use embedding::{EmbeddingLoader, EmbeddingLoaderBuilder, EmbeddingSpecificConfig};
#[doc(hidden)]
pub use execution::{StepLookahead, StepSubmission};
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
pub use hf_config::HfConfigOverrides;
use image::DynamicImage;
pub use inputs_processor::InputProcessorOutput;
pub use isq::{
    expand_isq_value, expand_uqff_shards, parse_isq_value, parse_uqff_shard,
    resolve_uqff_report_output, resolve_uqff_shorthand, IsqModel, IsqOrganization, UqffWriteConfig,
    UQFF_MULTI_FILE_DELIMITER,
};
pub(crate) use isq::{sanitize_quantized_weight_source_config, IsqModelLoader};
use llguidance::toktrie::TokEnv;
#[cfg(feature = "cuda")]
pub(crate) use loaders::HostBacking;
pub(crate) use loaders::{apply_model_activation_memory, checkpoint_runtime_size};
pub use loaders::{
    ActivationMemoryContext, ActivationMemoryEstimate, ActivationModelConfig, AdapterKind,
    AutoDeviceMapParams, AutoDeviceMapQuantization, AutoEmbeddingLoader, AutoMultimodalLoader,
    AutoNormalLoader, DeviceMappedModelLoader, DiffusionLoaderType, DiffusionModel,
    DiffusionModelLoader, EmbeddingGemmaLoader, EmbeddingLoaderType, EmbeddingModel,
    EmbeddingModelLoader, EmbeddingModelPaths, EmbeddingModule, EmbeddingModulePaths,
    EmbeddingModuleType, FluxLoader, GemmaLoader, Idefics2Loader, LLaVALoader, LLaVANextLoader,
    LlamaLoader, Loader, LocalModelPaths, MistralLoader, MixtralLoader, ModelDownload, ModelKind,
    ModelPaths, MultimodalLoaderType, MultimodalModel, MultimodalModelLoader, NormalLoaderType,
    NormalLoadingMetadata, NormalModel, NormalModelLoader, Phi2Loader, Phi3Loader, Phi3VLoader,
    PrettyName, QuantizationKind, Qwen2Loader, Qwen3EmbeddingLoader, Starcoder2Loader, TokenSource,
};
#[cfg(test)]
pub(crate) use loaders::{
    GraniteMoeHybridLoader, Lfm3OmniLoader, Lfm3VlLoader, Qwen3NextLoader, Qwen3_5TextLoader,
};
pub use loaders::{LayerResource, ResourceSource};
#[allow(clippy::too_many_arguments)]
pub(crate) fn get_device_layers_for_loader(
    loader: &dyn loaders::DeviceMappedModelLoader,
    config: &str,
    num_layers: usize,
    mut layer_sizes_in_bytes: Vec<usize>,
    non_mapped_size_in_bytes: usize,
    total_model_size_in_bytes: usize,
    devices: &[Device],
    dtype: DType,
    params: &loaders::AutoDeviceMapParams,
    paged_attn_config: Option<&mut PagedAttentionConfig>,
) -> Result<(crate::device_map::DeviceMapMetadata, usize)> {
    let resources = loader.layer_resources(config, loaders::ResourceSource::default())?;
    let total_model_size_in_bytes = loaders::add_resource_bytes(
        &mut layer_sizes_in_bytes,
        total_model_size_in_bytes,
        &resources,
    )?;
    let mapping = loaders::auto_device_map::get_device_layers(
        loader,
        config,
        num_layers,
        layer_sizes_in_bytes,
        non_mapped_size_in_bytes,
        total_model_size_in_bytes,
        devices,
        dtype,
        params,
        paged_attn_config,
        &resources,
    )?;
    Ok((mapping, total_model_size_in_bytes))
}

fn finish_dynamic_lora_runtime(
    paths: &dyn ModelPaths,
    layers: Arc<mistralrs_quant::LoraLayerRegistry>,
    runtime_config: crate::LoraRuntimeConfig,
    live_updates: bool,
) -> Result<Arc<crate::DynamicLoraRuntime>> {
    let AdapterPaths::Lora(adapter_paths) = paths.get_adapter_paths() else {
        unreachable!("LoRA loaders require resolved LoRA adapter paths")
    };

    layers.finalize()?;
    let runtime = Arc::new(crate::DynamicLoraRuntime::new(
        layers,
        runtime_config,
        live_updates,
    )?);
    for adapter in adapter_paths {
        let info = runtime.load_from_safetensors(
            adapter.alias.clone(),
            adapter.source.clone(),
            adapter.revision.clone(),
            &adapter.config_path,
            &adapter.weights_path,
        )?;
        tracing::info!(
            alias = %info.alias,
            generation = %info.generation,
            rank = info.rank,
            bytes = info.bytes,
            "LoRA adapter preloaded"
        );
    }
    Ok(runtime)
}

use mistralrs_quant::{IsqType, QuantMethod};
pub use multimodal::{MultimodalLoader, MultimodalLoaderBuilder, MultimodalSpecificConfig};
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub(crate) use paths::{
    get_adapter_paths, get_chat_template, get_model_paths, AdapterPathOptions, XLoraPreload,
};
pub use paths::{AdapterPaths, ResolvedLoraAdapter};
pub use processing::DecisionForwardContext;
pub(crate) use processing::{
    apply_chat_template, BasicProcessor, MessagesAction, Processor, ProcessorCreator,
};
use rand_isaac::Isaac64Rng;
pub use speech::{SpeechLoader, SpeechPipeline};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokenizers::Tokenizer;

use anyhow::Result;
use candle_core::{DType, Device, DeviceLocation, IndexOp, Tensor, Var};

use crate::paged_attention::block_hash::{
    adapter_generation_key, compute_block_hashes, MultimodalAttentionPolicy,
};
use crate::sequence::{SeqStepType, Sequence, SequenceState};

use prompt_chunks::{
    build_prompt_chunk_plan, next_prompt_chunk_group, recurrent_checkpoint_boundary,
    PromptChunkPlan,
};

pub(crate) use self::inputs_processor::{
    is_inputs_processor_validation_error, InputsProcessorValidationError,
};
pub use self::inputs_processor::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType,
};
use self::text_models_inputs_processor::{
    FlashParams, PagedAttentionInputMetadata, PagedAttentionMeta,
};

#[cfg(feature = "cuda")]
pub(crate) fn synchronize_cuda_contexts(primary: &Device, mapper: &dyn DeviceMapper) -> Result<()> {
    let mut devices = mapper.get_unique_devices();
    if !devices.iter().any(|device| device.same_device(primary)) {
        devices.push(primary.clone());
    }
    for device in devices {
        if let Device::Cuda(cuda) = device {
            cuda.cuda_stream().context().synchronize()?;
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
pub(crate) fn prepare_cuda_runtime(primary: &Device, mapper: &dyn DeviceMapper) -> Result<()> {
    mistralrs_quant::cublaslt::maybe_init_cublas_lt_wrapper(primary.clone());
    let mut devices = mapper.get_unique_devices();
    if !devices.iter().any(|device| device.same_device(primary)) {
        devices.push(primary.clone());
    }
    for device in devices.iter().filter(|device| device.is_cuda()) {
        match crate::cuda::preload::preload_candle_ptx(device) {
            Ok(count) if count > 0 => {
                tracing::info!("Preloaded {count} Candle CUDA PTX functions.")
            }
            Ok(_) => {}
            Err(err) => tracing::warn!("Failed to preload Candle CUDA PTX functions: {err}"),
        }
    }
    #[cfg(feature = "cutile")]
    if let Err(err) = mistralrs_quant::cutile::warmup_moe_kernels(
        primary,
        crate::utils::progress::progress_suppressed(),
    ) {
        tracing::warn!("Failed to warm up cuTile MoE kernels: {err}");
    }
    synchronize_cuda_contexts(primary, mapper)
}

pub(crate) fn resolve_lora_execution(
    runtime: Option<&crate::DynamicLoraRuntime>,
    input_ids: &Tensor,
    paged_attn_meta: Option<&PagedAttentionInputMetadata>,
    flash_meta: &FlashParams,
    adapter_leases: &[Option<crate::AdapterLease>],
) -> candle_core::Result<Option<Arc<mistralrs_quant::LoraExecution>>> {
    let (batch, sequence_length) = input_ids.dims2()?;
    let query_lens = if flash_meta.packed {
        if batch != 1 {
            candle_core::bail!("packed adapter routing requires a flat physical batch");
        }
        let query_lens = paged_attn_meta
            .and_then(|metadata| metadata.query_lens.as_deref())
            .ok_or_else(|| {
                candle_core::Error::msg("packed adapter routing requires logical query lengths")
            })?;
        if adapter_leases.len() != query_lens.len() {
            candle_core::bail!(
                "adapter lease count {} does not match packed logical sequence count {}",
                adapter_leases.len(),
                query_lens.len()
            );
        }
        let logical_tokens = query_lens.iter().sum::<usize>();
        if logical_tokens != sequence_length {
            candle_core::bail!(
                "packed logical query lengths total {logical_tokens} does not match physical sequence length {sequence_length}"
            );
        }
        Some(query_lens)
    } else {
        if adapter_leases.len() != batch {
            candle_core::bail!(
                "adapter lease count {} does not match model batch size {batch}",
                adapter_leases.len()
            );
        }
        None
    };
    if adapter_leases.iter().all(Option::is_none) {
        return Ok(None);
    }
    let runtime = runtime.ok_or_else(|| {
        candle_core::Error::msg("request selected an adapter on a pipeline without dynamic LoRA")
    })?;
    match query_lens {
        Some(query_lens) => runtime
            .ragged_execution(adapter_leases, query_lens)
            .map(Some),
        None => runtime.execution(adapter_leases, sequence_length).map(Some),
    }
}

pub(crate) fn validate_lora_loader_config(
    adapters: Option<&[crate::LoraAdapterSpec]>,
    runtime_config: Option<crate::LoraRuntimeConfig>,
) -> anyhow::Result<()> {
    if let Some(runtime_config) = runtime_config {
        runtime_config.validate()?;
    }
    let Some(adapters) = adapters else {
        return Ok(());
    };
    let mut aliases = std::collections::HashSet::new();
    for adapter in adapters {
        let alias = adapter.alias.trim();
        if alias.is_empty() {
            anyhow::bail!("LoRA adapter alias must not be empty");
        }
        if alias.len() > crate::MAX_LORA_ALIAS_BYTES {
            anyhow::bail!(
                "LoRA adapter alias must not exceed {} bytes",
                crate::MAX_LORA_ALIAS_BYTES
            );
        }
        if adapter.source.trim().is_empty() {
            anyhow::bail!(
                "LoRA adapter source for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter.revision().is_empty() {
            anyhow::bail!(
                "LoRA adapter revision for alias `{}` must not be empty",
                adapter.alias
            );
        }
        if adapter
            .base_model_name
            .as_deref()
            .is_some_and(|model| model.trim().is_empty())
        {
            anyhow::bail!(
                "LoRA adapter `{}` has an empty base_model_name",
                adapter.alias
            );
        }
        if !aliases.insert(alias) {
            anyhow::bail!(
                "LoRA adapter alias `{}` is specified more than once",
                adapter.alias
            );
        }
    }
    if let Some(config) = runtime_config.filter(|config| aliases.len() > config.max_adapters) {
        anyhow::bail!(
            "LoRA adapter preload count {} exceeds the configured maximum {}",
            aliases.len(),
            config.max_adapters
        );
    }
    Ok(())
}

pub use crate::kv_cache::{
    Cache, CacheManager, EitherCache, KvCache, LayerCaches, NormalCache, NormalCacheType,
};

pub(crate) const RECURRENT_GRAPH_PAD_SLOTS: usize = 1;
const AUTO_RECURRENT_KV_FLOOR_FRACTION: f64 = 0.05;
const AUTO_RECURRENT_PREFIX_BUDGET_DIVISOR: usize = 2;
const BYTES_PER_MIB: usize = 1024 * 1024;

#[derive(Clone, Copy)]
struct RecurrentCheckpointBudget {
    requested_lanes: usize,
    capacity: usize,
    snapshot_bytes: usize,
    current_capacity: usize,
    current_lanes: usize,
    allocation_available: usize,
    future_reserved_bytes: usize,
    kv_floor_bytes: usize,
}

pub(crate) fn disable_unsupported_paged_attention(
    config: &mut Option<PagedAttentionConfig>,
) -> Result<()> {
    if config
        .as_ref()
        .is_some_and(|config| config.memory_goal != MemoryGoal::Auto)
    {
        anyhow::bail!("The requested memory goal requires paged attention, which this model or device placement does not support");
    }
    *config = None;
    Ok(())
}

fn effective_recurrent_checkpoint_lanes(requested: usize, supported: bool) -> usize {
    if supported {
        requested
    } else {
        1
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn automatic_recurrent_checkpoint_lane_budget(budget: RecurrentCheckpointBudget) -> Result<usize> {
    let current_layout_bytes = budget
        .snapshot_bytes
        .checked_mul(budget.current_capacity)
        .and_then(|bytes| bytes.checked_mul(budget.current_lanes))
        .ok_or_else(|| anyhow::anyhow!("current recurrent layout size overflow"))?;
    let capacity = budget.capacity.max(budget.current_capacity);
    let bytes_per_lane = budget
        .snapshot_bytes
        .checked_mul(capacity)
        .ok_or_else(|| anyhow::anyhow!("recurrent checkpoint lane size overflow"))?;
    if bytes_per_lane == 0 {
        anyhow::bail!("recurrent checkpoint lane size must be nonzero");
    }
    let steady_layout_budget = budget
        .allocation_available
        .saturating_add(current_layout_bytes)
        .saturating_sub(budget.future_reserved_bytes)
        .saturating_sub(budget.kv_floor_bytes);
    let steady_lanes = steady_layout_budget / bytes_per_lane;
    let allocation_lanes = budget.allocation_available / bytes_per_lane;
    let checkpoint_lanes = budget
        .requested_lanes
        .min(steady_lanes)
        .min(allocation_lanes);
    Ok(if checkpoint_lanes >= 2 {
        checkpoint_lanes
    } else {
        1
    })
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
fn recurrent_kv_floor_bytes(
    memory_config: MemoryGoal,
    memory_total: usize,
    kv_bytes_per_token: usize,
) -> Result<usize> {
    let bytes = match memory_config {
        MemoryGoal::KvBytes(bytes) => bytes,
        MemoryGoal::Auto => (memory_total as f64 * AUTO_RECURRENT_KV_FLOOR_FRACTION) as usize,
        MemoryGoal::Utilization(fraction) => {
            let floor_fraction = f64::from(fraction).min(AUTO_RECURRENT_KV_FLOOR_FRACTION);
            (memory_total as f64 * floor_fraction) as usize
        }
        MemoryGoal::KvTokens(tokens) => {
            let bytes = tokens
                .checked_mul(kv_bytes_per_token)
                .ok_or_else(|| anyhow::anyhow!("PagedAttention context memory size overflow"))?;
            bytes
                .div_ceil(BYTES_PER_MIB)
                .checked_mul(BYTES_PER_MIB)
                .ok_or_else(|| anyhow::anyhow!("PagedAttention context memory size overflow"))?
        }
    };
    Ok(bytes)
}

fn paged_kv_bytes_per_token(
    paged_attn_config: &PagedAttentionConfig,
    dtype: DType,
    model_config: &dyn ModelConfigLike,
) -> Result<usize> {
    paged_attn_config
        .cache_type
        .to_dtype(dtype)
        .size_in_bytes()
        .checked_mul(model_config.total_kv_cache_elements_per_token())
        .ok_or_else(|| anyhow::anyhow!("PagedAttention token memory size overflow"))
}

fn automatic_recurrent_checkpoint_lanes(
    cache: &EitherCache,
    paged_attn_config: &PagedAttentionConfig,
    primary_device: &Device,
    capacity: usize,
    kv_bytes_per_token: usize,
) -> Result<usize> {
    let hybrid = cache.hybrid();
    let snapshot_bytes_by_device = hybrid.recurrent_snapshot_bytes_by_device()?;
    let recurrent_devices = hybrid.recurrent_devices();
    let current_capacity = hybrid.recurrent_capacity();
    let current_lanes = hybrid.checkpoint_lanes();
    drop(hybrid);
    let reservations =
        paged_attention_memory_reservations(cache, paged_attn_config, primary_device)?;
    let mut selected_lanes = paged_attn_config.recurrent_checkpoint_lanes;
    for device in recurrent_devices {
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            crate::utils::memory_usage::MemoryUsage.trim_cuda_memory_pool(&device, 0)?;
        }
        let memory = crate::utils::memory_usage::MemoryUsage.query(&device)?;
        let snapshot_bytes = snapshot_bytes_by_device
            .get(&device.location())
            .copied()
            .ok_or_else(|| {
                candle_core::Error::msg(
                    "recurrent device is missing from the snapshot memory inventory",
                )
            })?;
        let future_reserved_bytes = if device.same_device(primary_device) {
            reservations.primary_device_bytes
        } else {
            reservations.secondary_device_bytes
        };
        let kv_floor_bytes = recurrent_kv_floor_bytes(
            paged_attn_config.memory_goal,
            memory.total(),
            kv_bytes_per_token,
        )?;
        selected_lanes = selected_lanes.min(automatic_recurrent_checkpoint_lane_budget(
            RecurrentCheckpointBudget {
                requested_lanes: paged_attn_config.recurrent_checkpoint_lanes,
                capacity,
                snapshot_bytes,
                current_capacity,
                current_lanes,
                allocation_available: paged_attn_config.allocation_budget(&device, memory)?,
                future_reserved_bytes,
                kv_floor_bytes,
            },
        )?);
    }
    Ok(selected_lanes)
}

fn reserve_recurrent_serving_capacity(
    cache: &EitherCache,
    paged_attn_config: &PagedAttentionConfig,
    recurrent_checkpoints_supported: bool,
    recurrent_transitions_supported: bool,
    primary_device: &Device,
    kv_bytes_per_token: usize,
) -> Result<bool> {
    if !cache.is_hybrid() {
        return Ok(false);
    }
    let requested_lanes = effective_recurrent_checkpoint_lanes(
        paged_attn_config.recurrent_checkpoint_lanes,
        recurrent_checkpoints_supported || recurrent_transitions_supported,
    );
    let transition_log = recurrent_transitions_supported
        && requested_lanes > 1
        && requested_lanes <= crate::cuda::gdn::GDN_SPEC_FUSED_MAX_TOKENS;
    let Some(serving_capacity) = paged_attn_config.serving_capacity else {
        return Ok(if transition_log {
            cache.hybrid().configure_transition_lanes(requested_lanes)?
        } else {
            cache.hybrid().configure_checkpoint_lanes(requested_lanes)?
        });
    };
    let capacity = serving_capacity
        .checked_add(RECURRENT_GRAPH_PAD_SLOTS)
        .ok_or_else(|| candle_core::Error::msg("recurrent serving capacity overflow"))?;
    let checkpoint_lanes = if !transition_log
        && paged_attn_config.recurrent_checkpoint_lanes_auto
        && requested_lanes > 1
    {
        automatic_recurrent_checkpoint_lanes(
            cache,
            paged_attn_config,
            primary_device,
            capacity,
            kv_bytes_per_token,
        )?
    } else {
        requested_lanes
    };
    if checkpoint_lanes != paged_attn_config.recurrent_checkpoint_lanes {
        tracing::info!(
            requested_lanes = paged_attn_config.recurrent_checkpoint_lanes,
            checkpoint_lanes,
            serving_capacity,
            "Adjusted recurrent speculative checkpoint depth for the serving memory budget"
        );
    }
    Ok(if transition_log {
        cache
            .hybrid()
            .reserve_recurrent_transition_layout(capacity, checkpoint_lanes)?
    } else {
        cache
            .hybrid()
            .reserve_recurrent_layout(capacity, checkpoint_lanes)?
    })
}

fn uses_recurrent_transition_log(cache: &EitherCache) -> bool {
    cache.is_hybrid() && cache.hybrid().uses_recurrent_transition_log()
}

#[derive(Clone, Copy)]
struct RecurrentPrefixBudget {
    snapshot_bytes: usize,
    persistent_bytes: usize,
    execution_bytes: usize,
}

impl RecurrentPrefixBudget {
    fn reserve_runtime(mut self, enabled: bool, baseline_activation: usize) -> Self {
        let allowance = crate::paged_attention::automatic_runtime_allowance(
            enabled,
            baseline_activation,
            self.persistent_bytes.min(self.execution_bytes),
        );
        self.persistent_bytes = self.persistent_bytes.saturating_sub(allowance);
        self.execution_bytes = self.execution_bytes.saturating_sub(allowance);
        self
    }
}

fn automatic_recurrent_prefix_capacity(
    requested: usize,
    budgets: &[RecurrentPrefixBudget],
) -> usize {
    budgets.iter().fold(requested, |capacity, budget| {
        if budget.snapshot_bytes == 0 {
            return capacity;
        }
        let persistent = budget.persistent_bytes / AUTO_RECURRENT_PREFIX_BUDGET_DIVISOR;
        let retained = persistent.min(budget.execution_bytes.saturating_sub(budget.snapshot_bytes));
        capacity.min(retained / budget.snapshot_bytes)
    })
}

fn configure_automatic_recurrent_prefix_capacity(
    cache: &EitherCache,
    config: &mut PagedAttentionConfig,
    mut request: crate::paged_attention::CacheConfigRequest<'_>,
) -> Result<()> {
    if !config.recurrent_prefix_capacity_auto
        || config.recurrent_prefix_capacity == 0
        || !cache.is_hybrid()
    {
        return Ok(());
    }
    let hybrid = cache.hybrid();
    let snapshot_bytes = hybrid.recurrent_snapshot_bytes_by_device()?;
    let devices = hybrid.recurrent_devices();
    drop(hybrid);
    request.memory_reservations = config
        .memory_reservations()
        .map_err(candle_core::Error::msg)?;
    let mut budgets = Vec::with_capacity(devices.len());
    for device in devices {
        device.synchronize()?;
        #[cfg(feature = "cuda")]
        if device.is_cuda() {
            crate::MemoryUsage.synchronize_cuda_context(&device)?;
            crate::MemoryUsage.trim_cuda_memory_pool(&device, 0)?;
        }
        let (persistent, transient) =
            crate::paged_attention::optional_weight_reservation(config, request, &device)
                .map_err(candle_core::Error::msg)?;
        let memory = crate::MemoryUsage.query(&device)?;
        let budget = RecurrentPrefixBudget {
            snapshot_bytes: snapshot_bytes[&device.location()],
            persistent_bytes: config
                .allocation_budget(&device, memory)
                .map_err(candle_core::Error::msg)?
                .saturating_sub(persistent),
            execution_bytes: config
                .execution_budget(&device, memory)
                .map_err(candle_core::Error::msg)?
                .saturating_sub(persistent)
                .saturating_sub(transient),
        };
        let baseline_activation = if device.same_device(request.device) {
            config.primary_activation_memory_reservation_bytes
        } else {
            config.mapped_activation_memory_reservation_bytes
        };
        budgets.push(budget.reserve_runtime(
            request.reserve_decode_graphs && device.is_cuda(),
            baseline_activation,
        ));
    }
    let requested = config.recurrent_prefix_capacity;
    config.recurrent_prefix_capacity = automatic_recurrent_prefix_capacity(requested, &budgets);
    if !request.silent && requested != config.recurrent_prefix_capacity {
        tracing::info!(
            requested,
            capacity = config.recurrent_prefix_capacity,
            "Adjusted automatic recurrent prefix capacity for the serving memory budget"
        );
    }
    Ok(())
}

fn add_recurrent_prefix_memory_reservations(
    mut reservations: CacheMemoryReservations,
    bytes_by_device: HashMap<DeviceLocation, usize>,
    primary_device: DeviceLocation,
    prefix_capacity: usize,
) -> Result<CacheMemoryReservations> {
    if prefix_capacity == 0 {
        return Ok(reservations);
    }
    let peak_snapshots = prefix_capacity
        .checked_add(1)
        .ok_or_else(|| candle_core::Error::msg("recurrent prefix capacity overflow"))?;
    let mut secondary_prefix_bytes = 0usize;
    let mut secondary_staging_bytes = 0usize;
    for (device, bytes_per_snapshot) in bytes_by_device {
        let bytes = bytes_per_snapshot
            .checked_mul(peak_snapshots)
            .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
        if device == primary_device {
            reservations.primary_device_bytes = reservations
                .primary_device_bytes
                .checked_add(bytes)
                .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
            reservations.primary_staging_bytes = reservations
                .primary_staging_bytes
                .checked_add(bytes_per_snapshot)
                .ok_or_else(|| candle_core::Error::msg("recurrent prefix staging overflow"))?;
        } else {
            secondary_prefix_bytes = secondary_prefix_bytes.max(bytes);
            secondary_staging_bytes = secondary_staging_bytes.max(bytes_per_snapshot);
        }
    }
    reservations.secondary_device_bytes = reservations
        .secondary_device_bytes
        .checked_add(secondary_prefix_bytes)
        .ok_or_else(|| candle_core::Error::msg("recurrent prefix reservation overflow"))?;
    reservations.secondary_staging_bytes = reservations
        .secondary_staging_bytes
        .checked_add(secondary_staging_bytes)
        .ok_or_else(|| candle_core::Error::msg("recurrent prefix staging overflow"))?;
    Ok(reservations)
}

#[cfg(feature = "cuda")]
fn paged_attention_runtime_memory_reservation(
    cache: &EitherCache,
    config: Option<&PagedAttentionConfig>,
    primary: &Device,
    devices: &[Device],
) -> Result<Option<Arc<crate::resource_plan::RuntimeMemoryReservation>>> {
    let Some(config) = config else {
        return Ok(None);
    };
    let prefix = if config.recurrent_prefix_capacity > 0 && cache.is_hybrid() {
        Some(crate::resource_plan::RecurrentPrefixReservation::new(
            cache.hybrid().recurrent_snapshot_bytes_by_device()?,
            config.recurrent_prefix_capacity,
        ))
    } else {
        None
    };
    Ok(config.runtime_memory_reservation(primary, devices, prefix))
}

fn paged_attention_memory_reservations(
    cache: &EitherCache,
    paged_attn_config: &PagedAttentionConfig,
    primary_device: &Device,
) -> Result<CacheMemoryReservations> {
    let reservations = paged_attn_config
        .memory_reservations()
        .map_err(candle_core::Error::msg)?;
    if paged_attn_config.recurrent_prefix_capacity == 0 || !cache.is_hybrid() {
        return Ok(reservations);
    }
    let bytes_by_device = cache.hybrid().recurrent_snapshot_bytes_by_device()?;
    add_recurrent_prefix_memory_reservations(
        reservations,
        bytes_by_device,
        primary_device.location(),
        paged_attn_config.recurrent_prefix_capacity,
    )
}

pub(crate) type DeviceTensorMap = HashMap<DeviceLocation, Tensor>;

pub(crate) fn metadata_rope_positions<'a>(
    metadata: &'a PagedAttentionInputMetadata,
    device: &Device,
) -> Option<&'a Tensor> {
    metadata
        .rope_positions
        .as_ref()
        .and_then(|positions| positions.get(&device.location()))
}

pub(crate) enum ForwardMaskCache<'a> {
    Normal(&'a [KvCache]),
    Paged(&'a [usize]),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecurrentBatchKind {
    Prefill,
    Decode,
    SpeculativeDecode,
}

pub(crate) fn recurrent_batch_kind_for_input(
    is_prompt: bool,
    has_staged_speculative_batch: bool,
) -> RecurrentBatchKind {
    if is_prompt {
        RecurrentBatchKind::Prefill
    } else if has_staged_speculative_batch {
        RecurrentBatchKind::SpeculativeDecode
    } else {
        RecurrentBatchKind::Decode
    }
}

#[derive(Clone, Debug)]
pub(crate) struct RecurrentMetadata {
    batch_kind: RecurrentBatchKind,
    state_indices: Tensor,
    state_indices_host: Option<Vec<u32>>,
}

impl RecurrentMetadata {
    pub(crate) fn new(
        batch_kind: RecurrentBatchKind,
        state_indices: Tensor,
        state_indices_host: Option<Vec<u32>>,
    ) -> Self {
        Self {
            batch_kind,
            state_indices,
            state_indices_host,
        }
    }

    pub(crate) fn batch_kind(&self) -> RecurrentBatchKind {
        self.batch_kind
    }

    pub(crate) fn state_indices(&self) -> &Tensor {
        &self.state_indices
    }

    pub(crate) fn state_indices_host(&self) -> Option<&[u32]> {
        self.state_indices_host.as_deref()
    }
}

impl PastKvLenCache for ForwardMaskCache<'_> {
    fn get_past_kv_len(&self) -> candle_core::Result<usize> {
        match self {
            Self::Normal(cache) => Ok(cache
                .iter()
                .map(KvCache::current_seq_len)
                .max()
                .unwrap_or(0)),
            Self::Paged(offsets) => offsets.get_past_kv_len(),
        }
    }
}

pub(crate) struct ModelForwardContext<'a> {
    paged: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
    seqlen_offsets: &'a [usize],
    text_position_cache: HashMap<(DeviceLocation, usize), Tensor>,
    context_lens: &'a [(usize, usize)],
    position_ids: &'a [usize],
    flash_params: &'a FlashParams,
    recurrent_metadata: Option<RecurrentMetadata>,
    recurrent_batch_kind: Option<RecurrentBatchKind>,
    requires_full_prefill_queries: bool,
}

impl<'a> ModelForwardContext<'a> {
    pub(crate) fn new(
        seqlen_offsets: &'a [usize],
        context_lens: &'a [(usize, usize)],
        position_ids: &'a [usize],
        metadata: Option<(&'a [(Tensor, Tensor)], &'a PagedAttentionInputMetadata)>,
        flash_params: &'a FlashParams,
    ) -> Self {
        Self {
            paged: metadata,
            seqlen_offsets,
            text_position_cache: HashMap::new(),
            context_lens,
            position_ids,
            flash_params,
            recurrent_metadata: None,
            recurrent_batch_kind: None,
            requires_full_prefill_queries: false,
        }
    }

    pub(crate) fn with_recurrent_batch_kind(
        mut self,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> Self {
        self.recurrent_batch_kind = Some(recurrent_batch_kind);
        self
    }

    pub(crate) fn with_recurrent_metadata(
        mut self,
        recurrent_metadata: Option<RecurrentMetadata>,
    ) -> Self {
        if let Some(metadata) = recurrent_metadata.as_ref() {
            self.recurrent_batch_kind = Some(metadata.batch_kind());
        }
        self.recurrent_metadata = recurrent_metadata;
        self
    }

    pub(crate) fn require_full_prefill_queries(&mut self) {
        self.requires_full_prefill_queries = true;
    }

    pub(crate) fn requires_full_prefill_queries(&self) -> bool {
        self.requires_full_prefill_queries
    }

    pub(crate) fn rope_positions(&self, device: &Device) -> Option<&'a Tensor> {
        self.paged
            .and_then(|(_, metadata)| metadata_rope_positions(metadata, device))
    }

    pub(crate) fn is_paged(&self) -> bool {
        self.paged.is_some()
    }

    pub(crate) fn seqlen_offsets(&self) -> &[usize] {
        self.seqlen_offsets
    }

    pub(crate) fn context_lens(&self) -> &[(usize, usize)] {
        self.context_lens
    }

    pub(crate) fn position_ids(&self) -> &[usize] {
        self.position_ids
    }

    pub(crate) fn position_ids_vec(&self) -> Vec<usize> {
        self.position_ids.to_vec()
    }

    pub(crate) fn flash_params(&self) -> &FlashParams {
        self.flash_params
    }

    pub(crate) fn recurrent_metadata(&self) -> Option<&RecurrentMetadata> {
        self.recurrent_metadata.as_ref()
    }

    pub(crate) fn recurrent_batch_kind(&self) -> Option<RecurrentBatchKind> {
        self.recurrent_batch_kind
    }

    pub(crate) fn prompt_chunk_attention_policy(&self) -> MultimodalAttentionPolicy {
        self.paged_input_metadata()
            .map_or(MultimodalAttentionPolicy::Causal, |metadata| {
                metadata.prompt_chunk_attention_policy
            })
    }

    pub(crate) fn paged_input_metadata(&self) -> Option<&PagedAttentionInputMetadata> {
        self.paged.map(|(_, metadata)| metadata)
    }

    pub(crate) fn paged_layer(
        &self,
        layer_idx: usize,
    ) -> Option<((Tensor, Tensor), &PagedAttentionInputMetadata)> {
        self.paged
            .map(|(kv_cache, metadata)| (kv_cache[layer_idx].clone(), metadata))
    }

    pub(crate) fn text_positions(
        &mut self,
        device: &Device,
        seq_len: usize,
    ) -> candle_core::Result<Option<&Tensor>> {
        if self.flash_params.packed {
            let positions = self.rope_positions(device).ok_or_else(|| {
                candle_core::Error::msg("packed prefill is missing RoPE positions")
            })?;
            return Ok(Some(positions));
        }
        if self.rope_positions(device).is_some() {
            return Ok(self.rope_positions(device));
        }
        let seqlen_offsets = self.seqlen_offsets;
        let location = device.location();
        let key = (location, seq_len);
        if let std::collections::hash_map::Entry::Vacant(entry) =
            self.text_position_cache.entry(key)
        {
            entry.insert(text_positions_tensor(seqlen_offsets, seq_len, device)?);
        }
        Ok(self.text_position_cache.get(&key))
    }

    pub(crate) fn text_positions_from_offsets(
        &self,
        seqlen_offsets: &[usize],
        seq_len: usize,
        device: &Device,
    ) -> candle_core::Result<Tensor> {
        text_positions_tensor(seqlen_offsets, seq_len, device)
    }

    pub(crate) fn is_first_prompt_chunk(&self) -> bool {
        self.paged
            .is_none_or(|(_, metadata)| metadata.is_first_prompt_chunk)
    }

    pub(crate) fn is_final_prompt_chunk(&self) -> bool {
        self.paged
            .is_none_or(|(_, metadata)| metadata.is_final_prompt_chunk)
    }

    pub(crate) fn needs_logits(&self) -> bool {
        self.paged.is_none_or(|(_, metadata)| metadata.needs_logits)
    }

    /// Runs `head` on the selected rows, or returns a rank-preserving placeholder when the pipeline
    /// discards this forward's logits.
    pub(crate) fn lm_head(
        &self,
        head: &dyn QuantMethod,
        xs: &Tensor,
    ) -> candle_core::Result<Tensor> {
        if self.needs_logits() {
            return head.forward(xs);
        }
        let mut dims = xs.dims().to_vec();
        *dims
            .last_mut()
            .expect("selected hidden states have a feature axis") = 1;
        Tensor::zeros(dims, xs.dtype(), xs.device())
    }

    pub(crate) fn mask_cache<'b>(&'b self, normal_cache: &'b [KvCache]) -> ForwardMaskCache<'b> {
        if self.paged.is_some() {
            ForwardMaskCache::Paged(self.seqlen_offsets)
        } else {
            ForwardMaskCache::Normal(normal_cache)
        }
    }

    pub(crate) fn logits(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
        let devices = [logits.device().clone()];
        let selection = if self.flash_params.packed {
            let query_lens = self
                .paged_input_metadata()
                .and_then(|metadata| metadata.query_lens.as_deref())
                .ok_or_else(|| {
                    candle_core::Error::msg("packed prefill requires logical query lengths")
                })?;
            LogitsSelection::from_packed_context_lens(
                logits,
                self.context_lens,
                query_lens,
                &devices,
            )?
        } else {
            LogitsSelection::from_context_lens(logits, self.context_lens, &devices)?
        };
        selection.select(logits)
    }
}

pub(crate) fn text_positions_tensor(
    seqlen_offsets: &[usize],
    seq_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut positions = Vec::with_capacity(seqlen_offsets.len() * seq_len);
    for offset in seqlen_offsets {
        for seq_idx in 0..seq_len {
            positions.push(u32::try_from(offset + seq_idx).map_err(candle_core::Error::wrap)?);
        }
    }
    Tensor::from_vec(positions, (seqlen_offsets.len() * seq_len,), device)
}

pub(crate) fn decode_positions_tensor(
    position_ids: &[usize],
    seq_len: usize,
    device: &Device,
) -> candle_core::Result<Tensor> {
    let mut positions = Vec::with_capacity(position_ids.len() * seq_len);
    for end in position_ids {
        let start = end.checked_sub(seq_len).ok_or_else(|| {
            candle_core::Error::msg(format!(
                "decode position end {end} is smaller than query length {seq_len}"
            ))
        })?;
        for position in start..*end {
            positions.push(u32::try_from(position).map_err(candle_core::Error::wrap)?);
        }
    }
    Tensor::from_vec(positions, (position_ids.len() * seq_len,), device)
}

#[derive(Clone, Debug)]
pub(crate) enum LogitsSelection {
    Decode {
        start: usize,
        len: usize,
    },
    Indices {
        indices: DeviceTensorMap,
        batch: usize,
        len: usize,
    },
    PackedIndices {
        indices: DeviceTensorMap,
        batch: usize,
        len: usize,
    },
    All,
}

impl LogitsSelection {
    pub(crate) fn from_context_lens(
        source: &Tensor,
        context_lens: &[(usize, usize)],
        devices: &[Device],
    ) -> candle_core::Result<Self> {
        let dims = source.dims();
        if dims.len() < 2 {
            candle_core::bail!("logits selection source must have rank >= 2");
        }
        let batch = dims[0];
        let seq_len = dims[1];
        if context_lens.len() != batch {
            candle_core::bail!(
                "logits selection batch mismatch: {} spans for batch {batch}",
                context_lens.len()
            );
        }
        let Some((first_start, first_len)) = context_lens.first().copied() else {
            candle_core::bail!("logits selection requires at least one span");
        };
        for (start, len) in context_lens.iter().copied() {
            let end = start
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::msg("logits selection span overflow"))?;
            if end > seq_len {
                candle_core::bail!(
                    "logits selection span ({start}, {len}) exceeds sequence length {seq_len}"
                );
            }
        }
        if context_lens.iter().all(|span| *span == (0, seq_len)) {
            return Ok(Self::All);
        }
        if context_lens
            .iter()
            .all(|span| *span == (first_start, first_len))
        {
            return Ok(Self::Decode {
                start: first_start,
                len: first_len,
            });
        }

        if context_lens.iter().any(|(_, len)| *len != first_len) {
            candle_core::bail!("ragged logits selection spans are not supported");
        }

        let mut flat_indices = Vec::with_capacity(batch * first_len);
        for (batch_idx, (start, len)) in context_lens.iter().copied().enumerate() {
            let end = start + len;
            for pos in start..end {
                let idx = batch_idx
                    .checked_mul(seq_len)
                    .and_then(|idx| idx.checked_add(pos))
                    .ok_or_else(|| candle_core::Error::msg("logits selection index overflow"))?;
                flat_indices.push(u32::try_from(idx).map_err(candle_core::Error::wrap)?);
            }
        }

        let cpu_indices = Tensor::from_vec(flat_indices, (batch * first_len,), &Device::Cpu)?;
        let mut indices = HashMap::new();
        for device in devices {
            indices.insert(device.location(), cpu_indices.to_device(device)?);
        }
        Ok(Self::Indices {
            indices,
            batch,
            len: first_len,
        })
    }

    pub(crate) fn from_packed_context_lens(
        source: &Tensor,
        context_lens: &[(usize, usize)],
        query_lens: &[usize],
        devices: &[Device],
    ) -> candle_core::Result<Self> {
        let (physical_batch, physical_seq_len, _) = source.dims3()?;
        if context_lens.len() != query_lens.len() {
            candle_core::bail!(
                "packed logits selection length mismatch: {} spans for {} queries",
                context_lens.len(),
                query_lens.len()
            );
        }
        let total_tokens = query_lens.iter().sum::<usize>();
        if physical_batch * physical_seq_len != total_tokens {
            candle_core::bail!(
                "packed logits selection token mismatch: source has {} rows, queries have {total_tokens}",
                physical_batch * physical_seq_len
            );
        }
        let Some((_, output_len)) = context_lens.first().copied() else {
            candle_core::bail!("packed logits selection requires at least one span");
        };
        if context_lens.iter().any(|(_, len)| *len != output_len) {
            candle_core::bail!("ragged packed logits selection spans are not supported");
        }

        let mut indices = Vec::with_capacity(context_lens.len() * output_len);
        let mut base = 0usize;
        for ((start, len), query_len) in context_lens.iter().copied().zip(query_lens) {
            let end = start
                .checked_add(len)
                .ok_or_else(|| candle_core::Error::msg("packed logits selection span overflow"))?;
            if end > *query_len {
                candle_core::bail!(
                    "packed logits selection span ({start}, {len}) exceeds query length {query_len}"
                );
            }
            for position in start..end {
                indices.push(u32::try_from(base + position).map_err(candle_core::Error::wrap)?);
            }
            base += query_len;
        }

        let batch = query_lens.len();
        let cpu_indices = Tensor::from_vec(indices, (batch * output_len,), &Device::Cpu)?;
        let mut device_indices = HashMap::new();
        for device in devices {
            device_indices.insert(device.location(), cpu_indices.to_device(device)?);
        }
        Ok(Self::PackedIndices {
            indices: device_indices,
            batch,
            len: output_len,
        })
    }

    pub(crate) fn select(&self, logits: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            Self::All => Ok(logits.clone()),
            Self::Decode { start, len } => {
                let seq_len = logits.dim(1)?;
                if *start == 0 && *len == seq_len {
                    Ok(logits.clone())
                } else {
                    logits.narrow(1, *start, *len)
                }
            }
            Self::Indices {
                indices,
                batch,
                len,
            } => {
                let (logits_batch, seq_len, hidden) = logits.dims3()?;
                if logits_batch != *batch {
                    candle_core::bail!(
                        "logits selection batch mismatch: logits batch {logits_batch}, selection batch {batch}"
                    );
                }
                let indices = indices
                    .get(&logits.device().location())
                    .ok_or_else(|| candle_core::Error::msg("missing logits selection indices"))?;
                let flat = logits.reshape((logits_batch * seq_len, hidden))?;
                flat.index_select(indices, 0)?
                    .reshape((*batch, *len, hidden))
            }
            Self::PackedIndices {
                indices,
                batch,
                len,
            } => {
                let (physical_batch, physical_seq_len, hidden) = logits.dims3()?;
                let indices = indices
                    .get(&logits.device().location())
                    .ok_or_else(|| candle_core::Error::msg("missing logits selection indices"))?;
                logits
                    .reshape((physical_batch * physical_seq_len, hidden))?
                    .index_select(indices, 0)?
                    .reshape((*batch, *len, hidden))
            }
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum SupportedModality {
    Decision,
    Text,
    Audio,
    Vision,
    Video,
    Embedding,
}

impl Debug for SupportedModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Decision => write!(f, "Decision"),
            Self::Text => write!(f, "📝 Text"),
            Self::Audio => write!(f, "🔊 Audio"),
            Self::Vision => write!(f, "🖼️ Vision"),
            Self::Video => write!(f, "🎬 Video"),
            Self::Embedding => write!(f, "🔢 Embedding"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Modalities {
    pub input: Vec<SupportedModality>,
    pub output: Vec<SupportedModality>,
}

pub struct GeneralMetadata {
    /// Prompt-plus-output token limit per request, from `resolve_max_model_len`.
    pub max_model_len: usize,
    /// Only None if it doesn't make sense for the model
    pub llg_factory: Option<Arc<llguidance::ParserFactory>>,
    pub no_kv_cache: bool,
    pub no_prefix_cache: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub kind: ModelKind,
    // TODO: Replace is_xlora queries to check via kind instead:
    pub is_xlora: bool,
    pub activation_dtype: DType,
    pub sliding_window: Option<usize>,
    // PagedAttention stuff
    pub cache_config: Option<CacheConfig>,
    pub cache_engine: Option<CacheEngine>,
    pub memory_plan: Option<crate::memory_plan::RuntimeMemoryPlan>,
    #[cfg(feature = "cuda")]
    pub runtime_memory_reservation: Option<Arc<crate::resource_plan::RuntimeMemoryReservation>>,
    pub model_metadata: Option<Arc<dyn ModelConfigLike + Send + Sync>>,
    pub modalities: Modalities,
    // UQFF writes force the whole model onto CPU, so the pipeline is not servable afterwards.
    pub loaded_for_uqff_write: bool,
}

impl GeneralMetadata {
    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_capture_memory_ready(&self) -> bool {
        self.runtime_memory_reservation
            .as_ref()
            .is_none_or(|reservation| reservation.capture_ready())
    }

    pub fn tok_env(&self) -> Option<TokEnv> {
        self.llg_factory.as_ref().map(|f| f.tok_env().clone())
    }
}

/// Caps each request at the model's context, the paged KV capacity, and the requested `max_model_len`.
pub(crate) fn resolve_max_model_len(
    model_max: usize,
    requested: Option<usize>,
    paged: Option<(&CacheConfig, MemoryGoal)>,
    silent: bool,
) -> anyhow::Result<usize> {
    if let Some(requested) = requested.filter(|&len| len > model_max) {
        anyhow::bail!(
            "max_model_len {requested} exceeds the model's context length of {model_max} tokens"
        );
    }
    let kv = paged.map(|(cache, goal)| (cache.usable_tokens(), goal));
    if let (Some(requested), Some((kv_tokens, goal))) = (requested, kv) {
        if requested > kv_tokens {
            let remedy = match goal {
                MemoryGoal::KvBytes(_) | MemoryGoal::KvTokens(_) => "raise --kv-cache",
                MemoryGoal::Utilization(_) => {
                    "raise --gpu-memory-utilization, use a smaller --quant, or use --pa-cache-type f8e4m3"
                }
                MemoryGoal::Auto => "use a smaller --quant or --pa-cache-type f8e4m3",
            };
            anyhow::bail!(
                "max_model_len {requested} does not fit in the KV cache, which holds {kv_tokens} tokens. Lower max_model_len or {remedy}."
            );
        }
    }
    let kv_tokens = kv.map(|(tokens, _)| tokens);
    let limit = requested
        .unwrap_or(model_max)
        .min(kv_tokens.unwrap_or(usize::MAX));
    if silent {
        return Ok(limit);
    }
    let kv_summary = kv_tokens.map_or(String::new(), |kv| {
        let requests = kv / limit;
        let plural = if requests == 1 { "" } else { "s" };
        format!(" KV cache: {kv} tokens, room for {requests} full-length request{plural} at once.")
    });
    if requested.is_some() {
        tracing::info!("Context: up to {limit} tokens per request (max_model_len).{kv_summary}");
    } else if limit < model_max {
        tracing::warn!(
            "Context: up to {limit} tokens per request (model supports {model_max}; limited by KV cache memory).{kv_summary}"
        );
    } else {
        tracing::info!("Context: up to {limit} tokens per request.{kv_summary}");
    }
    Ok(limit)
}

#[derive(Clone, Copy)]
pub enum CacheInstruction {
    In,
    Out,
    /// load_preallocated_cache means to load the preallocated cache, if applicable.
    Reset {
        load_preallocated_cache: bool,
        reset_non_granular: bool,
    },
    Nothing,
}

pub trait PreProcessingMixin: MetadataMixin {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(BasicProcessor)
    }
    /// Only None if it doesnt make sense for the model
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>>;
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>>;
}

pub trait IsqPipelineMixin {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()>;

    /// Start collecting activation statistics from live traffic on every ISQ-tracked layer.
    fn begin_calibration(&mut self) -> Result<()> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    fn calibration_status(&self) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }

    /// Requantize with the collected statistics and swap the layers into the live model.
    fn apply_calibration(
        &mut self,
        _save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<isq_flow::CalibrationStatus> {
        anyhow::bail!("This pipeline does not support online calibration.")
    }
}

pub trait CacheManagerMixin {
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()>;
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> candle_core::Result<()>;
    fn cache(&self) -> &EitherCache;
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    /// Only None if it doesnt make sense for the model
    fn tokenizer(&self) -> Option<Arc<Tokenizer>>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    /// Destroy decode graphs at teardown, while the engine thread's cuTile modules are still loaded.
    fn cleanup_cuda_graphs(&self) {}
    /// Evict least-recently-used decode graphs without disturbing recurrent state.
    fn reclaim_cuda_graph_memory(&self, _max_entries: usize) -> usize {
        0
    }
    /// Capture the decode graphs for the common batch sizes up front, so live requests never pay for
    /// an eager step plus a capture when the batch composition changes.
    fn precapture_cuda_decode_graphs(&self, _ctx: &DecodeGraphPrecaptureCtx) {}
    fn get_metadata(&self) -> Arc<GeneralMetadata>;
    fn encoder_activation_memory_bytes(
        &self,
        _seq: &Sequence,
    ) -> candle_core::Result<Option<usize>> {
        Ok(None)
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        None
    }
    fn supports_reasoning_budget(&self) -> bool {
        true
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper>;
    fn execution_devices(&self) -> Vec<Device> {
        let primary = self.device();
        let mut devices = self
            .device_mapper()
            .map(DeviceMapper::get_unique_devices)
            .unwrap_or_default();
        if !devices.iter().any(|device| device.same_device(&primary)) {
            devices.push(primary);
        }
        devices
    }
}

/// Implemented by the base model of an AnyMoe.
pub trait AnyMoePipelineMixin {
    /// Get vars for each gating layer
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        unreachable!()
    }
    fn amoe_finish_training(&mut self, _gate_model_id: Option<String>) -> candle_core::Result<()> {
        unreachable!()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        unreachable!()
    }
    fn amoe_supported(&self) -> bool {
        false
    }
    /// Per-layer cached outputs.
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        unreachable!()
    }
    /// Inject the MoE layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_create_layers(
        &mut self,
        _model_ids: Vec<String>,
        _token: &TokenSource,
        _revision: Option<String>,
        _match_regex: &str,
        _config: AnyMoeConfig,
        _dtype: DType,
        _dev: &Device,
        (_prefix, _mlp): (String, String),
        _layers: Vec<usize>,
        _expert_type: AnyMoeExpertType,
        _silent: bool,
        _gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        unreachable!()
    }
    /// Pre-train the gating layers
    #[allow(clippy::too_many_arguments)]
    fn amoe_pre_train(
        &self,
        _inputs: AnyMoeTrainingInputs,
        (_prefix, _mlp): (String, String),
        _model_ids: Vec<String>,
        _token: TokenSource,
        _revision: Option<String>,
        _layers: Vec<usize>,
        _silent: bool,
    ) -> Result<Option<AnyMoeTrainingResult>, candle_core::Error> {
        unreachable!()
    }
}

/// Category of the model. This can also be used to extract model-category specific tools,
/// such as the multimodal model prompt prefixer.
#[derive(Clone)]
pub enum ModelCategory {
    Decision {
        video_sampling: crate::VideoFrameSampling,
    },
    Text,
    Multimodal {
        prefixer: Arc<dyn MultimodalPromptPrefixer>,
        video_sampling: crate::VideoFrameSampling,
    },
    Diffusion,
    Audio,
    Speech,
    Embedding,
}

impl std::fmt::Debug for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ModelCategory::Decision { .. } => write!(f, "ModelCategory::Decision"),
            ModelCategory::Text => write!(f, "ModelCategory::Text"),
            ModelCategory::Multimodal { .. } => {
                write!(f, "ModelCategory::Multimodal {{ prefixer: .. }}")
            }
            ModelCategory::Diffusion => write!(f, "ModelCategory::Diffusion"),
            ModelCategory::Audio => write!(f, "ModelCategory::Audio"),
            ModelCategory::Speech => write!(f, "ModelCategory::Speech"),
            ModelCategory::Embedding => write!(f, "ModelCategory::Embedding"),
        }
    }
}

impl PartialEq for ModelCategory {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Decision { .. }, Self::Decision { .. }) => true,
            (Self::Text, Self::Text) => true,
            (Self::Multimodal { .. }, Self::Multimodal { .. }) => true,
            (Self::Audio, Self::Audio) => true,
            (Self::Speech, Self::Speech) => true,
            (Self::Diffusion, Self::Diffusion) => true,
            (Self::Embedding, Self::Embedding) => true,
            (
                Self::Text
                | Self::Decision { .. }
                | Self::Multimodal { .. }
                | Self::Diffusion
                | Self::Audio
                | Self::Speech
                | Self::Embedding,
                _,
            ) => false,
        }
    }
}

/// Prepend a vision tag appropriate for the model to the prompt. Image indexing is assumed that start at 0.
pub trait MultimodalPromptPrefixer: Send + Sync {
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_image(&self, _image_indices: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_audio(&self, _audio_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
    /// Prefix for inclusion in messages (may do nothing if the chat template handles it).
    fn prefix_video(&self, _video_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

/// Paged-attention facts the engine knows once the KV cache exists, needed to fabricate a decode step.
#[derive(Clone, Debug)]
pub struct DecodeGraphPrecaptureCtx {
    pub block_size: usize,
    pub max_batch_size: usize,
    pub max_paged_context_len: usize,
    pub attention_backend: AttentionBackendKind,
    pub sliding_window: Option<usize>,
    pub num_kv_heads: usize,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)]
pub enum CacheBackendMetadata {
    DefaultInstructions {
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    },
    PagedAttention {
        metadata: PagedAttentionMeta,
    },
}

#[derive(Clone, Debug)]
pub enum ForwardInputsResult {
    RawLogits {
        logits: Tensor,
    },
    Embeddings {
        embeddings: Tensor,
    },
    CausalGeneration {
        logits: Tensor,
    },
    Image {
        images: Vec<DynamicImage>,
    },
    Speech {
        pcms: Vec<Arc<Vec<f32>>>,
        rates: Vec<usize>,
        channels: Vec<usize>,
    },
    BlockGeneration {
        token_blocks: Vec<Vec<u32>>,
        denoise_time: std::time::Duration,
    },
}

impl ForwardInputsResult {
    fn index_bs(&self, bs_idx: usize) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.i(bs_idx)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.i(bs_idx)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.i(bs_idx)?,
            }),
            Self::Image { images } => Ok(Self::Image {
                images: vec![images[bs_idx].clone()],
            }),
            Self::Speech {
                pcms,
                rates,
                channels,
            } => Ok(Self::Speech {
                pcms: vec![pcms[bs_idx].clone()],
                rates: vec![rates[bs_idx]],
                channels: vec![channels[bs_idx]],
            }),
            Self::BlockGeneration {
                token_blocks,
                denoise_time,
            } => Ok(Self::BlockGeneration {
                token_blocks: vec![token_blocks[bs_idx].clone()],
                denoise_time: *denoise_time,
            }),
        }
    }

    fn to_device(&self, device: &Device) -> candle_core::Result<Self> {
        match self {
            Self::CausalGeneration { logits } => Ok(Self::CausalGeneration {
                logits: logits.to_device(device)?,
            }),
            Self::RawLogits { logits } => Ok(Self::RawLogits {
                logits: logits.to_device(device)?,
            }),
            Self::Embeddings { embeddings } => Ok(Self::Embeddings {
                embeddings: embeddings.to_device(device)?,
            }),
            Self::Image { .. } => Ok(self.clone()),
            Self::Speech { .. } => Ok(self.clone()),
            Self::BlockGeneration { .. } => Ok(self.clone()),
        }
    }

    fn into_cpu_for_batch(
        self,
        batch_size: usize,
        preserve_causal_generation: bool,
    ) -> candle_core::Result<Self> {
        if batch_size <= 1
            || preserve_causal_generation && matches!(&self, Self::CausalGeneration { .. })
        {
            return Ok(self);
        }
        self.to_device(&Device::Cpu)
    }
}

#[doc(hidden)]
pub struct ForwardStepResult {
    pub output: ForwardInputsResult,
    #[cfg(feature = "cuda")]
    pub(crate) cuda_decode: Option<cuda_graph::CudaDecodeGraphLaunch>,
}

impl ForwardStepResult {
    pub fn eager(output: ForwardInputsResult) -> Self {
        Self {
            output,
            #[cfg(feature = "cuda")]
            cuda_decode: None,
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn cuda_decode(
        output: ForwardInputsResult,
        launch: Option<cuda_graph::CudaDecodeGraphLaunch>,
    ) -> Self {
        Self {
            output,
            cuda_decode: launch,
        }
    }
}

/// One sequence's slice of a prompt chunk the target just processed, for proposers that keep their
/// own KV cache; `tokens` is the full prompt so the shifted next token is available past `range`.
pub struct SpeculativePromptRow {
    pub seq_idx: usize,
    pub range: (usize, usize),
    pub tokens: Vec<u32>,
}

pub struct SpeculativePromptChunk {
    /// In forward-batch order (row `i` of the batch is `rows[i].seq_idx`)
    pub rows: Vec<SpeculativePromptRow>,
    pub is_final_prompt_chunk: bool,
}

fn should_sample_step(
    is_prompt: bool,
    scheduler_visible_prompt_step: bool,
    is_final_prompt_chunk: bool,
) -> bool {
    !is_prompt || !scheduler_visible_prompt_step || is_final_prompt_chunk
}

fn should_try_speculative_sampling(
    is_prompt: bool,
    scheduler_visible_prompt_step: bool,
    is_final_prompt_chunk: bool,
    return_raw_logits: bool,
    supports_prompt_bootstrap: bool,
) -> bool {
    !return_raw_logits
        && (!is_prompt || supports_prompt_bootstrap)
        && should_sample_step(
            is_prompt,
            scheduler_visible_prompt_step,
            is_final_prompt_chunk,
        )
}

fn prompt_chunk_is_final(
    scheduler_visible_prompt_step: bool,
    scheduler_visible_prompt_is_final: bool,
    planned_final_prompt_chunk: bool,
) -> bool {
    if scheduler_visible_prompt_step {
        scheduler_visible_prompt_is_final
    } else {
        planned_final_prompt_chunk
    }
}

fn next_pipeline_prompt_chunk_group(
    plan_indices: &[usize],
    chunk_plans: &[Vec<PromptChunkPlan>],
    requires_uniform_prompt_batch: bool,
    supports_packed_prefill: bool,
    hybrid_recurrent: bool,
) -> Option<(Vec<usize>, MultimodalAttentionPolicy, bool)> {
    let planned_final_prompt_chunk = plan_indices
        .iter()
        .zip(chunk_plans)
        .find_map(|(&plan_idx, plan)| plan.get(plan_idx).map(|_| plan_idx + 1 == plan.len()))?;
    let require_uniform_query_len = requires_uniform_prompt_batch
        || !supports_packed_prefill
        || (hybrid_recurrent && !planned_final_prompt_chunk);
    next_prompt_chunk_group(plan_indices, chunk_plans, require_uniform_query_len)
}

#[async_trait::async_trait]
pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + MetadataMixin
    + AnyMoePipelineMixin
{
    fn finalize_response(&self, _seq: &mut crate::sequence::Sequence) -> candle_core::Result<()> {
        Ok(())
    }
    fn prepare_streaming_response(
        &self,
        _seq: &mut crate::sequence::Sequence,
    ) -> candle_core::Result<()> {
        Ok(())
    }
    fn release_generation_sequences(&self, _sequence_ids: &[usize]) {}
    fn response_token_is_hidden(&self, _token: u32) -> bool {
        false
    }

    fn requires_uniform_prompt_batch(&self) -> bool {
        true
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        true
    }

    fn requires_uniform_media_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        false
    }

    /// Logs the attention, sampling and speculation paths this pipeline resolved to.
    fn log_runtime_plan(&self) {}

    fn supports_packed_prefill(&self) -> bool {
        false
    }

    fn adapter_runtime(&self) -> Option<Arc<crate::DynamicLoraRuntime>> {
        None
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error>;

    fn forward_decision(
        &self,
        _sequence: &mut Sequence,
    ) -> Result<crate::decision::DecisionResponse> {
        anyhow::bail!("This pipeline does not support decision requests")
    }

    #[doc(hidden)]
    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardStepResult, candle_core::Error> {
        self.forward_inputs(inputs, return_raw_logits)
            .map(ForwardStepResult::eager)
    }

    #[cfg(feature = "cuda")]
    #[doc(hidden)]
    fn replay_cuda_decode_one_token(
        &mut self,
        _launch: CudaDecodeGraphLaunch,
    ) -> Result<Option<ForwardStepResult>, candle_core::Error> {
        Ok(None)
    }

    fn attach_speculative(
        &mut self,
        _config: crate::speculative::SpeculativeConfig,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support speculative decoding attachment.")
    }

    #[doc(hidden)]
    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        _runtime: crate::speculative::MtpRuntimeConfig,
    ) -> Result<(), candle_core::Error> {
        self.attach_speculative(config)
    }

    fn release_speculative_sequences(&mut self, _seq_ids: &[usize]) -> candle_core::Result<()> {
        Ok(())
    }

    fn flush_recurrent_speculative_transitions(
        &self,
        _seq_ids: &[usize],
    ) -> candle_core::Result<()> {
        Ok(())
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        false
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        crate::speculative::SpeculativePrefixReplay::NotRequired
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        false
    }

    fn speculative_prefix_checkpoint_policy(
        &self,
    ) -> crate::speculative::SpeculativePrefixCheckpointPolicy {
        crate::speculative::SpeculativePrefixCheckpointPolicy::new(
            self.speculative_prefix_replay(),
            self.supports_paged_auxiliary_prefix_state(),
        )
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
    ) -> Result<Option<Arc<dyn PagedAuxiliaryPrefixState>>, candle_core::Error> {
        Ok(None)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        _sequence_id: usize,
        _cached_tokens: usize,
        _state: &dyn PagedAuxiliaryPrefixState,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support auxiliary paged prefix state.")
    }

    /// Called after a prompt chunk forward so a speculative proposer with its own KV cache can
    /// process the chunk. Default: nothing to do.
    fn speculative_prompt_chunk(
        &mut self,
        _seqs: &[&mut Sequence],
        _chunk: &SpeculativePromptChunk,
        _metadata: &PagedAttentionMeta,
    ) -> Result<(), candle_core::Error> {
        Ok(())
    }

    /// Append pre-sampled token blocks (block-diffusion canvases) to the sequences via the
    /// standard per-token finalize path. Overridden by pipelines whose models emit
    /// `ForwardInputsResult::BlockGeneration`.
    async fn sample_block_gen(
        &self,
        _input_seqs: &mut [&mut Sequence],
        _token_blocks: Vec<Vec<u32>>,
        _denoise_times: Vec<std::time::Duration>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("This pipeline does not support block generation.")
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        _input_seqs: &mut [&mut Sequence],
        _logits: &[Tensor],
        _batched_logits: Option<&Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        _metadata: Option<PagedAttentionMeta>,
        _logger: &IntervalLogger,
    ) -> Result<bool, candle_core::Error> {
        Ok(false)
    }

    fn snapshot_paged_recurrent_prefix(
        &mut self,
        seq: &Sequence,
        prefix_cacher: &mut PrefixCacheManagerV2,
        block_size: usize,
        cached_tokens: usize,
    ) -> Result<(), candle_core::Error> {
        if cached_tokens == 0
            || !cached_tokens.is_multiple_of(block_size)
            || !self.cache().is_hybrid()
            || !prefix_cacher.accepts_paged_recurrent_prefix()
        {
            return Ok(());
        }
        let Some(slot_idx) = seq.recurrent_state_idx() else {
            return Ok(());
        };

        self.flush_recurrent_speculative_transitions(&[*seq.id()])?;

        let snapshots = self
            .cache()
            .hybrid()
            .snapshot_recurrent_state(*seq.id(), slot_idx)?;
        if snapshots.is_empty() {
            return Ok(());
        }
        let auxiliary = if self
            .speculative_prefix_checkpoint_policy()
            .uses_auxiliary_state(crate::scheduler::modality_signature(seq))
        {
            self.capture_paged_auxiliary_prefix_state(*seq.id(), cached_tokens)?
        } else {
            None
        };
        let adapter_key = adapter_generation_key(seq.adapter_generation());
        let block_hashes = compute_block_hashes(
            seq.get_toks(),
            block_size,
            seq.mm_features(),
            adapter_key.as_slice(),
        );
        let n_blocks = cached_tokens / block_size;
        if block_hashes.len() >= n_blocks {
            let owner = *block_hashes
                .last()
                .expect("recurrent prefix owner requires a full block");
            prefix_cacher.add_paged_recurrent_prefix(
                owner,
                block_hashes[..n_blocks].to_vec(),
                snapshots,
                auxiliary,
            );
        }
        Ok(())
    }

    /// Returns the total of model execution time.
    #[allow(clippy::too_many_arguments)]
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
        logger: &IntervalLogger,
    ) -> Result<Duration, candle_core::Error> {
        let completion = self
            .submit_step(
                input_seqs,
                is_prompt,
                return_raw_logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                backend_metadata,
                logger,
                StepLookahead::Disabled,
            )
            .await?
            .into_ready()
            .expect("lookahead-disabled pipeline step must complete eagerly");
        Ok(completion.duration())
    }

    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    async fn submit_step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        return_raw_logits: bool,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        backend_metadata: CacheBackendMetadata,
        logger: &IntervalLogger,
        lookahead: StepLookahead,
    ) -> Result<StepSubmission, candle_core::Error> {
        if matches!(self.category(), ModelCategory::Decision { .. }) {
            return step::submit_decision_step(self, input_seqs).await;
        }
        let call = step::StepCall {
            input_seqs,
            is_prompt,
            return_raw_logits,
            prefix_cacher,
            disable_eos_stop,
            rng,
            logger,
            lookahead,
        };
        match backend_metadata {
            CacheBackendMetadata::DefaultInstructions { pre_op, post_op } => {
                step::submit_default_step(self, call, pre_op, post_op).await
            }
            CacheBackendMetadata::PagedAttention { metadata } => {
                step::submit_paged_step(self, call, metadata).await
            }
        }
    }

    async fn try_sample_causal_gen_batched(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: &Tensor,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<bool, candle_core::Error> {
        Ok(false)
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;

    fn category(&self) -> ModelCategory;

    /// Return encoder cache hit/miss counters (hits, misses) if this pipeline has an encoder cache.
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        None
    }
}

pub(crate) fn extract_logits(
    logits: &Tensor,
    context_lens: Vec<(usize, usize)>,
) -> candle_core::Result<Tensor> {
    LogitsSelection::from_context_lens(logits, &context_lens, &[logits.device().clone()])?
        .select(logits)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
