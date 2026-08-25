use super::llg::build_llg_factory;
use super::{
    get_model_paths, paged_attention_memory_reservations, reserve_recurrent_serving_capacity,
    text_models_inputs_processor::ModelInputs, AdapterKind, CacheManager, DecodeGraphPrecaptureCtx,
    GeneralMetadata, Loader, ModelKind, ModelPaths, NormalModel, NormalModelLoader, TokenSource,
};
use super::{
    AnyMoePipelineMixin, CacheManagerMixin, EitherCache, ForwardInputsResult, ForwardStepResult,
    IsqOrganization, IsqPipelineMixin, MetadataMixin, ModelCategory, PreProcessingMixin,
};
use super::{
    AutoNormalLoader, DeepSeekV2Loader, DeepSeekV3Loader, GLM4Loader, GLM4MoeLiteLoader,
    GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader, GraniteMoeHybridLoader,
    HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, LlamaLoader, MistralLoader,
    MixtralLoader, NormalLoaderType, Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader,
    Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3_5TextLoader, SmolLm3Loader,
    Starcoder2Loader,
};
use crate::amoe::AnyMoeExpertType;
use crate::attention::ATTENTION_CHUNK_SIZE;
#[cfg(feature = "cuda")]
use crate::cuda::gdn::GDN_PAD_SLOT;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
#[cfg(feature = "cuda")]
use crate::kv_cache::RecurrentCheckpointStateSnapshot;
#[cfg(feature = "cuda")]
type SeqRecurrentCheckpointSnapshots = Vec<(usize, RecurrentCheckpointStateSnapshot)>;
#[cfg(feature = "cuda")]
type HybridStateIndicesSnapshot = (Option<Tensor>, Option<Vec<u32>>);
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};
use crate::lora::Ordering;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    capture_cuda_decode_graph, cuda_decode_graph_batch_kind_supported,
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, cuda_graph_batch_bucket,
    cuda_graph_precapture_batches, hybrid_graph_slots, install_hybrid_graph_state_indices,
    record_cuda_graph_dispatch, CudaDecodeGraphCaptureCtx, CudaDecodeGraphKey,
    CudaDecodeGraphLaunch, CudaDecodeGraphReplay, CudaDecodeGraphReplayInput, CudaDecodeGraphState,
    CudaGraphComponent, CudaGraphDecodeStep, CudaGraphDecodeStepInputs, CudaGraphDispatchMode,
    CudaGraphDispatchReason, CudaGraphEvent, CudaGraphEventGuard, CudaGraphPrecaptureInputs,
    CUDA_GRAPH_PRECAPTURE_MAX_BATCH,
};
use crate::pipeline::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::{AutoDeviceMapQuantization, QuantizationConfigShim};
use crate::pipeline::sampling::{sample_and_add_toks, sample_and_add_toks_batched};
use crate::pipeline::text_models_inputs_processor::InputMetadata;
#[cfg(feature = "cuda")]
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{
    get_chat_template, hf::build_api, Modalities, ModelForwardContext, RecurrentBatchKind,
    RecurrentMetadata, SupportedModality,
};
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::xlora_models::NonGranularState;
use crate::{
    api_dir_list, api_get_file, get_mut_arcmutex, get_paths, get_uqff_paths, lora_model_loader,
    normal_model_loader, xlora_model_loader, DeviceMapSetting, DynamicLoraRuntime, LoraAdapterSpec,
    LoraRuntimeConfig, PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor, Var};
use either::Either;
use hf_hub::Cache;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::str::FromStr;
#[cfg(feature = "cuda")]
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub struct NormalPipeline {
    model: Box<dyn NormalModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    non_granular_state: Option<NonGranularState>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    #[cfg(feature = "cuda")]
    cuda_decode_graph: StdMutex<CudaDecodeGraphState>,
    #[cfg(feature = "cuda")]
    cuda_sparse_rejection: StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>>,
    generation_defaults: Option<crate::ModelGenerationDefaults>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    source_weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>>,
    dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
}

pub(crate) struct NormalPipelineBuildArgs {
    pub model: Box<dyn NormalModel + Send + Sync>,
    pub tokenizer: Tokenizer,
    pub chat_template: ChatTemplate,
    pub generation_config: Option<GenerationConfig>,
    pub paged_attn_config: Option<PagedAttentionConfig>,
    pub dtype: DType,
    pub device: Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub silent: bool,
    pub max_kv_tokens: Option<usize>,
    pub no_kv_cache: bool,
    pub is_xlora: bool,
    pub kind: ModelKind,
    pub non_granular_index: Option<usize>,
    pub model_id: String,
    pub loaded_for_uqff_write: bool,
    pub tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    pub source_weight_files: Vec<PathBuf>,
    pub source_weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>>,
    pub dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
}

pub(crate) fn build_normal_pipeline(
    args: NormalPipelineBuildArgs,
) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
    let NormalPipelineBuildArgs {
        model,
        tokenizer,
        chat_template,
        generation_config,
        paged_attn_config,
        dtype,
        device,
        mapper,
        silent,
        max_kv_tokens,
        no_kv_cache,
        is_xlora,
        kind,
        non_granular_index,
        model_id,
        loaded_for_uqff_write,
        tracked_modules,
        source_weight_files,
        source_weight_source,
        dynamic_lora,
    } = args;

    let model_metadata = model.model_config();
    let num_hidden_layers = match model.cache() {
        EitherCache::Full(full) => full.lock().len(),
        EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
        EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
    };
    let recurrent_checkpoints_supported = model.supports_recurrent_speculative_checkpoints();
    let recurrent_pool_grew = paged_attn_config
        .map(|config| {
            reserve_recurrent_serving_capacity(
                model.cache(),
                config,
                recurrent_checkpoints_supported,
            )
        })
        .transpose()?
        .unwrap_or(false);
    #[cfg(feature = "cuda")]
    if recurrent_pool_grew {
        super::synchronize_cuda_contexts(&device, mapper.as_ref())?;
    }
    #[cfg(not(feature = "cuda"))]
    let _ = recurrent_pool_grew;

    let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
        let cache_config = calculate_cache_config(
            paged_attn_config.mem_gpu,
            paged_attention_memory_reservations(model.cache(), paged_attn_config, &device)?,
            paged_attn_config.block_size,
            dtype,
            paged_attn_config.cache_type,
            model_metadata.as_ref(),
            &device,
            &mapper
                .get_unique_devices()
                .into_iter()
                .map(Some)
                .collect::<Vec<_>>(),
            silent,
            None,
            max_kv_tokens,
        )?;
        let layer_devices = (0..num_hidden_layers)
            .map(|layer| mapper.device_for(layer, false).cloned())
            .collect();
        let cache_engine = CacheEngine::new(
            model_metadata.as_ref(),
            &cache_config,
            dtype,
            model.device(),
            layer_devices,
        )?;
        (Some(cache_config), Some(cache_engine))
    } else {
        (None, None)
    };

    let max_seq_len = model.max_seq_len();
    let llg_factory = build_llg_factory(tokenizer.clone())?;
    let generation_defaults = generation_config
        .as_ref()
        .and_then(GenerationConfig::generation_defaults);
    let eos = calculate_eos_tokens(&chat_template, generation_config.as_ref(), &tokenizer);
    let sliding_window = model.config().sliding_window;

    Ok(Arc::new(Mutex::new(NormalPipeline {
        model,
        tokenizer: tokenizer.into(),
        no_kv_cache,
        chat_template: Arc::new(chat_template),
        non_granular_state: non_granular_index.map(|tgt_non_granular_index| NonGranularState {
            non_granular_index: Arc::new(Mutex::new(0)),
            tgt_non_granular_index,
        }),
        model_id,
        metadata: Arc::new(GeneralMetadata {
            max_seq_len,
            llg_factory: Some(llg_factory),
            no_kv_cache,
            no_prefix_cache: is_xlora,
            num_hidden_layers,
            eos_tok: eos,
            kind,
            is_xlora,
            activation_dtype: dtype,
            sliding_window,
            cache_config,
            cache_engine,
            model_metadata: Some(model_metadata),
            modalities: Modalities {
                input: vec![SupportedModality::Text],
                output: vec![SupportedModality::Text],
            },
            loaded_for_uqff_write,
        }),
        #[cfg(feature = "cuda")]
        cuda_decode_graph: StdMutex::new(CudaDecodeGraphState::default()),
        #[cfg(feature = "cuda")]
        cuda_sparse_rejection: StdMutex::new(None),
        generation_defaults,
        mapper,
        tracked_modules,
        source_weight_files,
        source_weight_source,
        dynamic_lora,
    })))
}

fn normal_model_requires_uniform_prompt_batch(
    is_hybrid: bool,
    packed_prefill_available: bool,
    is_xlora: bool,
    has_speculative_proposer: bool,
) -> bool {
    (is_hybrid && !packed_prefill_available)
        || is_xlora
        || (has_speculative_proposer && !packed_prefill_available)
}

/// A loader for a "normal" (non-quantized) model.
pub struct NormalLoader {
    inner: Box<dyn NormalModelLoader>,
    model_id: String,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    prepared_source: Option<PreparedNormalSource>,
    mtp: bool,
}

#[derive(Clone)]
pub(crate) struct PreparedNormalSource {
    pub config: String,
    pub weights: mistralrs_quant::ShardedVarBuilder,
    pub tokenizer: Tokenizer,
    pub generation_config: Option<GenerationConfig>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub source_weight_files: Vec<PathBuf>,
    pub rope_pairing: crate::gguf::normal_registry::RopePairing,
}

pub(crate) fn new_dynamic_lora_registry(
    config: &str,
) -> Result<Arc<mistralrs_quant::LoraLayerRegistry>> {
    let config = serde_json::from_str::<serde_json::Value>(config)?;
    let qwen35_moe_identity = config
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .and_then(|architectures| architectures.first())
        .and_then(serde_json::Value::as_str)
        == Some("Qwen3NextForCausalLM")
        && config
            .get(crate::gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY)
            .and_then(serde_json::Value::as_str)
            == Some("tiled");
    let registry = if qwen35_moe_identity {
        mistralrs_quant::LoraLayerRegistry::new_with_site_prefix_alias(
            "model",
            "model.language_model",
        )?
    } else {
        mistralrs_quant::LoraLayerRegistry::new()
    };
    Ok(Arc::new(registry))
}

#[derive(Default)]
/// A builder for a loader for a "normal" (non-quantized) model.
pub struct NormalLoaderBuilder {
    model_id: Option<String>,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    mtp: bool,
}

#[derive(Clone, Default)]
/// Config specific to loading a normal model.
pub struct NormalSpecificConfig {
    pub topology: Option<Topology>,
    pub organization: IsqOrganization,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub hf_config_overrides: Option<super::HfConfigOverrides>,
    pub max_model_len: Option<usize>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
}

impl NormalLoaderBuilder {
    pub fn new(
        config: NormalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        no_kv_cache: bool,
        jinja_explicit: Option<String>,
    ) -> Self {
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
            jinja_explicit,
            no_kv_cache,
            hf_cache_path,
            ..Default::default()
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    fn with_adapter(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.xlora_model_id = Some(xlora_model_id);
        self.xlora_order = Some(xlora_order);
        self.no_kv_cache = no_kv_cache;
        self.tgt_non_granular_index = tgt_non_granular_index;
        self.model_id = if let Some(id) = self.model_id {
            Some(id)
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                self.xlora_order.as_ref().unwrap().base_model_id
            );
            Some(self.xlora_order.as_ref().unwrap().base_model_id.clone())
        };
        self
    }

    pub fn with_xlora(
        mut self,
        xlora_model_id: String,
        xlora_order: Ordering,
        no_kv_cache: bool,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::XLora,
        };
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(
        mut self,
        adapters: Vec<LoraAdapterSpec>,
        runtime_config: LoraRuntimeConfig,
    ) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::Lora,
        };
        self.lora_adapters = Some(adapters);
        self.lora_runtime_config = Some(runtime_config);
        self
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    /// If the loader type is not specified, loader type is automatically determined from the
    /// `architectures` array in the config.
    fn build_inner(
        self,
        loader_tp: Option<NormalLoaderType>,
        prepared_source: Option<PreparedNormalSource>,
    ) -> anyhow::Result<NormalLoader> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        let loader: Box<dyn NormalModelLoader> = match loader_tp {
            Some(NormalLoaderType::Mistral) => Box::new(MistralLoader),
            Some(NormalLoaderType::Gemma) => Box::new(GemmaLoader),
            Some(NormalLoaderType::Llama) => Box::new(LlamaLoader),
            Some(NormalLoaderType::Mixtral) => Box::new(MixtralLoader),
            Some(NormalLoaderType::Phi2) => Box::new(Phi2Loader),
            Some(NormalLoaderType::Phi3) => Box::new(Phi3Loader),
            Some(NormalLoaderType::Qwen2) => Box::new(Qwen2Loader),
            Some(NormalLoaderType::Gemma2) => Box::new(Gemma2Loader),
            Some(NormalLoaderType::Starcoder2) => Box::new(Starcoder2Loader),
            Some(NormalLoaderType::Phi3_5MoE) => Box::new(Phi3_5MoELoader),
            Some(NormalLoaderType::DeepSeekV2) => Box::new(DeepSeekV2Loader),
            Some(NormalLoaderType::DeepSeekV3) => Box::new(DeepSeekV3Loader),
            Some(NormalLoaderType::Qwen3) => Box::new(Qwen3Loader),
            Some(NormalLoaderType::GLM4) => Box::new(GLM4Loader),
            Some(NormalLoaderType::GLM4MoeLite) => Box::new(GLM4MoeLiteLoader),
            Some(NormalLoaderType::GLM4Moe) => Box::new(GLM4MoeLoader),
            Some(NormalLoaderType::Qwen3Moe) => Box::new(Qwen3MoELoader),
            Some(NormalLoaderType::SmolLm3) => Box::new(SmolLm3Loader),
            Some(NormalLoaderType::GraniteMoeHybrid) => Box::new(GraniteMoeHybridLoader),
            Some(NormalLoaderType::GptOss) => Box::new(GptOssLoader),
            Some(NormalLoaderType::HunYuanDenseV1) => Box::new(HunYuanDenseV1Loader),
            Some(NormalLoaderType::HunYuanMoEV1) => Box::new(HunYuanMoEV1Loader),
            Some(NormalLoaderType::Qwen3Next) => Box::new(Qwen3NextLoader),
            Some(NormalLoaderType::Qwen3_5) => Box::new(Qwen3_5TextLoader),
            Some(NormalLoaderType::Lfm2) => Box::new(Lfm2Loader),
            Some(NormalLoaderType::Lfm2Moe) => Box::new(Lfm2Loader),
            None => Box::new(AutoNormalLoader),
        };
        Ok(NormalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
            jinja_explicit: self.jinja_explicit,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            prepared_source,
            mtp: self.mtp,
        })
    }

    pub fn build(self, loader_tp: Option<NormalLoaderType>) -> anyhow::Result<Box<dyn Loader>> {
        Ok(Box::new(self.build_inner(loader_tp, None)?))
    }

    pub(crate) fn build_with_source(
        mut self,
        loader_tp: NormalLoaderType,
        source: PreparedNormalSource,
        kind: ModelKind,
    ) -> anyhow::Result<Box<dyn Loader>> {
        self.kind = kind;
        Ok(Box::new(self.build_inner(Some(loader_tp), Some(source))?))
    }
}

impl Loader for NormalLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let cache = self
            .hf_cache_path
            .clone()
            .map(Cache::new)
            .unwrap_or_default();
        GLOBAL_HF_CACHE.get_or_init(|| cache);

        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision.clone(),
            self,
            None,
            None,
            silent,
            self.config.from_uqff.is_some(),
            crate::pipeline::AdapterPathOptions {
                xlora_model_id: self.xlora_model_id.as_ref(),
                lora_adapters: self.lora_adapters.as_deref(),
                xlora_order: self.xlora_order.as_ref(),
                xlora_preload: crate::pipeline::XLoraPreload::Skip,
            }
        );
        *self
            .token_source
            .write()
            .expect("Failed to write to token source") = Some(token_source);
        *self.revision.write().expect("Failed to write to revision") = revision.clone();
        if let Some(from_uqff) = self.config.from_uqff.clone() {
            *self.from_uqff.write().unwrap() = Some(get_uqff_paths!(&from_uqff, self, silent));
        }
        self.load_model_from_path(
            paths?.as_ref(),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let config = match self.prepared_source.as_ref() {
            Some(source) => source.config.clone(),
            None => std::fs::read_to_string(paths.get_config_filename())?,
        };
        let config = if self.config.from_uqff.is_some() {
            super::isq::sanitize_quantized_weight_source_config(&config)?
        } else {
            config
        };
        let config = match &self.config.hf_config_overrides {
            Some(overrides) => overrides.apply(&config)?,
            None => config,
        };
        let config = if self.mtp {
            super::loaders::inject_mtp_config_flag(&config)?
        } else {
            config
        };
        let config = self
            .inner
            .runtime_config(&config, self.config.max_model_len)?
            .into_owned();
        super::loaders::validate_lora_qk_rope_layout(
            &config,
            self.lora_adapters.is_some() || self.xlora_model_id.is_some(),
        )?;

        if !self.inner.supports_paged_attention(&config)? {
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let use_nccl = mistralrs_quant::distributed::use_nccl();
        let write_uqff = self.config.write_uqff.is_some();
        let tensor_parallelism = distributed::resolve_tensor_parallelism(
            self.inner.model_config(&config)?.as_ref(),
            use_nccl,
            write_uqff,
        )?;
        let use_distributed = tensor_parallelism.is_enabled();
        let device = device.clone();

        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { worker_rank, .. } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl && !write_uqff {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(&device)?
        };
        #[cfg(feature = "cuda")]
        for device in &available_devices {
            if let Device::Cuda(dev) = device {
                unsafe { dev.disable_event_tracking() };
            }
        }
        let device = if use_distributed {
            available_devices[0].clone()
        } else {
            device
        };
        let uqff_reader = if let Some(from_uqff) = &*self.from_uqff.read().unwrap() {
            Some(Arc::new(mistralrs_quant::UqffReader::open(from_uqff)?))
        } else {
            None
        };
        let prepared_weight_source = self
            .prepared_source
            .as_ref()
            .and_then(|source| source.weights.weight_source().cloned());
        let has_prepared_weight_source = prepared_weight_source.is_some();
        let weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>> = uqff_reader
            .clone()
            .map(|reader| reader as Arc<dyn mistralrs_quant::QuantizedWeightSource>)
            .or(prepared_weight_source.clone());

        // If auto, convert to Map if not using nccl
        let mut max_kv_tokens: Option<usize> = None;
        if write_uqff {
            mapper = DeviceMapSetting::dummy();
        } else if use_distributed {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(params) = mapper.clone() {
            max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());
            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                match super::isq_flow::resolve_auto_device_map_sizing(
                    uqff_reader.is_some(),
                    has_prepared_weight_source,
                    in_situ_quant,
                ) {
                    sizing @ (super::isq_flow::AutoDeviceMapSizing::Uqff
                    | super::isq_flow::AutoDeviceMapSizing::PreparedWeightSource) => {
                        let source = weight_source
                            .as_ref()
                            .expect("selected weight-source sizing requires a weight source");
                        let quantization =
                            if matches!(sizing, super::isq_flow::AutoDeviceMapSizing::Uqff) {
                                AutoDeviceMapQuantization::weight_source(source.as_ref())
                            } else {
                                AutoDeviceMapQuantization::weight_source_with_topology(
                                    source.as_ref(),
                                    self.config.topology.as_ref(),
                                )
                            };
                        let weight_pack_factor = quantization
                            .conservative_pack_factor(dtype, source.pack_factor(dtype)?);
                        let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                            &config,
                            dtype,
                            weight_pack_factor,
                            None,
                        )?;
                        let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                            &config,
                            dtype,
                            weight_pack_factor,
                            Some(&quantization),
                            None,
                        )?;
                        let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                        (
                            layer_sizes_in_bytes,
                            non_mapped_size_in_bytes,
                            layer_sizes_sum + non_mapped_size_in_bytes,
                        )
                    }
                    super::isq_flow::AutoDeviceMapSizing::Isq(isq) => {
                        let moqe =
                            matches!(self.config.organization, IsqOrganization::MoeExpertsOnly);
                        let source_pack_factor = if let Some(source) = &prepared_weight_source {
                            source.pack_factor(dtype)?
                        } else {
                            QuantizationConfigShim::get_quant_config_pack_factor(&config, dtype)?
                        };
                        let target_pack_factor = isq.pack_factor(dtype);
                        let (weight_pack_factor, non_mapped_pack_factor, quantization) = if moqe {
                            let quantization = prepared_weight_source.as_ref().map_or_else(
                                || {
                                    AutoDeviceMapQuantization::isq(
                                        None,
                                        self.config.topology.as_ref(),
                                    )
                                },
                                |source| {
                                    AutoDeviceMapQuantization::weight_source_with_topology(
                                        source.as_ref(),
                                        self.config.topology.as_ref(),
                                    )
                                },
                            );
                            (
                                quantization.conservative_moqe_pack_factor(
                                    dtype,
                                    source_pack_factor,
                                    isq,
                                ),
                                source_pack_factor,
                                quantization,
                            )
                        } else {
                            let quantization = AutoDeviceMapQuantization::isq(
                                Some(isq),
                                self.config.topology.as_ref(),
                            );
                            (
                                quantization.conservative_pack_factor(dtype, target_pack_factor),
                                target_pack_factor,
                                quantization,
                            )
                        };
                        let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                            &config,
                            dtype,
                            weight_pack_factor,
                            None,
                        )?;
                        let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                            &config,
                            dtype,
                            non_mapped_pack_factor,
                            Some(&quantization),
                            None,
                        )?;
                        let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                        (
                            layer_sizes_in_bytes,
                            non_mapped_size_in_bytes,
                            layer_sizes_sum + non_mapped_size_in_bytes,
                        )
                    }
                    super::isq_flow::AutoDeviceMapSizing::Checkpoint => {
                        let inventory =
                            if self.config.topology.is_none() && self.lora_adapters.is_none() {
                                let num_layers = self.inner.num_layers(&config)?;
                                crate::pipeline::loaders::checkpoint_device_map_sizes(
                                    paths.get_weight_filenames(),
                                    num_layers,
                                    dtype,
                                    |name| self.inner.checkpoint_layer_index(&config, name),
                                )?
                            } else {
                                None
                            };
                        if let Some(inventory) = inventory {
                            info!(
                                model_mib = inventory.total_model_size_in_bytes / (1024 * 1024),
                                "Using checkpoint tensor inventory for automatic device mapping"
                            );
                            (
                                inventory.layer_sizes_in_bytes,
                                inventory.non_mapped_size_in_bytes,
                                inventory.total_model_size_in_bytes,
                            )
                        } else {
                            // Be sure to get the weight pack factor here; we might be loading a prequantized model.
                            let weight_pack_factor =
                                QuantizationConfigShim::get_quant_config_pack_factor(
                                    &config, dtype,
                                )?;
                            let quantization = self.config.topology.as_ref().map(|topology| {
                                AutoDeviceMapQuantization::isq(None, Some(topology))
                            });
                            let weight_pack_factor =
                                quantization
                                    .as_ref()
                                    .map_or(weight_pack_factor, |quantization| {
                                        quantization
                                            .conservative_pack_factor(dtype, weight_pack_factor)
                                    });
                            let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                                &config,
                                dtype,
                                weight_pack_factor,
                                None,
                            )?;
                            let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                                &config,
                                dtype,
                                weight_pack_factor,
                                quantization.as_ref(),
                                None,
                            )?;
                            let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                            (
                                layer_sizes_in_bytes,
                                non_mapped_size_in_bytes,
                                layer_sizes_sum + non_mapped_size_in_bytes,
                            )
                        }
                    }
                };

            let new = auto_device_map::get_device_layers(
                &*self.inner,
                &config,
                self.inner.num_layers(&config)?,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &available_devices,
                dtype,
                &params,
                paged_attn_config.as_mut(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        let mapper_device = if write_uqff {
            Device::Cpu
        } else {
            device.clone()
        };
        let mapper_topology = if write_uqff {
            None
        } else {
            self.config.topology.as_ref()
        };

        let pipeline_mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &mapper_device,
            mapper_topology,
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &mapper_device,
            mapper_topology,
            &available_devices,
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..self.inner.num_layers(&config)? {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }
        let dtype = super::isq_flow::resolve_weight_load_dtype(
            dtype,
            mapper.as_ref(),
            &available_devices,
            write_uqff,
        )?;

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu && paged_attn_config.is_some() {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        trace!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let topology_overrides = self
            .config
            .topology
            .as_ref()
            .map(|topology| topology.immediate_overrides())
            .unwrap_or_default();

        let plan = super::isq_flow::resolve_and_install_isq_plan(super::isq_flow::IsqPlanInputs {
            in_situ_quant,
            has_imatrix: self.config.imatrix.is_some(),
            has_calibration: self.config.calibration_file.is_some(),
            write_uqff_types: self.config.write_uqff.as_ref().map(|c| c.types.clone()),
            has_write_uqff: self.config.write_uqff.is_some(),
            loading_from_uqff: self.config.from_uqff.is_some(),
            organization: self.config.organization,
            topology_overrides,
            loader: &*self.inner,
            config: &config,
            device: &device,
        })?;
        let use_immediate = plan.immediate_isq_installed;
        let loading_isq = plan.loading_isq;
        let load_device = plan.load_device.clone();

        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

        // Load matformer slicing config if provided
        let matformer_slicing_config = if let Some(matformer_path) =
            &self.config.matformer_config_path
        {
            use crate::matformer::{MatformerConfig, MatformerSliceConfig};
            info!("Loading Matformer config from {:?}", matformer_path);
            let config = Arc::new(MatformerConfig::from_file(matformer_path)?);

            if let Some(slice_name) = &self.config.matformer_slice_name {
                info!("Using Matformer slice: {}", slice_name);
                Some(MatformerSliceConfig::new(slice_name.clone(), config))
            } else {
                // If no slice name is provided but config exists, we'll need to handle this
                // For now, return None and let the model handle the default slice selection
                warn!("Matformer config loaded but no slice name specified. Models will use their default slice.");
                None
            }
        } else {
            None
        };

        info!(
            "{}",
            WeightLoadingMode::from(WeightLoadingState {
                from_uqff: self.config.from_uqff.is_some(),
                loading_isq,
                immediate_isq: use_immediate,
                write_uqff: self.config.write_uqff.is_some(),
            })
            .message("model")
        );

        let (model, tracker, dynamic_lora) = if use_distributed {
            let distributed_weights = match self.prepared_source.as_ref() {
                Some(source) => {
                    distributed::DistributedWeightSource::Prepared(source.weights.clone())
                }
                None => distributed::DistributedWeightSource::Paths(paths),
            };
            let (mapper, sharded_vb) =
                distributed::prepare_distributed_mapper(distributed::DistributedMapperConfig {
                    dtype,
                    device: &device,
                    available_devices: &available_devices,
                    global_world_size_override: tensor_parallelism.world_size(),
                    silent,
                    config: &config,
                    loading_isq,
                    from_uqff: self.config.from_uqff.is_some(),
                    write_uqff: self.config.write_uqff.is_some(),
                    organization: self.config.organization,
                    model: &*self.inner,
                    weights: distributed_weights,
                })?;
            let sharded_vb = if let Some(reader) = uqff_reader.clone() {
                sharded_vb.with_uqff_reader(reader)
            } else {
                sharded_vb
            };

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal | ModelKind::GgufQuantized { .. } => {
                    let tracker = sharded_vb.tracker().clone();
                    let model = self.inner.load(
                        &config,
                        sharded_vb,
                        crate::pipeline::NormalLoadingMetadata {
                            mapper,
                            loading_isq,
                            real_device: device.clone(),
                            multi_progress: multi_progress.clone(),
                            matformer_slicing_config: matformer_slicing_config.clone(),
                            rope_pairing: self
                                .prepared_source
                                .as_ref()
                                .map(|source| source.rope_pairing),
                        },
                        attention_mechanism,
                    )?;
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::XLora,
                }
                | ModelKind::GgufAdapter {
                    adapter: AdapterKind::XLora,
                    ..
                } => {
                    let (model, tracker) = xlora_model_loader!(
                        paths,
                        Some(dtype),
                        &load_device,
                        layer_devices.clone(),
                        config,
                        self.inner,
                        silent,
                        mapper,
                        loading_isq,
                        device.clone(),
                        multi_progress.clone(),
                        matformer_slicing_config.clone(),
                        uqff_reader.clone(),
                    );
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                }
                | ModelKind::GgufAdapter {
                    adapter: AdapterKind::Lora,
                    ..
                } => {
                    if let Some(source) = self.prepared_source.as_ref() {
                        let layers = new_dynamic_lora_registry(&config)?;
                        let sharded_vb = sharded_vb.with_lora_registry(layers.clone());
                        let tracker = sharded_vb.tracker().clone();
                        let model = self.inner.load(
                            &config,
                            sharded_vb,
                            crate::pipeline::NormalLoadingMetadata {
                                mapper,
                                loading_isq,
                                real_device: device.clone(),
                                multi_progress: multi_progress.clone(),
                                matformer_slicing_config: matformer_slicing_config.clone(),
                                rope_pairing: Some(source.rope_pairing),
                            },
                            attention_mechanism,
                        )?;
                        let dynamic_lora = super::finish_dynamic_lora_runtime(
                            paths,
                            layers,
                            self.lora_runtime_config
                                .expect("LoRA loaders have a runtime config"),
                            false,
                        )?;
                        (model, tracker, Some(dynamic_lora))
                    } else {
                        lora_model_loader!(
                            paths,
                            Some(dtype),
                            &load_device,
                            layer_devices.clone(),
                            config,
                            self.inner,
                            silent,
                            mapper,
                            loading_isq,
                            self.config.from_uqff.is_some(),
                            device.clone(),
                            attention_mechanism,
                            matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                            multi_progress.clone(),
                            matformer_slicing_config.clone(),
                            uqff_reader.clone(),
                            self.lora_runtime_config
                                .expect("LoRA loaders have a runtime config"),
                            false,
                        )
                    }
                }
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal | ModelKind::GgufQuantized { .. } => {
                    let (model, tracker) = if let Some(source) = self.prepared_source.as_ref() {
                        let vb = source
                            .weights
                            .clone()
                            .set_dtype(dtype)
                            .set_device(load_device.clone());
                        let tracker = vb.tracker().clone();
                        let model = self.inner.load(
                            &config,
                            vb,
                            crate::pipeline::NormalLoadingMetadata {
                                mapper,
                                loading_isq,
                                real_device: device.clone(),
                                multi_progress: multi_progress.clone(),
                                matformer_slicing_config: matformer_slicing_config.clone(),
                                rope_pairing: Some(source.rope_pairing),
                            },
                            attention_mechanism,
                        )?;
                        (model, tracker)
                    } else {
                        normal_model_loader!(
                            paths,
                            Some(dtype),
                            &load_device,
                            layer_devices.clone(),
                            config,
                            self.inner,
                            silent,
                            mapper,
                            loading_isq,
                            self.config.from_uqff.is_some(),
                            device.clone(),
                            attention_mechanism,
                            matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                            multi_progress.clone(),
                            matformer_slicing_config.clone(),
                            uqff_reader.clone(),
                        )
                    };
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::XLora,
                }
                | ModelKind::GgufAdapter {
                    adapter: AdapterKind::XLora,
                    ..
                } => {
                    let (model, tracker) = xlora_model_loader!(
                        paths,
                        Some(dtype),
                        &load_device,
                        layer_devices.clone(),
                        config,
                        self.inner,
                        silent,
                        mapper,
                        loading_isq,
                        device.clone(),
                        multi_progress.clone(),
                        matformer_slicing_config.clone(),
                        uqff_reader.clone(),
                    );
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                }
                | ModelKind::GgufAdapter {
                    adapter: AdapterKind::Lora,
                    ..
                } => {
                    if let Some(source) = self.prepared_source.as_ref() {
                        let layers = new_dynamic_lora_registry(&config)?;
                        let vb = source
                            .weights
                            .clone()
                            .set_dtype(dtype)
                            .set_device(load_device.clone())
                            .with_lora_registry(layers.clone());
                        let tracker = vb.tracker().clone();
                        let model = self.inner.load(
                            &config,
                            vb,
                            crate::pipeline::NormalLoadingMetadata {
                                mapper,
                                loading_isq,
                                real_device: device.clone(),
                                multi_progress: multi_progress.clone(),
                                matformer_slicing_config: matformer_slicing_config.clone(),
                                rope_pairing: Some(source.rope_pairing),
                            },
                            attention_mechanism,
                        )?;
                        let dynamic_lora = super::finish_dynamic_lora_runtime(
                            paths,
                            layers,
                            self.lora_runtime_config
                                .expect("LoRA loaders have a runtime config"),
                            true,
                        )?;
                        (model, tracker, Some(dynamic_lora))
                    } else {
                        lora_model_loader!(
                            paths,
                            Some(dtype),
                            &load_device,
                            layer_devices.clone(),
                            config,
                            self.inner,
                            silent,
                            mapper,
                            loading_isq,
                            self.config.from_uqff.is_some(),
                            device.clone(),
                            attention_mechanism,
                            matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                            multi_progress.clone(),
                            matformer_slicing_config.clone(),
                            uqff_reader.clone(),
                            self.lora_runtime_config
                                .expect("LoRA loaders have a runtime config"),
                            true,
                        )
                    }
                }
                _ => unreachable!(),
            }
        };

        let tokenizer = match self.prepared_source.as_ref() {
            Some(source) => source.tokenizer.clone(),
            None => get_tokenizer(paths.get_tokenizer_filename(), None)?,
        };
        let gen_conf: Option<GenerationConfig> = match self.prepared_source.as_ref() {
            Some(source) => source.generation_config.clone(),
            None => paths.get_gen_conf_filename().and_then(|f| {
                match serde_json::from_str::<GenerationConfig>(&fs::read_to_string(f).unwrap()) {
                    Ok(conf) => Some(conf),
                    Err(e) => {
                        warn!("Failed to parse generation_config.json: {}", e);
                        None
                    }
                }
            }),
        };
        let gen_conf = gen_conf.or_else(|| GenerationConfig::from_model_config(&config));

        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            self.prepared_source
                .as_ref()
                .and_then(|source| source.chat_template.clone()),
        );
        if let Some(source) = self.prepared_source.as_ref() {
            if chat_template.bos_token.is_none() {
                chat_template.bos_token = source
                    .bos_token
                    .clone()
                    .map(|token| BeginEndUnkPadTok(Either::Left(token)));
            }
            if chat_template.eos_token.is_none() {
                chat_template.eos_token = source
                    .eos_token
                    .clone()
                    .map(|token| BeginEndUnkPadTok(Either::Left(token)));
            }
            if chat_template.unk_token.is_none() {
                chat_template.unk_token = source
                    .unk_token
                    .clone()
                    .map(|token| BeginEndUnkPadTok(Either::Left(token)));
            }
        }

        plan.validate_tracked_selection(&tracker.get())?;

        let imatrix_map = if plan.wants_imatrix {
            let drive = super::isq_flow::NormalCalibrationDrive(&*model);
            Some(super::isq_flow::resolve_imatrix_map(
                &drive,
                &tracker.get().clone(),
                self.config.imatrix.as_ref(),
                self.config.calibration_file.as_ref(),
                &super::isq_flow::CalibrationCtx {
                    tokenizer: &tokenizer,
                    bos_tok_id: chat_template
                        .bos_tok()
                        .as_deref()
                        .and_then(|tok| tokenizer.token_to_id(tok)),
                    load_device: &load_device,
                    mapper: Some(pipeline_mapper.as_ref()),
                },
            )?)
        } else {
            None
        };

        if plan.capture == mistralrs_quant::IsqCaptureMode::CaptureMatches {
            let ty = in_situ_quant.context("imatrix quantization requires an ISQ type")?;
            super::isq_flow::complete_isq_capture(
                &tracker.get().clone(),
                ty,
                imatrix_map
                    .as_ref()
                    .expect("CaptureMatches requires imatrix data"),
            )?;
        }

        if let Some(write_uqff) = &self.config.write_uqff {
            let layers = tracker.get().clone();
            let uqff_types = plan
                .write_types
                .clone()
                .filter(|types| !types.is_empty())
                .context("UQFF serialization requires at least one ISQ type.")?;
            let residual = match self.config.organization {
                IsqOrganization::Default => model.residual_tensors(),
                IsqOrganization::MoeExpertsOnly => model
                    .residual_tensors_moe_experts_only()
                    .unwrap_or(model.residual_tensors()),
            };
            let full_ser = UqffFullSer {
                tokenizer: &tokenizer,
                template_filename: paths.get_template_filename(),
                effective_chat_template: Some(&chat_template),
                generation_config: match self.prepared_source.as_ref() {
                    Some(source) if source.generation_config.is_none() => None,
                    _ => paths.get_gen_conf_filename(),
                },
                config: config.clone(),
                processor_filename: &None,
                preprocessor_filename: &None,
                modules: None,
                module_paths: None,
            };
            write_uqff_artifacts(UqffWriteRequest {
                output: write_uqff.output.clone(),
                types: uqff_types,
                base_model: write_uqff.base_model.clone(),
                repo_id: write_uqff.repo_id.clone(),
                layers,
                quantize_predicates: plan.uqff_quantize_predicates.clone(),
                residual,
                full_ser,
                imatrix: imatrix_map.unwrap_or_default(),
            })?;
        }

        let paged_attn_config = if matches!(
            self.kind,
            ModelKind::Adapter {
                adapter: AdapterKind::XLora
            }
        ) {
            warn!(
                "Adapter parallel_models do not currently support PagedAttention, running without"
            );
            None
        } else {
            paged_attn_config
        };

        if plan.immediate_isq_installed {
            for module in tracker.get().clone() {
                module.ct.resolve()?;
            }
        }
        #[cfg(feature = "cuda")]
        super::synchronize_cuda_contexts(&device, pipeline_mapper.as_ref())?;

        let tracked_modules = tracker.get().clone();
        let source_weight_files = match self.prepared_source.as_ref() {
            Some(source) => source.source_weight_files.clone(),
            None if self.config.from_uqff.is_some() => Vec::new(),
            None => paths.get_weight_filenames().to_vec(),
        };

        build_normal_pipeline(NormalPipelineBuildArgs {
            model,
            tokenizer,
            chat_template,
            generation_config: gen_conf,
            paged_attn_config,
            dtype,
            device,
            mapper: pipeline_mapper,
            silent,
            max_kv_tokens,
            no_kv_cache: self.no_kv_cache,
            is_xlora,
            kind: self.kind.clone(),
            non_granular_index: self.tgt_non_granular_index,
            model_id: self.model_id.clone(),
            loaded_for_uqff_write: self.config.write_uqff.is_some(),
            tracked_modules,
            source_weight_files,
            source_weight_source: weight_source,
            dynamic_lora,
        })
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for NormalPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for NormalPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if self.tracked_modules.is_empty() {
            anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
        }
        tracing::info!(
            "Re-quantizing {} layers to {dtype}.",
            self.tracked_modules.len()
        );
        self.cleanup_cuda_graphs();
        super::isq_flow::requantize_and_swap(
            &self.tracked_modules,
            dtype,
            |module| module.default_type(dtype),
            &|_| None,
        )
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.tracked_modules)?;
        #[cfg(feature = "cuda")]
        self.cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned")
            .suspend();
        Ok(())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        self.cleanup_cuda_graphs();
        let result = super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            self.source_weight_source.as_deref(),
            save_cimatrix.as_deref(),
        );
        #[cfg(feature = "cuda")]
        if result.is_ok() || !super::isq_flow::calibration_status(&self.tracked_modules).collecting
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .resume();
        }
        result
    }
}

impl CacheManagerMixin for NormalPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) -> candle_core::Result<()> {
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_in_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_in_cache(self, seqs, false),
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        match self.model.cache() {
            EitherCache::Full(_) => FullCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Normal(_) => NormalCacheManager.clone_out_cache(self, seqs, false),
            EitherCache::Hybrid(_) => HybridCacheManager.clone_out_cache(self, seqs, false),
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,
        load_preallocated_cache: bool,
    ) -> candle_core::Result<()> {
        match self.model.cache() {
            EitherCache::Full(_) => {
                FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false)
            }
            EitherCache::Normal(_) => NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
            EitherCache::Hybrid(_) => HybridCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            ),
        }?;
        if reset_non_granular {
            self.reset_non_granular_state()
        }
        Ok(())
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for NormalPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .clear();
            if self.model.cache().is_hybrid() {
                if let Err(err) = self.model.cache().hybrid().release_graph_pad_slot() {
                    tracing::error!("Failed to release CUDA graph recurrent pad slot: {err}");
                }
            }
        }
    }
    fn reclaim_cuda_graph_memory(&self, max_entries: usize) -> usize {
        #[cfg(feature = "cuda")]
        {
            crate::pipeline::cuda_graph::reclaim_cuda_graph_entries(
                max_entries,
                |limit| {
                    self.cuda_decode_graph
                        .lock()
                        .expect("CUDA graph mutex poisoned")
                        .evict_lru_for_memory_pressure(limit)
                },
                |limit| self.model.evict_speculative_cuda_graphs(limit),
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = max_entries;
            0
        }
    }
    fn precapture_cuda_decode_graphs(&self, ctx: &DecodeGraphPrecaptureCtx) {
        #[cfg(feature = "cuda")]
        {
            if let Err(err) = self.precapture_cuda_decode_graphs_impl(ctx) {
                self.cuda_decode_graph
                    .lock()
                    .expect("CUDA graph mutex poisoned")
                    .clear();
                warn!("CUDA decode graph precapture failed, graphs will be captured lazily: {err}");
            }
            if let Err(err) = self.model.precapture_speculative_cuda_graphs() {
                warn!("Speculative CUDA graph precapture failed, graphs will be captured lazily: {err}");
            }
        }
        #[cfg(not(feature = "cuda"))]
        let _ = ctx;
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for NormalPipeline {
    fn has_speculative_proposer(&self) -> bool {
        self.model.has_speculative_proposer()
    }

    fn speculative_plan(
        &self,
        batch_size: usize,
    ) -> Option<crate::speculative::SpeculativeBatchPlan> {
        self.model.speculative_plan(batch_size)
    }

    fn speculative_observe(&self, observation: crate::speculative::SpeculativeBatchObservation) {
        self.model.speculative_observe(observation);
    }

    fn speculative_bypass(&mut self, seq_ids: &[usize]) {
        self.model.speculative_bypass(seq_ids);
    }

    fn speculative_target_hiddens(
        &self,
        rows: &[(usize, usize)],
    ) -> candle_core::Result<Option<Tensor>> {
        self.model.speculative_target_hiddens(rows)
    }

    fn speculative_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposeBatchCtx<'_>,
    ) -> candle_core::Result<Option<crate::speculative::SpeculativeProposalBatch>> {
        self.model.speculative_propose(ctx)
    }

    fn speculative_prepare_propose(
        &mut self,
        ctx: crate::speculative::SpeculativeProposePrepareCtx<'_>,
    ) -> candle_core::Result<Option<Box<dyn crate::speculative::SpeculativeProposePreparation>>>
    {
        self.model.speculative_prepare_propose(ctx)
    }

    fn speculative_commit(
        &mut self,
        rows: &[crate::speculative::SpeculativeCommitRow],
    ) -> candle_core::Result<()> {
        self.model.speculative_commit(rows)
    }

    fn build_speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>> {
        Ok(Box::new(ModelInputs {
            input_ids: input_meta.input,
            input_ids_full: None,
            seqlen_offsets: input_meta.positions,
            seqlen_offsets_full: None,
            context_lens: input_meta.context_lens,
            position_ids: input_meta.position_ids,
            paged_attn_meta: input_meta.paged_attn_meta,
            flash_meta: input_meta.flash_meta,
            flash_meta_full: None,
            recurrent_batch_kind: RecurrentBatchKind::SpeculativeDecode,
            adapter_leases: Arc::from([]),
        }))
    }

    #[cfg(feature = "cuda")]
    fn cuda_sparse_rejection_workspace(
        &self,
    ) -> &StdMutex<Option<crate::speculative::CudaSparseRejectionWorkspace>> {
        &self.cuda_sparse_rejection
    }
}

#[cfg(feature = "cuda")]
impl NormalPipeline {
    fn try_cuda_decode_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position_ids: &[usize],
        paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_meta: &FlashParams,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<Option<CudaDecodeGraphReplay>> {
        if !cuda_decode_graphs_enabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Disabled,
            );
            return Ok(None);
        }
        if !cuda_decode_graph_batch_kind_supported(recurrent_batch_kind) {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if !self.model.supports_cuda_decode_graphs()
            || !cuda_decode_graph_supported_for_model(self.metadata.model_metadata.as_deref())
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::ModelUnsupported,
            );
            return Ok(None);
        }
        if self.model.has_speculative_proposer() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::SpeculativeConflict,
            );
            return Ok(None);
        }
        let Some((kv_cache, metadata)) = paged_attn_meta else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::PagedAttentionUnavailable,
            );
            return Ok(None);
        };
        if metadata.is_first_prompt_chunk || metadata.num_cached_tokens.is_some() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::Prefill,
            );
            return Ok(None);
        }
        if metadata.decode_rows.is_none() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if q_len != 1
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || position_ids.len() != batch
            || !input_ids.device().is_cuda()
        {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::IncompatibleShape,
            );
            return Ok(None);
        }
        let Some(bucket) = cuda_graph_batch_bucket(batch) else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::BatchUnsupported,
            );
            return Ok(None);
        };
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Skipped,
                CudaGraphDispatchReason::CacheConfigUnavailable,
            );
            return Ok(None);
        };
        // Captured kernels require canonical strides, but an already contiguous input needs no copy.
        let input_ids = &input_ids.contiguous()?;

        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::RuntimeDisabled,
            );
            return Ok(None);
        }
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let slots = hybrid_graph_slots(&mut self.model.cache().hybrid())?;
            if let Some(slots) = &slots {
                state.observe_recurrent_storage_generation(slots.storage_generation);
            }
            slots
        } else {
            None
        };
        let Some(step) = CudaGraphDecodeStep::padded(
            CudaGraphDecodeStepInputs {
                input_ids,
                seqlen_offsets,
                context_lens,
                position_ids,
                metadata,
                state_indices: hybrid_slots.as_ref().map(|slots| slots.real.as_slice()),
                pad_slot: hybrid_slots.as_ref().map(|_| GDN_PAD_SLOT),
            },
            bucket,
        )?
        else {
            record_cuda_graph_dispatch(
                CudaGraphComponent::Target,
                CudaGraphDispatchMode::Eager,
                CudaGraphDispatchReason::PaddingUnavailable,
            );
            return Ok(None);
        };
        let key = CudaDecodeGraphKey::new(
            &step.input_ids,
            &step.metadata,
            cache_config.block_size,
            recurrent_batch_kind,
        )?;
        if let Some(replay) = state.replay(&key, &step, CudaDecodeGraphReplayInput::Host)? {
            return Ok(Some(replay));
        }

        let replay_key = key.clone();
        let _ = self.capture_cuda_decode_graph_step(
            &mut state,
            key,
            &step,
            kv_cache.as_slice(),
            flash_meta,
            cache_config.block_size,
            recurrent_batch_kind,
            true,
        )?;
        super::synchronize_cuda_contexts(step.input_ids.device(), self.mapper.as_ref()).map_err(
            |err| {
                candle_core::Error::msg(format!(
                    "CUDA graph rollback synchronization failed: {err}"
                ))
            },
        )?;
        let replay = state
            .replay(&replay_key, &step, CudaDecodeGraphReplayInput::Host)?
            .ok_or_else(|| {
                candle_core::Error::msg("newly captured CUDA decode graph was not replayable")
            })?;
        record_cuda_graph_dispatch(
            CudaGraphComponent::Target,
            CudaGraphDispatchMode::Eager,
            CudaGraphDispatchReason::CachePopulation,
        );
        Ok(Some(replay))
    }

    fn precapture_cuda_decode_graphs_impl(
        &self,
        ctx: &DecodeGraphPrecaptureCtx,
    ) -> candle_core::Result<()> {
        let device = self.device();
        if !cuda_decode_graphs_enabled()
            || !device.is_cuda()
            || !self.model.supports_cuda_decode_graphs()
            || !cuda_decode_graph_supported_for_model(self.metadata.model_metadata.as_deref())
            || self.model.has_speculative_proposer()
        {
            return Ok(());
        }
        let (Some(cache_config), Some(cache_engine)) =
            (&self.metadata.cache_config, &self.metadata.cache_engine)
        else {
            return Ok(());
        };
        let kv_cache = cache_engine.get_kv_cache().clone();
        let hybrid_slots = if self.model.cache().is_hybrid() {
            let mut cache = self.model.cache().hybrid();
            let Some(pad_slot) = cache.graph_pad_slot()? else {
                return Ok(());
            };
            let pad_slot = cache.active_physical_slot(pad_slot)?;
            let pad_slot = u32::try_from(pad_slot).map_err(|_| {
                candle_core::Error::msg(format!("recurrent graph pad slot {pad_slot} exceeds u32"))
            })?;
            Some(pad_slot)
        } else {
            None
        };
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            return Ok(());
        }
        let start = std::time::Instant::now();
        let mut captured = 0usize;
        let inputs = CudaGraphPrecaptureInputs::new(ctx, 1, &device, self.device_mapper())?;
        let live = hybrid_slots.map(|pad_slot| vec![pad_slot]);
        for bucket in cuda_graph_precapture_batches() {
            let Some(step) = CudaGraphDecodeStep::padded(
                inputs.step_inputs(live.as_deref(), hybrid_slots.map(|_| GDN_PAD_SLOT)),
                bucket,
            )?
            else {
                continue;
            };
            let key = CudaDecodeGraphKey::new(
                &step.input_ids,
                &step.metadata,
                cache_config.block_size,
                RecurrentBatchKind::Decode,
            )?;
            if state.contains(&key) {
                continue;
            }
            self.capture_cuda_decode_graph_step(
                &mut state,
                key,
                &step,
                kv_cache.as_slice(),
                &inputs.flash_meta,
                cache_config.block_size,
                RecurrentBatchKind::Decode,
                false,
            )?;
            captured += 1;
        }
        if captured > 0 {
            info!(
                "Captured {captured} CUDA decode graphs through batch bucket {} in {:.2?}",
                CUDA_GRAPH_PRECAPTURE_MAX_BATCH,
                start.elapsed()
            );
        }
        Ok(())
    }

    /// Captures after one eager warmup; live calls roll it back so the first replay is canonical.
    fn capture_cuda_decode_graph_step(
        &self,
        state: &mut CudaDecodeGraphState,
        key: CudaDecodeGraphKey,
        step: &CudaGraphDecodeStep,
        kv_cache: &[(Tensor, Tensor)],
        flash_meta: &FlashParams,
        block_size: usize,
        recurrent_batch_kind: RecurrentBatchKind,
        rollback_live_state: bool,
    ) -> candle_core::Result<Tensor> {
        let graph_event =
            CudaGraphEventGuard::new(CudaGraphComponent::Target, CudaGraphEvent::Capture);
        let Device::Cuda(cuda_device) = step.input_ids.device() else {
            candle_core::bail!("CUDA graph decode expected CUDA input ids");
        };
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();
        let metadata = step
            .metadata
            .materialize_decode_tensors()
            .map_err(candle_core::Error::msg)?;

        let recurrent_snapshots = self.snapshot_hybrid_recurrent_checkpoints()?;
        let live_state_indices = self.snapshot_hybrid_state_indices();
        let capture_attempt: candle_core::Result<_> = (|| {
            let state_index_buffers = match &step.state_indices {
                Some(host) => Some(install_hybrid_graph_state_indices(
                    &mut self.model.cache().hybrid(),
                    host,
                )?),
                None => None,
            };
            let mut ctx = ModelForwardContext::new(
                &step.seqlen_offsets,
                &step.context_lens,
                &step.position_ids,
                Some((kv_cache, &metadata)),
                flash_meta,
            )
            .with_recurrent_batch_kind(recurrent_batch_kind)
            .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
            let warmup_logits = self.model.forward(&step.input_ids, &mut ctx)?;
            step.input_ids.device().synchronize()?;
            let live_logits = step.narrow_rows(&warmup_logits)?;

            // CUDA stream capture records recurrent writes without executing them.
            let entry = capture_cuda_decode_graph(
                CudaDecodeGraphCaptureCtx {
                    key,
                    input_ids: &step.input_ids,
                    seqlen_offsets: &step.seqlen_offsets,
                    position_ids: &step.position_ids,
                    block_size,
                    kv_cache,
                    metadata: &metadata,
                    model_metadata: self.metadata.model_metadata.as_deref(),
                    activation_dtype: self.metadata.activation_dtype,
                    warmup_logits: &warmup_logits,
                    state_indices: state_index_buffers,
                    real_batch: step.real_batch,
                },
                |graph_input_ids, graph_metadata| {
                    let mut ctx = ModelForwardContext::new(
                        &step.seqlen_offsets,
                        &step.context_lens,
                        &step.position_ids,
                        Some((kv_cache, graph_metadata)),
                        flash_meta,
                    )
                    .with_recurrent_batch_kind(recurrent_batch_kind)
                    .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
                    self.model.forward(graph_input_ids, &mut ctx)
                },
            )?;
            Ok((live_logits, entry))
        })();
        let (logits, entry) = self.finish_cuda_graph_capture_attempt(
            state,
            capture_attempt,
            recurrent_snapshots.as_deref(),
            live_state_indices.as_ref(),
            rollback_live_state,
        )?;
        state.insert(entry);
        graph_event.success();
        Ok(logits)
    }

    fn snapshot_hybrid_state_indices(&self) -> Option<HybridStateIndicesSnapshot> {
        self.model.cache().is_hybrid().then(|| {
            let cache = self.model.cache().hybrid();
            (
                cache.state_indices().cloned(),
                cache.state_indices_host().map(ToOwned::to_owned),
            )
        })
    }

    fn restore_hybrid_state_indices(&self, snapshot: Option<&HybridStateIndicesSnapshot>) {
        if let Some((tensor, host)) = snapshot {
            self.model
                .cache()
                .hybrid()
                .set_physical_state_indices_with_host(tensor.clone(), host.clone());
        }
    }

    fn finish_cuda_graph_capture_attempt<T>(
        &self,
        state: &mut CudaDecodeGraphState,
        attempt: candle_core::Result<T>,
        recurrent_snapshots: Option<&[(usize, RecurrentCheckpointStateSnapshot)]>,
        live_state_indices: Option<&HybridStateIndicesSnapshot>,
        rollback_live_state: bool,
    ) -> candle_core::Result<T> {
        self.restore_hybrid_state_indices(live_state_indices);
        match attempt {
            Ok(value) if !rollback_live_state => Ok(value),
            Ok(value) => {
                self.restore_hybrid_recurrent_checkpoints(recurrent_snapshots)
                    .map_err(|restore_err| {
                        state.block_eager_retry();
                        candle_core::Error::msg(format!(
                            "CUDA graph captured, but recurrent checkpoint rollback failed: {restore_err}"
                        ))
                    })?;
                Ok(value)
            }
            Err(capture_err) => {
                if let Err(restore_err) =
                    self.restore_hybrid_recurrent_checkpoints(recurrent_snapshots)
                {
                    state.block_eager_retry();
                    return Err(candle_core::Error::msg(format!(
                        "CUDA graph capture failed: {capture_err}; recurrent checkpoint rollback failed: {restore_err}"
                    )));
                }
                Err(capture_err)
            }
        }
    }

    fn snapshot_hybrid_recurrent_checkpoints(
        &self,
    ) -> candle_core::Result<Option<SeqRecurrentCheckpointSnapshots>> {
        if !self.model.cache().is_hybrid() {
            return Ok(None);
        }
        let hybrid_cache = self.model.cache().hybrid();
        let Some(mut indices) = hybrid_cache
            .logical_state_indices_host()
            .map(ToOwned::to_owned)
        else {
            return Ok(None);
        };
        indices.retain(|&idx| idx != u32::MAX);
        indices.sort_unstable();
        indices.dedup();
        let mut snapshots = Vec::with_capacity(indices.len());
        for idx in indices {
            let idx = idx as usize;
            snapshots.push((idx, hybrid_cache.snapshot_recurrent_checkpoint_state(idx)?));
        }
        Ok(Some(snapshots))
    }

    fn restore_hybrid_recurrent_checkpoints(
        &self,
        snapshots: Option<&[(usize, RecurrentCheckpointStateSnapshot)]>,
    ) -> candle_core::Result<()> {
        let Some(snapshots) = snapshots else {
            return Ok(());
        };
        let mut hybrid_cache = self.model.cache().hybrid();
        for (idx, snapshot) in snapshots {
            hybrid_cache.restore_recurrent_checkpoint_state(*idx, snapshot)?;
        }
        Ok(())
    }

    fn disable_cuda_decode_graph(&self, err: &candle_core::Error) -> bool {
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        let eager_retry_allowed = state.take_eager_retry_allowed();
        if !state.disabled() {
            warn!("CUDA decode graphs disabled after capture/replay error: {err}");
        }
        state.disable();
        drop(state);
        if self.model.cache().is_hybrid() {
            if let Err(release_err) = self.model.cache().hybrid().release_graph_pad_slot() {
                tracing::error!(
                    "Failed to release recurrent graph pad after graph disable: {release_err}"
                );
            }
        }
        eager_retry_allowed
    }
}

impl NormalPipeline {
    fn recurrent_metadata(&self, batch_kind: RecurrentBatchKind) -> Option<RecurrentMetadata> {
        if !self.model.cache().is_hybrid() {
            return None;
        }
        let hybrid_cache = self.model.cache().hybrid();
        let state_indices_host = hybrid_cache.state_indices_host().map(ToOwned::to_owned);
        hybrid_cache.state_indices().cloned().map(|state_indices| {
            RecurrentMetadata::new(batch_kind, state_indices, state_indices_host)
        })
    }
}

#[async_trait::async_trait]
impl Pipeline for NormalPipeline {
    fn requires_uniform_prompt_batch(&self) -> bool {
        normal_model_requires_uniform_prompt_batch(
            self.model.cache().is_hybrid(),
            self.supports_packed_prefill(),
            self.model.is_xlora(),
            self.model.has_speculative_proposer(),
        )
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        false
    }

    fn supports_batched_cuda_sampling(&self) -> bool {
        !self.model.has_speculative_proposer()
    }

    fn supports_speculative_prompt_bootstrap(&self) -> bool {
        self.model.supports_speculative_prompt_bootstrap()
    }

    fn speculative_prefix_replay(&self) -> crate::speculative::SpeculativePrefixReplay {
        self.model.speculative_prefix_replay()
    }

    fn supports_paged_auxiliary_prefix_state(&self) -> bool {
        self.model.supports_paged_auxiliary_prefix_state()
    }

    fn capture_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
    ) -> candle_core::Result<Option<Arc<dyn crate::prefix_cacher::PagedAuxiliaryPrefixState>>> {
        self.model
            .capture_paged_auxiliary_prefix_state(sequence_id, cached_tokens)
    }

    fn restore_paged_auxiliary_prefix_state(
        &mut self,
        sequence_id: usize,
        cached_tokens: usize,
        state: &dyn crate::prefix_cacher::PagedAuxiliaryPrefixState,
    ) -> candle_core::Result<()> {
        self.model
            .restore_paged_auxiliary_prefix_state(sequence_id, cached_tokens, state)
    }

    fn supports_packed_prefill(&self) -> bool {
        self.model.supports_packed_prefill()
            && self.metadata.cache_engine.is_some()
            && !self.model.is_xlora()
            && (!self.model.has_speculative_proposer()
                || self.model.supports_speculative_packed_prefill())
            && self.model.device().is_cuda()
            && self.mapper.get_unique_devices().iter().all(Device::is_cuda)
            && crate::using_flash_attn()
            && crate::attention::flash_backend_supports_sdpa(
                self.model.config().k_head_dim,
                false,
                self.metadata.sliding_window.is_some(),
            )
            && matches!(self.metadata.activation_dtype, DType::F16 | DType::BF16)
    }

    fn adapter_runtime(&self) -> Option<Arc<DynamicLoraRuntime>> {
        self.dynamic_lora.clone()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardInputsResult, candle_core::Error> {
        Ok(self.forward_step(inputs, return_raw_logits)?.output)
    }

    fn forward_step(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> Result<ForwardStepResult, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            flash_meta_full,
            recurrent_batch_kind,
            adapter_leases,
        } = *inputs.downcast().expect("Downcast failed.");
        let lora_execution = super::resolve_lora_execution(
            self.dynamic_lora.as_deref(),
            &input_ids,
            paged_attn_meta.as_ref(),
            &flash_meta,
            &adapter_leases,
        )?;
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(cache_engine), Some(meta)) => Some((cache_engine, meta)),
            (Some(_), None) => {
                // This can happen if Rust-side user code is wrong
                candle_core::bail!("Forward step expected a PagedAttention input metadata. This was not provided, please ensure that the scheduler config is correctly configured for PagedAttention.")
            }
            (None, Some(_)) => {
                // This should never happen but we handle it anyway
                candle_core::bail!("Forward step got a PagedAttention input metadata but there is no cache engine. Please raise an issue.")
            }
            (None, None) => None,
        };
        let logits = match self.model.is_xlora() {
            false => {
                #[cfg(feature = "cuda")]
                let mut cuda_graph_eager_fallback = None;
                let paged_attn_meta = paged_attn_meta
                    .as_ref()
                    .map(|meta| (meta.0.get_kv_cache().clone(), meta.1.clone()));

                #[cfg(feature = "cuda")]
                if lora_execution.is_none() && !return_raw_logits {
                    match self.try_cuda_decode_graph_forward(
                        &input_ids,
                        &seqlen_offsets,
                        &context_lens,
                        &position_ids,
                        paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), b)),
                        &flash_meta,
                        recurrent_batch_kind,
                    ) {
                        Ok(Some(replay)) => {
                            return Ok(ForwardStepResult::cuda_decode(
                                ForwardInputsResult::CausalGeneration {
                                    logits: replay.logits,
                                },
                                replay.launch,
                            ))
                        }
                        Ok(None) => {}
                        Err(err) => {
                            if !self.disable_cuda_decode_graph(&err) {
                                return Err(err);
                            }
                            cuda_graph_eager_fallback = Some(CudaGraphEventGuard::new(
                                CudaGraphComponent::Target,
                                CudaGraphEvent::EagerFallback,
                            ));
                        }
                    }
                }

                let paged_attn_meta = paged_attn_meta
                    .map(|(kv_cache, metadata)| {
                        metadata
                            .materialize_decode_tensors()
                            .map(|metadata| (kv_cache, metadata))
                    })
                    .transpose()
                    .map_err(candle_core::Error::msg)?;

                let mut ctx = ModelForwardContext::new(
                    &seqlen_offsets,
                    &context_lens,
                    &position_ids,
                    paged_attn_meta
                        .as_ref()
                        .map(|(kv_cache, meta)| (kv_cache.as_slice(), meta)),
                    &flash_meta,
                )
                .with_recurrent_batch_kind(recurrent_batch_kind)
                .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
                let eager_result = mistralrs_quant::with_lora_execution(lora_execution, || {
                    self.model.forward(&input_ids, &mut ctx)
                });
                #[cfg(feature = "cuda")]
                if eager_result.is_ok() {
                    if let Some(graph_event) = cuda_graph_eager_fallback.take() {
                        graph_event.success();
                    }
                }
                eager_result?
            }
            true => self.model.xlora_forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                position_ids,
                &flash_meta,
                flash_meta_full.as_ref().unwrap_or(&flash_meta),
            )?,
        };
        let output = if return_raw_logits {
            ForwardInputsResult::RawLogits { logits }
        } else {
            ForwardInputsResult::CausalGeneration { logits }
        };
        Ok(ForwardStepResult::eager(output))
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_decode_one_token(
        &mut self,
        launch: CudaDecodeGraphLaunch,
    ) -> candle_core::Result<Option<ForwardStepResult>> {
        let replay = {
            let mut state = self
                .cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned");
            if state.disabled() {
                return Ok(None);
            }
            state.replay_one_token(launch)
        };
        match replay {
            Ok(Some(replay)) => Ok(Some(ForwardStepResult::cuda_decode(
                ForwardInputsResult::CausalGeneration {
                    logits: replay.logits,
                },
                replay.launch,
            ))),
            Ok(None) => Ok(None),
            Err(err) => {
                let _ = self.disable_cuda_decode_graph(&err);
                Err(err)
            }
        }
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
    ) -> candle_core::Result<()> {
        self.attach_speculative_with_runtime(
            config,
            crate::speculative::MtpRuntimeConfig::default(),
        )
    }

    fn attach_speculative_with_runtime(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
        runtime: crate::speculative::MtpRuntimeConfig,
    ) -> candle_core::Result<()> {
        if self.dynamic_lora.is_some() {
            candle_core::bail!("dynamic LoRA does not support speculative decoding");
        }
        if matches!(config, crate::speculative::SpeculativeConfig::Mtp(_))
            && self.get_metadata().cache_engine.is_none()
        {
            candle_core::bail!(
                "MTP speculative decoding currently requires PagedAttention for this pipeline."
            );
        }
        if let Some(info) = self
            .model
            .attach_speculative_with_runtime(config, runtime)?
        {
            self.model.log_speculative_attach(&info);
        }
        Ok(())
    }

    fn release_speculative_sequences(&mut self, seq_ids: &[usize]) {
        self.model.release_speculative_sequences(seq_ids);
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        seqs: &mut [&mut Sequence],
        logits: &[Tensor],
        batched_logits: Option<&Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::pipeline::text_models_inputs_processor::PagedAttentionMeta>,
        logger: &crate::IntervalLogger,
    ) -> candle_core::Result<bool> {
        if !self.model.has_speculative_proposer() {
            crate::speculative::driver::clear_staged_speculative_tokens(seqs);
            return Ok(false);
        }

        let general_metadata = self.get_metadata();
        if let Some(cache_engine) = general_metadata.cache_engine.as_ref() {
            let Some(metadata) = metadata else {
                crate::speculative::driver::clear_staged_speculative_tokens(seqs);
                return Ok(false);
            };
            let cache = crate::speculative::cache::PagedSpeculativeCacheAccess::new(
                &metadata,
                cache_engine,
            );
            return crate::speculative::driver::try_sample_speculative_causal_gen(
                self,
                seqs,
                logits,
                batched_logits,
                prefix_cacher,
                disable_eos_stop,
                rng,
                &cache,
                logger,
            )
            .await;
        }

        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Ok(false)
    }

    async fn try_sample_causal_gen_batched(
        &self,
        seqs: &mut [&mut Sequence],
        logits: &Tensor,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<bool, candle_core::Error> {
        if self.model.has_speculative_proposer() {
            return Ok(false);
        }
        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        sample_and_add_toks_batched(
            self,
            seqs,
            logits.clone(),
            prefix_cacher,
            disable_eos_stop,
            rng,
        )
        .await?;
        Ok(true)
    }

    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for NormalPipeline {
    fn amoe_finish_training(&mut self, gate_model_id: Option<String>) -> candle_core::Result<()> {
        self.model.finish_training(gate_model_id)
    }
    fn amoe_layer_vars(&self) -> Vec<Vec<Var>> {
        self.model.get_vars()
    }
    fn amoe_base_model_trainable_params(&self) -> usize {
        self.model.trainable_params()
    }
    fn amoe_take_cached_gating_outputs(&mut self) -> Vec<Tensor> {
        self.model.take_cached_gating_outputs()
    }
    fn amoe_create_layers(
        &mut self,
        model_ids: Vec<String>,
        token: &TokenSource,
        revision: Option<String>,
        match_regex: &str,
        config: crate::amoe::AnyMoeConfig,
        dtype: candle_core::DType,
        dev: &Device,
        (prefix, mlp): (String, String),
        layers: Vec<usize>,
        expert_type: AnyMoeExpertType,
        silent: bool,
        gate_model_id: Option<String>,
    ) -> candle_core::Result<()> {
        let mut vbs = Vec::new();
        // Precompile regex here
        let regex = Regex::new(match_regex).map_err(candle_core::Error::msg)?;
        for model_id in model_ids {
            let model_id_str = &model_id;
            let model_id = Path::new(&model_id);

            let api = build_api(token, !silent).map_err(candle_core::Error::msg)?;
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut filenames = vec![];
            for rfilename in api_dir_list!(api, model_id, true, &revision)
                .filter(|x| x.ends_with(".safetensors"))
            {
                filenames.push(api_get_file!(api, &rfilename, model_id, &revision));
            }

            let regex = regex.clone();
            let match_regex_clone = match_regex.to_string();
            let layers_clone = layers.clone();
            let vb = from_mmaped_safetensors(
                filenames,
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                move |key| {
                    if regex.is_match(&key) {
                        // Idx of the last char of the layer id, +1
                        // Assumes N.MLP
                        let last_layer_idx = key.find(&match_regex_clone).unwrap() - 1;
                        let first_layer_idx = key[..last_layer_idx].rfind('.').unwrap();
                        let layer_n = key[first_layer_idx + 1..last_layer_idx]
                            .parse::<usize>()
                            .unwrap();
                        layers_clone.contains(&layer_n) || layers_clone.is_empty()
                    } else {
                        false
                    }
                },
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            vbs.push(vb);
        }

        let gate_vb = if let Some(gate_model_id) = gate_model_id {
            let model_id_str = &gate_model_id;
            let model_id = Path::new(&gate_model_id);

            let api = build_api(token, !silent).map_err(candle_core::Error::msg)?;
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut gate_filenames = vec![];
            for rfilename in api_dir_list!(api, model_id, true, &revision)
                .filter(|x| x.ends_with(".safetensors"))
            {
                gate_filenames.push(api_get_file!(api, &rfilename, model_id, &revision));
            }
            assert_eq!(
                gate_filenames.len(),
                1,
                "Gate model ID must contain only one .safetensors file"
            );

            let vb = from_mmaped_safetensors(
                gate_filenames.clone(),
                vec![],
                Some(dtype),
                dev,
                vec![None],
                silent,
                None,
                |_| true,
                Arc::new(|_| DeviceForLoadTensor::Base),
            )?;
            info!(
                "Loaded gating layers from `{}`",
                gate_filenames[0].display()
            );
            Some(vb)
        } else {
            None
        };

        self.model.create_anymoe_layers(
            vbs.clone(),
            config.clone(),
            (prefix.clone(), mlp.clone()),
            layers.clone(),
            expert_type.clone(),
            gate_vb.clone(),
        )?;

        Ok(())
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe_supported()
    }
}

#[cfg(test)]
mod tests {
    use super::{new_dynamic_lora_registry, normal_model_requires_uniform_prompt_batch};
    use crate::pipeline::finish_dynamic_lora_runtime;
    use crate::pipeline::{AdapterPaths, LocalModelPaths};
    use crate::LoraRuntimeConfig;
    use candle_core::{DType, Device};
    use mistralrs_quant::{LoraLayerRegistry, LoraLinearSpec, LoraSiteKey};
    use std::{path::PathBuf, sync::Arc};

    fn empty_lora_paths() -> LocalModelPaths<PathBuf> {
        LocalModelPaths {
            tokenizer_filename: PathBuf::new(),
            config_filename: PathBuf::new(),
            template_filename: None,
            filenames: Vec::new(),
            adapter_paths: AdapterPaths::Lora(Vec::new()),
            gen_conf: None,
            preprocessor_config: None,
            video_preprocessor_config: None,
            processor_config: None,
            chat_template_json_filename: None,
        }
    }

    #[test]
    fn hybrid_models_require_uniform_prompts_until_packed_prefill_is_proven() {
        assert!(normal_model_requires_uniform_prompt_batch(
            true, false, false, false
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            true, true, false, false
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            false, false, false, false
        ));
    }

    #[test]
    fn xlora_and_unsupported_speculative_models_remain_uniform() {
        assert!(normal_model_requires_uniform_prompt_batch(
            true, true, true, false
        ));
        assert!(normal_model_requires_uniform_prompt_batch(
            true, false, false, true
        ));
        assert!(!normal_model_requires_uniform_prompt_batch(
            true, true, false, true
        ));
    }

    #[test]
    fn prepared_lora_runtime_finalizes_sites_and_preserves_update_policy() {
        let paths = empty_lora_paths();
        for live_updates in [false, true] {
            let layers = Arc::new(LoraLayerRegistry::new());
            let runtime = finish_dynamic_lora_runtime(
                &paths,
                layers.clone(),
                LoraRuntimeConfig::default(),
                live_updates,
            )
            .unwrap();

            assert_eq!(runtime.supports_live_updates(), live_updates);
            let error = layers
                .register(
                    LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                    LoraLinearSpec::replicated(2, 2),
                    DType::F32,
                    Device::Cpu,
                )
                .unwrap_err();
            assert!(error.to_string().contains("after registry finalization"));
        }
    }

    #[test]
    fn persisted_qwen35_moe_config_restores_lora_namespace_alias() {
        let config = r#"{
            "architectures":["Qwen3NextForCausalLM"],
            "_mistralrs_gdn_v_head_layout":"tiled"
        }"#;
        let registry = new_dynamic_lora_registry(config).unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(
            site.key().path(),
            "model.language_model.layers.0.self_attn.q_proj"
        );
    }

    #[test]
    fn dense_qwen35_config_does_not_alias_lora_namespace() {
        let registry = new_dynamic_lora_registry(
            r#"{
                "architectures":["Qwen3_5ForCausalLM"],
                "_mistralrs_gdn_v_head_layout":"tiled"
            }"#,
        )
        .unwrap();
        let site = registry
            .register(
                LoraSiteKey::new("model.layers.0.self_attn.q_proj"),
                LoraLinearSpec::replicated(2, 2),
                DType::F32,
                Device::Cpu,
            )
            .unwrap();

        assert_eq!(site.key().path(), "model.layers.0.self_attn.q_proj");
    }
}
