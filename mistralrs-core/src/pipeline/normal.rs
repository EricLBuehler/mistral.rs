use super::causal::{CausalModel, CausalPipeline, CausalStepInputs};
use super::llg::build_llg_factory;
use super::loaders::ModelDownload;
use super::{
    get_model_paths, paged_attention_memory_reservations, reserve_recurrent_serving_capacity,
    text_models_inputs_processor::ModelInputs, AdapterKind, GeneralMetadata, Loader, ModelKind,
    ModelPaths, NormalModel, NormalModelLoader, TokenSource,
};
use super::{
    AutoNormalLoader, DeepSeekV2Loader, DeepSeekV3Loader, FlintLoader, GLM4Loader,
    GLM4MoeLiteLoader, GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader,
    GraniteMoeHybridLoader, HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, Lfm3Loader,
    LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType, Phi2Loader, Phi3Loader,
    Phi3_5MoELoader, Qwen2Loader, Qwen3Loader, Qwen3MoELoader, Qwen3NextLoader, Qwen3_5TextLoader,
    SmolLm3Loader, Starcoder2Loader,
};
use super::{EitherCache, IsqOrganization, ModelCategory};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
use crate::lora::Ordering;
use crate::paged_attention::{allocate_cache, AttentionImplementation, CacheConfigRequest};
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkPadTok, GenerationConfig};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, CudaDecodeGraphState,
};
use crate::pipeline::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::{
    get_chat_template, Modalities, ModelForwardContext, RecurrentBatchKind, SupportedModality,
};
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths, get_uqff_paths, lora_model_loader, normal_model_loader,
    xlora_model_loader, DeviceMapSetting, DynamicLoraRuntime, LoraAdapterSpec, LoraRuntimeConfig,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use either::Either;
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub(crate) struct NormalPipelineBuildArgs<'a> {
    pub model: Box<dyn NormalModel + Send + Sync>,
    pub tokenizer: Tokenizer,
    pub chat_template: ChatTemplate,
    pub generation_config: Option<GenerationConfig>,
    pub paged_attn_config: Option<PagedAttentionConfig>,
    pub attention_features: Option<crate::paged_attention::PrefixPrefillAttentionFeatures>,
    pub capacity_activation_estimate:
        &'a dyn Fn(
            &PagedAttentionConfig,
            usize,
        ) -> Result<super::loaders::ActivationMemoryEstimate>,
    pub dtype: DType,
    pub device: Device,
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
    pub silent: bool,
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
    pub max_model_len: Option<usize>,
}

pub(crate) fn build_normal_pipeline(
    args: NormalPipelineBuildArgs<'_>,
) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
    let NormalPipelineBuildArgs {
        model,
        tokenizer,
        chat_template,
        generation_config,
        mut paged_attn_config,
        attention_features,
        capacity_activation_estimate,
        dtype,
        device,
        mapper,
        silent,
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
        max_model_len,
    } = args;

    let model_metadata =
        crate::paged_attention::with_attention_features(model.model_config(), attention_features);
    #[cfg(feature = "metal")]
    if !loaded_for_uqff_write {
        super::metal_warmup::prepare_model(super::metal_warmup::MetalModelWarmup {
            device: &device,
            mapper: mapper.as_ref(),
            model: model_metadata.as_ref(),
            paged: paged_attn_config.as_ref(),
            dtype,
            tracked_modules: &tracked_modules,
            packed_prefill: model.supports_packed_prefill(),
        })?;
    }
    let num_hidden_layers = match model.cache() {
        EitherCache::Full(full) => full.lock().len(),
        EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
        EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
    };
    let recurrent_checkpoints_supported = model.supports_recurrent_speculative_checkpoints();
    let recurrent_transitions_supported = model.supports_recurrent_speculative_transitions();
    let recurrent_pool_grew = paged_attn_config
        .as_ref()
        .map(|config| {
            let kv_bytes_per_token =
                super::paged_kv_bytes_per_token(config, dtype, model_metadata.as_ref())?;
            reserve_recurrent_serving_capacity(
                model.cache(),
                config,
                recurrent_checkpoints_supported,
                recurrent_transitions_supported,
                &device,
                kv_bytes_per_token,
            )
        })
        .transpose()?
        .unwrap_or(false);
    let recurrent_pool_grew =
        if recurrent_transitions_supported && super::uses_recurrent_transition_log(model.cache()) {
            model.reserve_recurrent_speculative_transition_storage()? || recurrent_pool_grew
        } else {
            recurrent_pool_grew
        };
    let recurrent_pool_grew =
        model.reserve_recurrent_decode_deferred_storage()? || recurrent_pool_grew;
    #[cfg(feature = "cuda")]
    if recurrent_pool_grew {
        super::synchronize_cuda_contexts(&device, mapper.as_ref())?;
    }
    #[cfg(not(feature = "cuda"))]
    let _ = recurrent_pool_grew;

    let llg_factory = build_llg_factory(tokenizer.clone())?;
    #[cfg(feature = "cuda")]
    if !loaded_for_uqff_write {
        super::prepare_cuda_runtime(&device, mapper.as_ref())?;
    }

    if let Some(source) = &source_weight_source {
        #[cfg(feature = "cuda")]
        super::synchronize_cuda_contexts(&device, mapper.as_ref())?;
        source.release_source_cache()?;
    }

    let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config.as_mut() {
        #[cfg(feature = "cuda")]
        let reserve_decode_graphs = device.is_cuda()
            && model.supports_cuda_decode_graphs()
            && cuda_decode_graphs_enabled()
            && cuda_decode_graph_supported_for_model(Some(model_metadata.as_ref()));
        #[cfg(not(feature = "cuda"))]
        let reserve_decode_graphs = false;
        let paged_snapshot = paged_attn_config.clone();
        let estimate_for_capacity =
            |usable_tokens| capacity_activation_estimate(&paged_snapshot, usable_tokens);
        let memory_reservations = paged_attn_config.memory_reservations()?;
        let layer_devices = (0..model_metadata.num_layers())
            .map(|layer| mapper.device_for(layer, false).cloned())
            .collect::<Vec<_>>();
        let mut cache_request = CacheConfigRequest {
            memory_reservations,
            dtype,
            model: model_metadata.as_ref(),
            device: &device,
            layer_devices: &layer_devices,
            capacity_activation_estimate: &estimate_for_capacity,
            reserve_decode_graphs,
            silent,
        };
        super::configure_automatic_recurrent_prefix_capacity(
            model.cache(),
            paged_attn_config,
            cache_request,
        )?;
        cache_request.memory_reservations =
            paged_attention_memory_reservations(model.cache(), paged_attn_config, &device)?;
        let (cache_config, cache_engine) = allocate_cache(paged_attn_config, cache_request)?;
        (Some(cache_config), Some(cache_engine))
    } else {
        (None, None)
    };

    let max_model_len = super::resolve_max_model_len(
        model.max_seq_len(),
        max_model_len,
        cache_config.as_ref().zip(
            paged_attn_config
                .as_ref()
                .map(PagedAttentionConfig::memory_goal),
        ),
        silent,
    )?;
    let generation_defaults = generation_config
        .as_ref()
        .and_then(GenerationConfig::generation_defaults);
    let eos = calculate_eos_tokens(&chat_template, generation_config.as_ref(), &tokenizer);
    let sliding_window = model.config().sliding_window;
    #[cfg(feature = "cuda")]
    let runtime_memory_reservation = super::paged_attention_runtime_memory_reservation(
        model.cache(),
        paged_attn_config.as_ref(),
        &device,
        &mapper.get_unique_devices(),
    )?;

    Ok(Arc::new(Mutex::new(NormalPipeline {
        model,
        tokenizer: tokenizer.into(),
        chat_template: Arc::new(chat_template),
        model_id,
        metadata: Arc::new(GeneralMetadata {
            max_model_len,
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
            memory_plan: paged_attn_config
                .as_ref()
                .map(|config| config.runtime_memory_plan()),
            #[cfg(feature = "cuda")]
            runtime_memory_reservation,
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
        speculative_autotuner: StdMutex::new(Default::default()),
        generation_defaults,
        mapper,
        tracked_modules,
        source_weight_files,
        source_weight_source,
        dynamic_lora,
        flavor: NormalFlavor {
            no_kv_cache,
            non_granular_state: non_granular_index.map(|tgt_non_granular_index| NonGranularState {
                non_granular_index: Arc::new(Mutex::new(0)),
                tgt_non_granular_index,
            }),
        },
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
            Some(NormalLoaderType::Flint) => Box::new(FlintLoader),
            Some(NormalLoaderType::SmolLm3) => Box::new(SmolLm3Loader),
            Some(NormalLoaderType::GraniteMoeHybrid) => Box::new(GraniteMoeHybridLoader),
            Some(NormalLoaderType::GptOss) => Box::new(GptOssLoader),
            Some(NormalLoaderType::HunYuanDenseV1) => Box::new(HunYuanDenseV1Loader),
            Some(NormalLoaderType::HunYuanMoEV1) => Box::new(HunYuanMoEV1Loader),
            Some(NormalLoaderType::Qwen3Next) => Box::new(Qwen3NextLoader),
            Some(NormalLoaderType::Qwen3_5) => Box::new(Qwen3_5TextLoader),
            Some(NormalLoaderType::Lfm2) => Box::new(Lfm2Loader),
            Some(NormalLoaderType::Lfm3) => Box::new(Lfm3Loader),
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
    fn download_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        silent: bool,
    ) -> Result<ModelDownload> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        if let Some(cache) = self.hf_cache_path.as_ref() {
            GLOBAL_HF_CACHE.get_or_init(|| cache.clone());
        }

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
        let mut download = ModelDownload::new(paths?);
        if let Some(paths) = self.from_uqff.read().unwrap().as_ref() {
            download.additional_files.extend(paths.iter().cloned());
        }
        Ok(download)
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
        paged_attn_config =
            paged_attn_config.map(|paged| paged.with_max_model_len(self.config.max_model_len));
        let config = match self.prepared_source.as_ref() {
            Some(source) => source.config.clone(),
            None => super::loaders::load_model_config(
                paths.get_config_filename(),
                self.config.from_uqff.is_none(),
            )?,
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
            super::disable_unsupported_paged_attention(&mut paged_attn_config)?;
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
        if !write_uqff {
            if let Some(paged) = paged_attn_config.as_ref() {
                paged.initialize_memory_budget(&available_devices)?;
            }
        }
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
        let weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>> = uqff_reader
            .clone()
            .map(|reader| reader as Arc<dyn mistralrs_quant::QuantizedWeightSource>)
            .or(prepared_weight_source.clone());

        // If auto, convert to Map if not using nccl
        if write_uqff {
            mapper = DeviceMapSetting::dummy();
        } else if use_distributed {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(params) = mapper.clone() {
            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;
            let sizing_source = super::loaders::ResourceSource {
                weights: self.prepared_source.as_ref().map(|source| &source.weights),
                paths: paths.get_weight_filenames(),
            };
            let checkpoint = sizing_source.checkpoint_tensors()?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (mut layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                super::loaders::auto_device_map_sizes(
                    &*self.inner,
                    super::loaders::AutoMapSizingInput {
                        config: &config,
                        dtype,
                        weight_files: paths.get_weight_filenames(),
                        checkpoint: &checkpoint,
                        uqff: uqff_reader.is_some(),
                        weight_source: weight_source.as_ref(),
                        prepared_weight_source: prepared_weight_source.as_ref(),
                        in_situ_quant,
                        topology: self.config.topology.as_ref(),
                        organization: self.config.organization,
                        has_lora_adapters: self.lora_adapters.is_some(),
                        matformer: None,
                        packed_non_mapped: true,
                    },
                )?;

            let resources = self.inner.layer_resources(&config, sizing_source)?;
            let total_model_size_in_bytes = super::loaders::add_resource_bytes(
                &mut layer_sizes_in_bytes,
                total_model_size_in_bytes,
                &resources,
            )?;
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
                &resources,
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

        if let Some(paged) = paged_attn_config.as_mut() {
            super::loaders::reserve_model_activation_memory(
                self.inner.as_ref(),
                &config,
                paged,
                dtype,
                &device,
                &super::AutoDeviceMapParams::default_text(),
            )?;
        }

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu && paged_attn_config.is_some() {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            super::disable_unsupported_paged_attention(&mut paged_attn_config)?;
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
                        let layers = Arc::new(self.inner.new_lora_registry(&config)?);
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
                        let layers = Arc::new(self.inner.new_lora_registry(&config)?);
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

        let activation_device = device.clone();
        let capacity_activation_estimate = |paged: &PagedAttentionConfig, usable_tokens| {
            super::loaders::estimate_model_activation_memory(
                self.inner.as_ref(),
                &config,
                super::loaders::ActivationMemoryContext {
                    params: &super::AutoDeviceMapParams::default_text(),
                    limits: paged.execution_limits,
                    dtype,
                    device: &activation_device,
                    paged,
                    cache_capacity_tokens: Some(usable_tokens),
                },
            )
        };
        build_normal_pipeline(NormalPipelineBuildArgs {
            model,
            tokenizer,
            chat_template,
            generation_config: gen_conf,
            paged_attn_config,
            attention_features: self
                .inner
                .activation_model_config(&config)?
                .attention_features,
            capacity_activation_estimate: &capacity_activation_estimate,
            dtype,
            device,
            mapper: pipeline_mapper,
            silent,
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
            max_model_len: self.config.max_model_len,
        })
    }

    fn get_id(&self) -> String {
        self.model_id.clone()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

pub(crate) struct NormalFlavor {
    no_kv_cache: bool,
    non_granular_state: Option<NonGranularState>,
}

pub(crate) struct NormalStepArgs {
    input_ids_full: Option<Tensor>,
    seqlen_offsets_full: Option<Vec<usize>>,
    flash_meta_full: Option<FlashParams>,
}

pub(crate) type NormalPipeline = CausalPipeline<dyn NormalModel + Send + Sync>;

impl CausalModel for dyn NormalModel + Send + Sync {
    type Flavor = NormalFlavor;
    type StepArgs = NormalStepArgs;
    #[cfg(feature = "cuda")]
    const SPECULATIVE_CUDA_GRAPHS: bool = false;

    fn as_speculative_target(&self) -> &dyn crate::speculative::SpeculativeTargetMixin {
        self
    }
    fn as_speculative_target_mut(&mut self) -> &mut dyn crate::speculative::SpeculativeTargetMixin {
        self
    }
    fn device(&self) -> &Device {
        NormalModel::device(self)
    }
    fn cache(&self) -> &EitherCache {
        NormalModel::cache(self)
    }
    fn config(&self) -> &crate::paged_attention::ModelConfigMetadata {
        NormalModel::config(self)
    }
    fn supports_packed_prefill(&self) -> bool {
        NormalModel::supports_packed_prefill(self)
    }

    fn unpack_step_inputs(inputs: Box<dyn Any>) -> CausalStepInputs<NormalStepArgs> {
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
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
        CausalStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            adapter_leases,
            args: NormalStepArgs {
                input_ids_full,
                seqlen_offsets_full,
                flash_meta_full,
            },
        }
    }
    fn forward_step_args(
        &self,
        input_ids: &Tensor,
        _args: NormalStepArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    #[cfg(feature = "cuda")]
    fn forward_text(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        self.forward(input_ids, ctx)
    }
    fn speculative_verify_inputs(
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
    fn supports_cuda_decode_graphs_for(
        &self,
        _args: Option<&NormalStepArgs>,
    ) -> candle_core::Result<bool> {
        Ok(self.supports_cuda_decode_graphs())
    }

    fn category(_pipeline: &NormalPipeline) -> ModelCategory {
        ModelCategory::Text
    }
    fn requires_uniform_prompt_batch(pipeline: &NormalPipeline) -> bool {
        normal_model_requires_uniform_prompt_batch(
            pipeline.model.cache().is_hybrid(),
            Pipeline::supports_packed_prefill(pipeline),
            pipeline.model.is_xlora(),
            pipeline.model.has_speculative_proposer(),
        )
    }
    fn reset_non_granular_state(pipeline: &NormalPipeline) {
        if let Some(s) = pipeline.flavor.non_granular_state.as_ref() {
            *pipeline.model.cache().full().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn forward_override(
        pipeline: &NormalPipeline,
        step: &CausalStepInputs<NormalStepArgs>,
    ) -> Option<candle_core::Result<Tensor>> {
        if !pipeline.model.is_xlora() {
            return None;
        }
        Some(
            pipeline.model.xlora_forward(
                &step.input_ids,
                step.args.input_ids_full.as_ref().unwrap_or(&step.input_ids),
                &step.seqlen_offsets,
                step.args
                    .seqlen_offsets_full
                    .as_ref()
                    .unwrap_or(&step.seqlen_offsets),
                pipeline.flavor.no_kv_cache,
                &pipeline.flavor.non_granular_state,
                step.context_lens.clone(),
                step.position_ids.clone(),
                &step.flash_meta,
                step.args
                    .flash_meta_full
                    .as_ref()
                    .unwrap_or(&step.flash_meta),
            ),
        )
    }

    fn blocks_packed_prefill(&self) -> bool {
        self.is_xlora()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn download_selects_uqff_shards_and_residual_without_loading_tensors() -> anyhow::Result<()> {
        let directory = tempfile::tempdir()?;
        let root = directory.path();
        for filename in [
            "q8_0-0.uqff",
            "q8_0-1.uqff",
            "q4_0-0.uqff",
            "residual.safetensors",
            "model.safetensors",
        ] {
            std::fs::write(root.join(filename), b"not a tensor file")?;
        }
        std::fs::write(root.join("config.json"), b"{}")?;
        std::fs::write(root.join("tokenizer.json"), b"not a tokenizer")?;
        let loader = super::NormalLoaderBuilder::new(
            super::NormalSpecificConfig {
                from_uqff: Some(vec![PathBuf::from("q8_0-0.uqff")]),
                ..Default::default()
            },
            None,
            None,
            Some(root.display().to_string()),
            false,
            None,
        )
        .build(Some(super::NormalLoaderType::Llama))?;
        let download = loader.download_model_from_hf(None, crate::TokenSource::None, true)?;
        assert_eq!(
            download.paths.get_weight_filenames(),
            &[root.join("residual.safetensors")]
        );
        let files = download.files();
        assert!(files.contains(&root.join("q8_0-0.uqff")));
        assert!(files.contains(&root.join("q8_0-1.uqff")));
        assert!(!files.contains(&root.join("q4_0-0.uqff")));
        assert!(!files.contains(&root.join("model.safetensors")));
        Ok(())
    }

    use super::normal_model_requires_uniform_prompt_batch;
    use crate::pipeline::finish_dynamic_lora_runtime;
    use crate::pipeline::loaders::exact_weight_source_sizes;
    use crate::pipeline::{
        AdapterPaths, AutoNormalLoader, LocalModelPaths, NormalModelLoader, Qwen3NextLoader,
        Qwen3_5TextLoader,
    };
    use crate::LoraRuntimeConfig;
    use candle_core::{DType, Device};
    use mistralrs_quant::{LoraLayerRegistry, LoraLinearSpec, LoraSiteKey};
    use std::{path::PathBuf, sync::Arc};

    const WIDTH: usize = 256;
    const UNBOUND_BYTES: usize = 1_024;
    const Q8_BLOCK_BYTES: usize = 34;
    const Q8_BLOCK_ELEMS: usize = 32;
    const Q3_BLOCK_BYTES: usize = 110;

    #[test]
    fn gguf_mapping_uses_mixed_precision_resident_sizes() -> anyhow::Result<()> {
        use candle_core::{
            quantized::{gguf_file, GgmlDType, QTensor},
            Tensor,
        };
        use mistralrs_quant::{
            GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource, QuantizedWeightSource,
        };

        let mut bindings = GgufBindingMap::new();
        let mut tensors = Vec::new();
        for (native, stored, shape, dtype) in [
            (
                "model.embed_tokens.weight",
                "token_embd.weight",
                vec![2, WIDTH],
                GgmlDType::F32,
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
                vec![WIDTH, WIDTH],
                GgmlDType::Q8_0,
            ),
            (
                "model.layers.1.mlp.experts.gate_proj.weight",
                "blk.1.ffn_gate_exps.weight",
                vec![2, WIDTH, WIDTH],
                GgmlDType::Q3K,
            ),
            (
                "model.layers.1.mlp.gate.weight",
                "blk.1.ffn_gate_inp.weight",
                vec![2, WIDTH],
                GgmlDType::F32,
            ),
            (
                "model.layers.1.input_layernorm.weight",
                "blk.1.attn_norm.weight",
                vec![WIDTH],
                GgmlDType::F32,
            ),
        ] {
            let tensor = Tensor::ones(shape, DType::F32, &Device::Cpu)?;
            tensors.push((stored, QTensor::quantize(&tensor, dtype)?));
            bindings.insert(native, GgufTensorBinding::tensor(stored));
        }
        let mut file = tempfile::NamedTempFile::new()?;
        gguf_file::write(
            file.as_file_mut(),
            &[],
            &tensors
                .iter()
                .map(|(name, tensor)| (*name, tensor))
                .collect::<Vec<_>>(),
        )?;
        let archive = Arc::new(GgufArchive::open_file(file.path())?);
        let source = GgufWeightSource::new(archive, &bindings, DType::BF16)?;
        assert_eq!(source.pack_factor(DType::BF16)?, 1);
        let (mut layers, outside, total) =
            exact_weight_source_sizes(&source, DType::BF16, 2)?.unwrap();
        let total = crate::pipeline::loaders::add_resource_bytes(
            &mut layers,
            total,
            &[crate::pipeline::loaders::LayerResource {
                name: "extra".to_owned(),
                layer: 1,
                device_bytes: UNBOUND_BYTES,
                host_backing: None,
            }],
        )?;
        let attention = WIDTH * WIDTH / Q8_BLOCK_ELEMS * Q8_BLOCK_BYTES;
        let experts = 2 * WIDTH * Q3_BLOCK_BYTES;
        let dense = 3 * WIDTH * DType::BF16.size_in_bytes();
        assert_eq!(layers, vec![attention, experts + dense + UNBOUND_BYTES]);
        assert_eq!(outside, 2 * WIDTH * DType::BF16.size_in_bytes());
        assert_eq!(total, attention + experts + dense + UNBOUND_BYTES + outside);
        assert!(total < 3 * WIDTH * WIDTH * DType::BF16.size_in_bytes());
        Ok(())
    }

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
        for loader in [
            &AutoNormalLoader as &dyn NormalModelLoader,
            &Qwen3NextLoader,
        ] {
            assert!(loader.new_lora_registry("").is_err());
            assert!(loader.new_lora_registry("{").is_err());
            let registry = loader.new_lora_registry(config).unwrap();
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
    }

    #[test]
    fn dense_qwen35_config_does_not_alias_lora_namespace() {
        let config = r#"{
            "architectures":["Qwen3_5ForCausalLM"],
            "_mistralrs_gdn_v_head_layout":"tiled"
        }"#;
        for loader in [
            &AutoNormalLoader as &dyn NormalModelLoader,
            &Qwen3_5TextLoader,
        ] {
            assert!(loader.new_lora_registry("").is_err());
            assert!(loader.new_lora_registry("{").is_err());
            let registry = loader.new_lora_registry(config).unwrap();
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
}
