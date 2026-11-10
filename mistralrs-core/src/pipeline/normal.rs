use super::causal::{CausalModel, CausalPipeline, CausalStepInputs};
use super::llg::build_llg_factory;
use super::loaders::ModelDownload;
use super::{
    get_model_paths, paged_attention_memory_reservations, reserve_recurrent_serving_capacity,
    text_models_inputs_processor::ModelInputs, AdapterKind, GeneralMetadata, Loader, ModelKind,
    ModelPaths, NormalModel, NormalModelLoader, TokenSource,
};
use super::{AutoNormalLoader, NormalLoaderType};
use super::{EitherCache, IsqOrganization, ModelCategory};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
use crate::distributed;
use crate::lora::Ordering;
use crate::paged_attention::{allocate_cache, AttentionImplementation, CacheConfigRequest};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, CudaDecodeGraphState,
};
use crate::pipeline::isq::{UqffWriteConfig, WeightLoadingMode, WeightLoadingState};
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::pipeline::{Modalities, ModelForwardContext, RecurrentBatchKind, SupportedModality};
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::xlora_models::NonGranularState;
use crate::{
    get_mut_arcmutex, get_paths, get_uqff_paths, xlora_model_loader, DeviceMapSetting,
    DynamicLoraRuntime, LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig, Pipeline,
    Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
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
            Some(arch) => arch.loader(),
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
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        paged_attn_config =
            paged_attn_config.map(|paged| paged.with_max_model_len(self.config.max_model_len));
        let config = super::loaders::layered_model_config(super::loaders::ModelConfigLayers {
            paths,
            prepared: self
                .prepared_source
                .as_ref()
                .map(|source| source.config.as_str()),
            from_uqff: self.config.from_uqff.is_some(),
            overrides: self.config.hf_config_overrides.as_ref(),
            mtp: self.mtp,
        })?;
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

        let write_uqff = self.config.write_uqff.is_some();
        let super::loaders::LoadDevices {
            tensor_parallelism,
            device,
            available: available_devices,
        } = super::loaders::resolve_load_devices(
            self.inner.model_config(&config)?.as_ref(),
            device,
            write_uqff,
            super::loaders::SingleCudaDevice::NcclRequested,
            paged_attn_config.as_ref(),
        )?;
        let use_distributed = tensor_parallelism.is_enabled();
        let sources = super::loaders::open_weight_sources(
            self.from_uqff.read().unwrap().as_ref(),
            self.prepared_source.as_ref().map(|source| &source.weights),
        )?;
        let uqff_reader = sources.uqff_reader.clone();

        let mapper = super::loaders::plan_device_map(
            super::loaders::DeviceMapPlanRequest {
                setting: mapper,
                loader: &*self.inner,
                config: &config,
                dtype,
                write_uqff,
                use_distributed,
                available_devices: &available_devices,
                paths,
                prepared_weights: self.prepared_source.as_ref().map(|source| &source.weights),
                sources: &sources,
                in_situ_quant,
                topology: self.config.topology.as_ref(),
                organization: self.config.organization,
                has_lora_adapters: self.lora_adapters.is_some(),
                matformer: None,
                packed_non_mapped: true,
                auto_params: None,
            },
            paged_attn_config.as_mut(),
        )?;
        let super::loaders::ResolvedDeviceMapping {
            pipeline_mapper,
            mapper,
            layer_devices,
            dtype,
        } = super::loaders::resolve_device_mapping(super::loaders::DeviceMappingRequest {
            setting: &mapper,
            num_layers: self.inner.num_layers(&config)?,
            device: &device,
            topology: self.config.topology.as_ref(),
            available_devices: &available_devices,
            dtype,
            write_uqff,
        })?;

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

        let matformer_slicing_config = super::loaders::load_matformer_slice(
            self.config.matformer_config_path.as_ref(),
            self.config.matformer_slice_name.as_ref(),
        )?;

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

        let (mapper, distributed_vb) = if use_distributed {
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
            let sharded_vb = match uqff_reader.clone() {
                Some(reader) => sharded_vb.with_uqff_reader(reader),
                None => sharded_vb,
            };
            (mapper, Some(sharded_vb))
        } else {
            (mapper, None)
        };
        // Sharded and prepared weights are already open; everything else is read from `paths`.
        let open_vb = distributed_vb.or_else(|| {
            self.prepared_source.as_ref().map(|source| {
                source
                    .weights
                    .clone()
                    .set_dtype(dtype)
                    .set_device(load_device.clone())
            })
        });
        let live_updates = !use_distributed;
        let rope_pairing = self
            .prepared_source
            .as_ref()
            .map(|source| source.rope_pairing);
        // LoRA without a prepared source reads from `paths` even when a shard is open.
        let open_weights = |mapper: &dyn DeviceMapper, reuse_open: bool| match open_vb {
            Some(vb) if reuse_open => Ok(vb),
            _ => super::loaders::open_mapped_weights(
                &*self.inner,
                super::loaders::MappedWeightsRequest {
                    paths,
                    dtype,
                    load_device: &load_device,
                    layer_devices: layer_devices.clone(),
                    silent,
                    config: &config,
                    dummy_isq_layers: loading_isq && self.config.from_uqff.is_some(),
                    moe_experts_only: matches!(
                        self.config.organization,
                        IsqOrganization::MoeExpertsOnly
                    ),
                    device_for_tensor: self.inner.get_device_for_tensor(
                        &config,
                        mapper,
                        loading_isq,
                    )?,
                    uqff_reader: uqff_reader.clone(),
                },
            ),
        };

        let (model, tracker, dynamic_lora) = match self.kind {
            ModelKind::Normal | ModelKind::GgufQuantized { .. } => {
                let vb = open_weights(mapper.as_ref(), true)?;
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
                        rope_pairing,
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
                let layers = Arc::new(self.inner.new_lora_registry(&config)?);
                let vb = open_weights(mapper.as_ref(), rope_pairing.is_some())?
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
                        rope_pairing,
                    },
                    attention_mechanism,
                )?;
                let dynamic_lora = super::finish_dynamic_lora_runtime(
                    paths,
                    layers,
                    self.lora_runtime_config
                        .expect("LoRA loaders have a runtime config"),
                    live_updates,
                )?;
                (model, tracker, Some(dynamic_lora))
            }
            _ => unreachable!(),
        };

        let tokenizer = match self.prepared_source.as_ref() {
            Some(source) => source.tokenizer.clone(),
            None => get_tokenizer(paths.get_tokenizer_filename(), None)?,
        };
        let (gen_conf, chat_template) =
            super::loaders::load_prompt_assets(super::loaders::PromptAssetsRequest {
                paths,
                config: &config,
                prepared: self.prepared_source.as_ref().map(|source| {
                    super::loaders::PreparedPromptAssets {
                        generation_config: source.generation_config.as_ref(),
                        chat_template: source.chat_template.as_ref(),
                        bos_token: source.bos_token.as_ref(),
                        eos_token: source.eos_token.as_ref(),
                        unk_token: source.unk_token.as_ref(),
                    }
                }),
                jinja_explicit: self.jinja_explicit.as_ref(),
                chat_template: self.chat_template.as_ref(),
            });

        super::isq_flow::finish_isq(super::isq_flow::IsqFinish {
            plan: &plan,
            modules: tracker.get().clone(),
            drive: &super::isq_flow::NormalCalibrationDrive(&*model),
            model: &*model,
            in_situ_quant,
            imatrix: self.config.imatrix.as_ref(),
            calibration_file: self.config.calibration_file.as_ref(),
            tokenizer: &tokenizer,
            chat_template: &chat_template,
            load_device: &load_device,
            mapper: pipeline_mapper.as_ref(),
            write_uqff: self.config.write_uqff.as_ref(),
            organization: self.config.organization,
            paths,
            omit_generation_config: self
                .prepared_source
                .as_ref()
                .is_some_and(|source| source.generation_config.is_none()),
            config: &config,
            processor_filename: &None,
            preprocessor_filename: &None,
        })?;

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
            source_weight_source: sources.active,
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
