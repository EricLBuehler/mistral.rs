use super::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use super::{
    get_model_paths, AdapterKind, AnyMoePipelineMixin, AutoMultimodalLoader, CacheManager,
    CacheManagerMixin, EitherCache, ForwardInputsResult, Gemma3Loader, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, MiniCpmOLoader, ModelCategory, ModelKind, ModelPaths,
    MultimodalModel, MultimodalModelLoader, MultimodalPromptPrefixer, Phi4MMLoader,
    PreProcessingMixin, Processor, Qwen2VLLoader, Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, TokenSource, VLlama4Loader, VLlamaLoader,
};
use super::{
    DiffusionGemmaLoader, Gemma3nLoader, Gemma4Loader, Idefics2Loader, Idefics3Loader, LLaVALoader,
    LLaVANextLoader, Lfm2VlLoader, Mistral3Loader, MultimodalLoaderType, Phi3VLoader,
    Qwen2_5VLLoader, VoxtralLoader,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
#[cfg(feature = "cuda")]
use crate::kv_cache::RecurrentStateSnapshot;
use crate::kv_cache::{FullCacheManager, HybridCacheManager, NormalCacheManager};

#[cfg(feature = "cuda")]
type SeqRecurrentStateSnapshots = Vec<(usize, Vec<RecurrentStateSnapshot>)>;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{
    calculate_eos_tokens, BeginEndUnkPadTok, ChatTemplateValue, GenerationConfig,
};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    capture_cuda_decode_graph, cuda_decode_graphs_enabled, prepare_cuda_graph_memory_pool,
    CudaDecodeGraphCaptureCtx, CudaDecodeGraphKey, CudaDecodeGraphState,
};
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::{AutoDeviceMapQuantization, QuantizationConfigShim};
use crate::pipeline::sampling::sample_and_add_toks;
#[cfg(feature = "cuda")]
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{
    get_chat_template, hf::build_api, ChatTemplate, IsqOrganization, LocalModelPaths,
    ModelForwardContext, RecurrentBatchKind, RecurrentMetadata,
};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    api_dir_list, api_get_file, get_paths, get_uqff_paths, lora_model_loader,
    multimodal_normal_model_loader, multimodal_normal_model_loader_sharded, AnyMoeExpertType,
    DeviceMapSetting, DynamicLoraRuntime, LoraAdapterSpec, LoraRuntimeConfig, PagedAttentionConfig,
    Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, Var};
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

pub struct MultimodalPipeline {
    model: Box<dyn MultimodalModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    processor: Arc<dyn Processor + Send + Sync>,
    preprocessor_config: Arc<PreProcessorConfig>,
    prefixer: Arc<dyn MultimodalPromptPrefixer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    #[cfg(feature = "cuda")]
    cuda_decode_graph: StdMutex<CudaDecodeGraphState>,

    generation_defaults: Option<crate::ModelGenerationDefaults>,
    tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    dynamic_lora: Option<Arc<DynamicLoraRuntime>>,
}

/// A loader for a multimodal (non-quantized) model.
pub struct MultimodalLoader {
    inner: Box<dyn MultimodalModelLoader>,
    model_id: String,
    config: MultimodalSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
    loader_type: Option<MultimodalLoaderType>,
}

#[derive(Default)]
/// A builder for a loader for a multimodal (non-quantized) model.
pub struct MultimodalLoaderBuilder {
    model_id: Option<String>,
    config: MultimodalSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapters: Option<Vec<LoraAdapterSpec>>,
    lora_runtime_config: Option<LoraRuntimeConfig>,
}

#[derive(Clone, Default)]
/// Config specific to loading a multimodal model.
pub struct MultimodalSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
    pub organization: IsqOrganization,
}

impl MultimodalLoaderBuilder {
    pub fn new(
        config: MultimodalSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        jinja_explicit: Option<String>,
    ) -> Self {
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            kind: ModelKind::Normal,
            hf_cache_path: None,
            lora_adapters: None,
            lora_runtime_config: None,
        }
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

    pub fn build(self, loader: Option<MultimodalLoaderType>) -> Box<dyn Loader> {
        let loader_type = loader.clone();
        let loader: Box<dyn MultimodalModelLoader> = match loader {
            Some(MultimodalLoaderType::Phi3V) => Box::new(Phi3VLoader),
            Some(MultimodalLoaderType::Idefics2) => Box::new(Idefics2Loader),
            Some(MultimodalLoaderType::LLaVANext) => Box::new(LLaVANextLoader),
            Some(MultimodalLoaderType::LLaVA) => Box::new(LLaVALoader),
            Some(MultimodalLoaderType::Lfm2Vl) => Box::new(Lfm2VlLoader),
            Some(MultimodalLoaderType::VLlama) => Box::new(VLlamaLoader),
            Some(MultimodalLoaderType::Qwen2VL) => Box::new(Qwen2VLLoader),
            Some(MultimodalLoaderType::Idefics3) => Box::new(Idefics3Loader),
            Some(MultimodalLoaderType::MiniCpmO) => Box::new(MiniCpmOLoader),
            Some(MultimodalLoaderType::Phi4MM) => Box::new(Phi4MMLoader),
            Some(MultimodalLoaderType::Qwen2_5VL) => Box::new(Qwen2_5VLLoader),
            Some(MultimodalLoaderType::Gemma3) => Box::new(Gemma3Loader),
            Some(MultimodalLoaderType::Mistral3) => Box::new(Mistral3Loader),
            Some(MultimodalLoaderType::Llama4) => Box::new(VLlama4Loader),
            Some(MultimodalLoaderType::Gemma3n) => Box::new(Gemma3nLoader),
            Some(MultimodalLoaderType::Qwen3VL) => Box::new(Qwen3VLLoader),
            Some(MultimodalLoaderType::Qwen3VLMoE) => Box::new(Qwen3VLMoELoader),
            Some(MultimodalLoaderType::Qwen3_5) => Box::new(Qwen3_5Loader),
            Some(MultimodalLoaderType::Qwen3_5Moe) => Box::new(Qwen3_5MoeLoader),
            Some(MultimodalLoaderType::Voxtral) => Box::new(VoxtralLoader),
            Some(MultimodalLoaderType::Gemma4) => Box::new(Gemma4Loader),
            Some(MultimodalLoaderType::DiffusionGemma) => Box::new(DiffusionGemmaLoader),
            None => Box::new(AutoMultimodalLoader),
        };
        Box::new(MultimodalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            jinja_explicit: self.jinja_explicit,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            lora_adapters: self.lora_adapters,
            lora_runtime_config: self.lora_runtime_config,
            loader_type,
        })
    }
}

impl MultimodalLoader {
    fn validate_dynamic_lora(&self) -> Result<()> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        if self.lora_adapters.is_some()
            && self.loader_type.as_ref() != Some(&MultimodalLoaderType::Qwen3_5Moe)
        {
            anyhow::bail!(
                "dynamic LoRA for multimodal pipelines is supported only for the Qwen3.5/3.6 MoE text submodel; vision-tower adapters and other multimodal architectures are not supported"
            );
        }
        Ok(())
    }
}

impl Loader for MultimodalLoader {
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
        self.validate_dynamic_lora()?;
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
                xlora_model_id: None,
                lora_adapters: self.lora_adapters.as_deref(),
                xlora_order: None,
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
            &paths?,
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
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mut mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        self.validate_dynamic_lora()?;
        let config = std::fs::read_to_string(paths.get_config_filename())?;
        let runtime_config = self
            .inner
            .runtime_config(&config, self.config.max_model_len)?;

        if !self.inner.supports_paged_attention(&config) {
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        // Tokenizer deserialization can briefly use far more memory than its final representation.
        let (processor, preprocessor_config, tokenizer, llg_factory) = {
            let processor_config_json = paths
                .get_processor_config()
                .as_ref()
                .map(|f| fs::read_to_string(f).unwrap());

            // Some models only ship nested preprocessor settings in processor_config.json.
            let preprocessor_config: PreProcessorConfig =
                match paths.get_preprocessor_config().as_ref() {
                    Some(preprocessor_config) => {
                        serde_json::from_str(&fs::read_to_string(preprocessor_config).unwrap())
                            .unwrap()
                    }
                    None => processor_config_json.as_deref().map_or_else(
                        PreProcessorConfig::default,
                        |json| match PreProcessorConfig::from_processor_config_json(json) {
                            Ok(config) => config,
                            Err(err) => {
                                warn!(
                                    "Failed to synthesize preprocessor config from processor_config.json: {err}"
                                );
                                PreProcessorConfig::default()
                            }
                        },
                    ),
                };
            let processor_config: Option<ProcessorConfig> = processor_config_json
                .as_deref()
                .map(|json| serde_json::from_str(json).unwrap());
            let processor = self.inner.get_processor(
                &config,
                processor_config,
                preprocessor_config.clone(),
                self.config.max_edge,
            );
            let tokenizer = get_tokenizer(
                paths.get_tokenizer_filename(),
                Some(processor.get_special_tokens()),
            )?;
            let llg_factory = build_llg_factory(tokenizer.clone())?;
            (processor, preprocessor_config, tokenizer, llg_factory)
        };

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
        } else if use_distributed {
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

        // If auto, convert to Map if not using nccl
        let mut max_kv_tokens: Option<usize> = None;
        if write_uqff {
            mapper = DeviceMapSetting::dummy();
        } else if use_distributed {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(mut params) = mapper.clone() {
            // We can promote to multimodal params if we get text params
            params = params.maybe_promote_to_multimodal();
            max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());

            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(reader) = uqff_reader.as_ref() {
                    let weight_pack_factor = reader.pack_factor(dtype)?;
                    let quantization = AutoDeviceMapQuantization::uqff(reader);
                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        matformer_slicing_config.as_ref(),
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        Some(&quantization),
                        matformer_slicing_config.as_ref(),
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else if let Some(isq) = in_situ_quant {
                    let weight_pack_factor = isq.pack_factor(dtype);
                    let quantization =
                        AutoDeviceMapQuantization::isq(Some(isq), self.config.topology.as_ref());
                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        matformer_slicing_config.as_ref(),
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        Some(&quantization),
                        matformer_slicing_config.as_ref(),
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else {
                    // Be sure to get the weight pack factor here; we might be loading a prequantized model.
                    let weight_pack_factor =
                        QuantizationConfigShim::get_quant_config_pack_factor(&config, dtype)?;
                    let quantization = self
                        .config
                        .topology
                        .as_ref()
                        .map(|topology| AutoDeviceMapQuantization::isq(None, Some(topology)));
                    let layer_sizes_in_bytes = self.inner.layer_sizes_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        matformer_slicing_config.as_ref(),
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        quantization.as_ref(),
                        matformer_slicing_config.as_ref(),
                    )?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
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
                paged_attn_config.as_ref(),
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

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

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
            let (mapper, sharded_vb) = distributed::prepare_distributed_mapper(
                dtype,
                &device,
                &available_devices,
                tensor_parallelism.world_size(),
                silent,
                &config,
                loading_isq,
                self.config.from_uqff.is_some(),
                self.config.write_uqff.is_some(),
                self.config.organization,
                &*self.inner,
                paths.as_ref(),
            )?;
            let sharded_vb = if let Some(reader) = uqff_reader.clone() {
                sharded_vb.with_uqff_reader(reader)
            } else {
                sharded_vb
            };

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal => {
                    let (model, tracker) = multimodal_normal_model_loader_sharded!(
                        sharded_vb,
                        runtime_config,
                        self.inner,
                        mapper,
                        loading_isq,
                        device.clone(),
                        attention_mechanism,
                        multi_progress.clone(),
                        matformer_slicing_config.clone(),
                        uqff_reader.clone(),
                    );
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                } => lora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    runtime_config,
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
                ),
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => {
                    let (model, tracker) = multimodal_normal_model_loader!(
                        paths,
                        Some(dtype),
                        &load_device,
                        layer_devices.clone(),
                        runtime_config,
                        self.inner,
                        silent,
                        mapper,
                        loading_isq,
                        self.config.from_uqff.is_some(),
                        device.clone(),
                        attention_mechanism,
                        matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                        multi_progress,
                        matformer_slicing_config.clone(),
                        uqff_reader.clone(),
                    );
                    (model, tracker, None)
                }
                ModelKind::Adapter {
                    adapter: AdapterKind::Lora,
                } => lora_model_loader!(
                    paths,
                    Some(dtype),
                    &load_device,
                    layer_devices.clone(),
                    runtime_config,
                    self.inner,
                    silent,
                    mapper,
                    loading_isq,
                    self.config.from_uqff.is_some(),
                    device.clone(),
                    attention_mechanism,
                    matches!(self.config.organization, IsqOrganization::MoeExpertsOnly),
                    multi_progress,
                    matformer_slicing_config.clone(),
                    uqff_reader.clone(),
                    self.lora_runtime_config
                        .expect("LoRA loaders have a runtime config"),
                    true,
                ),
                _ => unreachable!(),
            }
        };

        // Release Metal loader scratch buffers before constructing the remaining pipeline state.
        for device in &available_devices {
            if matches!(device, Device::Metal(_)) {
                device.synchronize()?;
            }
        }

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        if model.is_block_diffusion() {
            if let Some(raw) = paths
                .get_gen_conf_filename()
                .and_then(|f| fs::read_to_string(f).ok())
            {
                model.configure_block_diffusion(&raw);
            }
        }
        let chat_template_explicit = paths
            .get_chat_template_explicit()
            .as_ref()
            .map(|x| x.to_string_lossy().to_string());
        let mut chat_template = get_chat_template(
            paths,
            self.jinja_explicit.as_ref(),
            chat_template_explicit.as_ref(),
            self.chat_template.as_ref(),
            None,
        );

        // If no chat template was found, use the loader's built-in default (if any).
        if chat_template.chat_template.is_none() {
            if let Some(default_tmpl) = self.inner.default_chat_template(&config) {
                info!("Using loader's built-in default chat template.");
                chat_template.chat_template = Some(ChatTemplateValue(Either::Left(default_tmpl)));
            }
        }

        // If no bos/eos tokens are set, use the loader's defaults (e.g. for Voxtral
        // which has no tokenizer_config.json).
        if let Some((bos, eos)) = self.inner.default_bos_eos(&config) {
            if chat_template.bos_token.is_none() {
                chat_template.bos_token = Some(BeginEndUnkPadTok(Either::Left(bos)));
            }
            if chat_template.eos_token.is_none() {
                chat_template.eos_token = Some(BeginEndUnkPadTok(Either::Left(eos)));
            }
        }

        let imatrix_map = if plan.wants_imatrix {
            let drive = super::isq_flow::MultimodalCalibrationDrive(&*model);
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
                generation_config: paths.get_gen_conf_filename(),
                config: config.clone(),
                processor_filename: paths.get_processor_config(),
                preprocessor_filename: paths.get_preprocessor_config(),
                modules: None,
                module_paths: None,
            };
            write_uqff_artifacts(UqffWriteRequest {
                output: write_uqff.output.clone(),
                types: uqff_types,
                base_model: write_uqff.base_model.clone(),
                repo_id: write_uqff.repo_id.clone(),
                layers,
                residual,
                full_ser,
                imatrix: imatrix_map.unwrap_or_default(),
            })?;
        }

        let model_metadata = model.model_config();
        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                dtype,
                paged_attn_config.cache_type,
                model_metadata.as_ref(),
                &device,
                &layer_devices,
                silent,
                None,
                max_kv_tokens,
            )?;
            let cache_engine = CacheEngine::new(
                model_metadata.as_ref(),
                &cache_config,
                dtype,
                &device,
                layer_devices,
            )?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let num_hidden_layers = match model.cache() {
            EitherCache::Full(full) => full.lock().len(),
            EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
            EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
        };
        let mut generation_defaults = gen_conf
            .as_ref()
            .and_then(GenerationConfig::generation_defaults);
        // HF's `max_new_tokens` for block-diffusion checkpoints is the per-call generate()
        // default (a single canvas); applying it as a session cap truncates every answer.
        if model.is_block_diffusion() {
            if let Some(defaults) = generation_defaults.as_mut() {
                defaults.max_new_tokens = None;
                defaults.max_length = None;
            }
        }
        let eos = calculate_eos_tokens(&chat_template, gen_conf.as_ref(), &tokenizer);
        let sliding_window = model.config().sliding_window;
        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = if self.config.from_uqff.is_some() {
            Vec::new()
        } else {
            paths.get_weight_filenames().to_vec()
        };

        Ok(Arc::new(Mutex::new(MultimodalPipeline {
            model,
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: Some(llg_factory),
                is_xlora: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                no_kv_cache: false,
                no_prefix_cache: !self.inner.supports_prefix_cacher(&config),
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
                model_metadata: Some(model_metadata),
                modalities: self.inner.modalities(&config)?,
                loaded_for_uqff_write: self.config.write_uqff.is_some(),
            }),
            processor,
            prefixer: self.inner.prefixer(&config),
            preprocessor_config: Arc::new(preprocessor_config),
            #[cfg(feature = "cuda")]
            cuda_decode_graph: StdMutex::new(CudaDecodeGraphState::default()),
            generation_defaults,
            tracked_modules,
            source_weight_files,
            mapper: pipeline_mapper,
            dynamic_lora,
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for MultimodalPipeline {
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        Some(self.chat_template.clone())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        Some(self.preprocessor_config.clone())
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        self.processor.clone()
    }
}

impl IsqPipelineMixin for MultimodalPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if self.tracked_modules.is_empty() {
            anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
        }
        tracing::info!(
            "Re-quantizing {} layers to {dtype}.",
            self.tracked_modules.len()
        );
        super::isq_flow::requantize_and_swap(
            &self.tracked_modules,
            dtype,
            |module| module.default_type(dtype),
            &|_| None,
        )
    }

    fn begin_calibration(&mut self) -> Result<()> {
        super::isq_flow::begin_calibration(&self.tracked_modules).map(|_| ())
    }

    fn calibration_status(&self) -> Result<super::isq_flow::CalibrationStatus> {
        Ok(super::isq_flow::calibration_status(&self.tracked_modules))
    }

    fn apply_calibration(
        &mut self,
        save_cimatrix: Option<std::path::PathBuf>,
    ) -> Result<super::isq_flow::CalibrationStatus> {
        super::isq_flow::apply_calibration(
            &self.tracked_modules,
            &self.source_weight_files,
            save_cimatrix.as_deref(),
        )
    }
}

impl CacheManagerMixin for MultimodalPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
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
    ) {
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
        }
        // Always clear model-specific state (e.g. Voxtral audio_embeds_cache)
        // for new prompts. set_none_cache is "Only called for prompt seqs",
        // so this is always appropriate. Default impl is a no-op.
        self.model.reset_model_specific_state();

        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl MetadataMixin for MultimodalPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        self.model.reset_model_specific_state();
    }
    fn cleanup_cuda_graphs(&self) {
        #[cfg(feature = "cuda")]
        {
            self.cuda_decode_graph
                .lock()
                .expect("CUDA graph mutex poisoned")
                .clear();
        }
    }
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn generation_defaults(&self) -> Option<crate::ModelGenerationDefaults> {
        self.generation_defaults.clone()
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

impl crate::speculative::driver::SpeculativePipelineExt for MultimodalPipeline {
    fn has_speculative_proposer(&self) -> bool {
        self.model.has_speculative_proposer()
    }

    fn speculative_proposal_len(&self) -> Option<usize> {
        self.model.speculative_proposal_len()
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

    fn build_speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>> {
        let model_specific_args = self.model.default_model_specific_args(&input_meta.input);
        let adapter_leases = vec![None; input_meta.input.dim(0)?].into();
        Ok(Box::new(ModelInputs {
            input_ids: input_meta.input,
            seqlen_offsets: input_meta.positions,
            context_lens: input_meta.context_lens,
            position_ids: input_meta.position_ids,
            pixel_values: None,
            model_specific_args,
            paged_attn_meta: input_meta.paged_attn_meta,
            flash_meta: input_meta.flash_meta,
            recurrent_batch_kind: RecurrentBatchKind::Decode,
            adapter_leases,
        }))
    }
}

fn lora_batch_shape<T>(
    input_ids: &Tensor,
    adapter_leases: &[Option<T>],
) -> candle_core::Result<Option<(usize, usize)>> {
    let (batch, sequence_length) = input_ids.dims2()?;
    if adapter_leases.len() != batch {
        candle_core::bail!(
            "adapter lease count {} does not match model batch size {batch}",
            adapter_leases.len()
        );
    }
    if adapter_leases.iter().all(Option::is_none) {
        return Ok(None);
    }
    Ok(Some((batch, sequence_length)))
}

impl MultimodalPipeline {
    fn lora_execution(
        &self,
        input_ids: &Tensor,
        adapter_leases: &[Option<crate::AdapterLease>],
    ) -> candle_core::Result<Option<Arc<mistralrs_quant::LoraExecution>>> {
        let Some((_, sequence_length)) = lora_batch_shape(input_ids, adapter_leases)? else {
            return Ok(None);
        };
        let runtime = self.dynamic_lora.as_ref().ok_or_else(|| {
            candle_core::Error::msg(
                "request selected an adapter on a pipeline without dynamic LoRA",
            )
        })?;
        runtime.execution(adapter_leases, sequence_length).map(Some)
    }

    fn recurrent_batch_kind(
        &self,
        input_ids: &Tensor,
        paged_attn_meta: Option<&PagedAttentionInputMetadata>,
        recurrent_batch_kind: RecurrentBatchKind,
    ) -> candle_core::Result<RecurrentBatchKind> {
        let seq_len = input_ids.dim(1)?;
        if let Some(metadata) = paged_attn_meta {
            return Ok(
                if !metadata.is_first_prompt_chunk
                    && metadata.num_cached_tokens.is_none()
                    && seq_len == 1
                {
                    RecurrentBatchKind::Decode
                } else {
                    RecurrentBatchKind::Prefill
                },
            );
        }
        Ok(recurrent_batch_kind)
    }

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

#[cfg(feature = "cuda")]
impl MultimodalPipeline {
    fn hybrid_state_indices_host(&self) -> Option<Vec<u32>> {
        if !self.model.cache().is_hybrid() {
            return None;
        }
        self.model
            .cache()
            .hybrid()
            .state_indices_host()
            .map(ToOwned::to_owned)
    }

    fn hybrid_state_indices_tensor(&self) -> Option<Tensor> {
        if !self.model.cache().is_hybrid() {
            return None;
        }
        self.model.cache().hybrid().state_indices().cloned()
    }

    fn snapshot_hybrid_recurrent_state(
        &self,
    ) -> candle_core::Result<Option<SeqRecurrentStateSnapshots>> {
        if !self.model.cache().is_hybrid() {
            return Ok(None);
        }
        let hybrid_cache = self.model.cache().hybrid();
        let Some(indices) = hybrid_cache.state_indices_host().map(ToOwned::to_owned) else {
            return Ok(None);
        };
        let mut snapshots = Vec::with_capacity(indices.len());
        for idx in indices {
            let idx = idx as usize;
            snapshots.push((idx, hybrid_cache.snapshot_recurrent_state(idx)?));
        }
        Ok(Some(snapshots))
    }

    fn restore_hybrid_recurrent_state(
        &self,
        snapshots: Option<&[(usize, Vec<RecurrentStateSnapshot>)]>,
    ) -> candle_core::Result<()> {
        let Some(snapshots) = snapshots else {
            return Ok(());
        };
        let mut hybrid_cache = self.model.cache().hybrid();
        for (idx, snapshot) in snapshots {
            hybrid_cache.restore_recurrent_state(*idx, snapshot)?;
        }
        Ok(())
    }

    fn try_cuda_decode_graph_forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: &[(usize, usize)],
        position_ids: &[usize],
        paged_attn_meta: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_meta: &FlashParams,
    ) -> candle_core::Result<Option<Tensor>> {
        if !cuda_decode_graphs_enabled() || !self.model.supports_cuda_decode_graphs() {
            return Ok(None);
        }
        if self.model.has_speculative_proposer() {
            return Ok(None);
        }
        let Some((kv_cache, metadata)) = paged_attn_meta else {
            return Ok(None);
        };
        if metadata.is_first_prompt_chunk || metadata.num_cached_tokens.is_some() {
            return Ok(None);
        }
        let (batch, q_len) = input_ids.dims2()?;
        if q_len != 1
            || seqlen_offsets.len() != batch
            || context_lens.len() != batch
            || position_ids.len() != batch
            || !input_ids.device().is_cuda()
        {
            return Ok(None);
        }
        let Some(cache_config) = self.metadata.cache_config.as_ref() else {
            return Ok(None);
        };
        let hybrid_state_indices = self.hybrid_state_indices_host();
        let key = CudaDecodeGraphKey::new(input_ids, metadata, cache_config.block_size)?
            .with_state_key(hybrid_state_indices.clone());

        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if state.disabled() {
            return Ok(None);
        }
        if let Some(logits) = state.replay(&key, input_ids, metadata, seqlen_offsets)? {
            return Ok(Some(logits));
        }

        let Device::Cuda(cuda_device) = input_ids.device() else {
            return Ok(None);
        };
        prepare_cuda_graph_memory_pool(&cuda_device.cuda_stream())?;
        let _htod_cache_guard = cuda_device.enable_cuda_graph_htod_cache();

        let recurrent_snapshots = self.snapshot_hybrid_recurrent_state()?;
        let mut ctx = ModelForwardContext::new(
            seqlen_offsets,
            context_lens,
            position_ids,
            Some((kv_cache.as_slice(), metadata)),
            flash_meta,
        )
        .with_recurrent_batch_kind(RecurrentBatchKind::Decode)
        .with_recurrent_metadata(self.recurrent_metadata(RecurrentBatchKind::Decode));
        let warmup_logits = self.model.forward(
            input_ids,
            None,
            self.model.default_model_specific_args(input_ids),
            &mut ctx,
        )?;
        input_ids.device().synchronize()?;
        let warmup_recurrent_snapshots = self.snapshot_hybrid_recurrent_state()?;
        self.restore_hybrid_recurrent_state(recurrent_snapshots.as_deref())?;
        input_ids.device().synchronize()?;

        let retained_tensors = self.hybrid_state_indices_tensor().into_iter().collect();
        let capture_result = capture_cuda_decode_graph(
            CudaDecodeGraphCaptureCtx {
                key,
                input_ids,
                seqlen_offsets,
                block_size: cache_config.block_size,
                kv_cache: kv_cache.as_slice(),
                metadata,
                model_metadata: self.metadata.model_metadata.as_deref(),
                warmup_logits: &warmup_logits,
                retained_tensors,
            },
            |graph_input_ids, graph_metadata| {
                let mut ctx = ModelForwardContext::new(
                    seqlen_offsets,
                    context_lens,
                    position_ids,
                    Some((kv_cache.as_slice(), graph_metadata)),
                    flash_meta,
                )
                .with_recurrent_batch_kind(RecurrentBatchKind::Decode)
                .with_recurrent_metadata(self.recurrent_metadata(RecurrentBatchKind::Decode));
                self.model.forward(
                    graph_input_ids,
                    None,
                    self.model.default_model_specific_args(graph_input_ids),
                    &mut ctx,
                )
            },
        );
        match capture_result {
            Ok(entry) => {
                self.restore_hybrid_recurrent_state(warmup_recurrent_snapshots.as_deref())?;
                state.insert(entry);
            }
            Err(err) => {
                self.restore_hybrid_recurrent_state(recurrent_snapshots.as_deref())?;
                return Err(err);
            }
        }
        Ok(Some(warmup_logits))
    }

    fn disable_cuda_decode_graph(&self, err: &candle_core::Error) {
        let mut state = self
            .cuda_decode_graph
            .lock()
            .expect("CUDA graph mutex poisoned");
        if !state.disabled() {
            warn!("CUDA decode graphs disabled after capture/replay error: {err}");
        }
        state.disable();
    }
}

#[async_trait::async_trait]
impl Pipeline for MultimodalPipeline {
    fn adapter_runtime(&self) -> Option<Arc<DynamicLoraRuntime>> {
        self.dynamic_lora.clone()
    }

    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            adapter_leases,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
        let lora_execution = self.lora_execution(&input_ids, &adapter_leases)?;
        let metadata = self.get_metadata();
        let paged_attn_meta = match (&metadata.cache_engine, &paged_attn_meta) {
            (Some(engine), Some(meta)) => Some((engine.get_kv_cache().clone(), meta)),
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
        let recurrent_batch_kind = self.recurrent_batch_kind(
            &input_ids,
            paged_attn_meta.as_ref().map(|(_, meta)| *meta),
            recurrent_batch_kind,
        )?;
        #[cfg(feature = "cuda")]
        if lora_execution.is_none() && !return_raw_logits && pixel_values.is_none() {
            match self.try_cuda_decode_graph_forward(
                &input_ids,
                &seqlen_offsets,
                &context_lens,
                &position_ids,
                paged_attn_meta.as_ref().map(|(a, b)| (a.clone(), *b)),
                &flash_meta,
            ) {
                Ok(Some(logits)) => return Ok(ForwardInputsResult::CausalGeneration { logits }),
                Ok(None) => {}
                Err(err) => self.disable_cuda_decode_graph(&err),
            }
        }
        let mut ctx = ModelForwardContext::new(
            &seqlen_offsets,
            &context_lens,
            &position_ids,
            paged_attn_meta
                .as_ref()
                .map(|(kv_cache, meta)| (kv_cache.as_slice(), *meta)),
            &flash_meta,
        )
        .with_recurrent_batch_kind(recurrent_batch_kind)
        .with_recurrent_metadata(self.recurrent_metadata(recurrent_batch_kind));
        let logits = mistralrs_quant::with_lora_execution(lora_execution, || {
            self.model
                .forward(&input_ids, pixel_values, model_specific_args, &mut ctx)
        })?;
        if self.model.is_block_diffusion() && !return_raw_logits {
            return Ok(ForwardInputsResult::BlockGeneration {
                token_blocks: logits.to_dtype(candle_core::DType::U32)?.to_vec2::<u32>()?,
                denoise_time: self.model.take_block_denoise_time().unwrap_or_default(),
            });
        }
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
    }

    fn attach_speculative(
        &mut self,
        config: crate::speculative::SpeculativeConfig,
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
        if let Some(info) = self.model.attach_speculative(config)? {
            self.model.log_speculative_attach(&info);
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    async fn try_sample_speculative_causal_gen(
        &mut self,
        seqs: &mut [&mut Sequence],
        logits: &[Tensor],
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        metadata: Option<crate::pipeline::text_models_inputs_processor::PagedAttentionMeta>,
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
                prefix_cacher,
                disable_eos_stop,
                rng,
                &cache,
            )
            .await;
        }

        crate::speculative::driver::clear_staged_speculative_tokens(seqs);
        Ok(false)
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

    async fn sample_block_gen(
        &self,
        input_seqs: &mut [&mut Sequence],
        token_blocks: Vec<Vec<u32>>,
        denoise_times: Vec<std::time::Duration>,
        prefix_cacher: &mut PrefixCacheManagerV2,
        disable_eos_stop: bool,
    ) -> Result<(), candle_core::Error> {
        crate::pipeline::sampling::finalize_block_gen(
            self,
            input_seqs,
            token_blocks,
            denoise_times,
            prefix_cacher,
            disable_eos_stop,
        )
        .await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Multimodal {
            prefixer: self.prefixer.clone(),
        }
    }

    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
        std::sync::Arc<std::sync::atomic::AtomicUsize>,
    )> {
        self.model.encoder_cache_counters()
    }
}

impl AnyMoePipelineMixin for MultimodalPipeline {
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

        self.model
            .create_anymoe_layers(vbs, config, (prefix, mlp), layers, expert_type, gate_vb)
    }
    fn amoe_supported(&self) -> bool {
        self.model.amoe_supported()
    }
}

#[cfg(test)]
mod tests {
    use super::lora_batch_shape;
    use candle_core::{Device, Tensor};

    #[test]
    fn lora_batch_shape_accepts_base_and_mixed_batches() -> candle_core::Result<()> {
        let input_ids = Tensor::zeros((2, 3), candle_core::DType::U32, &Device::Cpu)?;
        assert_eq!(lora_batch_shape(&input_ids, &[None::<()>, None])?, None);
        assert_eq!(
            lora_batch_shape(&input_ids, &[Some(()), None])?,
            Some((2, 3))
        );
        Ok(())
    }

    #[test]
    fn lora_batch_shape_rejects_missing_base_lease() -> candle_core::Result<()> {
        let input_ids = Tensor::zeros((2, 3), candle_core::DType::U32, &Device::Cpu)?;
        let error = lora_batch_shape(&input_ids, &[None::<()>]).unwrap_err();
        assert!(error
            .to_string()
            .contains("adapter lease count 1 does not match model batch size 2"));
        Ok(())
    }
}
