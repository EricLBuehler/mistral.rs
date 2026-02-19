use super::isq::ImatrixDataSource;
use super::isq::UqffFullSer;
use super::{
    get_model_paths, get_xlora_paths, AdapterKind, AnyMoePipelineMixin, AutoVisionLoader,
    CacheManager, CacheManagerMixin, EitherCache, ForwardInputsResult, Gemma3Loader,
    GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin, MiniCpmOLoader, ModelCategory,
    ModelKind, ModelPaths, MultimodalPromptPrefixer, Phi4MMLoader, PreProcessingMixin, Processor,
    Qwen2VLLoader, Qwen3VLLoader, Qwen3VLMoELoader, TokenSource, VLlama4Loader, VLlamaLoader,
    VisionModel, VisionModelLoader,
};
use super::{
    Gemma3nLoader, Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Mistral3Loader,
    Phi3VLoader, Qwen2_5VLLoader, VisionLoaderType, VoxtralLoader,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, WorkerTransferData};
use crate::kv_cache::{FullCacheManager, NormalCacheManager};
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{
    calculate_eos_tokens, BeginEndUnkPadTok, ChatTemplateValue, GenerationConfig,
};
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::QuantizationConfigShim;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::text_models_inputs_processor::make_prompt_chunk;
use crate::pipeline::{get_chat_template, ChatTemplate, IsqOrganization, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    tokens::get_token,
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    api_dir_list, api_get_file, get_paths, get_uqff_paths, vision_normal_model_loader,
    vision_normal_model_loader_sharded, AnyMoeExpertType, DeviceMapSetting, Ordering,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::Result;
use candle_core::{Device, Tensor, Var};
use either::Either;
use hf_hub::Cache;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::{
    AfqLayer, GgufMatMul, HqqLayer, ImmediateIsqOverride, IsqType, QuantizedSerdeType,
};
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::borrow::Cow;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use std::{env, fs};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct VisionPipeline {
    model: Box<dyn VisionModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    processor: Arc<dyn Processor + Send + Sync>,
    preprocessor_config: Arc<PreProcessorConfig>,
    topology: Option<Topology>,
    silent: bool,
    prefixer: Arc<dyn MultimodalPromptPrefixer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    organization: IsqOrganization,

    // For full UQFF serialization
    template_filename: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    config: String,
    processor_filename: Option<PathBuf>,
    preprocessor_filename: Option<PathBuf>,
    imatrix: Option<PathBuf>,
}

/// A loader for a vision (non-quantized) model.
pub struct VisionLoader {
    inner: Box<dyn VisionModelLoader>,
    model_id: String,
    config: VisionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Default)]
/// A builder for a loader for a vision (non-quantized) model.
pub struct VisionLoaderBuilder {
    model_id: Option<String>,
    config: VisionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapter_ids: Option<Vec<String>>,
}

#[derive(Clone, Default)]
/// Config specific to loading a vision model.
pub struct VisionSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<PathBuf>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub max_edge: Option<u32>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
    pub matformer_config_path: Option<PathBuf>,
    pub matformer_slice_name: Option<String>,
    pub organization: IsqOrganization,
}

impl VisionLoaderBuilder {
    pub fn new(
        config: VisionSpecificConfig,
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
            ..Default::default()
        }
    }

    pub fn hf_cache_path(mut self, hf_cache_path: PathBuf) -> Self {
        self.hf_cache_path = Some(hf_cache_path);
        self
    }

    pub fn with_lora(mut self, lora_adapter_ids: Vec<String>) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::Lora,
        };
        self.lora_adapter_ids = Some(lora_adapter_ids);
        self
    }

    pub fn build(self, loader: Option<VisionLoaderType>) -> Box<dyn Loader> {
        let loader: Box<dyn VisionModelLoader> = match loader {
            Some(VisionLoaderType::Phi3V) => Box::new(Phi3VLoader),
            Some(VisionLoaderType::Idefics2) => Box::new(Idefics2Loader),
            Some(VisionLoaderType::LLaVANext) => Box::new(LLaVANextLoader),
            Some(VisionLoaderType::LLaVA) => Box::new(LLaVALoader),
            Some(VisionLoaderType::VLlama) => Box::new(VLlamaLoader),
            Some(VisionLoaderType::Qwen2VL) => Box::new(Qwen2VLLoader),
            Some(VisionLoaderType::Idefics3) => Box::new(Idefics3Loader),
            Some(VisionLoaderType::MiniCpmO) => Box::new(MiniCpmOLoader),
            Some(VisionLoaderType::Phi4MM) => Box::new(Phi4MMLoader),
            Some(VisionLoaderType::Qwen2_5VL) => Box::new(Qwen2_5VLLoader),
            Some(VisionLoaderType::Gemma3) => Box::new(Gemma3Loader),
            Some(VisionLoaderType::Mistral3) => Box::new(Mistral3Loader),
            Some(VisionLoaderType::Llama4) => Box::new(VLlama4Loader),
            Some(VisionLoaderType::Gemma3n) => Box::new(Gemma3nLoader),
            Some(VisionLoaderType::Qwen3VL) => Box::new(Qwen3VLLoader),
            Some(VisionLoaderType::Qwen3VLMoE) => Box::new(Qwen3VLMoELoader),
            Some(VisionLoaderType::Voxtral) => Box::new(VoxtralLoader),
            None => Box::new(AutoVisionLoader),
        };
        Box::new(VisionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            xlora_model_id: None,
            xlora_order: None,
            jinja_explicit: self.jinja_explicit,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            lora_adapter_ids: self.lora_adapter_ids,
        })
    }
}

impl Loader for VisionLoader {
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
            self.config.from_uqff.is_some()
        );
        if let Some(from_uqff) = self.config.from_uqff.clone() {
            *self.from_uqff.write().unwrap() = Some(get_uqff_paths!(&from_uqff, self, silent));
        }
        *self
            .token_source
            .write()
            .expect("Failed to write to token source") = Some(token_source);
        *self.revision.write().expect("Failed to write to revision") = revision;
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
        let config = std::fs::read_to_string(paths.get_config_filename())?;

        if !self.inner.supports_paged_attention(&config) {
            paged_attn_config = None;
        }

        info!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let use_nccl = mistralrs_quant::distributed::use_nccl();

        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            // Use new_cuda instead of new_cuda_with_stream for NCCL compatibility
            // NCCL manages its own streams, so explicit stream creation can cause conflicts
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };
        #[cfg(feature = "cuda")]
        for device in &available_devices {
            if let Device::Cuda(dev) = device {
                unsafe { dev.disable_event_tracking() };
            }
        }
        let device = if use_nccl {
            available_devices[0].clone()
        } else {
            device.clone()
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
        if use_nccl {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(mut params) = mapper.clone() {
            // We can promote to vision params if we get text params
            params = params.maybe_promote_to_vision();
            max_kv_tokens = Some(params.max_seq_len() * params.max_batch_size());

            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(serialized) = &*self.from_uqff.read().unwrap() {
                    let weight_pack_factor = {
                        let ser_artifacts = unsafe {
                            candle_core::safetensors::MmapedSafetensors::multi(serialized)?
                        };
                        let mut total_pack_factors = 0;
                        let total_tensors = ser_artifacts.tensors().len();
                        for (_, artifact) in ser_artifacts.tensors() {
                            let artifact = artifact.data();
                            // NOTE(EricLBuehler): isq type is ALWAYS byte 4 (5th) of the tensor.
                            let isq_type = artifact[mistralrs_quant::UQFF_QUANT_TYPE_OFFSET];
                            let pack_factor = match QuantizedSerdeType::try_from(isq_type as usize)?
                            {
                                QuantizedSerdeType::Hqq => {
                                    HqqLayer::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::Gguf => {
                                    GgufMatMul::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::Fp8 => IsqType::F8E4M3.pack_factor(dtype),
                                QuantizedSerdeType::Unquant => 1,
                                QuantizedSerdeType::Afq => {
                                    AfqLayer::get_isq_type_from_uqff(Cow::Borrowed(artifact))?
                                        .pack_factor(dtype)
                                }
                                QuantizedSerdeType::F8Q8 => IsqType::F8Q8.pack_factor(dtype),
                            };
                            total_pack_factors += pack_factor;
                        }

                        total_pack_factors / total_tensors
                    };

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

        let pipeline_mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mapper = mapper.into_mapper(
            self.inner.num_layers(&config)?,
            &device,
            self.config.topology.as_ref(),
            &available_devices,
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..self.inner.num_layers(&config)? {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }
        let dtype = mapper.get_min_dtype(dtype)?;

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu && paged_attn_config.is_some() {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        info!("Model config: {:?}", self.inner.get_config_repr(&config)?);
        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let topology_overrides = self
            .config
            .topology
            .as_ref()
            .map(|topology| {
                topology
                    .pattern_overrides()
                    .into_iter()
                    .map(|(regex, layer)| ImmediateIsqOverride {
                        predicate: regex,
                        ty: layer.isq,
                        device: layer.device.clone(),
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();
        let has_override_isq = topology_overrides
            .iter()
            .any(|override_entry| override_entry.ty.is_some());
        let topology_requires_post_quant = self
            .config
            .topology
            .as_ref()
            .is_some_and(|topology| topology.requires_post_quantization());

        let allow_immediate_cli = self.config.imatrix.is_none()
            && self.config.calibration_file.is_none()
            && in_situ_quant.is_some();

        let mut immediate_ty = None;
        let mut immediate_predicates = Vec::new();
        if allow_immediate_cli {
            immediate_ty = in_situ_quant;
            immediate_predicates =
                if matches!(self.config.organization, IsqOrganization::MoeExpertsOnly) {
                    self.inner.immediate_isq_predicates_moqe(&config)?
                } else {
                    self.inner.immediate_isq_predicates(&config)?
                };
            info!("Applying ISQ to {in_situ_quant:?}");
            if immediate_predicates.is_empty() {
                warn!("No predicates for this model and ISQ setting detected. ISQ will not be applied to any weights!");
            }
        }

        let use_immediate = allow_immediate_cli || has_override_isq;
        if use_immediate {
            let (pool, num_threads) = mistralrs_quant::create_isq_thread_pool(immediate_ty);
            info!("Applying immediate ISQ in parallel on {num_threads} threads.");
            mistralrs_quant::set_immediate_isq_with_pool(
                immediate_ty,
                immediate_predicates.clone(),
                topology_overrides.clone(),
                pool,
            );
        }

        // Logic for ISQ here: if no calibration (i.e imatrix), then allow immediate ISQ. Otherwise, back to normal.
        let mut loading_isq = if use_immediate {
            false
        } else {
            in_situ_quant.is_some()
        };
        if self.config.imatrix.is_some() || self.config.calibration_file.is_some() {
            loading_isq = true;
        }
        loading_isq |= topology_requires_post_quant;
        loading_isq |= self.config.from_uqff.is_some();

        if self.config.imatrix.is_some() && self.config.calibration_file.is_some() {
            anyhow::bail!(
                "`imatrix` and `calibration_file` were both specified, this is not allowed."
            );
        }

        // Load onto the regular device if not using isq or if the calibration file is specified.
        // For immediate ISQ on discrete GPUs, load to CPU: the mapper will set the correct target
        // device per-layer, and linear constructors will override to CPU for ISQ-targeted weights.
        // On integrated/unified memory systems (e.g. Grace Blackwell), CPU and GPU share memory,
        // so we load directly to the device.
        let load_device = if !loading_isq || self.config.calibration_file.is_some() {
            loading_isq = false;
            if use_immediate && !crate::utils::normal::is_integrated_gpu(&device) {
                Device::Cpu
            } else {
                device.clone()
            }
        } else {
            Device::Cpu
        };

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let multi_progress = Arc::new(new_multi_progress());

        let mut model = if use_nccl {
            let (mapper, sharded_vb) = distributed::prepare_distributed_mapper(
                dtype,
                &device,
                &available_devices,
                silent,
                &config,
                loading_isq,
                self.config.from_uqff.is_some(),
                self.config.organization,
                &*self.inner,
                paths.as_ref(),
            )?;

            // Special case for where things can be more optimially loaded.
            match self.kind {
                ModelKind::Normal => vision_normal_model_loader_sharded!(
                    sharded_vb,
                    config,
                    self.inner,
                    mapper,
                    loading_isq,
                    device.clone(),
                    attention_mechanism,
                    multi_progress.clone(),
                    matformer_slicing_config.clone(),
                ),
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => vision_normal_model_loader!(
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
                    multi_progress,
                    matformer_slicing_config.clone(),
                ),
                _ => unreachable!(),
            }
        };

        // Handle the Gemma 3 1b case here
        let preprocessor_config: PreProcessorConfig = match paths.get_preprocessor_config().as_ref()
        {
            Some(preprocessor_config) => {
                serde_json::from_str(&fs::read_to_string(preprocessor_config).unwrap()).unwrap()
            }
            None => PreProcessorConfig::default(),
        };
        let processor_config: Option<ProcessorConfig> = paths
            .get_processor_config()
            .as_ref()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());

        let processor = self.inner.get_processor(
            &config,
            processor_config,
            preprocessor_config.clone(),
            self.config.max_edge,
        ); //There are always some repos that don't properly handle config position, for example... LLaVA

        let tokenizer = get_tokenizer(
            paths.get_tokenizer_filename(),
            Some(processor.get_special_tokens()),
        )?;

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
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

        if let Some(calibration_file) = &self.config.calibration_file {
            let calibration_data = std::fs::read_to_string(calibration_file)?;
            // Tokenize, don't add bos yet
            let tokens = tokenizer
                .encode_fast(calibration_data, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            info!(
                "Collecting imatrix from calibration file `{}` of {} tokens.",
                calibration_file.display(),
                tokens.len()
            );
            let bos_tok_id = chat_template
                .bos_tok()
                .as_deref()
                .and_then(|tok| tokenizer.token_to_id(tok));

            // NOTE: We ONLY calibrate the text bits of these models!!
            // So only those should be tracked!
            match self.config.organization {
                IsqOrganization::Default => model.begin_track_stats()?,
                IsqOrganization::MoeExpertsOnly => model.begin_track_stats_moe_experts_only()?,
            }

            const CHUNK_SIZE: usize = 1024;
            let n_chunks: usize = tokens.len().div_ceil(CHUNK_SIZE);
            let start = Instant::now();
            for (i, chunk) in tokens.chunks(CHUNK_SIZE).enumerate() {
                let mut chunk = chunk.to_vec();
                if let Some(bos_tok_id) = bos_tok_id {
                    chunk.insert(0, bos_tok_id);
                }
                let chunk_len = chunk.len();

                let start = Instant::now();
                let inputs = make_prompt_chunk(
                    0,
                    vec![&chunk],
                    &[0],
                    &load_device,
                    None,
                    false,
                    None,
                    None,
                    None,
                )?;
                let _ = model.forward(
                    &inputs.input,
                    None, // NOTE: We ONLY calibrate the text bits of these models!!
                    &inputs.positions,
                    inputs.context_lens,
                    inputs.position_ids,
                    model.default_model_specific_args(&inputs.input),
                    None,
                    &inputs.flash_meta,
                )?;
                match model.cache_mut() {
                    EitherCache::Full(full) => {
                        for layer in &mut *full.lock() {
                            *layer = None
                        }
                    }
                    EitherCache::Normal(normal) => {
                        for layer in &mut *normal.lock().unwrap().0 {
                            layer.reset();
                        }
                    }
                    EitherCache::Hybrid(hybrid) => {
                        hybrid.lock().unwrap().reset();
                    }
                }
                let end = Instant::now();
                info!(
                    "Processed chunk {}/{n_chunks} ({chunk_len} tokens), {:.2}s",
                    i + 1,
                    end.duration_since(start).as_secs_f32()
                );
            }
            load_device.synchronize()?;
            let end = Instant::now();
            info!(
                "Finished collecting imatrix in {:.2}s",
                end.duration_since(start).as_secs_f32()
            );
        }

        let should_serialize = self.config.write_uqff.is_some();
        let should_quantize_pass = loading_isq;

        if (should_quantize_pass || should_serialize) && self.config.from_uqff.is_none() {
            let imatrix_source = if should_quantize_pass {
                match (
                    self.config.imatrix.as_ref(),
                    self.config.calibration_file.is_some(),
                ) {
                    (None, false) => None,
                    (Some(file), false) => Some(ImatrixDataSource::File(file)),
                    (None, true) => Some(ImatrixDataSource::Collected),
                    (Some(_), true) => unreachable!(),
                }
            } else {
                None
            };
            if should_quantize_pass {
                info!("Applying ISQ to all ranks.");
            } else {
                info!("Serializing existing ISQ tensors without additional quantization.");
            }
            model.quantize(
                in_situ_quant,
                device.clone(),
                self.config.topology.as_ref(),
                silent,
                imatrix_source,
                self.config.organization,
                should_quantize_pass,
                self.config.write_uqff.as_ref(),
                UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    generation_config: paths.get_gen_conf_filename(),
                    config: config.clone(),
                    processor_filename: paths.get_processor_config(),
                    preprocessor_filename: paths.get_preprocessor_config(),
                    modules: None,
                    module_paths: None,
                },
                Arc::new(new_multi_progress()),
            )?;
        } else if let Some(from_uqff) = &*self.from_uqff.read().unwrap() {
            model.load_from_artifacts(
                device.clone(),
                self.config.topology.as_ref(),
                silent,
                from_uqff,
            )?;
        }

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
            anyhow::ensure!(
                !matches!(self.kind, ModelKind::Adapter { .. }),
                "PagedAttention does not support adapter models."
            );
            let cache_config = calculate_cache_config(
                paged_attn_config.mem_gpu,
                paged_attn_config.block_size,
                dtype,
                paged_attn_config.cache_type,
                model.config(),
                &device,
                &layer_devices,
                silent,
                None,
                max_kv_tokens,
            )?;
            let cache_engine =
                CacheEngine::new(model.config(), &cache_config, dtype, &device, layer_devices)?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let llg_factory = build_llg_factory(tokenizer.clone())?;
        let num_hidden_layers = match model.cache() {
            EitherCache::Full(full) => full.lock().len(),
            EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
            EitherCache::Hybrid(hybrid) => hybrid.lock().unwrap().num_layers(),
        };
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        let sliding_window = model.config().sliding_window;
        let model_metadata = Arc::new(model.config().clone());
        Ok(Arc::new(Mutex::new(VisionPipeline {
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
            }),
            processor,
            prefixer: self.inner.prefixer(&config),
            preprocessor_config: Arc::new(preprocessor_config),
            topology: self.config.topology.clone(),
            silent,
            organization: self.config.organization,
            template_filename: paths.get_template_filename().clone(),
            generation_config: paths.get_gen_conf_filename().cloned(),
            config,
            processor_filename: paths.get_processor_config().clone(),
            preprocessor_filename: paths.get_preprocessor_config().clone(),
            mapper: pipeline_mapper,
            imatrix: self.config.imatrix.clone(),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for VisionPipeline {
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

impl IsqPipelineMixin for VisionPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        let device = self.device().clone();
        self.model
            .quantize(
                Some(dtype),
                device,
                self.topology.as_ref(),
                self.silent,
                self.imatrix.as_ref().map(ImatrixDataSource::File),
                self.organization,
                true,
                None,
                UqffFullSer {
                    tokenizer: &self.tokenizer,
                    template_filename: &self.template_filename,
                    generation_config: self.generation_config.as_ref(),
                    config: self.config.clone(),
                    processor_filename: &self.processor_filename,
                    preprocessor_filename: &self.preprocessor_filename,
                    modules: None,
                    module_paths: None,
                },
                Arc::new(new_multi_progress()),
            )
            .map_err(anyhow::Error::msg)
    }
}

impl CacheManagerMixin for VisionPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.model.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, false)
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence]) {
        if matches!(self.model.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, false)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, false)
        }
    }
    fn set_none_cache(
        &self,
        seqs: &mut [&mut Sequence],
        reset_non_granular: bool,
        modify_draft_cache: bool,

        load_preallocated_cache: bool,
    ) {
        if matches!(self.model.cache(), EitherCache::Full(_)) {
            FullCacheManager.set_none_cache(self, seqs, modify_draft_cache, false);
        } else {
            NormalCacheManager.set_none_cache(
                self,
                seqs,
                modify_draft_cache,
                load_preallocated_cache,
            );
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

impl MetadataMixin for VisionPipeline {
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
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for VisionPipeline {
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
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
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
        let logits = self.model.forward(
            &input_ids,
            pixel_values,
            &seqlen_offsets,
            context_lens,
            position_ids,
            model_specific_args,
            paged_attn_meta,
            &flash_meta,
        )?;
        if return_raw_logits {
            Ok(ForwardInputsResult::RawLogits { logits })
        } else {
            Ok(ForwardInputsResult::CausalGeneration { logits })
        }
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
        ModelCategory::Vision {
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

impl AnyMoePipelineMixin for VisionPipeline {
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

            let api = {
                let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
                let mut api = ApiBuilder::from_cache(cache)
                    .with_progress(!silent)
                    .with_token(get_token(token).map_err(candle_core::Error::msg)?);
                if let Some(cache_dir) = crate::hf_hub_cache_dir() {
                    api = api.with_cache_dir(cache_dir);
                }
                api.build().map_err(candle_core::Error::msg)?
            };
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut filenames = vec![];
            for rfilename in
                api_dir_list!(api, model_id, true).filter(|x| x.ends_with(".safetensors"))
            {
                filenames.push(api_get_file!(api, &rfilename, model_id));
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

            let api = {
                let cache = GLOBAL_HF_CACHE.get().cloned().unwrap_or_default();
                let mut api = ApiBuilder::from_cache(cache)
                    .with_progress(!silent)
                    .with_token(get_token(token).map_err(candle_core::Error::msg)?);
                if let Some(cache_dir) = crate::hf_hub_cache_dir() {
                    api = api.with_cache_dir(cache_dir);
                }
                api.build().map_err(candle_core::Error::msg)?
            };
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut gate_filenames = vec![];
            for rfilename in
                api_dir_list!(api, model_id, true).filter(|x| x.ends_with(".safetensors"))
            {
                gate_filenames.push(api_get_file!(api, &rfilename, model_id));
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
