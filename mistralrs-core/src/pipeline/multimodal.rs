use super::causal::{CausalModel, CausalPipeline, CausalStepInputs};
use super::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use super::loaders::ModelDownload;
use super::{
    get_model_paths, paged_attention_memory_reservations, reserve_recurrent_serving_capacity,
    AdapterKind, AutoMultimodalLoader, EitherCache, ForwardInputsResult, Gemma3Loader,
    GeneralMetadata, Loader, MetadataMixin, MiniCpmOLoader, ModelCategory, ModelKind, ModelPaths,
    MultimodalModel, MultimodalModelLoader, MultimodalPromptPrefixer, Phi4MMLoader,
    PreProcessingMixin, Processor, Qwen2VLLoader, Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, Qwen4ExpLoader, TokenSource, VLlama4Loader, VLlamaLoader,
};
use super::{
    DiffusionGemmaLoader, Gemma3nLoader, Gemma4Loader, Idefics2Loader, Idefics3Loader, LLaVALoader,
    LLaVANextLoader, Lfm2VlLoader, Lfm3AudioLoader, Lfm3D1Loader, Lfm3OmniLoader, Lfm3VlLoader,
    Mistral3Loader, MultimodalLoaderType, MuseGlimmerLoader, Phi3VLoader, Qwen2_5VLLoader,
    VoxtralLoader,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map;
use crate::distributed::{self, WorkerTransferData};

use crate::paged_attention::{allocate_cache, AttentionImplementation, CacheConfigRequest};
use crate::pipeline::chat_template::{
    calculate_eos_tokens, BeginEndUnkPadTok, ChatTemplateValue, GenerationConfig,
};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, CudaDecodeGraphState,
};
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::{AutoDeviceMapQuantization, QuantizationConfigShim};
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{
    get_chat_template, IsqOrganization, LocalModelPaths, ModelForwardContext, RecurrentBatchKind,
};
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    get_paths, get_uqff_paths, lora_model_loader, multimodal_normal_model_loader,
    multimodal_normal_model_loader_sharded, DeviceMapSetting, LoraAdapterSpec, LoraRuntimeConfig,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use either::Either;
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
use std::{env, fs};
use tokenizers::AddedToken;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

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
    prepared_source: Option<PreparedMultimodalSource>,
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Clone)]
pub(crate) struct PreparedMultimodalSource {
    pub config: String,
    pub weights: mistralrs_quant::ShardedVarBuilder,
    pub tokenizer: Tokenizer,
    pub generation_config: Option<GenerationConfig>,
    pub chat_template: Option<String>,
    pub bos_token: Option<String>,
    pub eos_token: Option<String>,
    pub unk_token: Option<String>,
    pub processor_config: Option<String>,
    pub preprocessor_config: Option<String>,
    pub source_weight_files: Vec<PathBuf>,
    pub rope_pairing: crate::gguf::normal_registry::RopePairing,
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
    mtp: bool,
    encoder_cache_memory_bytes: Option<usize>,
}

#[derive(Clone, Default)]
/// Config specific to loading a multimodal model.
pub struct MultimodalSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub max_edge: Option<u32>,
    pub max_model_len: Option<usize>,
    pub hf_config_overrides: Option<super::HfConfigOverrides>,
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
        let hf_cache_path = config.hf_cache_path.clone();
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            jinja_explicit,
            kind: ModelKind::Normal,
            hf_cache_path,
            lora_adapters: None,
            lora_runtime_config: None,
            mtp: false,
            encoder_cache_memory_bytes: None,
        }
    }

    /// Load the MTP head built into the checkpoint so it can drive speculative decoding.
    pub fn with_mtp(mut self, mtp: bool) -> Self {
        self.mtp = mtp;
        self
    }

    pub fn with_encoder_cache_memory_bytes(mut self, max_bytes: Option<usize>) -> Self {
        if let Some(max_bytes) = max_bytes {
            assert!(max_bytes > 0, "encoder cache memory must be nonzero");
        }
        self.encoder_cache_memory_bytes = max_bytes;
        self
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

    fn build_inner(
        self,
        loader: Option<MultimodalLoaderType>,
        prepared_source: Option<PreparedMultimodalSource>,
    ) -> Box<dyn Loader> {
        let loader_type = loader.clone();
        let loader: Box<dyn MultimodalModelLoader> = match loader {
            Some(MultimodalLoaderType::Phi3V) => Box::new(Phi3VLoader),
            Some(MultimodalLoaderType::Idefics2) => Box::new(Idefics2Loader),
            Some(MultimodalLoaderType::LLaVANext) => Box::new(LLaVANextLoader),
            Some(MultimodalLoaderType::LLaVA) => Box::new(LLaVALoader),
            Some(MultimodalLoaderType::Lfm2Vl) => Box::new(Lfm2VlLoader),
            Some(MultimodalLoaderType::Lfm3Vl) => Box::new(Lfm3VlLoader),
            Some(MultimodalLoaderType::Lfm3Audio) => Box::new(Lfm3AudioLoader),
            Some(MultimodalLoaderType::Lfm3Omni) => Box::new(Lfm3OmniLoader),
            Some(MultimodalLoaderType::Lfm3D1) => Box::new(Lfm3D1Loader),
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
            Some(MultimodalLoaderType::Qwen4Exp) => Box::new(Qwen4ExpLoader),
            Some(MultimodalLoaderType::Voxtral) => Box::new(VoxtralLoader),
            Some(MultimodalLoaderType::Gemma4) => Box::new(Gemma4Loader),
            Some(MultimodalLoaderType::MuseGlimmer) => Box::new(MuseGlimmerLoader),
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
            prepared_source,
            mtp: self.mtp,
            encoder_cache_memory_bytes: self.encoder_cache_memory_bytes,
        })
    }

    pub fn build(self, loader: Option<MultimodalLoaderType>) -> Box<dyn Loader> {
        self.build_inner(loader, None)
    }

    pub(crate) fn build_with_source(
        mut self,
        loader: MultimodalLoaderType,
        source: PreparedMultimodalSource,
        kind: ModelKind,
    ) -> Box<dyn Loader> {
        self.kind = kind;
        self.build_inner(Some(loader), Some(source))
    }
}

impl MultimodalLoader {
    fn validate_dynamic_lora(&self) -> Result<()> {
        super::validate_lora_loader_config(
            self.lora_adapters.as_deref(),
            self.lora_runtime_config,
        )?;
        if self.lora_adapters.is_some()
            && !self
                .loader_type
                .as_ref()
                .is_some_and(supports_dynamic_lora_loader)
        {
            anyhow::bail!("dynamic LoRA is not supported for this multimodal architecture");
        }
        Ok(())
    }
}

pub(super) fn supports_dynamic_lora_loader(loader: &MultimodalLoaderType) -> bool {
    matches!(
        loader,
        MultimodalLoaderType::Qwen2VL
            | MultimodalLoaderType::Qwen2_5VL
            | MultimodalLoaderType::Qwen3VL
            | MultimodalLoaderType::Qwen3VLMoE
            | MultimodalLoaderType::Qwen3_5
            | MultimodalLoaderType::Qwen3_5Moe
            | MultimodalLoaderType::Gemma3
            | MultimodalLoaderType::Gemma3n
            | MultimodalLoaderType::Idefics3
            | MultimodalLoaderType::Mistral3
            | MultimodalLoaderType::Llama4
            | MultimodalLoaderType::Lfm2Vl
            | MultimodalLoaderType::Lfm3Vl
            | MultimodalLoaderType::Gemma4
            | MultimodalLoaderType::MuseGlimmer
    )
}

impl Loader for MultimodalLoader {
    fn download_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        silent: bool,
    ) -> Result<ModelDownload> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        self.validate_dynamic_lora()?;
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
        self.validate_dynamic_lora()?;
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
        super::loaders::validate_lora_qk_rope_layout(&config, self.lora_adapters.is_some())?;
        let modalities = self.inner.modalities(&config)?;
        let is_decision = modalities
            .output
            .contains(&crate::SupportedModality::Decision);
        let runtime_config = self
            .inner
            .runtime_config(&config, self.config.max_model_len)?;

        if !self.inner.supports_paged_attention(&config) {
            super::disable_unsupported_paged_attention(&mut paged_attn_config)?;
        }
        let supports_encoder_cache = self.inner.supports_encoder_cache(&config);
        if self.encoder_cache_memory_bytes.is_some() && !supports_encoder_cache {
            mistralrs_quant::log::once_log_warn(
                "Configured encoder cache capacity ignored because this model has no encoder cache",
            );
        }
        if let (Some(bytes), Some(cache_config)) = (
            self.encoder_cache_memory_bytes
                .filter(|_| supports_encoder_cache),
            paged_attn_config.as_mut(),
        ) {
            *cache_config = cache_config
                .clone()
                .with_base_device_memory_reservation(bytes)?;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        // Tokenizer deserialization can briefly use far more memory than its final representation.
        let (processor, preprocessor_config, tokenizer, llg_factory) = {
            let processor_config_json = match self.prepared_source.as_ref() {
                Some(source) => source.processor_config.clone(),
                None => paths
                    .get_processor_config()
                    .as_ref()
                    .map(|f| fs::read_to_string(f).unwrap()),
            };

            // Some models only ship nested preprocessor settings in processor_config.json.
            let mut preprocessor_config: PreProcessorConfig = match self.prepared_source.as_ref() {
                Some(source) => source
                    .preprocessor_config
                    .as_deref()
                    .map(serde_json::from_str)
                    .transpose()?
                    .unwrap_or_else(|| {
                        processor_config_json.as_deref().map_or_else(
                            PreProcessorConfig::default,
                            |json| {
                                PreProcessorConfig::from_processor_config_json(json)
                                    .unwrap_or_default()
                            },
                        )
                    }),
                None => match paths.get_preprocessor_config().as_ref() {
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
                },
            };
            if let Some(video_config) = paths.get_video_preprocessor_config() {
                preprocessor_config.video = Some(Box::new(serde_json::from_str(
                    &fs::read_to_string(video_config).unwrap(),
                )?));
            }
            let processor_config: Option<ProcessorConfig> = processor_config_json
                .as_deref()
                .map(|json| serde_json::from_str(json).unwrap());
            let processor = self.inner.get_processor(
                &runtime_config,
                processor_config,
                preprocessor_config.clone(),
                self.config.max_edge,
            );
            let tokenizer = match self.prepared_source.as_ref() {
                Some(source) => {
                    let mut tokenizer = source.tokenizer.clone();
                    let special_tokens = processor
                        .get_special_tokens()
                        .iter()
                        .map(|token| AddedToken::from((*token).to_string(), true))
                        .collect::<Vec<_>>();
                    tokenizer.add_special_tokens(&special_tokens);
                    tokenizer
                }
                None => get_tokenizer(
                    paths.get_tokenizer_filename(),
                    Some(processor.get_special_tokens()),
                )?,
            };
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
        let has_prepared_weight_source = prepared_weight_source.is_some();
        let weight_source: Option<Arc<dyn mistralrs_quant::QuantizedWeightSource>> = uqff_reader
            .clone()
            .map(|reader| reader as Arc<dyn mistralrs_quant::QuantizedWeightSource>)
            .or(prepared_weight_source.clone());

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

        let activation_params = match &mapper {
            DeviceMapSetting::Auto(params) => self.inner.auto_device_map_params(&config, params)?,
            _ => super::AutoDeviceMapParams::default_multimodal(),
        };

        // If auto, convert to Map if not using nccl
        if write_uqff {
            mapper = DeviceMapSetting::dummy();
        } else if use_distributed {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if matches!(mapper, DeviceMapSetting::Auto(_)) {
            let params = activation_params.clone();

            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;
            let sizing_source = super::loaders::ResourceSource {
                weights: self.prepared_source.as_ref().map(|source| &source.weights),
                paths: paths.get_weight_filenames(),
            };
            let checkpoint = sizing_source.checkpoint_tensors()?;
            let source_quantization = QuantizationConfigShim::get_quantization_config(&config)?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (mut layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
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
                        let exact = if self.config.topology.is_none()
                            && matformer_slicing_config.is_none()
                        {
                            source.resident_layer_bytes(dtype)?
                        } else {
                            None
                        };
                        if let Some(exact) = exact {
                            let layer_sizes = (0..self.inner.num_layers(&config)?)
                                .map(|layer| exact.layers.get(&layer).copied().unwrap_or(0))
                                .collect::<Vec<_>>();
                            let total = layer_sizes.iter().sum::<usize>() + exact.outside;
                            (layer_sizes, exact.outside, total)
                        } else {
                            let quantization =
                                if matches!(sizing, super::isq_flow::AutoDeviceMapSizing::Uqff) {
                                    AutoDeviceMapQuantization::weight_source(source.as_ref())
                                } else {
                                    AutoDeviceMapQuantization::weight_source_with_topology(
                                        source.as_ref(),
                                        self.config.topology.as_ref(),
                                    )
                                };
                            let quantization = quantization
                                .with_checkpoint(&checkpoint)
                                .with_source_quantization(source_quantization.as_ref());
                            let weight_pack_factor = quantization
                                .conservative_pack_factor(dtype, source.pack_factor(dtype)?);
                            let non_mapped_pack_factor =
                                if matches!(sizing, super::isq_flow::AutoDeviceMapSizing::Uqff) {
                                    weight_pack_factor
                                } else {
                                    1
                                };
                            let layer_sizes_in_bytes =
                                self.inner.layer_sizes_in_bytes_with_quantization(
                                    &config,
                                    dtype,
                                    weight_pack_factor,
                                    Some(&quantization),
                                    matformer_slicing_config.as_ref(),
                                )?;
                            let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                                &config,
                                dtype,
                                non_mapped_pack_factor,
                                Some(&quantization),
                                matformer_slicing_config.as_ref(),
                            )?;
                            let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                            (
                                layer_sizes_in_bytes,
                                non_mapped_size_in_bytes,
                                layer_sizes_sum + non_mapped_size_in_bytes,
                            )
                        }
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
                                1,
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
                        let quantization = quantization
                            .with_checkpoint(&checkpoint)
                            .with_source_quantization(source_quantization.as_ref());
                        let layer_sizes_in_bytes =
                            self.inner.layer_sizes_in_bytes_with_quantization(
                                &config,
                                dtype,
                                weight_pack_factor,
                                (!moqe).then_some(&quantization),
                                matformer_slicing_config.as_ref(),
                            )?;
                        let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                            &config,
                            dtype,
                            non_mapped_pack_factor,
                            Some(&quantization),
                            matformer_slicing_config.as_ref(),
                        )?;
                        let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                        (
                            layer_sizes_in_bytes,
                            non_mapped_size_in_bytes,
                            layer_sizes_sum + non_mapped_size_in_bytes,
                        )
                    }
                    super::isq_flow::AutoDeviceMapSizing::Checkpoint => {
                        let inventory = if self.config.topology.is_none()
                            && self.lora_adapters.is_none()
                            && matformer_slicing_config.is_none()
                            && self.inner.supports_checkpoint_weight_inventory(&config)?
                        {
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
                            let checkpoint_quantization =
                                QuantizationConfigShim::get_quantization_config(&config)?;
                            let quantization = AutoDeviceMapQuantization::checkpoint(
                                checkpoint_quantization.as_ref(),
                                self.config.topology.as_ref(),
                            )
                            .map(|quantization| {
                                quantization
                                    .with_checkpoint(&checkpoint)
                                    .with_source_quantization(source_quantization.as_ref())
                            });
                            let weight_pack_factor =
                                quantization
                                    .as_ref()
                                    .map_or(weight_pack_factor, |quantization| {
                                        quantization
                                            .conservative_pack_factor(dtype, weight_pack_factor)
                                    });
                            let layer_sizes_in_bytes =
                                self.inner.layer_sizes_in_bytes_with_quantization(
                                    &config,
                                    dtype,
                                    weight_pack_factor,
                                    quantization.as_ref(),
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
                        }
                    }
                };

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
                &runtime_config,
                paged,
                dtype,
                &device,
                &activation_params,
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
                        self.prepared_source
                            .as_ref()
                            .map(|source| source.rope_pairing),
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
                        let layers = Arc::new(self.inner.new_lora_registry(&runtime_config)?);
                        let sharded_vb = sharded_vb.with_lora_registry(layers.clone());
                        let tracker = sharded_vb.tracker().clone();
                        let model = self.inner.load(
                            &runtime_config,
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
                            &runtime_config,
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
                        multimodal_normal_model_loader!(
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
                        )
                    };
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
                        let layers = Arc::new(self.inner.new_lora_registry(&runtime_config)?);
                        let vb = source
                            .weights
                            .clone()
                            .set_dtype(dtype)
                            .set_device(load_device.clone())
                            .with_lora_registry(layers.clone());
                        let tracker = vb.tracker().clone();
                        let model = self.inner.load(
                            &runtime_config,
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
                        )
                    }
                }
                _ => unreachable!(),
            }
        };

        if let Some(max_bytes) = self
            .encoder_cache_memory_bytes
            .filter(|_| supports_encoder_cache)
        {
            assert!(
                model.configure_encoder_cache_memory_bytes(max_bytes),
                "multimodal loader advertised an encoder cache but the model did not expose one"
            );
        }

        // Release Metal loader scratch buffers before constructing the remaining pipeline state.
        for device in &available_devices {
            if matches!(device, Device::Metal(_)) {
                device.synchronize()?;
            }
        }

        let gen_conf: Option<GenerationConfig> = match self.prepared_source.as_ref() {
            Some(source) => source.generation_config.clone(),
            None => paths
                .get_gen_conf_filename()
                .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap()),
        };
        let gen_conf = gen_conf.or_else(|| GenerationConfig::from_model_config(&config));
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

        plan.validate_tracked_selection(&tracker.get())?;

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
                effective_chat_template: Some(&chat_template),
                generation_config: match self.prepared_source.as_ref() {
                    Some(source) if source.generation_config.is_none() => None,
                    _ => paths.get_gen_conf_filename(),
                },
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
                quantize_predicates: plan.uqff_quantize_predicates.clone(),
                residual,
                full_ser,
                imatrix: imatrix_map.unwrap_or_default(),
            })?;
        }

        if plan.immediate_isq_installed {
            for module in tracker.get().clone() {
                module.ct.resolve()?;
            }
        }
        #[cfg(feature = "cuda")]
        let mut model = model;
        let model_metadata = crate::paged_attention::with_attention_features(
            model.model_config(),
            self.inner
                .activation_model_config(&runtime_config)?
                .attention_features,
        );
        // Layers past the mapped stack (e.g. an MTP head) live on the non-mapped device.
        while layer_devices.len() < model_metadata.num_layers() {
            layer_devices.push(Some(device.clone()));
        }
        #[cfg(feature = "metal")]
        if self.config.write_uqff.is_none() {
            super::metal_warmup::prepare_model(super::metal_warmup::MetalModelWarmup {
                device: &device,
                mapper: pipeline_mapper.as_ref(),
                model: model_metadata.as_ref(),
                paged: paged_attn_config.as_ref(),
                dtype,
                tracked_modules: &tracker.get(),
                packed_prefill: model.supports_packed_prefill(),
            })?;
        }
        // Parallel weight loading leaves stream-ordered frees pending across loader streams;
        // drain the whole context so the KV sizing and allocation see the real free VRAM.
        #[cfg(feature = "cuda")]
        super::synchronize_cuda_contexts(&device, pipeline_mapper.as_ref())?;

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
        let recurrent_pool_grew = if recurrent_transitions_supported
            && super::uses_recurrent_transition_log(model.cache())
        {
            model.reserve_recurrent_speculative_transition_storage()? || recurrent_pool_grew
        } else {
            recurrent_pool_grew
        };
        let recurrent_pool_grew =
            model.reserve_recurrent_decode_deferred_storage()? || recurrent_pool_grew;
        #[cfg(feature = "cuda")]
        if recurrent_pool_grew {
            super::synchronize_cuda_contexts(&device, pipeline_mapper.as_ref())?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = recurrent_pool_grew;

        #[cfg(feature = "cuda")]
        if self.config.write_uqff.is_none() {
            super::prepare_cuda_runtime(&device, pipeline_mapper.as_ref())?;
        }

        if let Some(source) = &weight_source {
            #[cfg(feature = "cuda")]
            super::synchronize_cuda_contexts(&device, pipeline_mapper.as_ref())?;
            source.release_source_cache()?;
        }

        let (cache_config, cache_engine) = if let Some(paged_attn_config) =
            paged_attn_config.as_mut()
        {
            #[cfg(feature = "cuda")]
            let reserve_decode_graphs = device.is_cuda()
                && model.supports_cuda_decode_graphs()
                && cuda_decode_graphs_enabled()
                && cuda_decode_graph_supported_for_model(Some(model_metadata.as_ref()));
            #[cfg(not(feature = "cuda"))]
            let reserve_decode_graphs = false;
            let paged_snapshot = paged_attn_config.clone();
            let capacity_activation_estimate = |usable_tokens| {
                super::loaders::estimate_model_activation_memory(
                    self.inner.as_ref(),
                    &runtime_config,
                    super::loaders::ActivationMemoryContext {
                        params: &activation_params,
                        limits: paged_snapshot.execution_limits,
                        dtype,
                        device: &device,
                        paged: &paged_snapshot,
                        cache_capacity_tokens: Some(usable_tokens),
                    },
                )
            };
            let memory_reservations = if paged_attn_config.recurrent_prefix_capacity_auto {
                paged_attn_config.memory_reservations()?
            } else {
                paged_attention_memory_reservations(model.cache(), paged_attn_config, &device)?
            };
            let mut cache_request = CacheConfigRequest {
                memory_reservations,
                dtype,
                model: model_metadata.as_ref(),
                device: &device,
                layer_devices: &layer_devices,
                capacity_activation_estimate: &capacity_activation_estimate,
                reserve_decode_graphs,
                silent,
            };
            #[cfg(feature = "cuda")]
            {
                model.prepare_optional_weights(&|device| {
                    crate::paged_attention::optional_weight_budget(
                        paged_attn_config,
                        cache_request,
                        device,
                    )
                    .map_err(candle_core::Error::msg)
                })?;
                for (device, bytes) in model.reclaimable_weight_cache_bytes() {
                    paged_attn_config.reserve_reclaimable_weight_cache(&device, bytes)?;
                }
            }
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
            #[cfg(feature = "cuda")]
            model.prepare_optional_weights(&|device| {
                let available = crate::MemoryUsage.query(device)?.available();
                Ok(crate::memory_plan::WeightAllocationBudget {
                    persistent_bytes: available,
                    execution_bytes: available,
                })
            })?;
            (None, None)
        };

        let max_model_len = super::resolve_max_model_len(
            model.max_seq_len(),
            self.config.max_model_len,
            cache_config.as_ref().zip(
                paged_attn_config
                    .as_ref()
                    .map(PagedAttentionConfig::memory_goal),
            ),
            silent,
        )?;
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
        #[cfg(feature = "cuda")]
        let runtime_memory_reservation = super::paged_attention_runtime_memory_reservation(
            model.cache(),
            paged_attn_config.as_ref(),
            &device,
            &pipeline_mapper.get_unique_devices(),
        )?;
        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = match self.prepared_source.as_ref() {
            Some(source) => source.source_weight_files.clone(),
            None if self.config.from_uqff.is_some() => Vec::new(),
            None => paths.get_weight_filenames().to_vec(),
        };

        let prefixer = self.inner.prefixer(&config);
        let video_sampling = self.inner.video_frame_sampling(&config);
        Ok(Arc::new(Mutex::new(MultimodalPipeline {
            model,
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_model_len,
                llg_factory: Some(llg_factory),
                is_xlora: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                no_kv_cache: is_decision,
                no_prefix_cache: !self.inner.supports_prefix_cacher(&config),
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
                modalities,
                loaded_for_uqff_write: self.config.write_uqff.is_some(),
            }),
            #[cfg(feature = "cuda")]
            cuda_decode_graph: StdMutex::new(CudaDecodeGraphState::default()),
            #[cfg(feature = "cuda")]
            cuda_sparse_rejection: StdMutex::new(None),
            speculative_autotuner: StdMutex::new(Default::default()),
            generation_defaults,
            tracked_modules,
            source_weight_files,
            source_weight_source: weight_source,
            mapper: pipeline_mapper,
            dynamic_lora,
            flavor: MultimodalFlavor {
                processor,
                preprocessor_config: Arc::new(preprocessor_config),
                prefixer,
                video_sampling,
                last_prompt_attention: StdMutex::new(None),
            },
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

pub(crate) struct MultimodalFlavor {
    processor: Arc<dyn Processor + Send + Sync>,
    preprocessor_config: Arc<PreProcessorConfig>,
    prefixer: Arc<dyn MultimodalPromptPrefixer>,
    video_sampling: crate::VideoFrameSampling,
    // Attention inputs of the last prompt-chunk forward, so a built-in drafter can prefill with them
    last_prompt_attention: StdMutex<Option<(PagedAttentionInputMetadata, FlashParams)>>,
}

pub(crate) struct MultimodalStepArgs {
    pixel_values: Option<Tensor>,
    model_specific_args: Box<dyn Any>,
}

pub(crate) type MultimodalPipeline = CausalPipeline<dyn MultimodalModel + Send + Sync>;

fn resolve_recurrent_batch_kind(
    input_ids: &Tensor,
    paged_attn_meta: Option<&PagedAttentionInputMetadata>,
    recurrent_batch_kind: RecurrentBatchKind,
) -> candle_core::Result<RecurrentBatchKind> {
    if recurrent_batch_kind != RecurrentBatchKind::Decode {
        return Ok(recurrent_batch_kind);
    }
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

impl CausalModel for dyn MultimodalModel + Send + Sync {
    type Flavor = MultimodalFlavor;
    type StepArgs = MultimodalStepArgs;
    #[cfg(feature = "cuda")]
    const SPECULATIVE_CUDA_GRAPHS: bool = true;

    fn device(&self) -> &Device {
        MultimodalModel::device(self)
    }
    fn cache(&self) -> &EitherCache {
        MultimodalModel::cache(self)
    }
    fn config(&self) -> &crate::paged_attention::ModelConfigMetadata {
        MultimodalModel::config(self)
    }
    fn supports_packed_prefill(&self) -> bool {
        MultimodalModel::supports_packed_prefill(self)
    }

    fn unpack_step_inputs(inputs: Box<dyn Any>) -> CausalStepInputs<MultimodalStepArgs> {
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
        CausalStepInputs {
            input_ids,
            seqlen_offsets,
            context_lens,
            position_ids,
            paged_attn_meta,
            flash_meta,
            recurrent_batch_kind,
            adapter_leases,
            args: MultimodalStepArgs {
                pixel_values,
                model_specific_args,
            },
        }
    }
    fn forward_step_args(
        &self,
        input_ids: &Tensor,
        args: MultimodalStepArgs,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        self.forward(input_ids, args.pixel_values, args.model_specific_args, ctx)
    }
    #[cfg(feature = "cuda")]
    fn forward_text(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
    ) -> candle_core::Result<Tensor> {
        self.forward(
            input_ids,
            None,
            self.default_model_specific_args(input_ids),
            ctx,
        )
    }
    fn speculative_verify_inputs(
        &self,
        input_meta: InputMetadata,
    ) -> candle_core::Result<Box<dyn Any>> {
        let model_specific_args = self.default_model_specific_args(&input_meta.input);
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
            recurrent_batch_kind: RecurrentBatchKind::SpeculativeDecode,
            adapter_leases,
        }))
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs_for(
        &self,
        args: Option<&MultimodalStepArgs>,
    ) -> candle_core::Result<bool> {
        Ok(match args {
            Some(args) => self.supports_cuda_decode_graphs_for_args(&*args.model_specific_args),
            None => {
                let probe = Tensor::zeros((1, 1), DType::U32, MultimodalModel::device(self))?;
                self.supports_cuda_decode_graphs_for_args(
                    &*self.default_model_specific_args(&probe),
                )
            }
        })
    }
    #[cfg(feature = "cuda")]
    fn cuda_graph_step_eligible(args: &MultimodalStepArgs) -> bool {
        args.pixel_values.is_none()
    }

    fn category(pipeline: &MultimodalPipeline) -> ModelCategory {
        let modalities = &pipeline.metadata.modalities;
        let video_sampling = pipeline.flavor.video_sampling;
        if modalities
            .output
            .contains(&crate::SupportedModality::Decision)
        {
            ModelCategory::Decision { video_sampling }
        } else if matches!(
            modalities.input.as_slice(),
            [crate::SupportedModality::Text]
        ) {
            ModelCategory::Text
        } else {
            ModelCategory::Multimodal {
                prefixer: pipeline.flavor.prefixer.clone(),
                video_sampling,
            }
        }
    }
    fn requires_uniform_prompt_batch(pipeline: &MultimodalPipeline) -> bool {
        !Pipeline::supports_packed_prefill(pipeline)
    }
    fn reset_non_granular_state(pipeline: &MultimodalPipeline) {
        pipeline.model.reset_model_specific_state();
    }
    fn input_processor_config(flavor: &MultimodalFlavor) -> Option<Arc<dyn Any>> {
        Some(flavor.preprocessor_config.clone())
    }
    fn processor(flavor: &MultimodalFlavor) -> Arc<dyn Processor> {
        flavor.processor.clone()
    }
    fn kv_cache_bypassed(metadata: &GeneralMetadata) -> bool {
        metadata.no_kv_cache
    }
    fn begin_step(
        pipeline: &MultimodalPipeline,
        step: &CausalStepInputs<MultimodalStepArgs>,
    ) -> candle_core::Result<RecurrentBatchKind> {
        let paged_attn_meta = step.paged_attn_meta.as_ref();
        let recurrent_batch_kind = resolve_recurrent_batch_kind(
            &step.input_ids,
            paged_attn_meta,
            step.recurrent_batch_kind,
        )?;
        if pipeline.model.has_speculative_proposer() {
            *pipeline
                .flavor
                .last_prompt_attention
                .lock()
                .expect("prompt attention mutex poisoned") = paged_attn_meta
                .filter(|meta| meta.is_first_prompt_chunk || meta.num_cached_tokens.is_some())
                .map(|meta| (meta.clone(), step.flash_meta.clone()));
        }
        Ok(recurrent_batch_kind)
    }
    fn forward_decision(
        pipeline: &MultimodalPipeline,
        sequence: &mut Sequence,
    ) -> Result<crate::decision::DecisionResponse> {
        let tokenizer = pipeline
            .tokenizer()
            .context("Decision models require a tokenizer")?;
        let processor = pipeline.get_processor();
        pipeline.model.forward_decision(
            sequence,
            super::DecisionForwardContext {
                tokenizer: &tokenizer,
                processor: processor.as_ref(),
                device: &MetadataMixin::device(pipeline),
                processor_config: pipeline.get_input_processor_config(),
                mapper: pipeline.device_mapper(),
            },
        )
    }
    fn speculative_prompt_chunk(
        pipeline: &mut MultimodalPipeline,
        seqs: &[&mut Sequence],
        chunk: &crate::pipeline::SpeculativePromptChunk,
        metadata: &crate::pipeline::text_models_inputs_processor::PagedAttentionMeta,
    ) -> candle_core::Result<()> {
        if !pipeline.model.has_speculative_proposer() {
            return Ok(());
        }
        let general_metadata = pipeline.get_metadata();
        let Some(cache_engine) = general_metadata.cache_engine.as_ref() else {
            return Ok(());
        };
        let kv_cache = cache_engine.get_kv_cache().clone();
        let seq_ids = chunk
            .rows
            .iter()
            .map(|row| *seqs[row.seq_idx].id())
            .collect::<Vec<_>>();
        let batch_indices = (0..chunk.rows.len()).collect::<Vec<_>>();
        let tokens = chunk
            .rows
            .iter()
            .map(|row| row.tokens.as_slice())
            .collect::<Vec<_>>();
        let chunk_ranges = chunk.rows.iter().map(|row| row.range).collect::<Vec<_>>();
        let target_attention = pipeline
            .flavor
            .last_prompt_attention
            .lock()
            .expect("prompt attention mutex poisoned")
            .take();
        pipeline
            .model
            .speculative_prefill(crate::speculative::SpeculativePrefillCtx {
                seq_ids: &seq_ids,
                batch_indices: &batch_indices,
                tokens: &tokens,
                chunk_ranges: &chunk_ranges,
                is_final_prompt_chunk: chunk.is_final_prompt_chunk,
                cache: crate::speculative::SpeculativeKvCache::Paged {
                    metadata,
                    kv_cache: &kv_cache,
                },
                target_attention: target_attention.as_ref().map(|(metadata, flash_params)| {
                    crate::speculative::TargetAttentionInputs {
                        metadata,
                        flash_params,
                    }
                }),
            })
    }

    fn requires_uniform_completion_batch(&self) -> bool {
        MultimodalModel::requires_uniform_completion_batch(self)
    }
    fn requires_uniform_media_batch(&self) -> bool {
        !self.supports_mixed_media_batches()
    }
    fn supports_reasoning_budget(&self) -> bool {
        !self.is_block_diffusion()
    }
    fn encoder_activation_memory_bytes(
        &self,
        seq: &Sequence,
    ) -> candle_core::Result<Option<usize>> {
        MultimodalModel::encoder_activation_memory_bytes(self, seq)
    }
    fn encoder_cache_counters(
        &self,
    ) -> Option<(
        Arc<std::sync::atomic::AtomicUsize>,
        Arc<std::sync::atomic::AtomicUsize>,
    )> {
        MultimodalModel::encoder_cache_counters(self)
    }
    fn reset_state_for_sequences(&self, sequence_ids: &[usize]) {
        self.reset_model_specific_state_for_sequences(sequence_ids);
    }
    fn block_generation_output(
        &self,
        logits: &Tensor,
    ) -> candle_core::Result<Option<ForwardInputsResult>> {
        if !self.is_block_diffusion() {
            return Ok(None);
        }
        Ok(Some(ForwardInputsResult::BlockGeneration {
            token_blocks: logits.to_dtype(DType::U32)?.to_vec2::<u32>()?,
            denoise_time: self.take_block_denoise_time().unwrap_or_default(),
        }))
    }
    fn has_custom_sampling(&self) -> bool {
        MultimodalModel::has_custom_sampling(self)
    }
    fn prepare_sampling(
        &self,
        seq: &Sequence,
        logits: Tensor,
        rng: &mut Isaac64Rng,
    ) -> candle_core::Result<(Tensor, Option<u32>)> {
        MultimodalModel::prepare_sampling(self, seq, logits, rng)
    }
    fn accept_sample(&self, seq: &Sequence, token: u32) -> candle_core::Result<()> {
        MultimodalModel::accept_sample(self, seq, token)
    }
    fn sampling_eos_tokens(&self, seq: &Sequence, tokens: Option<&[u32]>) -> Option<Vec<u32>> {
        MultimodalModel::sampling_eos_tokens(self, seq, tokens)
    }
    fn finalize_response(&self, seq: &mut Sequence) -> candle_core::Result<()> {
        MultimodalModel::finalize_response(self, seq)
    }
    fn prepare_streaming_response(&self, seq: &mut Sequence) -> candle_core::Result<()> {
        MultimodalModel::prepare_streaming_response(self, seq)
    }
    fn release_generation_sequences(&self, sequence_ids: &[usize]) {
        MultimodalModel::release_generation_sequences(self, sequence_ids);
    }
    fn response_token_is_hidden(&self, token: u32) -> bool {
        MultimodalModel::response_token_is_hidden(self, token)
    }
}
