use super::causal::{CausalModel, CausalPipeline, CausalStepInputs};
use super::isq::{UqffWriteConfig, WeightLoadingMode, WeightLoadingState};
use super::loaders::ModelDownload;
use super::MultimodalLoaderType;
use super::{
    get_model_paths, paged_attention_memory_reservations, reserve_recurrent_serving_capacity,
    AdapterKind, AutoMultimodalLoader, EitherCache, ForwardInputsResult, GeneralMetadata, Loader,
    MetadataMixin, ModelCategory, ModelKind, ModelPaths, MultimodalModel, MultimodalModelLoader,
    MultimodalPromptPrefixer, PreProcessingMixin, Processor, TokenSource,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::distributed;

use crate::paged_attention::{allocate_cache, AttentionImplementation, CacheConfigRequest};
use crate::pipeline::chat_template::{
    calculate_eos_tokens, BeginEndUnkPadTok, ChatTemplateValue, GenerationConfig,
};
#[cfg(feature = "cuda")]
use crate::pipeline::cuda_graph::{
    cuda_decode_graph_supported_for_model, cuda_decode_graphs_enabled, CudaDecodeGraphState,
};
use crate::pipeline::llg::build_llg_factory;
use crate::pipeline::text_models_inputs_processor::FlashParams;
use crate::pipeline::text_models_inputs_processor::InputMetadata;
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{IsqOrganization, LocalModelPaths, ModelForwardContext, RecurrentBatchKind};
use crate::sequence::Sequence;
use crate::utils::progress::{new_multi_progress, ProgressScopeGuard};
use crate::utils::tokenizer::get_tokenizer;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    get_paths, get_uqff_paths, DeviceMapSetting, LoraAdapterSpec, LoraRuntimeConfig,
    PagedAttentionConfig, Pipeline, Topology, TryIntoDType, GLOBAL_HF_CACHE,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use either::Either;
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Mutex as StdMutex;
use std::sync::{Arc, RwLock};
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
            Some(arch) => arch.loader(),
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
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        self.validate_dynamic_lora()?;
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

        let write_uqff = self.config.write_uqff.is_some();
        let super::loaders::LoadDevices {
            tensor_parallelism,
            device,
            available: available_devices,
        } = super::loaders::resolve_load_devices(
            self.inner.model_config(&config)?.as_ref(),
            device,
            write_uqff,
            super::loaders::SingleCudaDevice::TensorParallel,
            paged_attn_config.as_ref(),
        )?;
        let use_distributed = tensor_parallelism.is_enabled();
        let sources = super::loaders::open_weight_sources(
            self.from_uqff.read().unwrap().as_ref(),
            self.prepared_source.as_ref().map(|source| &source.weights),
        )?;
        let uqff_reader = sources.uqff_reader.clone();

        let matformer_slicing_config = super::loaders::load_matformer_slice(
            self.config.matformer_config_path.as_ref(),
            self.config.matformer_slice_name.as_ref(),
        )?;

        let activation_params = match &mapper {
            DeviceMapSetting::Auto(params) => self.inner.auto_device_map_params(&config, params)?,
            _ => super::AutoDeviceMapParams::default_multimodal(),
        };

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
                matformer: matformer_slicing_config.as_ref(),
                packed_non_mapped: false,
                auto_params: Some(&activation_params),
            },
            paged_attn_config.as_mut(),
        )?;
        let super::loaders::ResolvedDeviceMapping {
            pipeline_mapper,
            mapper,
            mut layer_devices,
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
        let open_weights =
            |mapper: &dyn crate::device_map::DeviceMapper, reuse_open: bool| match open_vb {
                Some(vb) if reuse_open => Ok(vb),
                _ => super::loaders::open_mapped_weights(
                    &*self.inner,
                    super::loaders::MappedWeightsRequest {
                        paths,
                        dtype,
                        load_device: &load_device,
                        layer_devices: layer_devices.clone(),
                        silent,
                        config: &runtime_config,
                        dummy_isq_layers: loading_isq && self.config.from_uqff.is_some(),
                        moe_experts_only: matches!(
                            self.config.organization,
                            IsqOrganization::MoeExpertsOnly
                        ),
                        device_for_tensor: self.inner.get_device_for_tensor(
                            &runtime_config,
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
                    &runtime_config,
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
                adapter: AdapterKind::Lora,
            }
            | ModelKind::GgufAdapter {
                adapter: AdapterKind::Lora,
                ..
            } => {
                let layers = Arc::new(self.inner.new_lora_registry(&runtime_config)?);
                let vb = open_weights(mapper.as_ref(), rope_pairing.is_some())?
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

        let (gen_conf, mut chat_template) =
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
        if model.is_block_diffusion() {
            if let Some(raw) = paths
                .get_gen_conf_filename()
                .and_then(|f| fs::read_to_string(f).ok())
            {
                model.configure_block_diffusion(&raw);
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

        super::isq_flow::finish_isq(super::isq_flow::IsqFinish {
            plan: &plan,
            modules: tracker.get().clone(),
            drive: &super::isq_flow::MultimodalCalibrationDrive(&*model),
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
            processor_filename: paths.get_processor_config(),
            preprocessor_filename: paths.get_preprocessor_config(),
        })?;

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

        if let Some(source) = &sources.active {
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
            source_weight_source: sources.active,
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

    fn as_speculative_target(&self) -> &dyn crate::speculative::SpeculativeTargetMixin {
        self
    }
    fn as_speculative_target_mut(&mut self) -> &mut dyn crate::speculative::SpeculativeTargetMixin {
        self
    }
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
