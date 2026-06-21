use super::isq::{
    write_uqff_artifacts, UqffFullSer, UqffWriteConfig, UqffWriteRequest, WeightLoadingMode,
    WeightLoadingState,
};
use super::{
    get_model_paths, get_xlora_paths, AdapterKind, AnyMoePipelineMixin, CacheManagerMixin,
    EitherCache, ForwardInputsResult, GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin,
    ModelCategory, ModelKind, ModelPaths, PreProcessingMixin, TokenSource,
};
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::{self, DeviceMapper};
use crate::distributed::{self, use_ring, WorkerTransferData};
use crate::embedding_models::inputs_processor::{EmbeddingProcessor, ModelInputs};
use crate::embedding_models::{Dense, DenseActivation, Normalize, Pooling};
use crate::embedding_normal_model_loader;
use crate::embedding_normal_model_loader_sharded;
use crate::get_embedding_paths;
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::loaders::auto_device_map;
use crate::pipeline::loaders::QuantizationConfigShim;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::EmbeddingLoaderType;
use crate::pipeline::EmbeddingModel;
use crate::pipeline::EmbeddingModelLoader;
use crate::pipeline::{AutoEmbeddingLoader, EmbeddingModulePaths};
use crate::pipeline::{ChatTemplate, EmbeddingModelPaths, IsqOrganization, Processor};
use crate::pipeline::{EmbeddingGemmaLoader, Qwen3EmbeddingLoader};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::Modalities;
use crate::SupportedModality;
use crate::{
    get_uqff_paths, DeviceMapSetting, PagedAttentionConfig, Pipeline, Topology, TryIntoDType,
    GLOBAL_HF_CACHE,
};
use anyhow::Context;
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Linear, Module};
use hf_hub::Cache;
use hf_hub::{Repo, RepoType};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::safetensors::MmapedSafetensors;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::env;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info, trace, warn};

pub struct EmbeddingPipeline {
    model: Box<dyn EmbeddingModel + Send + Sync>,
    tracked_modules: Vec<mistralrs_quant::TrackedModule>,
    source_weight_files: Vec<std::path::PathBuf>,
    tokenizer: Arc<Tokenizer>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,
    modules: Vec<Box<dyn Module + Send + Sync>>,
    processor: Arc<dyn Processor + Send + Sync>,
}

/// A loader for an embedding (non-quantized) model.
pub struct EmbeddingLoader {
    inner: Box<dyn EmbeddingModelLoader>,
    model_id: String,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    token_source: RwLock<Option<TokenSource>>,
    revision: RwLock<Option<String>>,
    from_uqff: RwLock<Option<Vec<PathBuf>>>,
    hf_cache_path: Option<PathBuf>,
    lora_adapter_ids: Option<Vec<String>>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Copy, Default)]
pub(crate) enum EmbeddingLoadContext {
    #[default]
    Primary,
    Search,
}

impl EmbeddingLoadContext {
    fn weight_target(self) -> &'static str {
        match self {
            Self::Primary => "model",
            Self::Search => "search embedding model",
        }
    }
}

#[derive(Default)]
/// A builder for a loader for an embedding (non-quantized) model.
pub struct EmbeddingLoaderBuilder {
    model_id: Option<String>,
    config: EmbeddingSpecificConfig,
    kind: ModelKind,
    tokenizer_json: Option<String>,
    hf_cache_path: Option<PathBuf>,
    lora_adapter_ids: Option<Vec<String>>,
    load_context: EmbeddingLoadContext,
}

#[derive(Clone, Default)]
/// Config specific to loading an embedding model.
pub struct EmbeddingSpecificConfig {
    pub topology: Option<Topology>,
    pub write_uqff: Option<UqffWriteConfig>,
    pub from_uqff: Option<Vec<PathBuf>>,
    pub imatrix: Option<PathBuf>,
    pub calibration_file: Option<PathBuf>,
    pub hf_cache_path: Option<PathBuf>,
}

impl EmbeddingLoaderBuilder {
    pub fn new(
        config: EmbeddingSpecificConfig,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        Self {
            config,
            tokenizer_json,
            model_id,
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

    pub(crate) fn with_load_context(mut self, load_context: EmbeddingLoadContext) -> Self {
        self.load_context = load_context;
        self
    }

    pub fn build(self, loader: Option<EmbeddingLoaderType>) -> Box<dyn Loader> {
        let loader: Box<dyn EmbeddingModelLoader> = match loader {
            Some(EmbeddingLoaderType::EmbeddingGemma) => Box::new(EmbeddingGemmaLoader),
            Some(EmbeddingLoaderType::Qwen3Embedding) => Box::new(Qwen3EmbeddingLoader),
            None => Box::new(AutoEmbeddingLoader),
        };
        Box::new(EmbeddingLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            tokenizer_json: self.tokenizer_json,
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
            hf_cache_path: self.hf_cache_path,
            lora_adapter_ids: self.lora_adapter_ids,
            load_context: self.load_context,
        })
    }
}

impl Loader for EmbeddingLoader {
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

        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_embedding_paths!(
            EmbeddingModelPaths,
            &token_source,
            revision.clone(),
            self,
            None,
            None,
            silent,
            self.config.from_uqff.is_some()
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
        let config = std::fs::read_to_string(paths.get_config_filename())?;

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for embedding models, disabling it.");
            paged_attn_config = None;
        }

        debug!("Prompt chunk size is {ATTENTION_CHUNK_SIZE}.");

        let use_nccl = mistralrs_quant::distributed::use_nccl();

        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl || use_ring() {
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
        let device = if use_nccl || use_ring() {
            available_devices[0].clone()
        } else {
            device.clone()
        };
        let uqff_reader = if let Some(from_uqff) = &*self.from_uqff.read().unwrap() {
            Some(Arc::new(mistralrs_quant::UqffReader::open(from_uqff)?))
        } else {
            None
        };

        // If auto, convert to Map if not using nccl
        if use_nccl || use_ring() {
            mapper = DeviceMapSetting::DummyNccl {
                nm_device: available_devices[0].clone(),
            };
        } else if let DeviceMapSetting::Auto(params) = mapper.clone() {
            // Initial dtype
            let dtype = dtype.try_into_dtype(&available_devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(reader) = uqff_reader.as_ref() {
                    let weight_pack_factor = reader.pack_factor(dtype)?;
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
                        None,
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
                        None,
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
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
                        None,
                    )?;
                    let non_mapped_size_in_bytes = self.inner.non_mapped_size_in_bytes(
                        &config,
                        dtype,
                        weight_pack_factor,
                        None,
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
            organization: Default::default(),
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

        let modules_config: Vec<_> = paths
            .get_modules()
            .context("Embedding models require the `modules.json` file.")?
            .to_vec();
        assert!(matches!(
            modules_config.first(),
            Some(EmbeddingModulePaths::Transformer { .. })
        ));

        let mut modules: Vec<Box<dyn Module + Send + Sync>> = Vec::new();
        for module in &modules_config {
            match module {
                EmbeddingModulePaths::Transformer { .. } => (),
                EmbeddingModulePaths::Pooling { config, .. } => {
                    let layer: Pooling = serde_json::from_str(&std::fs::read_to_string(config)?)?;
                    modules.push(Box::new(layer));
                }
                EmbeddingModulePaths::Dense { config, model, .. } => {
                    let config: Dense = serde_json::from_str(&std::fs::read_to_string(config)?)?;
                    let safetensors = unsafe { MmapedSafetensors::new(model)? };
                    let weight = safetensors.load("linear.weight", &device, Some(dtype))?;
                    let bias = if config.bias {
                        Some(safetensors.load("linear.bias", &device, Some(dtype))?)
                    } else {
                        None
                    };
                    let (out_f, in_f) = weight.dims2()?;
                    assert_eq!((out_f, in_f), (config.out_features, config.in_features));
                    if !matches!(config.activation_function, DenseActivation::Identity) {
                        anyhow::bail!("Expected Identity activation function.");
                    }

                    modules.push(Box::new(Linear::new(weight, bias)));
                }
                EmbeddingModulePaths::Normalize { .. } => {
                    modules.push(Box::new(Normalize));
                }
            }
        }
        info!(
            "{}",
            WeightLoadingMode::from(WeightLoadingState {
                from_uqff: self.config.from_uqff.is_some(),
                loading_isq,
                immediate_isq: use_immediate,
                write_uqff: self.config.write_uqff.is_some(),
            })
            .message(self.load_context.weight_target())
        );

        let (model, tracker) = if use_nccl || use_ring() {
            let (mapper, sharded_vb) = distributed::prepare_distributed_mapper(
                dtype,
                &device,
                &available_devices,
                silent,
                &config,
                loading_isq,
                self.config.from_uqff.is_some(),
                self.config.write_uqff.is_some(),
                IsqOrganization::Default,
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
                ModelKind::Normal => embedding_normal_model_loader_sharded!(
                    sharded_vb,
                    config,
                    self.inner,
                    mapper,
                    loading_isq,
                    device.clone(),
                    attention_mechanism,
                    multi_progress.clone(),
                    uqff_reader.clone(),
                ),
                _ => unreachable!(),
            }
        } else {
            match self.kind {
                ModelKind::Normal => embedding_normal_model_loader!(
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
                    multi_progress,
                    uqff_reader.clone(),
                ),
                _ => unreachable!(),
            }
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;

        let imatrix_map = if plan.wants_imatrix {
            let drive = super::isq_flow::EmbeddingCalibrationDrive(&*model);
            Some(super::isq_flow::resolve_imatrix_map(
                &drive,
                &tracker.get().clone(),
                self.config.imatrix.as_ref(),
                self.config.calibration_file.as_ref(),
                &super::isq_flow::CalibrationCtx {
                    tokenizer: &tokenizer,
                    bos_tok_id: None,
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
            let modules_json = EmbeddingModulePaths::serialize_modules(&modules_config);
            let full_ser = UqffFullSer {
                tokenizer: &tokenizer,
                template_filename: paths.get_template_filename(),
                generation_config: paths.get_gen_conf_filename(),
                config: config.clone(),
                processor_filename: &None,
                preprocessor_filename: &None,
                modules: Some(&modules_json),
                module_paths: Some(&modules_config),
            };
            write_uqff_artifacts(UqffWriteRequest {
                output: write_uqff.output.clone(),
                types: uqff_types,
                base_model: write_uqff.base_model.clone(),
                repo_id: write_uqff.repo_id.clone(),
                layers,
                residual: model.residual_tensors(),
                full_ser,
                imatrix: imatrix_map.unwrap_or_default(),
            })?;
        }

        let has_causal_attention = self.inner.has_causal_attention(&config)?;
        let max_seq_len = self.inner.model_config(&config)?.max_seq_len();
        let tracked_modules = tracker.get().clone();
        // rank-sliced layers re-slice at source read; inexpressible slices fall back per layer
        let source_weight_files = if self.config.from_uqff.is_some() {
            Vec::new()
        } else {
            paths.get_weight_filenames().to_vec()
        };

        Ok(Arc::new(Mutex::new(EmbeddingPipeline {
            model,
            tracked_modules,
            source_weight_files,
            tokenizer: tokenizer.into(),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: ModelKind::Normal,
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Embedding],
                },
            }),
            mapper: pipeline_mapper,
            modules,
            processor: Arc::new(EmbeddingProcessor {
                has_causal_attention,
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for EmbeddingPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        self.processor.clone()
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for EmbeddingPipeline {
    fn re_isq_model(&mut self, dtype: IsqType) -> Result<()> {
        if self.tracked_modules.is_empty() {
            anyhow::bail!("Runtime re-ISQ requires the model to have been loaded with ISQ.");
        }
        tracing::info!(
            "Re-quantizing {} layers to {dtype}.",
            self.tracked_modules.len()
        );
        super::isq_flow::requantize_and_swap(&self.tracked_modules, dtype, |_| dtype, &|_| None)
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

impl CacheManagerMixin for EmbeddingPipeline {
    fn clone_in_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn clone_out_cache(&self, _seqs: &mut [&mut Sequence]) {}
    fn set_none_cache(
        &self,
        _seqs: &mut [&mut Sequence],
        _reset_non_granular: bool,
        _modify_draft_cache: bool,
        _load_preallocated_cache: bool,
    ) {
    }
    fn cache(&self) -> &EitherCache {
        unreachable!()
    }
}

impl MetadataMixin for EmbeddingPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {}
    fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        Some(self.tokenizer.clone())
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        Some(&*self.mapper)
    }
}

#[async_trait::async_trait]
impl Pipeline for EmbeddingPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        _return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        let ModelInputs {
            input_ids,
            flash_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");

        let mut xs = self.model.forward(&input_ids, &flash_meta)?;
        for module in &self.modules {
            xs = module.forward(&xs)?;
        }

        Ok(ForwardInputsResult::Embeddings { embeddings: xs })
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
        ModelCategory::Embedding
    }
}

impl AnyMoePipelineMixin for EmbeddingPipeline {}
