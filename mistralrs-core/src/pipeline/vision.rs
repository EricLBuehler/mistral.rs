use super::cache_manager::{FullCacheManager, NormalCacheManager};
use super::isq::ImatrixDataSource;
use super::isq::UqffFullSer;
use super::{
    get_model_paths, get_xlora_paths, AdapterActivationMixin, AnyMoePipelineMixin, CacheManager,
    CacheManagerMixin, EitherCache, ForwardInputsResult, GeneralMetadata, IsqPipelineMixin, Loader,
    MetadataMixin, MiniCpmOLoader, ModelCategory, ModelKind, ModelPaths, PreProcessingMixin,
    Processor, Qwen2VLLoader, TokenSource, VLlamaLoader, VisionModel, VisionModelLoader,
    VisionPromptPrefixer, XLoraPaths,
};
use super::{
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Phi3VLoader, VisionLoaderType,
};
use crate::device_map::{self, DeviceMapper};
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::isq::UQFF_RESIDUAL_SAFETENSORS;
use crate::pipeline::llg::build_tok_env;
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::text_models_inputs_processor::make_prompt_chunk;
use crate::pipeline::{get_chat_template, ChatTemplate, IsqOrganization, LocalModelPaths};
use crate::prefix_cacher_v2::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    api_dir_list, api_get_file, get_paths, get_uqff_paths, vision_normal_model_loader,
    AnyMoeExpertType, DeviceMapSetting, Ordering, PagedAttentionConfig, Pipeline, Topology,
    TryIntoDType,
};
use anyhow::{Context, Result};
use candle_core::{Device, Tensor, Var};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::{Arc, RwLock};
use std::time::Instant;
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
    prefixer: Arc<dyn VisionPromptPrefixer>,
    mapper: Box<dyn DeviceMapper + Send + Sync>,

    // For full UQFF serialization
    template_filename: Option<PathBuf>,
    generation_config: Option<PathBuf>,
    config: String,
    processor_filename: Option<PathBuf>,
    preprocessor_filename: Option<PathBuf>,
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
    from_uqff: RwLock<Option<PathBuf>>,
}

#[derive(Default)]
/// A builder for a loader for a vision (non-quantized) model.
pub struct VisionLoaderBuilder {
    model_id: Option<String>,
    config: VisionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
}

#[derive(Clone, Default)]
/// Config specific to loading a vision model.
pub struct VisionSpecificConfig {
    pub use_flash_attn: bool,
    pub prompt_batchsize: Option<NonZeroUsize>,
    pub topology: Option<Topology>,
    pub write_uqff: Option<PathBuf>,
    pub from_uqff: Option<PathBuf>,
    pub max_edge: Option<u32>,
    pub calibration_file: Option<PathBuf>,
}

impl VisionLoaderBuilder {
    pub fn new(
        config: VisionSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
    ) -> Self {
        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind: ModelKind::Normal,
        }
    }

    pub fn build(self, loader: VisionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn VisionModelLoader> = match loader {
            VisionLoaderType::Phi3V => Box::new(Phi3VLoader),
            VisionLoaderType::Idefics2 => Box::new(Idefics2Loader),
            VisionLoaderType::LLaVANext => Box::new(LLaVANextLoader),
            VisionLoaderType::LLaVA => Box::new(LLaVALoader),
            VisionLoaderType::VLlama => Box::new(VLlamaLoader),
            VisionLoaderType::Qwen2VL => Box::new(Qwen2VLLoader),
            VisionLoaderType::Idefics3 => Box::new(Idefics3Loader),
            VisionLoaderType::MiniCpmO => Box::new(MiniCpmOLoader),
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
            token_source: RwLock::new(None),
            revision: RwLock::new(None),
            from_uqff: RwLock::new(None),
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
        let config = std::fs::read_to_string(paths.get_config_filename())?;

        if !self.inner.supports_paged_attention() {
            paged_attn_config = None;
        }

        // If auto, convert to Map
        if let DeviceMapSetting::Auto(params) = mapper.clone() {
            let devices = device_map::get_all_similar_devices(device)?;
            // Initial dtype
            let dtype = dtype.try_into_dtype(&devices.iter().collect::<Vec<_>>())?;

            // ISQ or UQFF: quantized path
            // Match logic below where UQFF has priority
            let (layer_sizes_in_bytes, non_mapped_size_in_bytes, total_model_size_in_bytes) =
                if let Some(serialized) = &*self.from_uqff.read().unwrap() {
                    let parent = serialized
                        .parent()
                        .context("Target UQFF path must have a filename!")?;
                    let residual = parent.join(UQFF_RESIDUAL_SAFETENSORS);

                    let ser_total_size = {
                        let ser_artifacts = unsafe {
                            candle_core::safetensors::MmapedSafetensors::new(serialized)?
                        };
                        ser_artifacts
                            .tensors()
                            .iter()
                            .map(|(_, t)| t.data().len())
                            .sum::<usize>()
                    };
                    let res_total_size = {
                        let res_artifacts =
                            unsafe { candle_core::safetensors::MmapedSafetensors::new(residual)? };
                        res_artifacts
                            .tensors()
                            .iter()
                            .map(|(_, t)| t.data().len())
                            .sum::<usize>()
                    };
                    let size_per_layer = ser_total_size / self.inner.num_layers(&config)?;

                    // This is not completely correct but hopefully close enough.
                    // For example, the norms are not necessarily correctly done.
                    (
                        vec![size_per_layer; self.inner.num_layers(&config)?],
                        res_total_size,
                        ser_total_size + res_total_size,
                    )
                } else if let Some(isq) = in_situ_quant {
                    let weight_pack_factor = isq.pack_factor(dtype);
                    let layer_sizes_in_bytes =
                        self.inner
                            .layer_sizes_in_bytes(&config, dtype, weight_pack_factor)?;
                    let non_mapped_size_in_bytes =
                        self.inner
                            .non_mapped_size_in_bytes(&config, dtype, weight_pack_factor)?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                } else {
                    let layer_sizes_in_bytes =
                        self.inner.layer_sizes_in_bytes(&config, dtype, 1)?;
                    let non_mapped_size_in_bytes =
                        self.inner.non_mapped_size_in_bytes(&config, dtype, 1)?;
                    let layer_sizes_sum = layer_sizes_in_bytes.iter().sum::<usize>();
                    (
                        layer_sizes_in_bytes,
                        non_mapped_size_in_bytes,
                        layer_sizes_sum + non_mapped_size_in_bytes,
                    )
                };

            let new = self.inner.get_device_layers(
                &config,
                self.inner.num_layers(&config)?,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                &devices,
                dtype,
                &params,
                paged_attn_config.as_ref(),
            )?;
            mapper = DeviceMapSetting::Map(new);
        }

        let pipeline_mapper = mapper.into_mapper(
            self.inner.get_total_device_mapping_num_layers(&config)?,
            device,
            self.config.topology.as_ref(),
        )?;
        let mapper = mapper.into_mapper(
            self.inner.get_total_device_mapping_num_layers(&config)?,
            device,
            self.config.topology.as_ref(),
        )?;
        let mut layer_devices = Vec::new();
        for layer in 0..self.inner.get_total_device_mapping_num_layers(&config)? {
            let device = mapper.device_for(layer, false).cloned();
            layer_devices.push(device);
        }
        let dtype = mapper.get_min_dtype(dtype)?;

        // TODO: PagedAttention is not supported with CPU for now.
        // This check is not really necessary because `get_device_layers` should prevent it.
        let mapping_uses_cpu = mapper.get_unique_devices().iter().any(Device::is_cpu);
        if mapping_uses_cpu {
            warn!("Device mapping contains a mix of GPU and CPU. There is no CPU support for PagedAttention, disabling PagedAttention.");
            paged_attn_config = None;
        }

        info!(
            "Model config: {:?}",
            self.inner
                .get_config_repr(&config, self.config.use_flash_attn)?
        );

        let mut loading_isq = in_situ_quant.is_some() || self.config.from_uqff.is_some();
        if let Some(ref topology) = self.config.topology {
            loading_isq |= topology
                .0
                .iter()
                .any(|layer| layer.as_ref().is_some_and(|layer| layer.isq.is_some()));
        }

        // Load onto the regular device if not using isq or if the calibration file is specified
        let load_device = if !loading_isq || self.config.calibration_file.is_some() {
            loading_isq = false;
            device.clone()
        } else {
            Device::Cpu
        };

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let mut model = match self.kind {
            ModelKind::Normal => vision_normal_model_loader!(
                paths,
                Some(dtype),
                &load_device,
                layer_devices.clone(),
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                loading_isq,
                self.config.from_uqff.is_some(),
                device.clone(),
                attention_mechanism
            ),
            _ => unreachable!(),
        };
        let preprocessor_config: PreProcessorConfig = serde_json::from_str(
            &fs::read_to_string(
                paths
                    .get_preprocessor_config()
                    .as_ref()
                    .expect("Need preprocessor config"),
            )
            .unwrap(),
        )
        .unwrap();
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

        let gen_conf: Option<GenerationConfig> = paths.get_gen_conf_filename().map(|f| {
            serde_json::from_str(&fs::read_to_string(f).unwrap())
                .expect("bos_token_id/eos_token_id missing in generation_config.json")
        });
        let chat_template = get_chat_template(
            paths,
            &paths
                .get_chat_template_json()
                .as_ref()
                .map(|x| x.to_string_lossy().to_string())
                .clone(),
            &self.chat_template,
            None,
        );

        if let Some(calibration_file) = &self.config.calibration_file {
            let calibration_data = std::fs::read_to_string(calibration_file)?;
            // Tokenize, don't add bos yet
            let tokens = tokenizer
                .encode(calibration_data, false)
                .map_err(anyhow::Error::msg)?
                .get_ids()
                .to_vec();
            info!(
                "Collecting imatrix from calibration file `{}` of {} tokens.",
                calibration_file.display(),
                tokens.len()
            );
            let bos_toks = chat_template.bos_tok().map(|b| vec![b]).unwrap_or_default();
            let bos_tok_id = tokenizer
                .token_to_id(&bos_toks[0])
                .expect("Somehow the bos token is not present.");

            // NOTE: We ONLY calibrate the text bits of these models!!
            // So only those should be tracked!
            model.begin_track_stats()?;

            const CHUNK_SIZE: usize = 1024;
            let n_chunks: usize = tokens.len().div_ceil(CHUNK_SIZE);
            let start = Instant::now();
            for (i, chunk) in tokens.chunks(CHUNK_SIZE).enumerate() {
                let chunk = [vec![bos_tok_id], chunk.to_vec()].concat();
                let chunk_len = chunk.len();

                let start = Instant::now();
                let inputs =
                    make_prompt_chunk(0, vec![chunk], &[0], &load_device, None, false, None, None)?;
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
                            layer.set_len(0);
                        }
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

        if (in_situ_quant.is_some() || self.config.topology.is_some())
            && self.config.from_uqff.is_none()
        {
            let imatrix_source = self
                .config
                .calibration_file
                .as_ref()
                .map(|_| ImatrixDataSource::Collected);
            model.quantize(
                in_situ_quant,
                device.clone(),
                self.config.topology.as_ref(),
                silent,
                imatrix_source,
                IsqOrganization::Default,
                self.config.write_uqff.as_ref(),
                UqffFullSer {
                    tokenizer: &tokenizer,
                    template_filename: paths.get_template_filename(),
                    generation_config: paths.get_gen_conf_filename(),
                    config: config.clone(),
                    processor_filename: paths.get_processor_config(),
                    preprocessor_filename: paths.get_preprocessor_config(),
                },
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
                paged_attn_config.mem_cpu,
                paged_attn_config.block_size,
                dtype,
                model.config(),
                device,
                &layer_devices,
                silent,
            )?;
            let cache_engine =
                CacheEngine::new(model.config(), &cache_config, dtype, device, layer_devices)?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let tok_env = build_tok_env(tokenizer.clone());
        let num_hidden_layers = match model.cache() {
            EitherCache::Full(full) => full.lock().len(),
            EitherCache::Normal(normal) => normal.lock().unwrap().0.len(),
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
                tok_env: Some(tok_env),
                is_xlora: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                no_kv_cache: false,
                no_prefix_cache: true, // TODO: evaluate. Do vision models need to not have prefix caching?
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
                prompt_batchsize: self.config.prompt_batchsize,
                model_metadata: Some(model_metadata),
            }),
            processor,
            prefixer: self.inner.prefixer(),
            preprocessor_config: Arc::new(preprocessor_config),
            topology: self.config.topology.clone(),
            silent,
            template_filename: paths.get_template_filename().clone(),
            generation_config: paths.get_gen_conf_filename().cloned(),
            config,
            processor_filename: paths.get_processor_config().clone(),
            preprocessor_filename: paths.get_preprocessor_config().clone(),
            mapper: pipeline_mapper,
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
                None,
                IsqOrganization::Default,
                None,
                UqffFullSer {
                    tokenizer: &self.tokenizer,
                    template_filename: &self.template_filename,
                    generation_config: self.generation_config.as_ref(),
                    config: self.config.clone(),
                    processor_filename: &self.processor_filename,
                    preprocessor_filename: &self.preprocessor_filename,
                },
            )
            .map_err(anyhow::Error::msg)
    }
}

impl CacheManagerMixin for VisionPipeline {
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        if matches!(self.model.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
        } else {
            NormalCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
        }
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        if matches!(self.model.cache(), EitherCache::Full(_)) {
            FullCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
        } else {
            NormalCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
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
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &EitherCache {
        self.model.cache()
    }
}

impl AdapterActivationMixin for VisionPipeline {
    fn activate_adapters(&mut self, _adapters: Vec<String>) -> Result<usize> {
        anyhow::bail!("Vision models do not support adapter activation.");
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
    fn reset_non_granular_state(&self) {}
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
            mut paged_attn_meta,
            flash_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
        let paged_attn_meta = match (
            self.get_metadata().cache_engine.as_ref(),
            &mut paged_attn_meta,
        ) {
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
        #[cfg(feature = "metal")]
        let logits = objc::rc::autoreleasepool(|| {
            self.model.forward(
                &input_ids,
                pixel_values,
                &seqlen_offsets,
                context_lens,
                position_ids,
                model_specific_args,
                paged_attn_meta,
                &flash_meta,
            )
        })?;
        #[cfg(not(feature = "metal"))]
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
        let has_conv2d = self.model.has_conv2d();
        ModelCategory::Vision {
            has_conv2d,
            prefixer: self.prefixer.clone(),
        }
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

            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(token).map_err(candle_core::Error::msg)?)
                .build()
                .map_err(candle_core::Error::msg)?;
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut filenames = vec![];
            for rfilename in api_dir_list!(api, model_id).filter(|x| x.ends_with(".safetensors")) {
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

            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(token).map_err(candle_core::Error::msg)?)
                .build()
                .map_err(candle_core::Error::msg)?;
            let revision = revision.clone().unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                model_id_str.clone(),
                RepoType::Model,
                revision.clone(),
            ));

            let mut gate_filenames = vec![];
            for rfilename in api_dir_list!(api, model_id).filter(|x| x.ends_with(".safetensors")) {
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
