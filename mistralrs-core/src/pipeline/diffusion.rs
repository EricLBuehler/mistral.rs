use super::loaders::{DiffusionModelPaths, DiffusionModelPathsInner};
use super::{
    AnyMoePipelineMixin, Cache, CacheManagerMixin, DiffusionLoaderType, DiffusionModel,
    DiffusionModelLoader, EitherCache, FluxLoader, ForwardInputsResult, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, Processor, TokenSource,
};
use crate::device_map::{self, DeviceMapper};
use crate::diffusion_models::processor::{DiffusionProcessor, ModelInputs};
use crate::distributed::{self, WorkerTransferData};
use crate::paged_attention::AttentionImplementation;
use crate::pipeline::{ChatTemplate, Modalities, SupportedModality};
use crate::prefix_cacher::PrefixCacheManagerV2;
use crate::sequence::Sequence;
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::utils::{
    progress::{new_multi_progress, ProgressScopeGuard},
    tokens::get_token,
    varbuilder_utils::from_mmaped_safetensors,
};
use crate::{DeviceMapSetting, PagedAttentionConfig, Pipeline, TryIntoDType};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use image::{DynamicImage, RgbImage};
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::sync::Arc;
use std::{env, io};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::warn;

pub struct DiffusionPipeline {
    model: Box<dyn DiffusionModel + Send + Sync>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: EitherCache,
}

/// A loader for a vision (non-quantized) model.
pub struct DiffusionLoader {
    inner: Box<dyn DiffusionModelLoader>,
    model_id: String,
    kind: ModelKind,
}

#[derive(Default)]
/// A builder for a loader for a vision (non-quantized) model.
pub struct DiffusionLoaderBuilder {
    model_id: Option<String>,
    kind: ModelKind,
}

impl DiffusionLoaderBuilder {
    pub fn new(model_id: Option<String>) -> Self {
        Self {
            model_id,
            kind: ModelKind::Normal,
        }
    }

    pub fn build(self, loader: DiffusionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn DiffusionModelLoader> = match loader {
            DiffusionLoaderType::Flux => Box::new(FluxLoader { offload: false }),
            DiffusionLoaderType::FluxOffloaded => Box::new(FluxLoader { offload: true }),
        };
        Box::new(DiffusionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            kind: self.kind,
        })
    }
}

impl Loader for DiffusionLoader {
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
        let paths: anyhow::Result<Box<dyn ModelPaths>> = {
            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(&token_source)?)
                .build()?;
            let revision = revision.unwrap_or("main".to_string());
            let api = api.repo(Repo::with_revision(
                self.model_id.clone(),
                RepoType::Model,
                revision.clone(),
            ));
            let model_id = std::path::Path::new(&self.model_id);
            let filenames = self.inner.get_model_paths(&api, model_id)?;
            let config_filenames = self.inner.get_config_filenames(&api, model_id)?;
            Ok(Box::new(DiffusionModelPaths(DiffusionModelPathsInner {
                config_filenames,
                filenames,
            })))
        };
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
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let _progress_guard = ProgressScopeGuard::new(silent);
        let paths = &paths
            .as_ref()
            .as_any()
            .downcast_ref::<DiffusionModelPaths>()
            .expect("Path downcast failed.")
            .0;

        if matches!(mapper, DeviceMapSetting::Map(_)) {
            anyhow::bail!("Device mapping is not supported for diffusion models.")
        }

        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for Diffusion models.");
        }

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for Diffusion models, disabling it.");

            paged_attn_config = None;
        }

        if crate::using_flash_attn() {
            once_log_info("FlashAttention is enabled.");
        }

        let configs = paths
            .config_filenames
            .iter()
            .map(std::fs::read_to_string)
            .collect::<io::Result<Vec<_>>>()?;

        #[cfg(feature = "cuda")]
        if let Device::Cuda(dev) = &device {
            unsafe { dev.disable_event_tracking() };
        }
        let use_nccl = mistralrs_quant::distributed::use_nccl();
        let available_devices = if let Ok(payload) = env::var(distributed::IS_DAEMON_FLAG) {
            let payload: WorkerTransferData = serde_json::from_str(&payload)?;
            let WorkerTransferData::Init { id: _, worker_rank } = payload;
            vec![candle_core::Device::new_cuda(worker_rank + 1)?]
        } else if use_nccl {
            vec![candle_core::Device::new_cuda(0)?]
        } else {
            device_map::get_all_similar_devices(device)?
        };

        let mapper =
            DeviceMapSetting::dummy().into_mapper(usize::MAX, device, None, &available_devices)?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let model = match self.kind {
            ModelKind::Normal => {
                let vbs = paths
                    .filenames
                    .iter()
                    .zip(self.inner.force_cpu_vb())
                    .map(|(path, force_cpu)| {
                        let dev = if force_cpu { &Device::Cpu } else { device };
                        from_mmaped_safetensors(
                            vec![path.clone()],
                            Vec::new(),
                            Some(dtype),
                            dev,
                            vec![None],
                            silent,
                            None,
                            |_| true,
                            Arc::new(|_| DeviceForLoadTensor::Base),
                        )
                    })
                    .collect::<candle_core::Result<Vec<_>>>()?;

                self.inner.load(
                    configs,
                    vbs,
                    crate::pipeline::NormalLoadingMetadata {
                        mapper,
                        loading_isq: false,
                        real_device: device.clone(),
                        multi_progress: Arc::new(new_multi_progress()),
                        matformer_slicing_config: None,
                    },
                    attention_mechanism,
                    silent,
                )?
            }
            _ => unreachable!(),
        };

        let max_seq_len = model.max_seq_len();
        Ok(Arc::new(Mutex::new(DiffusionPipeline {
            model,
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                llg_factory: None,
                is_xlora: false,
                no_prefix_cache: false,
                num_hidden_layers: 1, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: vec![],
                kind: self.kind.clone(),
                no_kv_cache: true, // NOTE(EricLBuehler): no cache for these.
                activation_dtype: dtype,
                sliding_window: None,
                cache_config: None,
                cache_engine: None,
                model_metadata: None,
                modalities: Modalities {
                    input: vec![SupportedModality::Text],
                    output: vec![SupportedModality::Vision],
                },
            }),
            dummy_cache: EitherCache::Full(Cache::new(0, false)),
        })))
    }

    fn get_id(&self) -> String {
        self.model_id.to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for DiffusionPipeline {
    fn get_processor(&self) -> Arc<dyn Processor> {
        Arc::new(DiffusionProcessor)
    }
    fn get_chat_template(&self) -> Option<Arc<ChatTemplate>> {
        None
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for DiffusionPipeline {
    fn re_isq_model(&mut self, _dtype: IsqType) -> Result<()> {
        anyhow::bail!("Diffusion models do not support ISQ for now.")
    }
}

impl CacheManagerMixin for DiffusionPipeline {
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
        &self.dummy_cache
    }
}

impl MetadataMixin for DiffusionPipeline {
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
        None
    }
    fn device_mapper(&self) -> Option<&dyn DeviceMapper> {
        None
    }
}

#[async_trait::async_trait]
impl Pipeline for DiffusionPipeline {
    fn forward_inputs(
        &mut self,
        inputs: Box<dyn Any>,
        return_raw_logits: bool,
    ) -> candle_core::Result<ForwardInputsResult> {
        assert!(!return_raw_logits);

        let ModelInputs { prompts, params } = *inputs.downcast().expect("Downcast failed.");
        let img = self.model.forward(prompts, params)?.to_dtype(DType::U8)?;
        let (_b, c, h, w) = img.dims4()?;
        let mut images = Vec::new();
        for b_img in img.chunk(img.dim(0)?, 0)? {
            let flattened = b_img.squeeze(0)?.permute((1, 2, 0))?.flatten_all()?;
            if c != 3 {
                candle_core::bail!("Expected 3 channels in image output");
            }
            #[allow(clippy::cast_possible_truncation)]
            images.push(DynamicImage::ImageRgb8(
                RgbImage::from_raw(w as u32, h as u32, flattened.to_vec1::<u8>()?).ok_or(
                    candle_core::Error::Msg("RgbImage has invalid capacity.".to_string()),
                )?,
            ));
        }
        Ok(ForwardInputsResult::Image { images })
    }
    async fn sample_causal_gen(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Vec<Tensor>,
        _prefix_cacher: &mut PrefixCacheManagerV2,
        _disable_eos_stop: bool,
        _srng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        candle_core::bail!("`sample_causal_gen` is incompatible with `DiffusionPipeline`");
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Diffusion
    }
}

impl AnyMoePipelineMixin for DiffusionPipeline {}
