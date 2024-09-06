use super::cache_manager::DefaultCacheManager;
use super::{
    get_model_paths, get_xlora_paths, AdapterActivationMixin, AnyMoePipelineMixin, Cache,
    CacheManager, CacheManagerMixin, DiffusionLoaderType, DiffusionModel, DiffusionModelLoader,
    FluxLoader, ForwardInputsResult, GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin,
    ModelCategory, ModelKind, ModelPaths, PreProcessingMixin, TokenSource, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::sampling::sample_and_add_toks;
use crate::pipeline::{get_chat_template, ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::{
    get_paths, normal_model_loader, DeviceMapMetadata, Ordering, PagedAttentionConfig, Pipeline,
    TryIntoDType,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use mistralrs_quant::IsqType;
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use std::{fs, usize};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{info, warn};

pub struct DiffusionPipeline {
    model: Box<dyn DiffusionModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    dummy_cache: Cache,
}

/// A loader for a vision (non-quantized) model.
pub struct DiffusionLoader {
    inner: Box<dyn DiffusionModelLoader>,
    model_id: String,
    config: DiffusionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
}

#[derive(Default)]
/// A builder for a loader for a vision (non-quantized) model.
pub struct DiffusionLoaderBuilder {
    model_id: Option<String>,
    config: DiffusionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
}

#[derive(Clone, Default)]
/// Config specific to loading a vision model.
pub struct DiffusionSpecificConfig {
    pub use_flash_attn: bool,
}

impl DiffusionLoaderBuilder {
    pub fn new(
        config: DiffusionSpecificConfig,
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

    pub fn build(self, loader: DiffusionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn DiffusionModelLoader> = match loader {
            DiffusionLoaderType::Flux => Box::new(FluxLoader),
        };
        Box::new(DiffusionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            xlora_model_id: None,
            xlora_order: None,
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
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            None,
            None,
            silent
        );
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
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        mut paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config = std::fs::read_to_string(paths.get_config_filename())?;

        // Otherwise, the device mapper will print it
        if mapper.is_dummy() {
            info!(
                "Loading model `{}` on {}.",
                self.get_id(),
                device.device_pretty_repr()
            );
        } else {
            anyhow::bail!("Device mapping is not supported for Diffusion models.");
        }

        if in_situ_quant.is_some() {
            anyhow::bail!("ISQ is not supported for Diffusion models.");
        }

        if paged_attn_config.is_some() {
            warn!("PagedAttention is not supported for Diffusion models, disabling it.");

            paged_attn_config = None;
        }

        info!(
            "Model config: {:?}",
            self.inner
                .get_config_repr(&config, self.config.use_flash_attn)?
        );

        let mapper = mapper.into_mapper(
            self.inner.get_total_device_mapping_num_layers(&config)?,
            device,
            None,
        )?;
        let dtype = mapper.get_min_dtype(dtype)?;

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let model = match self.kind {
            ModelKind::Normal => normal_model_loader!(
                paths,
                Some(dtype),
                &device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                false,
                device.clone(),
                attention_mechanism
            ),
            _ => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = get_chat_template(paths, &self.chat_template, None);

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
            )?;
            let cache_engine = CacheEngine::new(model.config(), &cache_config, dtype, device)?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        let sliding_window = model.config().sliding_window;
        Ok(Arc::new(Mutex::new(DiffusionPipeline {
            model,
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                tok_trie,
                is_xlora: false,
                num_hidden_layers: usize::MAX, // FIXME(EricLBuehler): we know this is only for caching, so its OK.
                eos_tok: eos,
                kind: self.kind.clone(),
                has_no_kv_cache: false,
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
                prompt_batchsize: None,
            }),
            dummy_cache: Cache::new(0, false),
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
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
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
    fn clone_in_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
    }
    fn clone_out_cache(&self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
    }
    fn set_none_cache(&self, reset_non_granular: bool, modify_draft_cache: bool) {
        DefaultCacheManager.set_none_cache(self, modify_draft_cache);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &Cache {
        &self.dummy_cache
    }
}

impl AdapterActivationMixin for DiffusionPipeline {
    fn activate_adapters(&mut self, _adapters: Vec<String>) -> Result<usize> {
        anyhow::bail!("Diffusion models do not support adapter activation.");
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
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
}

#[async_trait::async_trait]
impl Pipeline for DiffusionPipeline {
    fn forward_inputs(&self, inputs: Box<dyn Any>) -> candle_core::Result<ForwardInputsResult> {
        todo!()
    }
    async fn sample_causal_gen(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Vec<Tensor>,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        sample_and_add_toks(self, seqs, logits, prefix_cacher, disable_eos_stop, rng).await
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Diffusion
    }
}

impl AnyMoePipelineMixin for DiffusionPipeline {}
