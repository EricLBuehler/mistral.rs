use super::cache_manager::DefaultCacheManager;
use super::vision_loaders::{
    Idefics2Loader, LLaVALoader, LLaVANextLoader, Phi3VLoader, VisionLoaderType,
};
use super::{
    get_model_paths, get_xlora_paths, AdapterActivationMixin, AnyMoePipelineMixin, Cache,
    CacheManager, CacheManagerMixin, GeneralMetadata, IsqPipelineMixin, Loader, MetadataMixin,
    ModelCategory, ModelKind, ModelPaths, PreProcessingMixin, Processor, TokenSource, VisionModel,
    VisionModelLoader, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::{get_chat_template, ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::ModelInputs;
use crate::{
    api_dir_list, api_get_file, do_sample, get_paths, vision_normal_model_loader, AnyMoeExpertType,
    DeviceMapMetadata, Ordering, PagedAttentionConfig, Pipeline, TryIntoDType,
};
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{Device, Tensor, Var};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use regex_automata::meta::Regex;
use std::any::Any;
use std::fs;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

pub struct VisionPipeline {
    model: Box<dyn VisionModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
    processor: Arc<dyn Processor + Send + Sync>,
    preprocessor_config: Arc<PreProcessorConfig>,
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

#[derive(Clone, Copy, Default)]
/// Config specific to loading a vision model.
pub struct VisionSpecificConfig {
    pub use_flash_attn: bool,
    pub repeat_last_n: usize,
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
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
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
        in_situ_quant: Option<GgmlDType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config = std::fs::read_to_string(paths.get_config_filename())?;
        let dtype = dtype.try_into_dtype(device)?;

        // Otherwise, the device mapper will print it
        if mapper.is_dummy() {
            info!(
                "Loading model `{}` on {}.",
                self.get_id(),
                device.device_pretty_repr()
            );
        }

        info!(
            "Model config: {:?}",
            self.inner
                .get_config_repr(&config, self.config.use_flash_attn)?
        );

        let load_device = if in_situ_quant.is_none() {
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
                dtype,
                &load_device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                in_situ_quant.is_some(),
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

        let processor =
            self.inner
                .get_processor(&config, processor_config, preprocessor_config.clone()); //There are always some repos that don't properly handle config position, for example... LLaVA

        let tokenizer = get_tokenizer(
            paths.get_tokenizer_filename(),
            Some(processor.get_special_tokens()),
        )?;

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = get_chat_template(paths, &self.chat_template, None);

        if let Some(in_situ_quant) = in_situ_quant {
            model.quantize(in_situ_quant, device.clone())?;
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
            )?;
            let cache_engine = CacheEngine::new(model.config(), &cache_config, dtype, device)?;
            (Some(cache_config), Some(cache_engine))
        } else {
            (None, None)
        };

        let max_seq_len = model.max_seq_len();
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let num_hidden_layers = model.cache().lock().len();
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        let sliding_window = model.config().sliding_window;
        Ok(Arc::new(Mutex::new(VisionPipeline {
            model,
            tok_trie: tok_trie.clone(),
            tokenizer: tokenizer.into(),
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                repeat_last_n: self.config.repeat_last_n,
                tok_trie,
                is_xlora: false,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                has_no_kv_cache: false,
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
            }),
            processor,
            preprocessor_config: Arc::new(preprocessor_config),
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
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        Some(self.preprocessor_config.clone())
    }
    fn get_processor(&self) -> Arc<dyn super::Processor> {
        self.processor.clone()
    }
}

impl IsqPipelineMixin for VisionPipeline {
    fn re_isq_model(&mut self, dtype: GgmlDType) -> Result<()> {
        let device = self.device().clone();
        self.model
            .quantize(dtype, device)
            .map_err(anyhow::Error::msg)
    }
}

impl CacheManagerMixin for VisionPipeline {
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
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
}

#[async_trait::async_trait]
impl Pipeline for VisionPipeline {
    fn forward_inputs(&self, inputs: Box<dyn Any>) -> candle_core::Result<Tensor> {
        let ModelInputs {
            input_ids,
            seqlen_offsets,
            seqlen_offsets_kernel,
            context_lens,
            position_ids,
            pixel_values,
            model_specific_args,
            mut paged_attn_meta,
        } = *inputs.downcast::<ModelInputs>().expect("Downcast failed.");
        self.model.forward(
            &input_ids,
            pixel_values,
            &seqlen_offsets,
            seqlen_offsets_kernel,
            context_lens,
            position_ids,
            model_specific_args,
            self.get_metadata().cache_engine.as_ref().map(|engine| {
                (
                    engine.get_kv_cache().clone(),
                    paged_attn_meta.as_mut().unwrap(),
                )
            }),
        )
    }
    async fn sample(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        do_sample!(self, seqs, logits, prefix_cacher, disable_eos_stop, rng)
    }
    fn category(&self) -> ModelCategory {
        let has_conv2d = self.model.has_conv2d();
        ModelCategory::Vision { has_conv2d }
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
        let regex = Regex::new(match_regex).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
        for model_id in model_ids {
            let model_id_str = &model_id;
            let model_id = Path::new(&model_id);

            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(token).map_err(|e| candle_core::Error::Msg(e.to_string()))?)
                .build()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
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
            let vb = from_mmaped_safetensors(filenames, vec![], dtype, dev, silent, move |key| {
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
            })?;
            vbs.push(vb);
        }

        let gate_vb = if let Some(gate_model_id) = gate_model_id {
            let model_id_str = &gate_model_id;
            let model_id = Path::new(&gate_model_id);

            let api = ApiBuilder::new()
                .with_progress(!silent)
                .with_token(get_token(token).map_err(|e| candle_core::Error::Msg(e.to_string()))?)
                .build()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
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
                dtype,
                dev,
                silent,
                |_| true,
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
