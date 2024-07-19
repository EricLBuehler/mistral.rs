use super::cache_manager::DefaultCacheManager;
use super::normal_loaders::{
    Gemma2Loader, GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType,
    Phi2Loader, Phi3Loader, Qwen2Loader, Starcoder2Loader,
};
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, NormalModel, NormalModelLoader,
    TokenSource, XLoraPaths,
};
use super::{
    AdapterActivationMixin, AnyMoePipelineMixin, CacheManagerMixin, IsqPipelineMixin,
    MetadataMixin, ModelCategory, PreProcessingMixin,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::amoe::AnyMoeExpertType;
use crate::lora::Ordering;
use crate::paged_attention::{calculate_cache_config, AttentionImplementation, CacheEngine};
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::{get_chat_template, Cache};
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::DeviceRepr;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::xlora_models::NonGranularState;
use crate::{
    api_dir_list, api_get_file, do_sample, get_mut_arcmutex, get_paths, lora_model_loader,
    normal_model_loader, xlora_model_loader, DeviceMapMetadata, PagedAttentionConfig, Pipeline,
    TryIntoDType,
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
use tracing::{info, warn};

pub struct NormalPipeline {
    model: Box<dyn NormalModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    non_granular_state: Option<NonGranularState>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
}

/// A loader for a "normal" (non-quantized) model.
pub struct NormalLoader {
    inner: Box<dyn NormalModelLoader>,
    model_id: String,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
}

#[derive(Default)]
/// A builder for a loader for a "normal" (non-quantized) model.
pub struct NormalLoaderBuilder {
    model_id: Option<String>,
    config: NormalSpecificConfig,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
}

#[derive(Clone, Copy, Default)]
/// Config specific to loading a normal model.
pub struct NormalSpecificConfig {
    pub use_flash_attn: bool,
    pub repeat_last_n: usize,
}

impl NormalLoaderBuilder {
    pub fn new(
        config: NormalSpecificConfig,
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
            ..Default::default()
        }
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

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = ModelKind::Adapter {
            adapter: AdapterKind::Lora,
        };
        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self, loader: NormalLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn NormalModelLoader> = match loader {
            NormalLoaderType::Mistral => Box::new(MistralLoader),
            NormalLoaderType::Gemma => Box::new(GemmaLoader),
            NormalLoaderType::Llama => Box::new(LlamaLoader),
            NormalLoaderType::Mixtral => Box::new(MixtralLoader),
            NormalLoaderType::Phi2 => Box::new(Phi2Loader),
            NormalLoaderType::Phi3 => Box::new(Phi3Loader),
            NormalLoaderType::Qwen2 => Box::new(Qwen2Loader),
            NormalLoaderType::Gemma2 => Box::new(Gemma2Loader),
            NormalLoaderType::Starcoder2 => Box::new(Starcoder2Loader),
        };
        Box::new(NormalLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
        })
    }
}

impl Loader for NormalLoader {
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

        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let attention_mechanism = if paged_attn_config.is_some() {
            AttentionImplementation::PagedAttention
        } else {
            AttentionImplementation::Eager
        };

        let mut model = match self.kind {
            ModelKind::Normal => normal_model_loader!(
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
            ModelKind::Adapter {
                adapter: AdapterKind::XLora,
            } => xlora_model_loader!(
                paths,
                dtype,
                &load_device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                in_situ_quant.is_some(),
                device.clone()
            ),
            ModelKind::Adapter {
                adapter: AdapterKind::Lora,
            } => lora_model_loader!(
                paths,
                dtype,
                &load_device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                in_situ_quant.is_some(),
                device.clone()
            ),
            _ => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = get_chat_template(paths, &self.chat_template, None);

        if let Some(in_situ_quant) = in_situ_quant {
            model.quantize(in_situ_quant, device.clone())?;
        }

        let paged_attn_config = if matches!(self.kind, ModelKind::Adapter { .. }) {
            warn!("Adapter models do not currently support PagedAttention, running without");
            None
        } else {
            paged_attn_config
        };

        let (cache_config, cache_engine) = if let Some(paged_attn_config) = paged_attn_config {
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
        Ok(Arc::new(Mutex::new(NormalPipeline {
            model,
            tok_trie: tok_trie.clone(),
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            model_id: self.model_id.clone(),
            metadata: Arc::new(GeneralMetadata {
                max_seq_len,
                repeat_last_n: self.config.repeat_last_n,
                tok_trie,
                has_no_kv_cache: self.no_kv_cache,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
                activation_dtype: dtype,
                sliding_window,
                cache_config,
                cache_engine,
            }),
        })))
    }

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(&self.model_id)
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for NormalPipeline {
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for NormalPipeline {
    fn re_isq_model(&mut self, dtype: GgmlDType) -> Result<()> {
        let device = self.device().clone();
        self.model
            .quantize(dtype, device)
            .map_err(anyhow::Error::msg)
    }
}

impl CacheManagerMixin for NormalPipeline {
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

impl AdapterActivationMixin for NormalPipeline {
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        self.model
            .activate_adapters(adapter_names)
            .map_err(anyhow::Error::msg)
    }
}

impl MetadataMixin for NormalPipeline {
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn get_metadata(&self) -> Arc<GeneralMetadata> {
        self.metadata.clone()
    }
}

#[async_trait::async_trait]
impl Pipeline for NormalPipeline {
    fn forward_inputs(&self, inputs: Box<dyn Any>) -> Result<Tensor, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
            position_ids,
            mut paged_attn_meta,
        } = *inputs.downcast().expect("Downcast failed.");
        match self.model.is_xlora() {
            false => self.model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
                position_ids,
                self.get_metadata().cache_engine.as_ref().map(|engine| {
                    (
                        engine.get_kv_cache().clone(),
                        paged_attn_meta.as_mut().unwrap(),
                    )
                }),
            ),
            true => self.model.xlora_forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
                position_ids,
            ),
        }
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
        ModelCategory::Text
    }
}

impl AnyMoePipelineMixin for NormalPipeline {
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
