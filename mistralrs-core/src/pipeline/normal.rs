use super::cache_manager::DefaultCacheManager;
use super::loaders::{
    GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType, Phi2Loader,
    Phi3Loader, Qwen2Loader,
};
use super::{
    get_model_paths, get_xlora_paths, CacheManager, GeneralMetadata, Loader, ModelInputs,
    ModelKind, ModelPaths, NormalModel, NormalModelLoader, Pipeline, TokenSource, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::lora::Ordering;
use crate::pipeline::chat_template::calculate_eos_tokens;
use crate::pipeline::Cache;
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::xlora_models::NonGranularState;
use crate::{
    deserialize_chat_template, do_sample, get_mut_arcmutex, get_paths, lora_model_loader,
    normal_model_loader, xlora_model_loader, DeviceMapMetadata,
};
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

pub struct NormalPipeline {
    model: Box<dyn NormalModel + Send + Sync>,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    non_granular_state: Option<NonGranularState>,
    model_id: String,
    metadata: GeneralMetadata,
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
        self.kind = ModelKind::XLoraNormal;
        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = ModelKind::LoraNormal;
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
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
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
        self.load_model_from_path(&paths?, _dtype, device, silent, mapper, in_situ_quant)
    }

    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let config = std::fs::read_to_string(paths.get_config_filename())?;
        let default_dtype = if device.is_cuda() && mapper.is_dummy() {
            DType::BF16
        } else if !mapper.is_dummy() {
            DType::F16
        } else {
            DType::F32
        };

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

        let mut is_lora = false;
        let mut model = match self.kind {
            ModelKind::QuantizedGGUF => unreachable!(),
            ModelKind::QuantizedGGML => unreachable!(),
            ModelKind::Normal => normal_model_loader!(
                paths,
                dtype,
                default_dtype,
                &load_device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                in_situ_quant.is_some(),
                device.clone()
            ),
            ModelKind::XLoraNormal => xlora_model_loader!(
                paths,
                dtype,
                default_dtype,
                &load_device,
                config,
                self.inner,
                self.config.use_flash_attn,
                silent,
                mapper,
                in_situ_quant.is_some(),
                device.clone()
            ),
            ModelKind::LoraNormal => {
                is_lora = true;
                lora_model_loader!(
                    paths,
                    dtype,
                    default_dtype,
                    &load_device,
                    config,
                    self.inner,
                    self.config.use_flash_attn,
                    silent,
                    mapper,
                    in_situ_quant.is_some(),
                    device.clone()
                )
            }
            ModelKind::XLoraGGUF => unreachable!(),
            ModelKind::XLoraGGML => unreachable!(),
            ModelKind::LoraGGUF => unreachable!(),
            ModelKind::LoraGGML => unreachable!(),
            ModelKind::Speculative {
                target: _,
                draft: _,
            } => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename())?;

        let (chat_template, gen_conf) = deserialize_chat_template!(paths, self);

        if let Some(in_situ_quant) = in_situ_quant {
            model.quantize(in_situ_quant, device.clone())?;
        }

        let max_seq_len = model.max_seq_len();
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let is_xlora = model.is_xlora() && !is_lora;
        let num_hidden_layers = model.cache().lock().len();
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
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
            metadata: GeneralMetadata {
                max_seq_len,
                repeat_last_n: self.config.repeat_last_n,
                tok_trie,
                has_no_kv_cache: self.no_kv_cache,
                is_xlora,
                num_hidden_layers,
                eos_tok: eos,
                is_lora,
            },
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

#[async_trait::async_trait]
impl Pipeline for NormalPipeline {
    fn forward_inputs(
        &mut self,
        ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
            position_ids,
        }: ModelInputs,
    ) -> Result<Tensor, candle_core::Error> {
        match self.model.is_xlora() {
            false => self.model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
                position_ids,
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
    fn device(&self) -> Device {
        self.model.device().clone()
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        self.tokenizer.clone()
    }
    fn name(&self) -> String {
        self.model_id.clone()
    }
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn reset_non_granular_state(&self) {
        if let Some(s) = self.non_granular_state.as_ref() {
            *self.cache().get_scalings_cache() = None;
            *get_mut_arcmutex!(s.non_granular_index) = 0;
        }
    }
    fn re_isq_model(&mut self, dtype: GgmlDType) -> Result<()> {
        let device = self.device().clone();
        self.model
            .quantize(dtype, device)
            .map_err(anyhow::Error::msg)
    }
    fn get_metadata(&self) -> &GeneralMetadata {
        &self.metadata
    }
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_in_cache(self, seqs, modify_draft_cache)
    }
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool) {
        DefaultCacheManager.clone_out_cache(self, seqs, modify_draft_cache)
    }
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool) {
        DefaultCacheManager.set_none_cache(self, modify_draft_cache);
        if reset_non_granular {
            self.reset_non_granular_state()
        }
    }
    fn cache(&self) -> &Cache {
        self.model.cache()
    }
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        self.model
            .activate_adapters(adapter_names)
            .map_err(anyhow::Error::msg)
    }
}
