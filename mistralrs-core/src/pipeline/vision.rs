use super::inputs_processor::InputsProcessor;
use super::vision_loaders::{Idefics2Loader, VisionLoaderType};
use super::{
    get_model_paths, get_xlora_paths, AdapterActivationMixin, CacheManagerMixin, GeneralMetadata,
    IsqPipelineMixin, Loader, MetadataMixin, ModelCategory, ModelKind, ModelPaths,
    PreProcessingMixin, TokenSource, VisionModel, VisionModelLoader, XLoraPaths,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::pipeline::chat_template::calculate_eos_tokens;
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::tokenizer::get_tokenizer;
use crate::utils::{tokens::get_token, varbuilder_utils::from_mmaped_safetensors};
use crate::xlora_models::NonGranularState;
use crate::{
    deserialize_chat_template, get_paths, vision_normal_model_loader, DeviceMapMetadata, Ordering,
    Pipeline,
};
use anyhow::Result;
use candle_core::quantized::GgmlDType;
use candle_core::{DType, Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use serde_json::Value;
use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
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
    non_granular_state: Option<NonGranularState>,
    model_id: String,
    metadata: GeneralMetadata,
}

/// A loader for a vision (non-quantized) model.
pub struct VisionLoader {
    inner: Box<dyn VisionModelLoader>,
    model_id: String,
    config: VisionSpecificConfig,
    kind: ModelKind,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
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
    tgt_non_granular_index: Option<usize>,
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
            ..Default::default()
        }
    }

    pub fn build(self, loader: VisionLoaderType) -> Box<dyn Loader> {
        let loader: Box<dyn VisionModelLoader> = match loader {
            VisionLoaderType::Idefics2 => Box::new(Idefics2Loader),
        };
        Box::new(VisionLoader {
            inner: loader,
            model_id: self.model_id.unwrap(),
            config: self.config,
            kind: self.kind,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
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
        // Otherwise, the device mapper will print it
        if mapper.is_dummy() {
            info!("Loading model `{}` on {device:?}...", self.get_id());
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

        let mut model = match self.kind {
            ModelKind::QuantizedGGUF => unreachable!(),
            ModelKind::QuantizedGGML => unreachable!(),
            ModelKind::Normal => vision_normal_model_loader!(
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
            ModelKind::XLoraNormal => unreachable!(),
            ModelKind::LoraNormal => unreachable!(),
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
        let num_hidden_layers = model.cache().lock().len();
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(VisionPipeline {
            model,
            tok_trie: tok_trie.clone(),
            tokenizer: tokenizer.into(),
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
                is_xlora: false,
                num_hidden_layers,
                eos_tok: eos,
                is_lora: false,
                has_no_kv_cache: false,
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

impl PreProcessingMixin for VisionPipeline {
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        todo!()
    }
    fn get_input_processor(&self) -> Box<dyn InputsProcessor> {
        todo!()
    }
}

impl IsqPipelineMixin for VisionPipeline {
    fn re_isq_model(&mut self, _dtype: GgmlDType) -> Result<()> {
        todo!()
    }
}

impl CacheManagerMixin for VisionPipeline {
    fn cache(&self) -> &super::Cache {
        todo!()
    }
    fn clone_in_cache(&mut self, _seqs: &mut [&mut Sequence], _modify_draft_cache: bool) {
        todo!()
    }
    fn clone_out_cache(&mut self, _seqs: &mut [&mut Sequence], _modify_draft_cache: bool) {
        todo!()
    }
    fn set_none_cache(&mut self, _reset_non_granular: bool, _modify_draft_cache: bool) {
        todo!()
    }
}

impl AdapterActivationMixin for VisionPipeline {
    fn activate_adapters(&mut self, _adapters: Vec<String>) -> Result<usize> {
        todo!()
    }
}

impl MetadataMixin for VisionPipeline {
    fn device(&self) -> Device {
        todo!()
    }
    fn get_metadata(&self) -> &GeneralMetadata {
        todo!()
    }
    fn name(&self) -> String {
        todo!()
    }
    fn reset_non_granular_state(&self) {
        todo!()
    }
    fn tokenizer(&self) -> Arc<Tokenizer> {
        todo!()
    }
}

#[async_trait::async_trait]
impl Pipeline for VisionPipeline {
    fn forward_inputs(&mut self, _inputs: Box<dyn Any>) -> candle_core::Result<Tensor> {
        todo!()
    }
    async fn sample(
        &self,
        _seqs: &mut [&mut Sequence],
        _logits: Tensor,
        _prefix_cacher: &mut PrefixCacheManager,
        _disable_eos_stop: bool,
        _rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error> {
        todo!()
    }
    fn category(&self) -> ModelCategory {
        ModelCategory::Vision
    }
}
