use super::cache_manager::DefaultCacheManager;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, QuantizationKind, TokenSource,
    XLoraPaths,
};
use super::{
    AdapterActivationMixin, CacheManagerMixin, IsqPipelineMixin, MetadataMixin, ModelCategory,
    PreProcessingMixin,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::lora::Ordering;
use crate::pipeline::chat_template::{calculate_eos_tokens, GenerationConfig};
use crate::pipeline::{get_chat_template, Cache};
use crate::pipeline::{ChatTemplate, LocalModelPaths};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::setup_logger_and_debug;
use crate::utils::model_config as ModelConfig;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    do_sample, get_mut_arcmutex, get_paths, DeviceMapMetadata, Pipeline, TryIntoDType, DEBUG,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama, utils::tokens::get_token,
    xlora_models::XLoraQLlama,
};
use anyhow::Result;
use candle_core::quantized::{ggml_file, GgmlDType};
use candle_core::{Device, Tensor};
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::level_filters::LevelFilter;
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

enum Model {
    Llama(QLlama),
    XLoraLlama(XLoraQLlama),
}

pub struct GGMLPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: GeneralMetadata,
}

/// A loader for a GGML model.
pub struct GGMLLoader {
    model_id: String,
    config: GGMLSpecificConfig,
    quantized_model_id: Option<String>,
    quantized_filename: Option<String>,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
}

#[derive(Clone, Copy, Default)]
/// Config for a GGML loader.
pub struct GGMLSpecificConfig {
    pub repeat_last_n: usize,
    pub gqa: usize,
}

#[derive(Default)]
/// A builder for a GGML loader.
pub struct GGMLLoaderBuilder {
    model_id: Option<String>,
    config: GGMLSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tokenizer_json: Option<String>,
    tgt_non_granular_index: Option<usize>,
}

impl GGMLLoaderBuilder {
    pub fn new(
        config: GGMLSpecificConfig,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
    ) -> Self {
        let kind = ModelKind::Quantized {
            quant: QuantizationKind::Ggml,
        };

        Self {
            config,
            chat_template,
            tokenizer_json,
            model_id,
            kind,
            quantized_filename,
            quantized_model_id,
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
        self.kind = (AdapterKind::XLora, QuantizationKind::Ggml).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Ggml).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        setup_logger_and_debug();

        Box::new(GGMLLoader {
            model_id: self.model_id.unwrap(),
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tokenizer_json: self.tokenizer_json,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filename: Some(self.quantized_filename),
            quantized_model_id: Some(self.quantized_model_id),
        })
    }
}

impl GGMLLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        config: GGMLSpecificConfig,
        quantized_model_id: Option<String>,
        quantized_filename: Option<String>,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        setup_logger_and_debug();

        let model_id = if let Some(id) = model_id {
            id
        } else {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.as_ref().unwrap().base_model_id
            );
            xlora_order.as_ref().unwrap().base_model_id.clone()
        };
        Self {
            model_id,
            config,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            chat_template,
            tokenizer_json,
            kind,
            tgt_non_granular_index,
        }
    }
}

impl Loader for GGMLLoader {
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let is_debug = std::env::var("MISTRALRS_DEBUG")
            .unwrap_or_default()
            .contains('1');
        DEBUG.store(is_debug, std::sync::atomic::Ordering::Relaxed);

        let filter = EnvFilter::builder()
            .with_default_directive(if is_debug {
                LevelFilter::INFO.into()
            } else {
                LevelFilter::DEBUG.into()
            })
            .from_env_lossy();
        tracing_subscriber::fmt().with_env_filter(filter).init();

        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGML model. This will not do anything."
            );
        }
        if !mapper.is_dummy() {
            warn!("GGML models do not support device mapping. Device mapping will not work. Please consider using a GGUF model.");
        }
        info!("Loading model `{}` on {device:?}...", self.get_id());

        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = ggml_file::Content::read(&mut file, device)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;

        info!("Model config: {:?}", model.hparams);

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            let mut tensors = Vec::new();
            for (name, t) in &model.tensors {
                tensors.push(format!(
                    "name = `{name}`, shape = {:?}, dtype = {:?}",
                    t.shape().clone(),
                    t.dtype(),
                ));
            }
            fs::write(
                "mistralrs_ggml_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!("Debug is enabled, wrote the names and information about each tensor to `mistralrs_ggml_tensors.txt`.");
        }

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGML((model, self.config.gqa).into());

            // With optional adapter config:
            let mut adapter = None;
            if has_adapter {
                adapter.replace(ModelConfig::Adapter::try_new(
                    paths, device, silent, is_xlora,
                )?);
            }

            ModelConfig::ModelParams::builder()
                .quant(quant)
                .and_adapter(adapter)
                .build()
        };

        // Config into model:
        // NOTE: No architecture to infer like GGUF, Llama model is implicitly matched
        let model = match self.kind {
            ModelKind::Quantized { .. } => Model::Llama(QLlama::try_from(model_config)?),
            ModelKind::AdapterQuantized { .. } => {
                Model::XLoraLlama(XLoraQLlama::try_from(model_config)?)
            }
            _ => unreachable!(),
        };

        let tokenizer = get_tokenizer(paths.get_tokenizer_filename(), None)?;
        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let chat_template = get_chat_template(paths, &self.chat_template, None);

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
        };
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.lock().len(),
            Model::XLoraLlama(ref model) => model.cache.lock().len(),
        };
        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(GGMLPipeline {
            model,
            tok_trie: tok_trie.clone(),
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self.model_id.clone(),
            non_granular_state: self.tgt_non_granular_index.map(|tgt_non_granular_index| {
                NonGranularState {
                    non_granular_index: Arc::new(Mutex::new(0)),
                    tgt_non_granular_index,
                }
            }),
            metadata: GeneralMetadata {
                max_seq_len,
                repeat_last_n: self.config.repeat_last_n,
                tok_trie,
                has_no_kv_cache: self.no_kv_cache,
                num_hidden_layers,
                eos_tok: eos,
                kind: self.kind.clone(),
                is_xlora,
            },
        })))
    }

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
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id,
            Some(vec![self.quantized_filename.as_ref().unwrap().clone()]),
            silent
        );
        self.load_model_from_path(&paths?, dtype, device, silent, mapper, in_situ_quant)
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

impl PreProcessingMixin for GGMLPipeline {
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGMLPipeline {
    fn re_isq_model(&mut self, _dtype: GgmlDType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGMLPipeline {
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
        match self.model {
            Model::Llama(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
        }
    }
}

impl AdapterActivationMixin for GGMLPipeline {
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        let is_lora = self.metadata.kind.is_adapted_and(|a| a.is_lora());
        if !is_lora {
            anyhow::bail!("Activating adapters is only supported for models fine-tuned with LoRA.")
        }

        match self.model {
            Model::XLoraLlama(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            _ => unreachable!(),
        }
    }
}

impl MetadataMixin for GGMLPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
        }
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
    fn get_metadata(&self) -> &GeneralMetadata {
        &self.metadata
    }
}

#[async_trait::async_trait]
impl Pipeline for GGMLPipeline {
    fn forward_inputs(&mut self, inputs: Box<dyn Any>) -> Result<Tensor, candle_core::Error> {
        let ModelInputs {
            input_ids,
            input_ids_full,
            seqlen_offsets,
            seqlen_offsets_full,
            seqlen_offsets_kernel,
            seqlen_offsets_kernel_full,
            context_lens,
            position_ids: _, // NOTE(EricLBuehler): ignore, it is for phi3
        } = *inputs.downcast().expect("Downcast failed.");
        match self.model {
            Model::Llama(ref mut model) => model.forward(
                &input_ids,
                &seqlen_offsets,
                seqlen_offsets_kernel,
                context_lens,
            ),
            Model::XLoraLlama(ref mut model) => model.forward(
                &input_ids,
                input_ids_full.as_ref().unwrap_or(&input_ids),
                &seqlen_offsets,
                seqlen_offsets_full.as_ref().unwrap_or(&seqlen_offsets),
                seqlen_offsets_kernel.clone(),
                seqlen_offsets_kernel_full.unwrap_or(seqlen_offsets_kernel),
                self.no_kv_cache,
                &self.non_granular_state,
                context_lens,
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
