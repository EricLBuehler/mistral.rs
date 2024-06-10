use super::cache_manager::DefaultCacheManager;
use super::{
    get_model_paths, get_xlora_paths, text_models_inputs_processor::ModelInputs, AdapterKind,
    CacheManager, GeneralMetadata, Loader, ModelKind, ModelPaths, PrettyName, QuantizationKind,
    TokenSource, XLoraPaths,
};
use super::{
    AdapterActivationMixin, CacheManagerMixin, IsqPipelineMixin, MetadataMixin, ModelCategory,
    PreProcessingMixin,
};
use crate::aici::bintokens::build_tok_trie;
use crate::aici::toktree::TokTrie;
use crate::gguf::{
    get_gguf_chat_template, {convert_gguf_to_hf_tokenizer, GgufTokenizerConversion},
};
use crate::lora::Ordering;
use crate::pipeline::chat_template::{calculate_eos_tokens, BeginEndUnkTok, GenerationConfig};
use crate::pipeline::ChatTemplate;
use crate::pipeline::{get_chat_template, Cache};
use crate::prefix_cacher::PrefixCacheManager;
use crate::sequence::Sequence;
use crate::utils::debug::setup_logger_and_debug;
use crate::utils::model_config as ModelConfig;
use crate::utils::tokenizer::get_tokenizer;
use crate::xlora_models::NonGranularState;
use crate::{
    do_sample, get_mut_arcmutex, get_paths_gguf, DeviceMapMetadata, LocalModelPaths, Pipeline,
    TryIntoDType, DEBUG,
};
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    utils::tokens::get_token,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use anyhow::{bail, Context, Result};
use candle_core::quantized::{
    gguf_file::{self, Value as GgufValue},
    GgmlDType,
};
use candle_core::{Device, Tensor};
use either::Either;
use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};
use rand_isaac::Isaac64Rng;
use std::any::Any;
use std::fs;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::Arc;
use strum::EnumString;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
}

pub struct GGUFPipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    tok_trie: Arc<TokTrie>,
    no_kv_cache: bool,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    non_granular_state: Option<NonGranularState>,
    metadata: GeneralMetadata,
}

/// Loader for a GGUF model.
pub struct GGUFLoader {
    model_id: Option<String>,
    config: GGUFSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    xlora_model_id: Option<String>,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    kind: ModelKind,
    tgt_non_granular_index: Option<usize>,
}

#[derive(Debug, EnumString)]
#[strum(serialize_all = "kebab-case")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl GGUFArchitecture {
    fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown GGUF architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }
}

#[derive(Clone, Copy, Default)]
/// A config for a GGUF loader.
pub struct GGUFSpecificConfig {
    pub repeat_last_n: usize,
}

#[derive(Default)]
/// A builder for a GGUF loader.
pub struct GGUFLoaderBuilder {
    model_id: Option<String>,
    config: GGUFSpecificConfig,
    quantized_model_id: String,
    quantized_filename: String,
    xlora_model_id: Option<String>,
    kind: ModelKind,
    xlora_order: Option<Ordering>,
    no_kv_cache: bool,
    chat_template: Option<String>,
    tgt_non_granular_index: Option<usize>,
}

impl GGUFLoaderBuilder {
    /// Create a loader builder for a GGUF model. `tok_model_id` is the model ID where you can find a
    /// `tokenizer_config.json` file. If the `chat_template` is specified, then it will be treated as a
    /// path and used over remote files, removing all remote accesses.
    pub fn new(
        config: GGUFSpecificConfig,
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
    ) -> Self {
        let kind = ModelKind::Quantized {
            quant: QuantizationKind::Gguf,
        };

        Self {
            config,
            chat_template,
            model_id: tok_model_id,
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
        self.kind = (AdapterKind::XLora, QuantizationKind::Gguf).into();

        self.with_adapter(
            xlora_model_id,
            xlora_order,
            no_kv_cache,
            tgt_non_granular_index,
        )
    }

    pub fn with_lora(mut self, lora_model_id: String, lora_order: Ordering) -> Self {
        self.kind = (AdapterKind::Lora, QuantizationKind::Gguf).into();

        self.with_adapter(lora_model_id, lora_order, false, None)
    }

    pub fn build(self) -> Box<dyn Loader> {
        setup_logger_and_debug();

        Box::new(GGUFLoader {
            model_id: self.model_id,
            config: self.config,
            xlora_model_id: self.xlora_model_id,
            kind: self.kind,
            xlora_order: self.xlora_order,
            no_kv_cache: self.no_kv_cache,
            chat_template: self.chat_template,
            tgt_non_granular_index: self.tgt_non_granular_index,
            quantized_filename: self.quantized_filename,
            quantized_model_id: self.quantized_model_id,
        })
    }
}

impl GGUFLoader {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: Option<String>,
        config: GGUFSpecificConfig,
        quantized_model_id: String,
        quantized_filename: String,
        xlora_model_id: Option<String>,
        kind: ModelKind,
        xlora_order: Option<Ordering>,
        no_kv_cache: bool,
        chat_template: Option<String>,
        tgt_non_granular_index: Option<usize>,
    ) -> Self {
        setup_logger_and_debug();

        let model_id = if let Some(id) = model_id {
            Some(id)
        } else if let Some(xlora_order) = xlora_order.clone() {
            info!(
                "Using adapter base model ID: `{}`",
                xlora_order.base_model_id
            );
            Some(xlora_order.base_model_id.clone())
        } else {
            None
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
            kind,
            tgt_non_granular_index,
        }
    }
}

fn parse_gguf_value(value: &GgufValue) -> String {
    match value {
        GgufValue::Array(vs) => vs
            .iter()
            .map(parse_gguf_value)
            .collect::<Vec<String>>()
            .join(", "),
        GgufValue::Bool(b) => b.to_string(),
        GgufValue::F32(x) => x.to_string(),
        GgufValue::F64(x) => x.to_string(),
        GgufValue::I8(x) => x.to_string(),
        GgufValue::I16(x) => x.to_string(),
        GgufValue::I32(x) => x.to_string(),
        GgufValue::I64(x) => x.to_string(),
        GgufValue::String(x) => x.to_string(),
        GgufValue::U8(x) => x.to_string(),
        GgufValue::U16(x) => x.to_string(),
        GgufValue::U32(x) => x.to_string(),
        GgufValue::U64(x) => x.to_string(),
    }
}

impl Loader for GGUFLoader {
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
        let paths: anyhow::Result<Box<dyn ModelPaths>> = get_paths_gguf!(
            LocalModelPaths,
            &token_source,
            revision,
            self,
            self.quantized_model_id.clone(),
            self.quantized_filename.clone(),
            silent
        );
        self.load_model_from_path(&paths?, dtype, device, silent, mapper, in_situ_quant)
    }

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
        if in_situ_quant.is_some() {
            anyhow::bail!(
                "You are trying to in-situ quantize a GGUF model. This will not do anything."
            );
        }
        // Otherwise, the device mapper will print it
        if mapper.is_dummy() {
            info!("Loading model `{}` on {device:?}...", self.get_id());
        }

        let mut file = std::fs::File::open(paths.get_weight_filenames().first().unwrap())?;
        let model = gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(paths.get_weight_filenames().first().unwrap()))?;
        let arch = model.metadata["general.architecture"]
            .to_string()
            .context("Model metadata should have declared an architecture")
            .and_then(GGUFArchitecture::from_value)?;

        info!("Model config:");
        let mut sorted_keys = model.metadata.keys().collect::<Vec<_>>();
        sorted_keys.sort();
        for name in sorted_keys {
            if !name.contains("tokenizer") {
                let value = parse_gguf_value(&model.metadata[name]);
                println!("{name}: {}", value);
            }
        }

        if DEBUG.load(std::sync::atomic::Ordering::Relaxed) {
            let mut tensors = Vec::new();
            for (name, info) in &model.tensor_infos {
                tensors.push(format!(
                    "name = `{name}`, shape = {:?}, dtype = {:?}",
                    info.shape.clone(),
                    info.ggml_dtype
                ));
            }
            fs::write(
                "mistralrs_gguf_tensors.txt",
                serde_json::to_string_pretty(&tensors).expect("Serialization failed."),
            )?;

            info!("Debug is enabled, wrote the names and information about each tensor to `mistralrs_gguf_tensors.txt`.");
        }

        let GgufTokenizerConversion {
            tokenizer,
            bos,
            eos,
            unk,
        } = if paths.get_tokenizer_filename().to_string_lossy().is_empty() {
            convert_gguf_to_hf_tokenizer(&model)?
        } else {
            GgufTokenizerConversion {
                tokenizer: get_tokenizer(paths.get_tokenizer_filename(), None)?,
                bos: None,
                eos: None,
                unk: None,
            }
        };

        // Only load gguf chat template if there is nothing else
        let gguf_chat_template =
            if paths.get_template_filename().is_none() && self.chat_template.is_none() {
                get_gguf_chat_template(&model)?
            } else {
                None
            };

        let has_adapter = self.kind.is_adapted();
        let is_xlora = self.kind.is_adapted_and(|a| a.is_x_lora());

        let model_config = {
            // Base config (quantization only):
            let quant = ModelConfig::ParamsGGUF((model, &mut file).into(), (device, mapper).into());

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
        let model = match self.kind {
            ModelKind::Quantized { .. } => match arch {
                GGUFArchitecture::Llama => Model::Llama(QLlama::try_from(model_config)?),
                GGUFArchitecture::Phi2 => Model::Phi2(QPhi::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::Phi3(QPhi3::try_from(model_config)?),
                a => bail!("Unsupported architecture `{a:?}` for GGUF"),
            },
            ModelKind::AdapterQuantized { adapter, .. } => match arch {
                GGUFArchitecture::Llama => Model::XLoraLlama(XLoraQLlama::try_from(model_config)?),
                GGUFArchitecture::Phi3 => Model::XLoraPhi3(XLoraQPhi3::try_from(model_config)?),
                a => bail!(
                    "Unsupported architecture `{a:?}` for GGUF {kind}",
                    kind = adapter.pretty_name()
                ),
            },
            _ => unreachable!(),
        };

        let gen_conf: Option<GenerationConfig> = paths
            .get_gen_conf_filename()
            .map(|f| serde_json::from_str(&fs::read_to_string(f).unwrap()).unwrap());
        let mut chat_template = get_chat_template(paths, &self.chat_template, gguf_chat_template);

        let max_seq_len = match model {
            Model::Llama(ref l) => l.max_seq_len,
            Model::Phi2(ref p) => p.max_seq_len,
            Model::XLoraLlama(ref xl) => xl.max_seq_len,
            Model::Phi3(ref p) => p.max_seq_len,
            Model::XLoraPhi3(ref p) => p.max_seq_len,
        };
        let tok_trie: Arc<TokTrie> = build_tok_trie(tokenizer.clone()).into();
        let num_hidden_layers = match model {
            Model::Llama(ref model) => model.cache.lock().len(),
            Model::Phi2(ref model) => model.cache.lock().len(),
            Model::XLoraLlama(ref model) => model.cache.lock().len(),
            Model::Phi3(ref model) => model.cache.lock().len(),
            Model::XLoraPhi3(ref model) => model.cache.lock().len(),
        };

        if chat_template.bos_token.is_none() && bos.is_some() {
            chat_template.bos_token = Some(BeginEndUnkTok(Either::Left(bos.unwrap())));
        }
        if chat_template.eos_token.is_none() && eos.is_some() {
            chat_template.eos_token = Some(BeginEndUnkTok(Either::Left(eos.unwrap())));
        }
        if chat_template.unk_token.is_none() && unk.is_some() {
            chat_template.unk_token = Some(BeginEndUnkTok(Either::Left(unk.unwrap())));
        }

        let eos = calculate_eos_tokens(&chat_template, gen_conf, &tokenizer);
        Ok(Arc::new(Mutex::new(GGUFPipeline {
            model,
            tok_trie: tok_trie.clone(),
            tokenizer: tokenizer.into(),
            no_kv_cache: self.no_kv_cache,
            chat_template: Arc::new(chat_template),
            model_id: self
                .model_id
                .clone()
                .unwrap_or(self.quantized_model_id.clone()),
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

    fn get_id(&self) -> String {
        self.xlora_model_id
            .as_deref()
            .unwrap_or(self.model_id.as_ref().unwrap_or(&self.quantized_model_id))
            .to_string()
    }

    fn get_kind(&self) -> ModelKind {
        self.kind.clone()
    }
}

impl PreProcessingMixin for GGUFPipeline {
    fn get_chat_template(&self) -> Arc<ChatTemplate> {
        self.chat_template.clone()
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>> {
        None
    }
}

impl IsqPipelineMixin for GGUFPipeline {
    fn re_isq_model(&mut self, _dtype: GgmlDType) -> Result<()> {
        anyhow::bail!(
            "You are trying to in-situ requantize a GGML model. This will not do anything."
        )
    }
}

impl CacheManagerMixin for GGUFPipeline {
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
            Model::Phi2(ref model) => &model.cache,
            Model::XLoraLlama(ref model) => &model.cache,
            Model::Phi3(ref model) => &model.cache,
            Model::XLoraPhi3(ref model) => &model.cache,
        }
    }
}

impl AdapterActivationMixin for GGUFPipeline {
    fn activate_adapters(&mut self, adapter_names: Vec<String>) -> anyhow::Result<usize> {
        let is_lora = self.metadata.kind.is_adapted_and(|a| a.is_lora());
        if !is_lora {
            anyhow::bail!("Activating adapters is only supported for models fine-tuned with LoRA.")
        }

        match self.model {
            Model::XLoraLlama(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            Model::XLoraPhi3(ref mut model) => model
                .activate_adapters(adapter_names)
                .map_err(anyhow::Error::msg),
            _ => unreachable!(),
        }
    }
}

impl MetadataMixin for GGUFPipeline {
    fn device(&self) -> Device {
        match self.model {
            Model::Llama(ref model) => model.device.clone(),
            Model::Phi2(ref model) => model.device.clone(),
            Model::XLoraLlama(ref model) => model.device.clone(),
            Model::Phi3(ref model) => model.device.clone(),
            Model::XLoraPhi3(ref model) => model.device.clone(),
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
impl Pipeline for GGUFPipeline {
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
            Model::Phi2(ref mut model) => model.forward(&input_ids, &seqlen_offsets, context_lens),
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
            Model::Phi3(ref mut model) => model.forward(&input_ids, &seqlen_offsets),
            Model::XLoraPhi3(ref mut model) => model.forward(
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
