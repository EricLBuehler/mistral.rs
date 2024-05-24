mod cache_manager;
mod chat_template;
mod ggml;
mod gguf;
mod inputs_processor;
mod isq;
mod macros;
mod normal;
mod normal_loaders;
mod paths;
mod sampling;
mod speculative;
mod vision;
mod vision_loaders;
use crate::aici::toktree::TokTrie;
use crate::prefix_cacher::PrefixCacheManager;
mod sampling_pipeline;
use crate::lora::{LoraConfig, Ordering};
use crate::DeviceMapMetadata;
use candle_core::quantized::GgmlDType;
use chat_template::{apply_chat_template_to, ChatTemplate};
use core::fmt;
use either::Either;
pub use ggml::{GGMLLoader, GGMLLoaderBuilder, GGMLSpecificConfig};
pub use gguf::{GGUFLoader, GGUFLoaderBuilder, GGUFSpecificConfig};
use indexmap::IndexMap;
pub use isq::IsqModel;
pub use normal::{NormalLoader, NormalLoaderBuilder, NormalSpecificConfig};
pub use normal_loaders::{
    GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader, NormalLoaderType, NormalModelLoader,
    Phi2Loader, Phi3Loader, Qwen2Loader,
};
pub(crate) use paths::{get_model_paths, get_xlora_paths, XLoraPaths};
use rand_isaac::Isaac64Rng;
pub use speculative::{SpeculativeConfig, SpeculativeLoader, SpeculativePipeline};
use std::any::Any;
use std::fmt::{Debug, Display};
use std::sync::Arc;
use std::{collections::HashMap, path::PathBuf, str::FromStr};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
pub use vision::{VisionLoader, VisionLoaderBuilder, VisionSpecificConfig};
pub use vision_loaders::VisionModelLoader;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use crate::{
    sequence::Sequence,
    xlora_models::{NonGranularState, XLoraConfig},
};

pub use self::cache_manager::{Cache, CacheManager, LayerCaches};
pub use self::inputs_processor::{
    text_models_inputs_processor, InputsProcessor, InputsProcessorType,
};

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `LocalModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths {
    /// Model weights files (multiple files supported).
    fn get_weight_filenames(&self) -> &[PathBuf];

    /// Retrieve the PretrainedConfig file.
    /// See: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/configuration#transformers.PretrainedConfig
    fn get_config_filename(&self) -> &PathBuf;

    /// A serialised `tokenizers.Tokenizer` HuggingFace object.
    /// See: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/tokenizer
    fn get_tokenizer_filename(&self) -> &PathBuf;

    /// Jinja format chat templating for chat completion.
    /// See: https://huggingface.co/docs/transformers/chat_templating
    fn get_template_filename(&self) -> &PathBuf;

    /// Optional adapter files. `(String, PathBuf)` is of the form `(id name, path)`.
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>>;

    /// Configuration of optional adapters. `(String, String)` is of the form `(id name, name)`.
    fn get_adapter_configs(&self) -> &Option<Vec<((String, String), LoraConfig)>>;

    /// Filepath for the XLORA classifier
    fn get_classifier_path(&self) -> &Option<PathBuf>;

    /// `XLoraConfig` for the XLORA classifier
    fn get_classifier_config(&self) -> &Option<XLoraConfig>;

    /// Return the defined ordering of adapters and layers within the model.
    fn get_ordering(&self) -> &Option<Ordering>;

    /// Filepath for general model configuration.
    fn get_gen_conf_filename(&self) -> Option<&PathBuf>;

    /// Information for preloading LoRA adapters (adapter name, the weight file, and the config).
    fn get_lora_preload_adapter_info(&self) -> &Option<HashMap<String, (PathBuf, LoraConfig)>>;

    /// Get the preprocessor config (for the vision models). This is used to pre process images.
    fn get_preprocessor_config(&self) -> &Option<PathBuf>;

    /// Get the processor config (for the vision models). This is primarily used for the chat template.
    fn get_processor_config(&self) -> &Option<PathBuf>;
}

#[derive(Clone)]
pub struct LocalModelPaths<P> {
    tokenizer_filename: P,
    config_filename: P,
    template_filename: P,
    filenames: Vec<P>,
    xlora_adapter_filenames: Option<Vec<(String, P)>>,
    xlora_adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
    classifier_path: Option<P>,
    classifier_config: Option<XLoraConfig>,
    xlora_ordering: Option<Ordering>,
    gen_conf: Option<P>,
    lora_preload_adapter_info: Option<HashMap<String, (P, LoraConfig)>>,
    preprocessor_config: Option<P>,
    processor_config: Option<P>,
}

impl<P> LocalModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        template_filename: P,
        filenames: Vec<P>,
        xlora_adapter_filenames: Option<Vec<(String, P)>>,
        xlora_adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
        classifier_path: Option<P>,
        classifier_config: Option<XLoraConfig>,
        xlora_ordering: Option<Ordering>,
        gen_conf: Option<P>,
        lora_preload_adapter_info: Option<HashMap<String, (P, LoraConfig)>>,
        preprocessor_config: Option<P>,
        processor_config: Option<P>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            template_filename,
            filenames,
            xlora_adapter_filenames,
            xlora_adapter_configs,
            classifier_path,
            classifier_config,
            xlora_ordering,
            gen_conf,
            lora_preload_adapter_info,
            preprocessor_config,
            processor_config,
        }
    }
}

impl ModelPaths for LocalModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
    fn get_adapter_filenames(&self) -> &Option<Vec<(String, PathBuf)>> {
        &self.xlora_adapter_filenames
    }
    fn get_adapter_configs(&self) -> &Option<Vec<((String, String), LoraConfig)>> {
        &self.xlora_adapter_configs
    }
    fn get_classifier_config(&self) -> &Option<XLoraConfig> {
        &self.classifier_config
    }
    fn get_classifier_path(&self) -> &Option<PathBuf> {
        &self.classifier_path
    }
    fn get_ordering(&self) -> &Option<Ordering> {
        &self.xlora_ordering
    }
    fn get_template_filename(&self) -> &PathBuf {
        &self.template_filename
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        self.gen_conf.as_ref()
    }
    fn get_lora_preload_adapter_info(&self) -> &Option<HashMap<String, (PathBuf, LoraConfig)>> {
        &self.lora_preload_adapter_info
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        &self.preprocessor_config
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        &self.processor_config
    }
}

#[derive(Debug, Clone)]
/// The source of the HF token.
pub enum TokenSource {
    Literal(String),
    EnvVar(String),
    Path(String),
    CacheToken,
    None,
}

impl FromStr for TokenSource {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.splitn(2, ':').collect();
        match parts[0] {
            "literal" => parts
                .get(1)
                .map(|&value| TokenSource::Literal(value.to_string()))
                .ok_or_else(|| "Expected a value for 'literal'".to_string()),
            "env" => Ok(TokenSource::EnvVar(
                parts
                    .get(1)
                    .unwrap_or(&"HUGGING_FACE_HUB_TOKEN")
                    .to_string(),
            )),
            "path" => parts
                .get(1)
                .map(|&value| TokenSource::Path(value.to_string()))
                .ok_or_else(|| "Expected a value for 'path'".to_string()),
            "cache" => Ok(TokenSource::CacheToken),
            "none" => Ok(TokenSource::None),
            _ => Err("Invalid token source format".to_string()),
        }
    }
}

impl fmt::Display for TokenSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenSource::Literal(value) => write!(f, "literal:{}", value),
            TokenSource::EnvVar(value) => write!(f, "env:{}", value),
            TokenSource::Path(value) => write!(f, "path:{}", value),
            TokenSource::CacheToken => write!(f, "cache"),
            TokenSource::None => write!(f, "none"),
        }
    }
}

#[derive(Clone, Default)]
/// The kind of model to build.
pub enum ModelKind {
    #[default]
    Normal,
    XLoraNormal,
    XLoraGGUF,
    XLoraGGML,
    QuantizedGGUF,
    QuantizedGGML,
    LoraGGUF,
    LoraGGML,
    LoraNormal,
    Speculative {
        target: Box<ModelKind>,
        draft: Box<ModelKind>,
    },
}

// TODO: Future replacement for `ModelKind` above:
#[derive(Default, derive_more::From, strum::Display)]
pub enum ModelKindB {
    #[default]
    #[strum(to_string = "normal (no quant, no adapters)")]
    Plain,

    #[strum(to_string = "quantized from {quant} (no adapters)")]
    Quantized { quant: QuantizationKind },

    #[strum(to_string = "{adapter}, (no quant)")]
    Adapter { adapter: AdapterKind },

    #[strum(to_string = "{adapter}, quantized from {quant}")]
    AdapterQuantized {
        adapter: AdapterKind,
        quant: QuantizationKind,
    },

    // TODO: This would need to be later changed to reference `Self`, but this current way
    // avoids having to handle the conversion logic with `ModelKind`.
    #[strum(to_string = "speculative: target: `{target}`, draft: `{draft}`")]
    Speculative {
        target: Box<ModelKind>,
        draft: Box<ModelKind>,
    },
}

#[derive(Clone, Copy, strum::Display, strum::EnumIs)]
#[strum(serialize_all = "kebab-case")]
pub enum QuantizationKind {
    Ggml,
    Gguf,
}

#[derive(Clone, Copy, strum::Display, strum::EnumIs)]
#[strum(serialize_all = "kebab-case")]
pub enum AdapterKind {
    Lora,
    XLora,
}

impl ModelKindB {
    // Quantized helpers:
    pub fn is_quantized(&self) -> bool {
        self.quantized_kind().iter().any(|q| q.is_some())
    }

    pub fn is_quantized_and(&self, mut f: impl FnMut(QuantizationKind) -> bool) -> bool {
        self.quantized_kind().iter().any(|q| q.is_some_and(&mut f))
    }

    pub fn quantized_kind(&self) -> Vec<Option<QuantizationKind>> {
        use ModelKindB::*;

        match self {
            Plain | Adapter { .. } => vec![None],
            Quantized { quant } | AdapterQuantized { quant, .. } => vec![Some(*quant)],
            Speculative { target, draft } => {
                let t = ModelKindB::from(*target.clone());
                let d = ModelKindB::from(*draft.clone());

                [t.quantized_kind(), d.quantized_kind()].concat()
            }
        }
    }

    // Adapter helpers:
    pub fn is_adapted(&self) -> bool {
        self.adapted_kind().iter().any(|a| a.is_some())
    }

    pub fn is_adapted_and(&self, mut f: impl FnMut(AdapterKind) -> bool) -> bool {
        self.adapted_kind().iter().any(|a| a.is_some_and(&mut f))
    }

    pub fn adapted_kind(&self) -> Vec<Option<AdapterKind>> {
        use ModelKindB::*;

        match self {
            Plain | Quantized { .. } => vec![None],
            Adapter { adapter } | AdapterQuantized { adapter, .. } => vec![Some(*adapter)],
            Speculative { target, draft } => {
                let t = ModelKindB::from(*target.clone());
                let d = ModelKindB::from(*draft.clone());

                [t.adapted_kind(), d.adapted_kind()].concat()
            }
        }
    }
}

// TODO: Temporary compatibility layers follow (until a future PR follow-up introduces a breaking change)
impl Display for ModelKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", ModelKindB::from(self.clone()))
    }
}

// Delegate to `ModelKindB` methods:
impl ModelKind {
    // Quantized helpers:
    pub fn is_quantized(&self) -> bool {
        let k = ModelKindB::from(self.clone());
        k.is_quantized()
    }

    pub fn is_quantized_and(&self, f: impl FnMut(QuantizationKind) -> bool) -> bool {
        let k = ModelKindB::from(self.clone());
        k.is_quantized_and(f)
    }

    pub fn quantized_kind(&self) -> Vec<Option<QuantizationKind>> {
        let k = ModelKindB::from(self.clone());
        k.quantized_kind()
    }

    // Adapter helpers:
    pub fn is_adapted(&self) -> bool {
        let k = ModelKindB::from(self.clone());
        k.is_adapted()
    }

    pub fn is_adapted_and(&self, f: impl FnMut(AdapterKind) -> bool) -> bool {
        let k = ModelKindB::from(self.clone());
        k.is_adapted_and(f)
    }

    pub fn adapted_kind(&self) -> Vec<Option<AdapterKind>> {
        let k = ModelKindB::from(self.clone());
        k.adapted_kind()
    }
}

impl From<ModelKind> for ModelKindB {
    fn from(kind: ModelKind) -> Self {
        match kind {
            ModelKind::Normal => ModelKindB::Plain,
            ModelKind::QuantizedGGML => (QuantizationKind::Ggml).into(),
            ModelKind::QuantizedGGUF => (QuantizationKind::Gguf).into(),
            ModelKind::XLoraNormal => (AdapterKind::XLora).into(),
            ModelKind::XLoraGGML => (AdapterKind::XLora, QuantizationKind::Ggml).into(),
            ModelKind::XLoraGGUF => (AdapterKind::XLora, QuantizationKind::Gguf).into(),
            ModelKind::LoraNormal => (AdapterKind::Lora).into(),
            ModelKind::LoraGGML => (AdapterKind::Lora, QuantizationKind::Ggml).into(),
            ModelKind::LoraGGUF => (AdapterKind::Lora, QuantizationKind::Gguf).into(),
            ModelKind::Speculative { target, draft } => (target, draft).into(),
        }
    }
}

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use mistralrs_core::{Loader, TokenSource, DeviceMapMetadata};
/// use candle_core::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     None,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapMetadata::dummy(),
///     None,
/// ).unwrap();
/// ```
pub trait Loader {
    /// If `revision` is None, then it defaults to `main`.
    /// If `dtype` is None, then it defaults to the model default (usually BF16).
    /// If model is not found on HF, will attempt to resolve locally.
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    fn load_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        _dtype: Option<DType>,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<GgmlDType>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    fn get_id(&self) -> String;
    fn get_kind(&self) -> ModelKind;
}

#[derive(Clone)]
pub struct GeneralMetadata {
    pub max_seq_len: usize,
    pub repeat_last_n: usize,
    pub tok_trie: Arc<TokTrie>,
    pub has_no_kv_cache: bool,
    pub is_xlora: bool,
    pub num_hidden_layers: usize,
    pub eos_tok: Vec<u32>,
    pub is_lora: bool,
}

pub enum AdapterInstruction {
    Activate(Vec<String>),
    None,
}

pub enum CacheInstruction {
    In(AdapterInstruction),
    Out,
    Reset {
        reset_non_granular: bool,
        adapter_inst: AdapterInstruction,
    },
    Nothing(AdapterInstruction),
}

pub trait PreProcessingMixin: MetadataMixin {
    fn apply_chat_template(
        &self,
        messages: Vec<IndexMap<String, String>>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let chat_template = self.get_chat_template();
        let template = chat_template.chat_template.as_ref().unwrap();
        let bos_tok = if let Some(ref bos) = self.get_chat_template().bos_token {
            match bos.0 {
                Either::Left(ref lit) => Some(lit.to_string()),
                Either::Right(ref added) => Some(added.content.to_string()),
            }
        } else {
            None
        };
        let eos_tok = match chat_template.eos_token {
            Either::Left(ref lit) => lit,
            Either::Right(ref added) => &added.content,
        };
        let unk_tok = if let Some(ref unk) = self.get_chat_template().unk_token {
            match unk.0 {
                Either::Left(ref lit) => Some(lit.to_string()),
                Either::Right(ref added) => Some(added.content.to_string()),
            }
        } else {
            None
        };
        apply_chat_template_to(
            messages,
            add_generation_prompt,
            template,
            bos_tok,
            eos_tok,
            unk_tok,
        )
    }
    fn get_chat_template(&self) -> Arc<ChatTemplate>;
    fn get_input_processor(&self) -> Box<dyn InputsProcessor>;
    fn tokenize_prompt(&self, prompt: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn get_input_processor_config(&self) -> Option<Arc<dyn Any>>;
}

pub trait IsqPipelineMixin {
    fn re_isq_model(&mut self, dtype: GgmlDType) -> Result<()>;
}

pub trait CacheManagerMixin {
    /// Clone the cache FROM the sequences' cache TO the model cache. Only called for completion seqs.
    /// It is not a guarantee that this will be called for each completion step.
    fn clone_in_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    /// Clone the cache FROM the model cache TO the sequences. Called for prompt and completion seqs.
    /// It is not a guarantee that this will be called for each step.
    fn clone_out_cache(&mut self, seqs: &mut [&mut Sequence], modify_draft_cache: bool);
    /// Set the model cache to all None. Only called for prompt seqs.
    /// It is not a guarantee that this will be called for each prompt step.
    /// This may also reset the non granular state if applicable.
    fn set_none_cache(&mut self, reset_non_granular: bool, modify_draft_cache: bool);
    fn cache(&self) -> &Cache;
}

pub trait AdapterActivationMixin {
    /// Returns the number of activated adapters.
    fn activate_adapters(&mut self, adapters: Vec<String>) -> Result<usize>;
}

pub trait MetadataMixin {
    fn device(&self) -> Device;
    fn tokenizer(&self) -> Arc<Tokenizer>;
    fn name(&self) -> String;
    fn reset_non_granular_state(&self);
    fn get_metadata(&self) -> &GeneralMetadata;
}

#[derive(PartialEq, Copy, Clone)]
pub enum ModelCategory {
    Text,
    Vision,
}

#[async_trait::async_trait]
pub trait Pipeline:
    Send
    + Sync
    + PreProcessingMixin
    + IsqPipelineMixin
    + CacheManagerMixin
    + AdapterActivationMixin
    + MetadataMixin
{
    fn forward_inputs(&mut self, inputs: Box<dyn Any>) -> Result<Tensor, candle_core::Error>;

    #[allow(clippy::too_many_arguments)]
    async fn step(
        &mut self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
        pre_op: CacheInstruction,
        post_op: CacheInstruction,
    ) -> Result<(), candle_core::Error> {
        let inputs = self
            .get_input_processor()
            .process_inputs(
                input_seqs,
                is_prompt,
                self.get_metadata().is_xlora,
                &self.device(),
                self.get_metadata().has_no_kv_cache,
                None,
                self.get_input_processor_config(),
            )
            .unwrap();

        match pre_op {
            CacheInstruction::In(adapter_inst) => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
                self.clone_in_cache(input_seqs, false)
            }
            CacheInstruction::Nothing(adapter_inst) => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
            }
            CacheInstruction::Reset {
                reset_non_granular,
                adapter_inst,
            } => {
                match adapter_inst {
                    AdapterInstruction::Activate(adapters) => {
                        self.activate_adapters(adapters).map_err(|e| {
                            candle_core::Error::msg(<anyhow::Error as AsRef<
                                dyn std::error::Error,
                            >>::as_ref(&e))
                        })?
                    }
                    AdapterInstruction::None => 0,
                };
                self.set_none_cache(reset_non_granular, false)
            }
            _ => unreachable!("Unreachable PRE cache op."),
        }

        let logits = self.forward_inputs(Box::new(inputs))?;

        match post_op {
            CacheInstruction::Out => self.clone_out_cache(input_seqs, false),
            CacheInstruction::Nothing(_) => (),
            CacheInstruction::Reset {
                reset_non_granular,
                adapter_inst: _,
            } => self.set_none_cache(reset_non_granular, false),
            _ => unreachable!("Unreachable POST cache op."),
        }

        self.sample(input_seqs, logits, prefix_cacher, disable_eos_stop, rng)
            .await?;
        Ok(())
    }

    async fn sample(
        &self,
        seqs: &mut [&mut Sequence],
        logits: Tensor,
        prefix_cacher: &mut PrefixCacheManager,
        disable_eos_stop: bool,
        rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    ) -> Result<(), candle_core::Error>;

    fn category(&self) -> ModelCategory;
}

pub trait NormalModel: IsqModel {
    fn forward(
        &mut self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &mut self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
    fn activate_adapters(&mut self, _: Vec<String>) -> candle_core::Result<usize> {
        candle_core::bail!("Unable to activate adapters for model without adapters");
    }
}

pub trait VisionModel: IsqModel {
    // pixel_values and pixel_attention_mask only specified for prompt seqs
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &mut self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        pixel_attention_mask: Option<Tensor>,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
}

pub(crate) fn extract_logits(
    logits: &Tensor,
    context_lens: Vec<(usize, usize)>,
) -> candle_core::Result<Tensor> {
    let mut toks = Vec::new();
    for (dim, (start, len)) in logits.chunk(logits.dims()[0], 0)?.iter().zip(context_lens) {
        toks.push(dim.narrow(1, start, len)?);
    }
    Tensor::cat(&toks, 0)
}

mod tests {
    #[test]
    /// Generating these cases:
    /// ```py
    /// >>> t=transformers.AutoTokenizer.from_pretrained(...)
    /// # If non-system prompt model
    /// >>> t.apply_chat_template([{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// # If system prompt model
    /// >>> t.apply_chat_template([{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"Hello"},{"role":"assistant","content":"Hi there"},{"role":"user","content":"Who are you"},{"role":"assistant","content":"   I am an assistant   "},{"role":"user","content":"Another question"}], add_generation_prompt=True, tokenize=False)
    /// ```
    fn test_chat_templates() {
        use indexmap::IndexMap;

        use crate::pipeline::apply_chat_template_to;
        let templates = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            (true, "<s>", "</s>", "<unk>", "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"),
            // mistralai/Mistral-7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // meta-llama/Llama-2-13b-chat-hf
            (true, "<s>", "</s>", "<unk>", "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if loop.index0 == 0 and system_message != false %}{% set content = '<<SYS>>\\n' + system_message + '\\n<</SYS>>\\n\\n' + message['content'] %}{% else %}{% set content = message['content'] %}{% endif %}{% if message['role'] == 'user' %}{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' '  + content.strip() + ' ' + eos_token }}{% endif %}{% endfor %}"),
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            (false, "<s>", "</s>", "<unk>", "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token}}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}"),
            // google/gemma-7b-it
            (false, "<bos>", "<eos>", "<unk>", "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"),
        ];
        let expected_outputs = [
            // ChatML: https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\nHi there<|im_end|>\n<|im_start|>user\nWho are you<|im_end|>\n<|im_start|>assistant\n   I am an assistant   <|im_end|>\n<|im_start|>user\nAnother question<|im_end|>\n<|im_start|>assistant\n",
            // mistralai/Mistral-7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s> [INST] Who are you [/INST]   I am an assistant   </s> [INST] Another question [/INST]",
            // meta-llama/Llama-2-13b-chat-hf
            "<s>[INST] <<SYS>>\nYou are a helpful assistant\n<</SYS>>\n\nHello [/INST] Hi there </s><s>[INST] Who are you [/INST] I am an assistant </s><s>[INST] Another question [/INST]",
            // mistralai/Mixtral-8x7B-Instruct-v0.1
            "<s>[INST] Hello [/INST]Hi there</s>[INST] Who are you [/INST]   I am an assistant   </s>[INST] Another question [/INST]",
            // google/gemma-7b-it
            "<bos><start_of_turn>user\nHello<end_of_turn>\n<start_of_turn>model\nHi there<end_of_turn>\n<start_of_turn>user\nWho are you<end_of_turn>\n<start_of_turn>model\nI am an assistant<end_of_turn>\n<start_of_turn>user\nAnother question<end_of_turn>\n<start_of_turn>model\n",
        ];
        let messages = [
            ["system", "You are a helpful assistant"],
            ["user", "Hello"],
            ["assistant", "Hi there"],
            ["user", "Who are you"],
            ["assistant", "   I am an assistant   "],
            ["user", "Another question"],
        ];
        let mut inputs = Vec::new();
        for [role, content] in messages {
            let mut message = IndexMap::new();
            message.insert("role".to_string(), role.to_string());
            message.insert("content".to_string(), content.to_string());
            inputs.push(message);
        }
        for ((i, (has_system, bos, eos, unk, template)), expected) in
            templates.into_iter().enumerate().zip(expected_outputs)
        {
            let output = apply_chat_template_to(
                if !has_system {
                    inputs[1..].to_vec()
                } else {
                    inputs.clone()
                },
                true,
                template,
                Some(bos.to_string()),
                eos,
                Some(unk.to_string()),
            )
            .unwrap_or_else(|_| panic!("Template number {i}"));
            assert_eq!(output, expected, "Template number {i}");
        }
    }
}
