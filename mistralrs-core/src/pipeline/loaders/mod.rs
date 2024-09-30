mod diffusion_loaders;
mod normal_loaders;
mod vision_loaders;

use std::{
    collections::HashMap,
    fmt::{self, Debug},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::Result;
use as_any::AsAny;
use candle_core::Device;
use mistralrs_quant::IsqType;
use tokio::sync::Mutex;

pub use normal_loaders::{
    AutoLoader, Gemma2Loader, GemmaLoader, LlamaLoader, MistralLoader, MixtralLoader,
    NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader, Phi2Loader,
    Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Starcoder2Loader,
};

pub use vision_loaders::{
    Idefics2Loader, LLaVALoader, LLaVANextLoader, Phi3VLoader, VLlamaLoader, VisionLoaderType,
    VisionModel, VisionModelLoader,
};

pub use diffusion_loaders::{
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, DiffusionModelPaths,
    DiffusionModelPathsInner, FluxLoader,
};

use crate::{
    lora::LoraConfig, xlora_models::XLoraConfig, DeviceMapMetadata, Ordering, PagedAttentionConfig,
    TryIntoDType,
};

use super::Pipeline;

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `LocalModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths: AsAny + Debug {
    /// Model weights files (multiple files supported).
    fn get_weight_filenames(&self) -> &[PathBuf];

    /// Retrieve the [`PretrainedConfig`] file.
    ///
    /// [`PretrainedConfig`]: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/configuration#transformers.PretrainedConfig
    fn get_config_filename(&self) -> &PathBuf;

    /// A serialised [`tokenizers.Tokenizer`] HuggingFace object.
    ///
    /// [`tokenizers.Tokenizer`]: https://huggingface.co/docs/transformers/v4.40.2/en/main_classes/tokenizer
    fn get_tokenizer_filename(&self) -> &PathBuf;

    /// File where the content is expected to deserialize to [`ChatTemplate`].
    ///
    /// [`ChatTemplate`]: crate::ChatTemplate
    fn get_template_filename(&self) -> &Option<PathBuf>;

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

#[derive(Clone, Debug)]
/// All local paths and metadata necessary to load a model.
pub struct LocalModelPaths<P: Debug> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub template_filename: Option<P>,
    pub filenames: Vec<P>,
    pub xlora_adapter_filenames: Option<Vec<(String, P)>>,
    pub xlora_adapter_configs: Option<Vec<((String, String), LoraConfig)>>,
    pub classifier_path: Option<P>,
    pub classifier_config: Option<XLoraConfig>,
    pub xlora_ordering: Option<Ordering>,
    pub gen_conf: Option<P>,
    pub lora_preload_adapter_info: Option<HashMap<String, (P, LoraConfig)>>,
    pub preprocessor_config: Option<P>,
    pub processor_config: Option<P>,
}

impl<P: Debug> LocalModelPaths<P> {
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
            template_filename: Some(template_filename),
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
    fn get_template_filename(&self) -> &Option<PathBuf> {
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

/// The kind of model to build.
#[derive(Clone, Default, derive_more::From, strum::Display)]
pub enum ModelKind {
    #[default]
    #[strum(to_string = "normal (no quant, no adapters)")]
    Normal,

    #[strum(to_string = "quantized from {quant} (no adapters)")]
    Quantized { quant: QuantizationKind },

    #[strum(to_string = "{adapter}, (no quant)")]
    Adapter { adapter: AdapterKind },

    #[strum(to_string = "{adapter}, quantized from {quant}")]
    AdapterQuantized {
        adapter: AdapterKind,
        quant: QuantizationKind,
    },

    #[strum(to_string = "speculative: target: `{target}`, draft: `{draft}`")]
    Speculative {
        target: Box<ModelKind>,
        draft: Box<ModelKind>,
    },

    #[strum(to_string = "anymoe: target: `{target}`")]
    AnyMoe { target: Box<ModelKind> },
}

#[derive(Clone, Copy, strum::Display, strum::EnumIs, strum::EnumMessage)]
#[strum(serialize_all = "kebab-case")]
pub enum QuantizationKind {
    /// GGML
    Ggml,
    /// GGUF
    Gguf,
    /// GPTQ
    Gptq,
}

#[derive(Clone, Copy, strum::Display, strum::EnumIs, strum::EnumMessage)]
#[strum(serialize_all = "kebab-case")]
pub enum AdapterKind {
    /// LoRA
    Lora,
    /// X-LoRA
    XLora,
}

// For the proper name as formatted via doc comment for a variant
pub trait PrettyName: strum::EnumMessage + ToString {
    fn pretty_name(&self) -> String {
        match self.get_documentation() {
            Some(s) => s.to_string(),
            // Instead of panic via expect(),
            // fallback to default kebab-case:
            None => self.to_string(),
        }
    }
}

impl PrettyName for AdapterKind {}
impl PrettyName for QuantizationKind {}

impl ModelKind {
    // Quantized helpers:
    pub fn is_quantized(&self) -> bool {
        self.quantized_kind().iter().any(|q| q.is_some())
    }

    pub fn is_quantized_and(&self, mut f: impl FnMut(QuantizationKind) -> bool) -> bool {
        self.quantized_kind().iter().any(|q| q.is_some_and(&mut f))
    }

    pub fn quantized_kind(&self) -> Vec<Option<QuantizationKind>> {
        use ModelKind::*;

        match self {
            Normal | Adapter { .. } => vec![None],
            Quantized { quant } | AdapterQuantized { quant, .. } => vec![Some(*quant)],
            Speculative { target, draft } => {
                let t = *target.clone();
                let d = *draft.clone();

                [t.quantized_kind(), d.quantized_kind()].concat()
            }
            AnyMoe { target } => target.quantized_kind(),
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
        use ModelKind::*;

        match self {
            Normal | Quantized { .. } => vec![None],
            Adapter { adapter } | AdapterQuantized { adapter, .. } => vec![Some(*adapter)],
            Speculative { target, draft } => {
                let t = *target.clone();
                let d = *draft.clone();

                [t.adapted_kind(), d.adapted_kind()].concat()
            }
            AnyMoe { target } => target.adapted_kind(),
        }
    }
}

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use mistralrs_core::{Loader, TokenSource, DeviceMapMetadata, ModelDType};
/// use candle_core::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     &ModelDType::Auto,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapMetadata::dummy(),
///     None,
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
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    /// Load a model from the specified paths.
    /// Also initializes `DEBUG`.
    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &Box<dyn ModelPaths>,
        dtype: &dyn TryIntoDType,
        device: &Device,
        silent: bool,
        mapper: DeviceMapMetadata,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    fn get_id(&self) -> String;
    fn get_kind(&self) -> ModelKind;
}
