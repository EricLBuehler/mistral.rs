pub(crate) mod auto_device_map;
mod diffusion_loaders;
mod embedding_loaders;
mod normal_loaders;
mod vision_loaders;
pub use auto_device_map::AutoDeviceMapParams;
use auto_device_map::NonMappedSubModel;

use std::{
    fmt::{self, Debug},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::Result;
use as_any::AsAny;
use candle_core::{DType, Device};
use mistralrs_quant::{IsqType, QuantizedConfig};
use serde::Deserialize;
use tokio::sync::Mutex;

pub use normal_loaders::{
    AutoNormalLoader, DeepSeekV2Loader, DeepSeekV3Loader, GLM4Loader, GLM4MoeLiteLoader,
    GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader, GraniteMoeHybridLoader, LlamaLoader,
    MistralLoader, MixtralLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel,
    NormalModelLoader, Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Qwen3Loader,
    Qwen3MoELoader, SmolLm3Loader, Starcoder2Loader,
};

pub use vision_loaders::{
    AutoVisionLoader, Gemma3Loader, Gemma3nLoader, Idefics2Loader, Idefics3Loader, LLaVALoader,
    LLaVANextLoader, MiniCpmOLoader, Mistral3Loader, Phi3VLoader, Phi4MMLoader, Qwen2VLLoader,
    Qwen2_5VLLoader, Qwen3VLLoader, Qwen3VLMoELoader, VLlama4Loader, VLlamaLoader,
    VisionLoaderType, VisionModel, VisionModelLoader,
};

pub use embedding_loaders::{
    AutoEmbeddingLoader, EmbeddingGemmaLoader, EmbeddingLoaderType, EmbeddingModel,
    EmbeddingModelLoader, EmbeddingModule, EmbeddingModulePaths, EmbeddingModuleType,
    Qwen3EmbeddingLoader,
};

pub use diffusion_loaders::{
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, DiffusionModelPaths,
    DiffusionModelPathsInner, FluxLoader,
};

use crate::{
    matformer::MatformerSliceConfig, paged_attention::ModelConfigLike, DeviceMapMetadata,
    DeviceMapSetting, PagedAttentionConfig, TryIntoDType,
};

use super::{paths::AdapterPaths, Pipeline};

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `LocalModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths: AsAny + Debug + Send + Sync {
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

    /// Filepath for general model configuration.
    fn get_gen_conf_filename(&self) -> Option<&PathBuf>;

    /// Get the preprocessor config (for the vision models). This is used to pre process images.
    fn get_preprocessor_config(&self) -> &Option<PathBuf>;

    /// Get the processor config (for the vision models). This is primarily used for the chat template.
    fn get_processor_config(&self) -> &Option<PathBuf>;

    /// Get the explicit chat template.
    fn get_chat_template_explicit(&self) -> &Option<PathBuf>;

    /// Get adapter paths.
    fn get_adapter_paths(&self) -> &AdapterPaths;

    /// Get embedding model `modules.json` compatible with sentence-transformers
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]>;
}

#[derive(Clone, Debug)]
/// All local paths and metadata necessary to load a model.
pub struct LocalModelPaths<P: Debug> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub template_filename: Option<P>,
    pub filenames: Vec<P>,
    pub adapter_paths: AdapterPaths,
    pub gen_conf: Option<P>,
    pub preprocessor_config: Option<P>,
    pub processor_config: Option<P>,
    pub chat_template_json_filename: Option<P>,
}

impl<P: Debug> LocalModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        template_filename: P,
        filenames: Vec<P>,
        adapter_paths: AdapterPaths,
        gen_conf: Option<P>,
        preprocessor_config: Option<P>,
        processor_config: Option<P>,
        chat_template_json_filename: Option<P>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            template_filename: Some(template_filename),
            filenames,
            adapter_paths,
            gen_conf,
            preprocessor_config,
            processor_config,
            chat_template_json_filename,
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
    fn get_template_filename(&self) -> &Option<PathBuf> {
        &self.template_filename
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        self.gen_conf.as_ref()
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        &self.preprocessor_config
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        &self.processor_config
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        &self.chat_template_json_filename
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        &self.adapter_paths
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        None
    }
}

#[derive(Clone, Debug)]
/// All local paths and metadata necessary to load an embedding model.
pub struct EmbeddingModelPaths<P: Debug> {
    pub tokenizer_filename: P,
    pub config_filename: P,
    pub modules: Vec<EmbeddingModulePaths>,
    pub filenames: Vec<P>,
    pub adapter_paths: AdapterPaths,
}

impl<P: Debug> EmbeddingModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        filenames: Vec<P>,
        adapter_paths: AdapterPaths,
        modules: Vec<EmbeddingModulePaths>,
    ) -> Self {
        Self {
            tokenizer_filename,
            config_filename,
            filenames,
            adapter_paths,
            modules,
        }
    }
}

impl ModelPaths for EmbeddingModelPaths<PathBuf> {
    fn get_config_filename(&self) -> &PathBuf {
        &self.config_filename
    }
    fn get_tokenizer_filename(&self) -> &PathBuf {
        &self.tokenizer_filename
    }
    fn get_weight_filenames(&self) -> &[PathBuf] {
        &self.filenames
    }
    fn get_template_filename(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_gen_conf_filename(&self) -> Option<&PathBuf> {
        None
    }
    fn get_preprocessor_config(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_processor_config(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_chat_template_explicit(&self) -> &Option<PathBuf> {
        &None
    }
    fn get_adapter_paths(&self) -> &AdapterPaths {
        &self.adapter_paths
    }
    fn get_modules(&self) -> Option<&[EmbeddingModulePaths]> {
        Some(&self.modules)
    }
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
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
            TokenSource::Literal(value) => write!(f, "literal:{value}"),
            TokenSource::EnvVar(value) => write!(f, "env:{value}"),
            TokenSource::Path(value) => write!(f, "path:{value}"),
            TokenSource::CacheToken => write!(f, "cache"),
            TokenSource::None => write!(f, "none"),
        }
    }
}

/// The kind of model to build.
#[derive(Clone, Default, derive_more::From, strum::Display)]
pub enum ModelKind {
    #[default]
    #[strum(to_string = "normal (no adapters)")]
    Normal,

    #[strum(to_string = "gguf quantized from {quant} (no adapters)")]
    GgufQuantized { quant: QuantizationKind },

    #[strum(to_string = "{adapter}")]
    Adapter { adapter: AdapterKind },

    #[strum(to_string = "{adapter}, gguf quantized from {quant}")]
    GgufAdapter {
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
            GgufQuantized { quant } | GgufAdapter { quant, .. } => vec![Some(*quant)],
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
            Normal | GgufQuantized { .. } => vec![None],
            Adapter { adapter } | GgufAdapter { adapter, .. } => vec![Some(*adapter)],
            Speculative { target, draft } => {
                let t = *target.clone();
                let d = *draft.clone();

                [t.adapted_kind(), d.adapted_kind()].concat()
            }
            AnyMoe { target } => target.adapted_kind(),
        }
    }
}

#[derive(Deserialize)]
pub struct QuantizationConfigShim {
    quantization_config: Option<QuantizedConfig>,
}

impl QuantizationConfigShim {
    pub fn get_quant_config_pack_factor(config: &str, dtype: DType) -> Result<usize> {
        let QuantizationConfigShim {
            quantization_config,
        } = serde_json::from_str(config)?;

        if let Some(quantization_config) = quantization_config {
            Ok(quantization_config.pack_factor(dtype))
        } else {
            Ok(1)
        }
    }
}

pub trait DeviceMappedModelLoader {
    /// Maximum activation size of non-mapped parts of this model.
    /// Useful for the vision models which may prefer to keep the vison components on the GPU.
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize>;
    /// Maximum activation size of mapped parts of the model
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize>;
    /// weight_pack_factor only applies to quantized weights.
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize>;
    /// weight_pack_factor only applies to quantized weights.
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>>;
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        None
    }
    fn num_layers(&self, config: &str) -> Result<usize>;
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>>;

    #[allow(clippy::too_many_arguments)]
    fn get_device_layers(
        &self,
        config: &str,
        num_layers: usize,
        layer_sizes_in_bytes: Vec<usize>,
        non_mapped_size_in_bytes: usize,
        total_model_size_in_bytes: usize,
        devices: &[Device],
        dtype: DType,
        params: &AutoDeviceMapParams,
        paged_attn_config: Option<&PagedAttentionConfig>,
    ) -> Result<DeviceMapMetadata>
    where
        Self: Sized,
    {
        auto_device_map::get_device_layers(
            self,
            config,
            num_layers,
            layer_sizes_in_bytes,
            non_mapped_size_in_bytes,
            total_model_size_in_bytes,
            devices,
            dtype,
            params,
            paged_attn_config,
        )
    }
}

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use mistralrs_core::{Loader, TokenSource, DeviceMapSetting, AutoDeviceMapParams, ModelDType};
/// use candle_core::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     &ModelDType::Auto,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapSetting::Auto(AutoDeviceMapParams::default_text()),
///     None,
///     None,
/// ).unwrap();
/// ```
pub trait Loader: Send + Sync {
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
        mapper: DeviceMapSetting,
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
        mapper: DeviceMapSetting,
        in_situ_quant: Option<IsqType>,
        paged_attn_config: Option<PagedAttentionConfig>,
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>>;

    fn get_id(&self) -> String;
    fn get_kind(&self) -> ModelKind;
}
