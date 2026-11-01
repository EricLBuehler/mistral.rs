pub(crate) mod activation_memory;
pub(crate) mod auto_device_map;
mod checkpoint_inventory;
mod diffusion_loaders;
mod embedding_loaders;
mod multimodal_loaders;
mod normal_loaders;
mod qwen_vision_memory;
mod resources;
pub(crate) use activation_memory::{
    apply_model_activation_memory, estimate_model_activation_memory,
    reserve_model_activation_memory,
};
pub use activation_memory::{
    ActivationMemoryContext, ActivationMemoryEstimate, ActivationModelConfig,
};
pub use auto_device_map::AutoDeviceMapParams;
use auto_device_map::NonMappedSubModel;
pub(crate) use checkpoint_inventory::{checkpoint_device_map_sizes, checkpoint_runtime_size};
pub(crate) use resources::add_resource_bytes;
#[cfg(any(feature = "cuda", test))]
pub use resources::HostBacking;
pub use resources::{CheckpointTensors, LayerResource, ResourceSource};

use std::{
    fmt::{self, Debug},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::Result;
use as_any::AsAny;
use candle_core::{DType, Device};
use mistralrs_quant::{IsqType, QuantizedConfig, QuantizedWeightSource};
use serde::Deserialize;
use tokio::sync::Mutex;

pub use normal_loaders::{
    AutoNormalLoader, DeepSeekV2Loader, DeepSeekV3Loader, FlintLoader, GLM4Loader,
    GLM4MoeLiteLoader, GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader,
    GraniteMoeHybridLoader, HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, LlamaLoader,
    MistralLoader, MixtralLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel,
    NormalModelLoader, Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Qwen3Loader,
    Qwen3MoELoader, Qwen3NextLoader, Qwen3_5TextLoader, SmolLm3Loader, Starcoder2Loader,
};

pub use multimodal_loaders::{
    AutoMultimodalLoader, DiffusionGemmaLoader, Gemma3Loader, Gemma3nLoader, Gemma4Loader,
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Lfm2VlLoader, MiniCpmOLoader,
    Mistral3Loader, MultimodalLoaderType, MultimodalModel, MultimodalModelLoader,
    MuseGlimmerLoader, Phi3VLoader, Phi4MMLoader, Qwen2VLLoader, Qwen2_5VLLoader, Qwen3VLLoader,
    Qwen3VLMoELoader, Qwen3_5Loader, Qwen3_5MoeLoader, Qwen4ExpLoader, VLlama4Loader, VLlamaLoader,
    VoxtralLoader,
};

pub use embedding_loaders::{
    AutoEmbeddingLoader, EmbeddingGemmaLoader, EmbeddingLoaderType, EmbeddingModel,
    EmbeddingModelLoader, EmbeddingModule, EmbeddingModulePaths, EmbeddingModuleType,
    Qwen3EmbeddingLoader,
};

pub use diffusion_loaders::{
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, FluxLoader,
};

use crate::{
    matformer::MatformerSliceConfig, paged_attention::ModelConfigLike, DeviceMapMetadata,
    DeviceMapSetting, PagedAttentionConfig, Topology, TryIntoDType,
};

use super::{paths::AdapterPaths, Pipeline};

pub(crate) const QK_ROPE_LAYOUT_CONFIG_KEY: &str = "_mistralrs_qk_rope_layout";
const LEGACY_MODEL_OPT_CONFIG: &str = "hf_quant_config.json";
const LFM3_CODEC_CONFIG: &str = "audio_tokenizer/config.json";
const MODEL_COMPANION_FILES: &[&str] = &[
    "model.safetensors.index.json",
    "tokenizer_config.json",
    "hf_quant_config.json",
    "calibration.json",
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors.index.json",
    "modules.json",
    "config_sentence_transformers.json",
    mistralrs_quant::UQFF_REPORT_JSON,
];
/// Set on the model config JSON when the checkpoint's built-in MTP head should be loaded.
pub const MTP_CONFIG_KEY: &str = "_mistralrs_mtp";

pub(crate) fn load_model_config(
    config_path: &std::path::Path,
    use_checkpoint_quantization: bool,
) -> Result<String> {
    let config = std::fs::read_to_string(config_path)?;
    let config = inject_auxiliary_model_configs(&config, config_path)?;
    if !use_checkpoint_quantization {
        return Ok(config);
    }
    let config = normalize_compression_config(&config)?;
    let Some(parent) = config_path.parent() else {
        return Ok(config);
    };
    let model_opt_path = parent.join(LEGACY_MODEL_OPT_CONFIG);
    if !model_opt_path.is_file() {
        return Ok(config);
    }
    let model_opt = std::fs::read_to_string(model_opt_path)?;
    inject_legacy_model_opt_config(&config, &model_opt)
}

fn inject_auxiliary_model_configs(config: &str, config_path: &std::path::Path) -> Result<String> {
    let mut value: serde_json::Value = serde_json::from_str(config)?;
    if value
        .get("architectures")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|architectures| {
            architectures
                .iter()
                .any(|architecture| architecture.as_str() == Some("D1Model"))
        })
    {
        crate::vision_models::lfm3_d1::config::inject_calibration(&mut value, config_path)?;
        return Ok(serde_json::to_string(&value)?);
    }
    match value.get("model_type").and_then(serde_json::Value::as_str) {
        Some("lfm3_audio") => {
            crate::vision_models::lfm3_audio::config::normalize_config(&mut value)?;
        }
        Some("lfm3_omni") => {}
        _ => return Ok(config.to_string()),
    }
    if value
        .get("codec_config")
        .is_some_and(|config| !config.is_null())
    {
        return Ok(serde_json::to_string(&value)?);
    }
    let codec_path = config_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."))
        .join(LFM3_CODEC_CONFIG);
    let codec: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&codec_path).map_err(|error| {
            anyhow::anyhow!(
                "Cannot load LFM3 codec config {}: {error}",
                codec_path.display()
            )
        })?)?;
    value["codec_config"] = codec;
    Ok(serde_json::to_string(&value)?)
}

fn normalize_compression_config(config: &str) -> Result<String> {
    let mut value: serde_json::Value = serde_json::from_str(config)?;
    if canonicalize_quantization_config(&mut value)? {
        Ok(serde_json::to_string(&value)?)
    } else {
        Ok(config.to_string())
    }
}

fn canonicalize_quantization_config(value: &mut serde_json::Value) -> Result<bool> {
    let object = value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    let root_quantization = object
        .get("quantization_config")
        .filter(|value| !value.is_null())
        .cloned();
    let text_quantization = object
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .and_then(|text| text.get("quantization_config"))
        .filter(|value| !value.is_null())
        .cloned();
    let compression = object
        .get("compression_config")
        .filter(|value| !value.is_null())
        .cloned();
    let text_compression = object
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .and_then(|text| text.get("compression_config"))
        .filter(|value| !value.is_null())
        .cloned();
    let Some(effective) = root_quantization
        .or(text_quantization)
        .or(compression)
        .or(text_compression)
    else {
        return Ok(false);
    };

    let mut changed = false;
    if object
        .get("quantization_config")
        .is_none_or(serde_json::Value::is_null)
    {
        object.insert("quantization_config".to_string(), effective.clone());
        changed = true;
    }
    if let Some(text) = object
        .get_mut("text_config")
        .and_then(serde_json::Value::as_object_mut)
    {
        if text.get("quantization_config") != Some(&effective) {
            text.insert("quantization_config".to_string(), effective);
            changed = true;
        }
    }
    Ok(changed)
}

fn inject_legacy_model_opt_config(config: &str, model_opt: &str) -> Result<String> {
    let mut config_value: serde_json::Value = serde_json::from_str(config)?;
    let canonicalized = canonicalize_quantization_config(&mut config_value)?;
    let config_object = config_value
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    let embedded = config_object
        .get("quantization_config")
        .filter(|value| !value.is_null())
        .cloned();
    if embedded.as_ref().is_some_and(|value| {
        !value
            .get("quant_method")
            .and_then(serde_json::Value::as_str)
            .is_some_and(|method| method.to_ascii_lowercase().starts_with("modelopt"))
    }) {
        return if canonicalized {
            Ok(serde_json::to_string(&config_value)?)
        } else {
            Ok(config.to_string())
        };
    }

    let model_opt_value: serde_json::Value = serde_json::from_str(model_opt)?;
    let quantization = match embedded.as_ref() {
        Some(embedded) => QuantizedConfig::from_modelopt_configs(Some(embedded), &model_opt_value),
        None => QuantizedConfig::from_modelopt_config(&model_opt_value),
    }
    .map_err(anyhow::Error::msg)?;
    config_object.insert(
        "quantization_config".to_string(),
        serde_json::to_value(quantization)?,
    );
    canonicalize_quantization_config(&mut config_value)?;
    Ok(serde_json::to_string(&config_value)?)
}

pub(crate) fn inject_mtp_config_flag(config: &str) -> anyhow::Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    object.insert(MTP_CONFIG_KEY.to_string(), serde_json::Value::Bool(true));
    Ok(config.to_string())
}

pub(crate) fn qk_rope_layout_from_config(
    config: &str,
) -> Result<Option<crate::gguf::normal_registry::RopePairing>> {
    let config: serde_json::Value = serde_json::from_str(config)?;
    let Some(layout) = config
        .get(QK_ROPE_LAYOUT_CONFIG_KEY)
        .and_then(serde_json::Value::as_str)
    else {
        return Ok(None);
    };
    match layout {
        "adjacent" => Ok(Some(
            crate::gguf::normal_registry::RopePairing::Adjacent,
        )),
        "half_split" => Ok(Some(
            crate::gguf::normal_registry::RopePairing::HalfSplit,
        )),
        layout => anyhow::bail!(
            "model config `{QK_ROPE_LAYOUT_CONFIG_KEY}` must be `adjacent` or `half_split`, got `{layout}`"
        ),
    }
}

pub(crate) fn stamp_qk_rope_layout(
    config: &str,
    layout: crate::gguf::normal_registry::RopePairing,
) -> Result<String> {
    let mut config: serde_json::Value = serde_json::from_str(config)?;
    let object = config
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("model config must be a JSON object"))?;
    object.insert(
        QK_ROPE_LAYOUT_CONFIG_KEY.to_string(),
        serde_json::Value::String(
            match layout {
                crate::gguf::normal_registry::RopePairing::Adjacent => "adjacent",
                crate::gguf::normal_registry::RopePairing::HalfSplit => "half_split",
            }
            .to_string(),
        ),
    );
    Ok(serde_json::to_string(&config)?)
}

pub(crate) fn validate_lora_qk_rope_layout(config: &str, has_adapter: bool) -> Result<()> {
    if has_adapter
        && qk_rope_layout_from_config(config)?
            == Some(crate::gguf::normal_registry::RopePairing::Adjacent)
    {
        anyhow::bail!(
            "LoRA and X-LoRA adapters are not supported when Q/K tensors use adjacent RoPE layout; load the original safetensors model or omit the adapter"
        );
    }
    Ok(())
}

/// `ModelPaths` abstracts the mechanism to get all necessary files for running a model. For
/// example `LocalModelPaths` implements `ModelPaths` when all files are in the local file system.
pub trait ModelPaths: AsAny + Debug + Send + Sync {
    fn files(&self) -> Vec<PathBuf> {
        let mut files = self.get_weight_filenames().to_vec();
        files.extend([
            self.get_config_filename().clone(),
            self.get_tokenizer_filename().clone(),
        ]);
        files.extend(self.get_template_filename().iter().cloned());
        files.extend(self.get_gen_conf_filename().cloned());
        files.extend(self.get_preprocessor_config().iter().cloned());
        files.extend(self.get_video_preprocessor_config().cloned());
        files.extend(self.get_processor_config().iter().cloned());
        files.extend(self.get_chat_template_explicit().iter().cloned());
        match self.get_adapter_paths() {
            AdapterPaths::Lora(adapters) => {
                for adapter in adapters {
                    files.extend([adapter.config_path.clone(), adapter.weights_path.clone()]);
                }
            }
            AdapterPaths::XLora {
                metadata_paths,
                adapter_safetensors,
                classifier_path,
                lora_preload_adapter_info,
                ..
            } => {
                files.extend(metadata_paths.iter().cloned());
                if let Some(adapters) = adapter_safetensors {
                    files.extend(adapters.iter().map(|(_, path)| path.clone()));
                }
                files.extend(classifier_path.iter().cloned());
                if let Some(adapters) = lora_preload_adapter_info {
                    files.extend(adapters.values().map(|(path, _)| path.clone()));
                }
            }
            AdapterPaths::None => {}
        }
        for module in self.get_modules().unwrap_or_default() {
            match module {
                EmbeddingModulePaths::Pooling { config, .. } => files.push(config.clone()),
                EmbeddingModulePaths::Dense { config, model, .. } => {
                    files.extend([config.clone(), model.clone()]);
                }
                _ => {}
            }
        }
        if let Some(parent) = self.get_config_filename().parent() {
            files.extend(
                MODEL_COMPANION_FILES
                    .iter()
                    .map(|filename| parent.join(filename))
                    .filter(|path| path.is_file()),
            );
        }
        files.retain(|path| !path.as_os_str().is_empty());
        files
    }

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

    /// Get the preprocessor config (for the multimodal models). This is used to pre process images.
    fn get_preprocessor_config(&self) -> &Option<PathBuf>;

    /// Get the video preprocessor config, for multimodal models that ship separate video settings.
    fn get_video_preprocessor_config(&self) -> Option<&PathBuf> {
        None
    }

    /// Get the processor config (for the multimodal models). This is primarily used for the chat template.
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
    pub video_preprocessor_config: Option<P>,
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
            video_preprocessor_config: None,
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
    fn get_video_preprocessor_config(&self) -> Option<&PathBuf> {
        self.video_preprocessor_config.as_ref()
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

impl<P: Debug + AsRef<std::path::Path>> EmbeddingModelPaths<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        tokenizer_filename: P,
        config_filename: P,
        mut filenames: Vec<P>,
        adapter_paths: AdapterPaths,
        modules: Vec<EmbeddingModulePaths>,
    ) -> Self {
        filenames.retain(|filename| {
            !modules.iter().any(|module| {
                matches!(module, EmbeddingModulePaths::Dense { model, .. } if model == filename.as_ref())
            })
        });
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
            AnyMoe { target } => target.adapted_kind(),
        }
    }
}

#[derive(Deserialize)]
pub struct QuantizationConfigShim {
    quantization_config: Option<QuantizedConfig>,
}

impl QuantizationConfigShim {
    pub fn get_quantization_config(config: &str) -> Result<Option<QuantizedConfig>> {
        Ok(serde_json::from_str::<Self>(config)?.quantization_config)
    }

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

#[derive(Clone, Copy)]
pub struct AutoDeviceMapQuantization<'a> {
    source: AutoDeviceMapQuantizationSource<'a>,
    topology: Option<&'a Topology>,
    checkpoint: Option<&'a CheckpointTensors<'a>>,
    source_quantization: Option<&'a QuantizedConfig>,
}

#[derive(Clone, Copy)]
enum AutoDeviceMapQuantizationSource<'a> {
    Isq(Option<IsqType>),
    WeightSource(&'a dyn QuantizedWeightSource),
    Checkpoint(&'a QuantizedConfig),
}

impl<'a> AutoDeviceMapQuantization<'a> {
    pub fn checkpoint(
        config: Option<&'a QuantizedConfig>,
        topology: Option<&'a Topology>,
    ) -> Option<Self> {
        match config {
            Some(
                config @ (QuantizedConfig::Fp8 { .. }
                | QuantizedConfig::CompressedTensors { .. }
                | QuantizedConfig::ModelOpt { .. }),
            ) => Some(Self {
                source: AutoDeviceMapQuantizationSource::Checkpoint(config),
                topology,
                checkpoint: None,
                source_quantization: Some(config),
            }),
            _ => topology.map(|topology| Self::isq(None, Some(topology))),
        }
    }

    pub fn isq(isq: Option<IsqType>, topology: Option<&'a Topology>) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::Isq(isq),
            topology,
            checkpoint: None,
            source_quantization: None,
        }
    }

    pub fn weight_source(source: &'a dyn QuantizedWeightSource) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::WeightSource(source),
            topology: None,
            checkpoint: None,
            source_quantization: None,
        }
    }

    pub fn weight_source_with_topology(
        source: &'a dyn QuantizedWeightSource,
        topology: Option<&'a Topology>,
    ) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::WeightSource(source),
            topology,
            checkpoint: None,
            source_quantization: None,
        }
    }

    pub fn with_checkpoint(mut self, checkpoint: &'a CheckpointTensors<'a>) -> Self {
        self.checkpoint = Some(checkpoint);
        self
    }

    pub fn with_source_quantization(mut self, config: Option<&'a QuantizedConfig>) -> Self {
        self.source_quantization = config;
        self
    }

    pub(crate) fn checkpoint_contains(&self, name: &str) -> bool {
        self.checkpoint
            .is_some_and(|checkpoint| checkpoint.contains(name))
            || matches!(self.source, AutoDeviceMapQuantizationSource::WeightSource(source) if source.contains(name))
    }

    pub fn uqff(source: &'a dyn QuantizedWeightSource) -> Self {
        Self::weight_source(source)
    }

    #[cfg(test)]
    fn unpromoted_pack_factor_for(
        &self,
        name: &str,
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        self.pack_factor_for_candidates(&[name], dtype, fallback, false)
    }

    pub fn promoted_pack_factor_for(
        &self,
        name: &str,
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        self.pack_factor_for_candidates(&[name], dtype, fallback, true)
    }

    pub fn tensor_size_in_bytes(
        &self,
        name: &str,
        shape: &[usize],
        dtype: DType,
        fallback: usize,
        promote_default: bool,
    ) -> Result<usize> {
        self.tensor_size_in_bytes_for_candidates(&[name], shape, dtype, fallback, promote_default)
    }

    pub(crate) fn tensor_size_in_bytes_for_candidates(
        &self,
        names: &[&str],
        shape: &[usize],
        dtype: DType,
        fallback: usize,
        promote_default: bool,
    ) -> Result<usize> {
        if let Some(ty) = self.topology_isq_for_candidates(names) {
            return Ok(mistralrs_quant::estimate_isq_weight_bytes(shape, dtype, ty));
        }
        if let AutoDeviceMapQuantizationSource::Checkpoint(config) = self.source {
            for name in names {
                let prefix = name.strip_suffix(".weight").unwrap_or(name);
                if let Some(spec) = config.resolve_checkpoint(prefix)? {
                    return Ok(spec.size_in_bytes(shape)?);
                }
            }
            return Ok(shape.iter().product::<usize>() * dtype.size_in_bytes());
        }
        if let AutoDeviceMapQuantizationSource::Isq(Some(default)) = self.source {
            let ty = if promote_default {
                default.promote_for_sensitive_tensor()
            } else {
                default
            };
            return Ok(mistralrs_quant::estimate_isq_weight_bytes(shape, dtype, ty));
        }
        let factor = self.pack_factor_for_candidates(names, dtype, fallback, promote_default)?;
        Ok(shape.iter().product::<usize>() * dtype.size_in_bytes() / factor)
    }

    pub(crate) fn expert_weights_size_in_bytes(
        &self,
        prefix: &str,
        [experts, hidden, intermediate]: [usize; 3],
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        let keys = mistralrs_quant::QuantizedExpertKeys::new(prefix);
        let stacked_names = [keys.gate, keys.up, keys.down].map(|name| format!("{name}.weight"));
        let per_expert_names = crate::moe::ExpertProjNames::KNOWN
            .into_iter()
            .find(|names| self.checkpoint_contains(&format!("{prefix}.0.{}.weight", names.gate)));
        let source_names = if self.checkpoint_contains(&stacked_names[0]) {
            stacked_names.clone()
        } else if let Some(names) = per_expert_names {
            [names.gate, names.up, names.down].map(|name| format!("{prefix}.0.{name}.weight"))
        } else if matches!(
            self.source,
            AutoDeviceMapQuantizationSource::Checkpoint(
                QuantizedConfig::ModelOpt { .. } | QuantizedConfig::CompressedTensors { .. }
            )
        ) {
            ["gate_proj", "up_proj", "down_proj"].map(|name| format!("{prefix}.0.{name}.weight"))
        } else {
            ["gate_up_proj", "gate_up_proj", "down_proj"]
                .map(|name| format!("{prefix}.{name}.weight"))
        };
        let prequantized = match self.source_quantization {
            None => false,
            Some(QuantizedConfig::Fp8 { .. }) if self.checkpoint.is_some() => {
                self.checkpoint_contains(&format!("{prefix}.gate_up_proj.weight_scale_inv"))
                    || crate::moe::ExpertProjNames::KNOWN.iter().any(|names| {
                        self.checkpoint_contains(&format!(
                            "{prefix}.0.{}.weight_scale_inv",
                            names.gate
                        ))
                    })
            }
            Some(_) => true,
        };
        let shapes = [
            [experts, intermediate, hidden],
            [experts, intermediate, hidden],
            [experts, hidden, intermediate],
        ];
        source_names.iter().zip(shapes).zip(stacked_names).try_fold(
            0,
            |total, ((source_name, shape), stacked_name)| {
                // Prequantized experts apply ISQ after stacking; dense checkpoints match their source names.
                let bytes = if matches!(self.source, AutoDeviceMapQuantizationSource::Checkpoint(_))
                {
                    if let Some(ty) = self.topology_isq_for_candidates(&[&stacked_name]) {
                        mistralrs_quant::estimate_isq_weight_bytes(&shape, dtype, ty)
                    } else {
                        Self {
                            topology: None,
                            ..*self
                        }
                        .tensor_size_in_bytes(
                            source_name,
                            &shape,
                            dtype,
                            fallback,
                            false,
                        )?
                    }
                } else {
                    let name = if prequantized {
                        &stacked_name
                    } else {
                        source_name
                    };
                    self.tensor_size_in_bytes(name, &shape, dtype, fallback, false)?
                };
                Ok(total + bytes)
            },
        )
    }

    pub(crate) fn embedding_size_in_bytes_for_candidates(
        &self,
        names: &[&str],
        shape: &[usize],
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        if matches!(self.source, AutoDeviceMapQuantizationSource::Checkpoint(_))
            && self.topology_isq_for_candidates(names).is_none()
        {
            return Ok(shape.iter().product::<usize>() * dtype.size_in_bytes());
        }
        self.tensor_size_in_bytes_for_candidates(names, shape, dtype, fallback, true)
    }

    fn embedding_pack_factor_for_candidates(
        &self,
        names: &[&str],
        dtype: DType,
        fallback: usize,
    ) -> Result<usize> {
        if matches!(self.source, AutoDeviceMapQuantizationSource::Checkpoint(_))
            && self.topology_isq_for_candidates(names).is_none()
        {
            return Ok(1);
        }
        self.pack_factor_for_candidates(names, dtype, fallback, true)
    }

    fn topology_isq_for_candidates(&self, names: &[&str]) -> Option<IsqType> {
        names.iter().find_map(|name| {
            let topology = self.topology?;
            topology
                .patterns
                .iter()
                .rev()
                .find(|(pattern, entry)| {
                    (entry.isq.is_some() || entry.device.is_some()) && pattern.is_match(name)
                })
                .map(|(_, entry)| entry)
                .or_else(|| topology.layer_for(mistralrs_quant::layer_index_from_prefix(name)?))
                .and_then(|entry| entry.isq)
        })
    }

    pub fn conservative_pack_factor(&self, dtype: DType, fallback: usize) -> usize {
        let topology_pack_factors = self.topology.into_iter().flat_map(|topology| {
            topology
                .layers
                .iter()
                .filter_map(|entry| entry.as_ref().and_then(|entry| entry.isq))
                .chain(topology.patterns.iter().filter_map(|(_, entry)| entry.isq))
        });
        topology_pack_factors.fold(fallback, |factor, ty| factor.min(ty.pack_factor(dtype)))
    }

    pub fn conservative_moqe_pack_factor(
        &self,
        dtype: DType,
        source_pack_factor: usize,
        target: IsqType,
    ) -> usize {
        self.conservative_pack_factor(dtype, source_pack_factor.min(target.pack_factor(dtype)))
    }

    fn pack_factor_for_candidates(
        &self,
        names: &[&str],
        dtype: DType,
        fallback: usize,
        promote_default: bool,
    ) -> Result<usize> {
        let topology_ty = self.topology_isq_for_candidates(names);
        match self.source {
            AutoDeviceMapQuantizationSource::Checkpoint(config) => {
                if let Some(ty) = topology_ty {
                    return Ok(ty.pack_factor(dtype));
                }
                for name in names {
                    let prefix = name.strip_suffix(".weight").unwrap_or(name);
                    if let Some(spec) = config.resolve_checkpoint(prefix)? {
                        return Ok(spec.pack_factor(dtype));
                    }
                }
                Ok(1)
            }
            AutoDeviceMapQuantizationSource::WeightSource(source) => {
                if let Some(ty) = topology_ty {
                    return Ok(ty.pack_factor(dtype));
                }
                for name in names {
                    if let Some(pack_factor) = source.pack_factor_for(name, dtype)? {
                        return Ok(pack_factor);
                    }
                }
                Ok(1)
            }
            AutoDeviceMapQuantizationSource::Isq(default) => {
                let ty = topology_ty.or_else(|| {
                    default.map(|ty| {
                        if promote_default {
                            ty.promote_for_sensitive_tensor()
                        } else {
                            ty
                        }
                    })
                });
                Ok(ty.map(|ty| ty.pack_factor(dtype)).unwrap_or(fallback))
            }
        }
    }
}

fn promoted_tensor_pack_factor(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    name: &str,
    dtype: DType,
    fallback: usize,
) -> Result<usize> {
    quantization.map_or(Ok(fallback), |quantization| {
        quantization.promoted_pack_factor_for(name, dtype, fallback)
    })
}

fn tied_promoted_tensor_pack_factor(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_name: &str,
    legacy_head_name: &str,
    dtype: DType,
    fallback: usize,
) -> Result<usize> {
    quantization.map_or(Ok(fallback), |quantization| match quantization.source {
        AutoDeviceMapQuantizationSource::WeightSource(_) => quantization
            .pack_factor_for_candidates(&[embedding_name, legacy_head_name], dtype, fallback, true),
        AutoDeviceMapQuantizationSource::Isq(_)
        | AutoDeviceMapQuantizationSource::Checkpoint(_) => {
            quantization.embedding_pack_factor_for_candidates(&[embedding_name], dtype, fallback)
        }
    })
}

fn language_model_pack_factors(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_name: &str,
    head_name: &str,
    tied: bool,
    dtype: DType,
    fallback: usize,
) -> Result<(usize, usize)> {
    let embedding = if tied {
        tied_promoted_tensor_pack_factor(quantization, embedding_name, head_name, dtype, fallback)?
    } else {
        quantization.map_or(Ok(fallback), |quantization| {
            quantization.embedding_pack_factor_for_candidates(&[embedding_name], dtype, fallback)
        })?
    };
    let head = promoted_tensor_pack_factor(quantization, head_name, dtype, fallback)?;
    Ok((embedding, head))
}

fn language_model_pack_factors_with_aliases(
    quantization: Option<&AutoDeviceMapQuantization<'_>>,
    embedding_names: &[&str],
    head_names: &[&str],
    tied: bool,
    dtype: DType,
    fallback: usize,
) -> Result<(usize, usize)> {
    let embedding = quantization.map_or(Ok(fallback), |quantization| {
        if tied
            && matches!(
                quantization.source,
                AutoDeviceMapQuantizationSource::WeightSource(_)
            )
        {
            let mut candidates = embedding_names.to_vec();
            candidates.extend_from_slice(head_names);
            quantization.pack_factor_for_candidates(&candidates, dtype, fallback, true)
        } else {
            quantization.embedding_pack_factor_for_candidates(embedding_names, dtype, fallback)
        }
    })?;
    let head = quantization.map_or(Ok(fallback), |quantization| {
        quantization.pack_factor_for_candidates(head_names, dtype, fallback, true)
    })?;
    Ok((embedding, head))
}

pub trait DeviceMappedModelLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig>;

    /// Maximum activation size of non-mapped parts of this model.
    /// Useful for the multimodal models which may prefer to keep the vison components on the GPU.
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize>;

    fn non_mapped_activation_memory_bytes(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _dtype: DType,
        _device: &Device,
    ) -> Result<usize> {
        self.non_mapped_max_act_size_elems(config, params)?
            .checked_mul(DType::F32.size_in_bytes())
            .ok_or_else(|| anyhow::anyhow!("non-mapped activation memory size overflow"))
    }
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
        quantization: Option<&AutoDeviceMapQuantization<'_>>,
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
    fn layer_sizes_in_bytes_with_quantization(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _quantization: Option<&AutoDeviceMapQuantization<'_>>,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        self.layer_sizes_in_bytes(config, dtype, weight_pack_factor, matformer_config)
    }
    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        None
    }
    fn non_mapped_sub_models_for_config(
        &self,
        _config: &str,
    ) -> Result<Option<Vec<NonMappedSubModel>>> {
        Ok(self.non_mapped_sub_models())
    }
    fn num_layers(&self, config: &str) -> Result<usize>;

    fn layer_resources(
        &self,
        _config: &str,
        _source: ResourceSource<'_>,
    ) -> Result<Vec<LayerResource>> {
        Ok(Vec::new())
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>>;

    fn supports_checkpoint_weight_inventory(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }

    fn checkpoint_layer_index(&self, _config: &str, tensor_name: &str) -> Option<usize> {
        checkpoint_inventory::standard_layer_index(tensor_name)
    }

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
        paged_attn_config: Option<&mut PagedAttentionConfig>,
    ) -> Result<DeviceMapMetadata>
    where
        Self: Sized,
    {
        super::get_device_layers_for_loader(
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
        .map(|(mapping, _)| mapping)
    }
}

#[derive(Debug)]
pub struct ModelDownload {
    pub paths: Box<dyn ModelPaths>,
    pub projector_paths: Vec<PathBuf>,
    pub additional_files: Vec<PathBuf>,
}

impl ModelDownload {
    pub fn new(paths: Box<dyn ModelPaths>) -> Self {
        Self {
            paths,
            projector_paths: Vec::new(),
            additional_files: Vec::new(),
        }
    }

    pub fn files(&self) -> Vec<PathBuf> {
        let mut files = self.paths.files();
        files.extend(self.projector_paths.iter().cloned());
        files.extend(self.additional_files.iter().cloned());
        files.sort();
        files.dedup();
        files
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
    /// Resolve and download inference assets without loading tensors or initializing a device.
    fn download_model_from_hf(
        &self,
        revision: Option<String>,
        token_source: TokenSource,
        silent: bool,
    ) -> Result<ModelDownload>;

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
    ) -> Result<Arc<Mutex<dyn Pipeline + Send + Sync>>> {
        let download = self.download_model_from_hf(revision, token_source, silent)?;
        self.load_model_from_path(
            download.paths.as_ref(),
            dtype,
            device,
            silent,
            mapper,
            in_situ_quant,
            paged_attn_config,
        )
    }

    /// Load a model from the specified paths.
    /// Also initializes `DEBUG`.
    #[allow(
        clippy::type_complexity,
        clippy::too_many_arguments,
        clippy::borrowed_box
    )]
    fn load_model_from_path(
        &self,
        paths: &dyn ModelPaths,
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

#[cfg(test)]
mod auto_device_map_quantization_tests {
    use super::*;

    #[test]
    fn legacy_model_opt_config_is_embedded() -> Result<()> {
        let config = r#"{"model_type":"llama"}"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"0.19.0"},
            "quantization":{"quant_algo":"FP8","exclude_modules":["lm_head"]}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        let quantization = &merged["quantization_config"];
        assert_eq!(quantization["quant_method"], "modelopt");
        assert_eq!(quantization["quantization"]["quant_algo"], "FP8");
        assert_eq!(merged["model_type"], "llama");
        assert_eq!(
            QuantizationConfigShim::get_quant_config_pack_factor(&merged.to_string(), DType::BF16)?,
            IsqType::F8E4M3.pack_factor(DType::BF16)
        );
        Ok(())
    }

    #[test]
    fn compression_config_alias_is_loaded_with_explicit_precedence() -> Result<()> {
        let compressed = serde_json::json!({
            "format": "float-quantized",
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "dynamic": false,
                        "num_bits": 8,
                        "strategy": "tensor",
                        "symmetric": true,
                        "type": "float"
                    }
                }
            }
        });
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("config.json");
        std::fs::write(
            &config_path,
            serde_json::to_vec(&serde_json::json!({
                "model_type": "llama",
                "compression_config": compressed
            }))?,
        )?;
        let loaded: serde_json::Value =
            serde_json::from_str(&load_model_config(&config_path, true)?)?;
        let quantized: QuantizedConfig =
            serde_json::from_value(loaded["quantization_config"].clone())?;
        assert!(matches!(
            quantized,
            QuantizedConfig::CompressedTensors { .. }
        ));

        let explicit = serde_json::json!({"quant_method": "fp8"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "quantization_config": explicit,
                "compression_config": compressed,
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], explicit);
        assert_eq!(normalized["text_config"]["quantization_config"], explicit);

        let conflicting = serde_json::json!({"quant_method": "fp8", "activation_scheme": "static"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "quantization_config": explicit,
                "text_config": {"quantization_config": conflicting}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], explicit);
        assert_eq!(normalized["text_config"]["quantization_config"], explicit);

        let nested = serde_json::json!({"quant_method": "fp8", "activation_scheme": "dynamic"});
        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "compression_config": compressed,
                "text_config": {"quantization_config": nested}
            })
            .to_string(),
        )?)?;
        assert_eq!(normalized["quantization_config"], nested);
        assert_eq!(normalized["text_config"]["quantization_config"], nested);

        let normalized: serde_json::Value = serde_json::from_str(&normalize_compression_config(
            &serde_json::json!({
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
        )?)?;
        assert_eq!(
            normalized["quantization_config"]["format"],
            "float-quantized"
        );
        assert_eq!(
            normalized["text_config"]["quantization_config"],
            normalized["quantization_config"]
        );
        let merged: serde_json::Value = serde_json::from_str(&inject_legacy_model_opt_config(
            &serde_json::json!({
                "text_config": {"compression_config": compressed}
            })
            .to_string(),
            r#"{"quantization":{"quant_algo":"FP8"}}"#,
        )?)?;
        assert_eq!(merged["quantization_config"]["format"], "float-quantized");
        Ok(())
    }

    #[test]
    fn uqff_config_loading_skips_checkpoint_quantization_metadata() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("config.json");
        let config = r#"{"model_type":"llama","compression_config":{"unsupported":true}}"#;
        std::fs::write(&config_path, config)?;
        std::fs::write(dir.path().join(LEGACY_MODEL_OPT_CONFIG), "not json")?;
        assert_eq!(load_model_config(&config_path, false)?, config);
        Ok(())
    }

    #[test]
    fn lfm3_codec_config_and_legacy_fields_survive_uqff_roundtrip() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let config_path = dir.path().join("config.json");
        let codec_path = dir.path().join(LFM3_CODEC_CONFIG);
        std::fs::create_dir(codec_path.parent().unwrap())?;
        let codec = serde_json::json!({"model_type":"liquid_codec", "n_q":12});
        for model_type in ["lfm3_audio", "lfm3_omni"] {
            let source = serde_json::json!({
                "model_type": model_type, "lfm": {"hidden_size": 2048},
                "codebooks": 12, "audio_vocab_size": 2017,
            });
            std::fs::write(&config_path, source.to_string())?;
            std::fs::write(&codec_path, codec.to_string())?;
            let config = load_model_config(&config_path, false)?;
            assert_eq!(load_model_config(&config_path, true)?, config);
            let parsed: serde_json::Value = serde_json::from_str(&config)?;
            assert_eq!(parsed["codec_config"], codec);
            if model_type == "lfm3_audio" {
                assert_eq!(parsed["text_config"]["hidden_size"], 2048);
                assert_eq!(parsed["audio_eos_code"], 2016);
                assert_eq!(parsed["maskgit_schedule"], serde_json::json!([4, 4, 4]));
            }
            std::fs::write(&config_path, &config)?;
            std::fs::remove_file(&codec_path)?;
            assert_eq!(load_model_config(&config_path, false)?, config);
        }
        Ok(())
    }

    #[test]
    fn embedded_quantization_config_wins_over_legacy_model_opt() -> Result<()> {
        let config = r#"{"quantization_config":{"quant_method":"fp8"}}"#;
        let legacy = r#"{"quantization":{"quant_algo":"FP8"}}"#;
        assert_eq!(inject_legacy_model_opt_config(config, legacy)?, config);
        Ok(())
    }

    #[test]
    fn embedded_model_opt_fields_override_legacy_config() -> Result<()> {
        let config = r#"{
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN",
                "ignore": ["new_head"]
            }
        }"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"legacy"},
            "quantization":{"quant_algo":"FP8","exclude_modules":["old_head"]}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        let quantization = &merged["quantization_config"];
        assert_eq!(quantization["quant_method"], "modelopt");
        assert_eq!(quantization["quant_algo"], "FP8_PER_CHANNEL_PER_TOKEN");
        assert_eq!(quantization["ignore"], serde_json::json!(["new_head"]));
        assert_eq!(quantization["producer"]["version"], "legacy");
        Ok(())
    }

    #[test]
    fn legacy_model_opt_config_is_propagated_to_text_config() -> Result<()> {
        let config = r#"{"model_type":"test","text_config":{"hidden_size":128}}"#;
        let legacy = r#"{"quantization":{"quant_algo":"FP8"}}"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        assert_eq!(
            merged["text_config"]["quantization_config"],
            merged["quantization_config"]
        );
        Ok(())
    }

    #[test]
    fn nested_model_opt_config_is_merged_and_propagated() -> Result<()> {
        let config = r#"{
            "model_type":"test",
            "text_config":{"quantization_config":{
                "quant_method":"modelopt",
                "quant_algo":"FP8_PER_CHANNEL_PER_TOKEN"
            }}
        }"#;
        let legacy = r#"{
            "producer":{"name":"modelopt","version":"0.19.0"},
            "quantization":{"quant_algo":"FP8"}
        }"#;
        let merged: serde_json::Value =
            serde_json::from_str(&inject_legacy_model_opt_config(config, legacy)?)?;
        assert_eq!(
            merged["quantization_config"]["quant_algo"],
            "FP8_PER_CHANNEL_PER_TOKEN"
        );
        assert_eq!(
            merged["quantization_config"]["producer"]["version"],
            "0.19.0"
        );
        assert_eq!(
            merged["text_config"]["quantization_config"],
            merged["quantization_config"]
        );
        Ok(())
    }

    #[test]
    fn qk_rope_layout_marker_round_trips() -> Result<()> {
        for layout in [
            crate::gguf::normal_registry::RopePairing::Adjacent,
            crate::gguf::normal_registry::RopePairing::HalfSplit,
        ] {
            let stamped = stamp_qk_rope_layout(r#"{"hidden_size":16}"#, layout)?;
            assert_eq!(qk_rope_layout_from_config(&stamped)?, Some(layout));
            assert_eq!(
                serde_json::from_str::<serde_json::Value>(&stamped)?["hidden_size"],
                16
            );
        }
        assert!(qk_rope_layout_from_config(r#"{"_mistralrs_qk_rope_layout":"invalid"}"#).is_err());
        Ok(())
    }

    #[test]
    fn adjacent_qk_rope_layout_rejects_lora_adapters() -> Result<()> {
        assert!(
            validate_lora_qk_rope_layout(r#"{"_mistralrs_qk_rope_layout":"adjacent"}"#, true,)
                .is_err()
        );
        assert!(
            validate_lora_qk_rope_layout(r#"{"_mistralrs_qk_rope_layout":"adjacent"}"#, false,)
                .is_ok()
        );
        assert!(validate_lora_qk_rope_layout(r#"{"hidden_size":16}"#, true).is_ok());
        Ok(())
    }

    struct PackFactorWeightSource(usize);

    impl QuantizedWeightSource for PackFactorWeightSource {
        fn contains(&self, _name: &str) -> bool {
            true
        }

        fn load_linear(
            &self,
            _key: &str,
            _device: &Device,
            _shard: mistralrs_quant::Shard,
        ) -> candle_core::Result<Option<std::sync::Arc<dyn mistralrs_quant::QuantMethod>>> {
            unreachable!()
        }

        fn load_optional_tensor(
            &self,
            _name: &str,
            _device: &Device,
        ) -> candle_core::Result<Option<candle_core::Tensor>> {
            unreachable!()
        }

        fn shard_alignment(&self, _key: &str) -> candle_core::Result<usize> {
            Ok(1)
        }

        fn pack_factor(&self, _dtype: DType) -> candle_core::Result<usize> {
            Ok(self.0)
        }

        fn pack_factor_for(&self, _key: &str, _dtype: DType) -> candle_core::Result<Option<usize>> {
            Ok(Some(self.0))
        }
    }

    const EMBEDDING: &str = "model.embed_tokens.weight";
    const HEAD: &str = "lm_head.weight";

    #[test]
    fn explicit_promotion_and_topology_overrides_resolve_in_estimates() -> Result<()> {
        let dtype = DType::BF16;
        for (default, sensitive) in [
            (IsqType::AFQ4, IsqType::AFQ6),
            (IsqType::Q4K, IsqType::Q6K),
            (IsqType::Q5K, IsqType::Q8_0),
            (IsqType::Q6K, IsqType::Q8_0),
        ] {
            let automatic = AutoDeviceMapQuantization::isq(Some(default), None);
            assert_eq!(
                automatic.promoted_pack_factor_for(EMBEDDING, dtype, 1)?,
                sensitive.pack_factor(dtype),
                "{default}"
            );
            assert_eq!(
                automatic.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
                default.pack_factor(dtype),
                "{default}"
            );
            assert_eq!(
                automatic.unpromoted_pack_factor_for(
                    "model.layers.0.mlp.down_proj.weight",
                    dtype,
                    1,
                )?,
                default.pack_factor(dtype),
                "{default}"
            );
        }

        let topology = Topology::from_str(
            "'/^model\\.embed_tokens\\.weight$/':\n  isq: Q2K\n'/^lm_head\\.weight$/':\n  isq: Q8_0\n",
        )?;
        let overridden = AutoDeviceMapQuantization::isq(Some(IsqType::Q4K), Some(&topology));
        assert_eq!(
            overridden.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
            IsqType::Q2K.pack_factor(dtype)
        );
        assert_eq!(
            overridden.unpromoted_pack_factor_for(HEAD, dtype, 1)?,
            IsqType::Q8_0.pack_factor(dtype)
        );
        assert_eq!(
            tied_promoted_tensor_pack_factor(Some(&overridden), EMBEDDING, HEAD, dtype, 1,)?,
            IsqType::Q2K.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn topology_only_quantization_uses_fallback_for_unmatched_tensors() -> Result<()> {
        let dtype = DType::BF16;
        let topology = Topology::from_str("'/^model\\.embed_tokens\\.weight$/':\n  isq: AFQ8\n")?;
        let quantization = AutoDeviceMapQuantization::isq(None, Some(&topology));
        assert_eq!(
            quantization.unpromoted_pack_factor_for(EMBEDDING, dtype, 1)?,
            IsqType::AFQ8.pack_factor(dtype)
        );
        assert_eq!(quantization.unpromoted_pack_factor_for(HEAD, dtype, 3)?, 3);
        Ok(())
    }

    #[test]
    fn topology_pack_factor_is_conservative_for_mapped_layers() -> Result<()> {
        let dtype = DType::BF16;
        let topology = Topology::from_str("'0':\n  isq: Q8_0\n")?;
        let quantization = AutoDeviceMapQuantization::isq(Some(IsqType::Q2K), Some(&topology));
        assert_eq!(
            quantization.conservative_pack_factor(dtype, IsqType::Q2K.pack_factor(dtype)),
            IsqType::Q8_0.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn topology_isq_overlays_prepared_weight_source_sizing() -> Result<()> {
        let dtype = DType::BF16;
        let source = PackFactorWeightSource(IsqType::Q2K.pack_factor(dtype));
        let topology = Topology::from_str("'/^model\\.embed_tokens\\.weight$/':\n  isq: Q8_0\n")?;
        let quantization =
            AutoDeviceMapQuantization::weight_source_with_topology(&source, Some(&topology));

        assert_eq!(
            quantization.promoted_pack_factor_for(
                EMBEDDING,
                dtype,
                IsqType::Q2K.pack_factor(dtype),
            )?,
            IsqType::Q8_0.pack_factor(dtype)
        );
        assert_eq!(
            quantization.conservative_pack_factor(dtype, source.pack_factor(dtype)?),
            IsqType::Q8_0.pack_factor(dtype)
        );
        Ok(())
    }

    #[test]
    fn moqe_sizing_keeps_source_precision_for_the_unquantized_trunk() -> Result<()> {
        let dtype = DType::BF16;
        let source_factor = IsqType::Q4K.pack_factor(dtype);
        let source = PackFactorWeightSource(source_factor);
        let prepared = AutoDeviceMapQuantization::weight_source(&source);
        assert_eq!(
            prepared.conservative_moqe_pack_factor(dtype, source_factor, IsqType::Q2K),
            source_factor.min(IsqType::Q2K.pack_factor(dtype))
        );

        let checkpoint = AutoDeviceMapQuantization::isq(None, None);
        assert_eq!(
            checkpoint.conservative_moqe_pack_factor(dtype, 1, IsqType::Q2K),
            1
        );
        Ok(())
    }

    #[test]
    fn expert_sizing_uses_prepared_separate_projection_names() -> Result<()> {
        use candle_core::{
            quantized::{gguf_file, GgmlDType, QTensor},
            Tensor,
        };
        use mistralrs_quant::{GgufArchive, GgufBindingMap, GgufTensorBinding, GgufWeightSource};

        const PREFIX: &str = "model.layers.0.mlp.experts";
        const EXPERTS: usize = 2;
        const WIDTH: usize = 256;
        const DENSE_BYTES: usize = EXPERTS * WIDTH * WIDTH * 2;
        const Q8_BYTES: usize = EXPERTS * WIDTH * WIDTH / 32 * 34;
        let mut bindings = GgufBindingMap::new();
        let mut tensors = Vec::new();
        for (projection, stored, dtype) in [
            ("gate_proj", "blk.0.ffn_gate_exps.weight", GgmlDType::Q4_0),
            ("up_proj", "blk.0.ffn_up_exps.weight", GgmlDType::Q4_0),
            ("down_proj", "blk.0.ffn_down_exps.weight", GgmlDType::Q8_0),
        ] {
            let tensor = Tensor::ones((EXPERTS, WIDTH, WIDTH), DType::F32, &Device::Cpu)?;
            tensors.push((stored, QTensor::quantize(&tensor, dtype)?));
            bindings.insert(
                format!("{PREFIX}.{projection}.weight"),
                GgufTensorBinding::tensor(stored),
            );
        }
        let mut file = tempfile::NamedTempFile::new()?;
        gguf_file::write(
            file.as_file_mut(),
            &[],
            &tensors
                .iter()
                .map(|(name, tensor)| (*name, tensor))
                .collect::<Vec<_>>(),
        )?;
        let source = GgufWeightSource::new(
            Arc::new(GgufArchive::open_file(file.path())?),
            &bindings,
            DType::BF16,
        )?;
        let topology = Topology::from_str(
            r"'/^model\.layers\.0\.mlp\.experts\.gate_proj\.weight$/':
  isq: Q8_0
",
        )?;
        for topology in [None, Some(&topology)] {
            let sizing = AutoDeviceMapQuantization::weight_source_with_topology(&source, topology);
            let gate_bytes = if topology.is_some() {
                Q8_BYTES
            } else {
                DENSE_BYTES / 3
            };
            assert_eq!(
                sizing.expert_weights_size_in_bytes(
                    PREFIX,
                    [EXPERTS, WIDTH, WIDTH],
                    DType::BF16,
                    1
                )?,
                gate_bytes + DENSE_BYTES / 3 + DENSE_BYTES,
            );
        }
        Ok(())
    }

    #[test]
    fn expert_sizing_accounts_for_separate_checkpoint_projections() -> Result<()> {
        const PREFIX: &str = "model.layers.0.mlp.experts";
        const EXPERTS: usize = 2;
        const WIDTH: usize = 256;
        const DENSE_BYTES: usize = EXPERTS * WIDTH * WIDTH * 2;
        const NVFP4_BYTES: usize =
            EXPERTS * (WIDTH * WIDTH / 2 + WIDTH * WIDTH / 16 + WIDTH * 4 + 4);
        let config: QuantizedConfig = serde_json::from_value(serde_json::json!({
            "quant_method": "modelopt", "quant_algo": "NVFP4", "group_size": 16,
            "exclude_modules": ["model.layers.0.mlp.experts.*.up_proj"]
        }))?;
        let sizing = AutoDeviceMapQuantization::checkpoint(Some(&config), None).unwrap();
        assert_eq!(
            sizing.expert_weights_size_in_bytes(PREFIX, [EXPERTS, WIDTH, WIDTH], DType::BF16, 1)?,
            2 * NVFP4_BYTES + DENSE_BYTES,
        );
        Ok(())
    }

    #[test]
    fn checkpoint_sizing_accounts_for_mixed_weights_and_runtime_scales() -> Result<()> {
        let config: QuantizedConfig = serde_json::from_value(serde_json::json!({
            "quant_method": "modelopt",
            "quant_algo": "MIXED_PRECISION",
            "quantized_layers": {
                "model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "weight_scale_encoding": "csf"},
                "model.layers.0.self_attn.q_proj": {"quant_algo": "MXFP8", "group_size": 32},
                "model.layers.0.self_attn.k_proj": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}
            }
        }))?;
        let sizing = AutoDeviceMapQuantization::checkpoint(Some(&config), None).unwrap();
        for (name, shape, expected) in [
            (
                "model.layers.0.mlp.experts.gate_up_proj.weight",
                vec![3, 4, 32],
                3 * (64 + 8 + 16 + 4),
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                vec![4, 32],
                128 + 4,
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                vec![4, 32],
                128 + 16,
            ),
            ("model.layers.0.mlp.gate.weight", vec![4, 32], 256),
        ] {
            assert_eq!(
                sizing.tensor_size_in_bytes(name, &shape, DType::BF16, 1, false)?,
                expected,
                "{name}"
            );
        }
        Ok(())
    }
}

mod lfm3;
mod lfm3_audio;
mod lfm3_d1;
mod lfm3_omni;
mod lfm3_vl;
pub use lfm3::Lfm3Loader;
pub use lfm3_audio::Lfm3AudioLoader;
pub use lfm3_d1::Lfm3D1Loader;
pub use lfm3_omni::Lfm3OmniLoader;
pub use lfm3_vl::Lfm3VlLoader;
