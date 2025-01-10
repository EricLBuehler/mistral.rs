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

use anyhow::{Context, Result};
use as_any::AsAny;
use candle_core::{DType, Device};
use mistralrs_quant::IsqType;
use tokio::sync::Mutex;

pub use normal_loaders::{
    AutoLoader, DeepSeekV2Loader, Gemma2Loader, GemmaLoader, LlamaLoader, MistralLoader,
    MixtralLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Starcoder2Loader,
};

pub use vision_loaders::{
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Phi3VLoader, Qwen2VLLoader,
    VLlamaLoader, VisionLoaderType, VisionModel, VisionModelLoader,
};

pub use diffusion_loaders::{
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, DiffusionModelPaths,
    DiffusionModelPathsInner, FluxLoader,
};

use crate::{
    lora::LoraConfig, utils::debug::DeviceRepr, xlora_models::XLoraConfig, DeviceLayerMapMetadata,
    DeviceMapMetadata, DeviceMapSetting, MemoryUsage, Ordering, PagedAttentionConfig, TryIntoDType,
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

    /// Get the explicit chat template. If specified, this overwrites anything in the tokenizer_config.json
    fn get_chat_template_json(&self) -> &Option<PathBuf>;
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
    pub chat_template_json_filename: Option<P>,
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
        chat_template_json_filename: Option<P>,
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
    fn get_chat_template_json(&self) -> &Option<PathBuf> {
        &self.chat_template_json_filename
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

pub trait DeviceMappedModelLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize>;
    fn per_layer_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize>;
    fn num_layers(&self, config: &str) -> Result<usize>;

    /// weight_pack_factor only applies to quantized weights.
    fn get_device_layers(
        &self,
        _config: &str,
        num_layers: usize,
        per_layer_size_in_bytes: usize,
        non_mapped_size_in_bytes: usize,
        total_model_size_in_bytes: usize,
        devices: &[Device],
    ) -> Result<DeviceMapMetadata> {
        let mut remaining_to_map = total_model_size_in_bytes;

        // Always add the CPU as fallback
        let devices = [devices, &[Device::Cpu]].concat();

        let mut per_layer_avail = Vec::new();
        for dev in devices.clone() {
            let avail = MemoryUsage.get_memory_available(&dev)?;
            per_layer_avail.push((avail, dev));
        }
        // Reverse so we don't use the cpu first!
        per_layer_avail.reverse();

        let mut device_layers = Vec::new();

        let mut current_ordinal = 0;
        let mut current_layer = 0;
        while remaining_to_map > 0 && !per_layer_avail.is_empty() {
            let (device_capacity, device) = per_layer_avail
                .pop()
                .context("No more devices to map to. The model does not fit on this system.")?;
            // All usage of 90% of the memory as a maximum.
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let device_capacity = (device_capacity as f64 * 0.90) as usize;
            let layers_on_device = if device_capacity >= remaining_to_map {
                num_layers - current_layer
            } else if current_ordinal == 0 {
                (device_capacity - non_mapped_size_in_bytes) / per_layer_size_in_bytes
            } else {
                device_capacity / per_layer_size_in_bytes
            };

            // CPU mappings are automatically handled by the traditional device mapper, we can just leave them out here.
            if !device.is_cpu() {
                device_layers.push(DeviceLayerMapMetadata {
                    ordinal: current_ordinal,
                    layers: layers_on_device,
                });
                current_ordinal += 1;
            }

            current_layer += layers_on_device;
            remaining_to_map =
                remaining_to_map.saturating_sub(per_layer_size_in_bytes * layers_on_device);
            if current_ordinal - 1 == 0 {
                remaining_to_map = remaining_to_map.saturating_sub(non_mapped_size_in_bytes);
            }
        }
        if remaining_to_map > 0 {
            anyhow::bail!(
                "This model does not fit on the devices {:?}, and exceeds total capacity by {}MB",
                devices
                    .iter()
                    .map(|dev| dev.device_pretty_repr())
                    .collect::<Vec<_>>(),
                remaining_to_map / (1024 * 1024)
            );
        }

        Ok(DeviceMapMetadata::from_num_device_layers(device_layers))
    }
}

/// The `Loader` trait abstracts the loading process. The primary entrypoint is the
/// `load_model` method.
///
/// # Example
/// ```no_run
/// use mistralrs_core::{Loader, TokenSource, DeviceMapSetting, ModelDType};
/// use candle_core::Device;
///
/// let loader: Box<dyn Loader> = todo!();
/// let pipeline = loader.load_model_from_hf(
///     None,
///     TokenSource::CacheToken,
///     &ModelDType::Auto,
///     &Device::cuda_if_available(0).unwrap(),
///     false,
///     DeviceMapSetting::Auto,
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
