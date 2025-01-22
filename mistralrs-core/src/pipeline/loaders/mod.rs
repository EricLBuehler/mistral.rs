mod diffusion_loaders;
mod normal_loaders;
mod vision_loaders;

use std::{
    collections::HashMap,
    fmt::{self, Debug, Display},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::{Context, Result};
use as_any::AsAny;
use candle_core::{DType, Device};
use itertools::Itertools;
use mistralrs_quant::IsqType;
use tokio::sync::Mutex;

pub use normal_loaders::{
    AutoLoader, DeepSeekV2Loader, DeepSeekV3Loader, Gemma2Loader, GemmaLoader, LlamaLoader,
    MistralLoader, MixtralLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel,
    NormalModelLoader, Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Starcoder2Loader,
};

use tracing::{info, warn};
pub use vision_loaders::{
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, MiniCpmOLoader, Phi3VLoader,
    Qwen2VLLoader, VLlamaLoader, VisionLoaderType, VisionModel, VisionModelLoader,
};

pub use diffusion_loaders::{
    DiffusionLoaderType, DiffusionModel, DiffusionModelLoader, DiffusionModelPaths,
    DiffusionModelPathsInner, FluxLoader,
};

use crate::{
    lora::LoraConfig,
    paged_attention::{calculate_cache_config, ModelConfigLike},
    utils::debug::DeviceRepr,
    xlora_models::XLoraConfig,
    DeviceLayerMapMetadata, DeviceMapMetadata, DeviceMapSetting, MemoryGpuConfig, MemoryUsage,
    Ordering, PagedAttentionConfig, TryIntoDType,
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

macro_rules! b_to_mb {
    ($x:expr) => {
        $x / (1024 * 1024)
    };
}

#[derive(Debug, Clone)]
pub enum AutoDeviceMapParams {
    Text {
        max_seq_len: usize,
        max_batch_size: usize,
    },
    Vision {
        max_seq_len: usize,
        max_batch_size: usize,
        max_image_shape: (usize, usize),
        max_num_images: usize,
    },
}

impl Display for AutoDeviceMapParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text {
                max_seq_len,
                max_batch_size,
            } => write!(
                f,
                "text[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}]"
            ),
            Self::Vision {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images
            } => write!(
                f,
                "vision[max_seq_len: {max_seq_len}, max_batch_size: {max_batch_size}, max_image_shape: {max_image_shape:?}, max_num_images: {max_num_images}]"
            ),
        }
    }
}

impl AutoDeviceMapParams {
    pub const DEFAULT_MAX_SEQ_LEN: usize = 4 * 1024;
    pub const DEFAULT_MAX_BATCH_SIZE: usize = 1;
    pub const DEFAULT_MAX_NUM_IMAGES: usize = 1;
    pub const DEFAULT_MAX_IMAGE_LENGTH: usize = 1024;

    pub fn default_text() -> Self {
        Self::Text {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
        }
    }

    pub fn default_vision() -> Self {
        Self::Vision {
            max_seq_len: Self::DEFAULT_MAX_SEQ_LEN,
            max_batch_size: Self::DEFAULT_MAX_BATCH_SIZE,
            max_num_images: Self::DEFAULT_MAX_NUM_IMAGES,
            max_image_shape: (
                Self::DEFAULT_MAX_IMAGE_LENGTH,
                Self::DEFAULT_MAX_IMAGE_LENGTH,
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) enum NonMappedSubModel {
    Vision,
}

impl Display for NonMappedSubModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vision => write!(f, "vision"),
        }
    }
}

fn calculate_key_block_shape(
    model_config: &dyn ModelConfigLike,
    dtype: DType,
    block_size: usize,
) -> (usize, usize, usize, usize) {
    let element_size = dtype.size_in_bytes();
    let x = 16 / element_size;
    (
        model_config.num_kv_heads(),
        model_config.k_head_dim() / x,
        block_size,
        x,
    )
}

fn calculate_value_block_shape(
    model_config: &dyn ModelConfigLike,
    block_size: usize,
) -> (usize, usize, usize) {
    (
        model_config.num_kv_heads(),
        model_config.v_head_dim(),
        block_size,
    )
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
    ) -> Result<usize>;
    /// weight_pack_factor only applies to quantized weights.
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
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
        mut layer_sizes_in_bytes: Vec<usize>,
        non_mapped_size_in_bytes: usize,
        total_model_size_in_bytes: usize,
        devices: &[Device],
        dtype: DType,
        params: &AutoDeviceMapParams,
        paged_attn_config: Option<&PagedAttentionConfig>,
    ) -> Result<DeviceMapMetadata> {
        let mapped_max_act_size_in_bytes =
            self.mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();
        let non_mapped_max_act_size_in_bytes =
            self.non_mapped_max_act_size_elems(config, params)? * dtype.size_in_bytes();

        let mut remaining_to_map = total_model_size_in_bytes;

        let max_seq_len = match params {
            AutoDeviceMapParams::Text { max_seq_len, .. }
            | AutoDeviceMapParams::Vision { max_seq_len, .. } => *max_seq_len,
        };
        let max_batch_size = match params {
            AutoDeviceMapParams::Text { max_batch_size, .. }
            | AutoDeviceMapParams::Vision { max_batch_size, .. } => *max_batch_size,
        };

        let model_cfg = self.model_config(config)?;
        let kv_cache_size_elems = match paged_attn_config {
            Some(paged_attn_config) => {
                let cache_config = calculate_cache_config(
                    MemoryGpuConfig::ContextSize(max_seq_len),
                    0,
                    Some(paged_attn_config.block_size.unwrap_or(32)),
                    dtype,
                    &*model_cfg,
                    &devices[0],
                    &devices.iter().map(|x| Some(x.clone())).collect::<Vec<_>>(),
                    true,
                )?;

                let key_block_shape =
                    calculate_key_block_shape(&*model_cfg, dtype, cache_config.block_size);
                let key_block_size = cache_config.num_gpu_blocks
                    * key_block_shape.0
                    * key_block_shape.1
                    * key_block_shape.2
                    * key_block_shape.3;

                let value_block_shape = calculate_value_block_shape(
                    &*self.model_config(config)?,
                    cache_config.block_size,
                );
                let value_block_size = cache_config.num_gpu_blocks
                    * value_block_shape.0
                    * value_block_shape.1
                    * value_block_shape.2;

                key_block_size + value_block_size
            }
            None => {
                // Non-paged KV cache
                let key_block_shape = [
                    max_batch_size,
                    model_cfg.num_kv_heads(),
                    max_seq_len,
                    model_cfg.k_head_dim(),
                ];
                let value_block_shape = [
                    max_batch_size,
                    model_cfg.num_kv_heads(),
                    max_seq_len,
                    model_cfg.v_head_dim(),
                ];

                key_block_shape.into_iter().product::<usize>()
                    + value_block_shape.iter().product::<usize>()
            }
        };
        let kv_cache_size_in_bytes = kv_cache_size_elems * dtype.size_in_bytes();

        let mut per_layer_avail = Vec::new();
        // Always add the CPU as fallback
        for dev in [devices, &[Device::Cpu]].concat() {
            let avail = MemoryUsage.get_memory_available(&dev)?;
            per_layer_avail.push((avail, dev));
        }
        // Reverse so we don't use the cpu first!
        per_layer_avail.reverse();

        // Reverse layer sizes so we can pop
        layer_sizes_in_bytes.reverse();

        let mut device_layers = Vec::new();

        info!("Using automatic device mapping parameters: {params}.");
        if let Some(sub_models) = self.non_mapped_sub_models() {
            let (_, last) = per_layer_avail.last().unwrap();
            info!(
                "The following sub-models will not be device mapped and will be loaded on {}: {}",
                last.device_pretty_repr(),
                sub_models.iter().map(|x| x.to_string()).join(", ")
            );
        }

        let mut current_ordinal = 0;
        let mut current_layer = 0;
        let per_layer_avail_cpy = per_layer_avail.clone();
        let mut mapping_includes_cpu = false;
        while remaining_to_map > 0 && !per_layer_avail.is_empty() {
            let (device_capacity, device) = per_layer_avail
                .pop()
                .context("No more devices to map to. The model does not fit on this system.")?;
            // All usage of 90% of the memory as a maximum.
            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let device_capacity = (device_capacity as f64 * 0.90) as usize;

            // Algorithm is to check the following:
            // 1) (no mapping) if *everything* fits on the first dev (non mapped and mapped)
            // 2) if the mapped activations plus remaining fits on the nth device
            // 3) common case, iteratively find the optimal amount of layers to put on the nth device
            //   - if this is the first dev: must hold the non-mapped act and non-mapped model
            //   - otherwise, must hold the mapped act
            #[allow(clippy::if_same_then_else)]
            let layers_on_device = if current_ordinal == 0
                && device_capacity
                    >= remaining_to_map
                        + non_mapped_max_act_size_in_bytes
                        + mapped_max_act_size_in_bytes
            {
                remaining_to_map = 0;

                num_layers - current_layer
            } else if current_ordinal != 0
                && device_capacity >= remaining_to_map + mapped_max_act_size_in_bytes
            {
                remaining_to_map = 0;

                num_layers - current_layer
            } else {
                // All devices need to account for the max mapped act size
                let mut used_capacity = mapped_max_act_size_in_bytes;
                let mut used_capacity_no_act = mapped_max_act_size_in_bytes;
                let mut layers_on_device = 0;

                // Device w/ ordinal 0 carries the non-mapped things
                if current_ordinal == 0 {
                    // Ensure the activations are properly handled
                    used_capacity = used_capacity.max(non_mapped_max_act_size_in_bytes);
                    used_capacity += non_mapped_size_in_bytes;
                }

                while let Some(&last) = layer_sizes_in_bytes.last() {
                    let delta = last + kv_cache_size_in_bytes;
                    if used_capacity + delta > device_capacity {
                        break;
                    }
                    let _ = layer_sizes_in_bytes.pop().unwrap();
                    used_capacity += delta;
                    used_capacity_no_act += delta;
                    layers_on_device += 1;
                }

                // Do not reduce amount to map if this device can't fit any
                // If the device cannot fit any layers, warn the user.
                if layers_on_device > 0 {
                    remaining_to_map = remaining_to_map.saturating_sub(used_capacity_no_act);
                } else {
                    warn!(
                        "Device {} can fit 0 layers. Consider reducing auto map params from current: {params} (ex. reducing max seq len or max num images)",
                        device.device_pretty_repr(),
                    );
                    current_ordinal += 1;
                    continue;
                }
                layers_on_device
            };

            // CPU mappings are automatically handled by the traditional device mapper, we can just leave them out here.
            if !device.is_cpu() {
                device_layers.push(DeviceLayerMapMetadata {
                    ordinal: current_ordinal,
                    layers: layers_on_device,
                });
                current_ordinal += 1;
            } else {
                mapping_includes_cpu = true;
            }

            current_layer += layers_on_device;
        }
        if remaining_to_map > 0 {
            anyhow::bail!(
                "This model does not fit on the devices {:?}, and exceeds total capacity by {}MB. Auto device mapping params: {params}",
                per_layer_avail_cpy
                    .iter()
                    .rev()
                    .map(|(avail, dev)| format!(
                        "{} (avail: {}MB)",
                        dev.device_pretty_repr(),
                        avail / (1024 * 1024),
                    ))
                    .collect::<Vec<_>>(),
                b_to_mb!(remaining_to_map)
            );
        }

        // TODO: PagedAttention is not supported with CPU for now.
        // Recalculate without PagedAttention metadata.
        if paged_attn_config.is_some_and(|_| mapping_includes_cpu) {
            return self.get_device_layers(
                config,
                num_layers,
                layer_sizes_in_bytes,
                non_mapped_size_in_bytes,
                total_model_size_in_bytes,
                devices,
                dtype,
                params,
                None,
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
