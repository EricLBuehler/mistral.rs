pub(crate) mod auto_device_map;
mod diffusion_loaders;
mod embedding_loaders;
mod multimodal_loaders;
mod normal_loaders;
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
    GLM4MoeLoader, Gemma2Loader, GemmaLoader, GptOssLoader, GraniteMoeHybridLoader,
    HunYuanDenseV1Loader, HunYuanMoEV1Loader, Lfm2Loader, LlamaLoader, MistralLoader,
    MixtralLoader, NormalLoaderType, NormalLoadingMetadata, NormalModel, NormalModelLoader,
    Phi2Loader, Phi3Loader, Phi3_5MoELoader, Qwen2Loader, Qwen3Loader, Qwen3MoELoader,
    Qwen3NextLoader, SmolLm3Loader, Starcoder2Loader,
};

pub use multimodal_loaders::{
    AutoMultimodalLoader, DiffusionGemmaLoader, Gemma3Loader, Gemma3nLoader, Gemma4Loader,
    Idefics2Loader, Idefics3Loader, LLaVALoader, LLaVANextLoader, Lfm2VlLoader, MiniCpmOLoader,
    Mistral3Loader, MultimodalLoaderType, MultimodalModel, MultimodalModelLoader, Phi3VLoader,
    Phi4MMLoader, Qwen2VLLoader, Qwen2_5VLLoader, Qwen3VLLoader, Qwen3VLMoELoader, Qwen3_5Loader,
    Qwen3_5MoeLoader, VLlama4Loader, VLlamaLoader, VoxtralLoader,
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
    DeviceMapSetting, PagedAttentionConfig, Topology, TryIntoDType,
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

    /// Get the preprocessor config (for the multimodal models). This is used to pre process images.
    fn get_preprocessor_config(&self) -> &Option<PathBuf>;

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
}

#[derive(Clone, Copy)]
enum AutoDeviceMapQuantizationSource<'a> {
    Isq(Option<IsqType>),
    Uqff(&'a mistralrs_quant::UqffReader),
}

impl<'a> AutoDeviceMapQuantization<'a> {
    pub fn isq(isq: Option<IsqType>, topology: Option<&'a Topology>) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::Isq(isq),
            topology,
        }
    }

    pub fn uqff(reader: &'a mistralrs_quant::UqffReader) -> Self {
        Self {
            source: AutoDeviceMapQuantizationSource::Uqff(reader),
            topology: None,
        }
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

    fn pack_factor_for_candidates(
        &self,
        names: &[&str],
        dtype: DType,
        fallback: usize,
        promote_default: bool,
    ) -> Result<usize> {
        match self.source {
            AutoDeviceMapQuantizationSource::Uqff(reader) => {
                for name in names {
                    if let Some(pack_factor) = reader.pack_factor_for(name, dtype)? {
                        return Ok(pack_factor);
                    }
                }
                Ok(1)
            }
            AutoDeviceMapQuantizationSource::Isq(default) => {
                let ty = names
                    .iter()
                    .find_map(|name| {
                        self.topology
                            .and_then(|topology| topology.match_for_name(name))
                            .and_then(|topology| topology.isq)
                    })
                    .or_else(|| {
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
        AutoDeviceMapQuantizationSource::Uqff(_) => quantization.pack_factor_for_candidates(
            &[embedding_name, legacy_head_name],
            dtype,
            fallback,
            true,
        ),
        AutoDeviceMapQuantizationSource::Isq(_) => {
            quantization.promoted_pack_factor_for(embedding_name, dtype, fallback)
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
        promoted_tensor_pack_factor(quantization, embedding_name, dtype, fallback)?
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
                AutoDeviceMapQuantizationSource::Uqff(_)
            )
        {
            let mut candidates = embedding_names.to_vec();
            candidates.extend_from_slice(head_names);
            quantization.pack_factor_for_candidates(&candidates, dtype, fallback, true)
        } else {
            quantization.pack_factor_for_candidates(embedding_names, dtype, fallback, true)
        }
    })?;
    let head = quantization.map_or(Ok(fallback), |quantization| {
        quantization.pack_factor_for_candidates(head_names, dtype, fallback, true)
    })?;
    Ok((embedding, head))
}

pub trait DeviceMappedModelLoader {
    /// Maximum activation size of non-mapped parts of this model.
    /// Useful for the multimodal models which may prefer to keep the vison components on the GPU.
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

#[cfg(test)]
mod auto_device_map_quantization_tests {
    use super::*;

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
}
