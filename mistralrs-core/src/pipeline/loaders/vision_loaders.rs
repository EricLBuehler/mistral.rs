use std::any::Any;
use std::sync::atomic::AtomicUsize;
use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::Conv2dConfig;
use image::{ColorType, DynamicImage};
use itertools::Itertools;
use mistralrs_quant::log::once_log_info;
use mistralrs_quant::ShardedVarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use self::minicpmo::{MiniCpmOConfig, MiniCpmOModel, MiniCpmOProcessor};

use super::{DeviceMappedModelLoader, NonMappedSubModel, NormalLoadingMetadata};
use crate::amoe::AnyMoeBaseModelMixin;
use crate::attention::ATTENTION_CHUNK_SIZE;
use crate::device_map::DeviceMapper;
use crate::layers::Conv3dConfig;
use crate::matformer::MatformerSliceConfig;
use crate::paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{
    EitherCache, IsqModel, Modalities, MultimodalPromptPrefixer, Processor, ProcessorCreator,
    SupportedModality,
};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::vision_models::clip::ClipConfig;
use crate::vision_models::gemma3::config::Gemma3Config;
use crate::vision_models::gemma3::{Gemma3Model, Gemma3Processor};
use crate::vision_models::gemma3n::config::{Gemma3nConfig, IntermediateSize};
use crate::vision_models::gemma3n::{Gemma3nModel, Gemma3nProcessor};
use crate::vision_models::idefics2::{Config as Idefics2Config, Idefics2};
use crate::vision_models::idefics2_input_processor::Idefics2Processor;
use crate::vision_models::idefics3::{Idefics3Config, Idefics3Model, Idefics3Processor};
use crate::vision_models::image_processor::ImagePreProcessor;
use crate::vision_models::inputs_processor::Phi4MMProcessor;
use crate::vision_models::llama4::{
    self, Llama4Config, Llama4ImageProcessor, Llama4Model, Llama4Processor,
};
use crate::vision_models::llava::config::Config as LLaVAConfig;
use crate::vision_models::llava15::Model as LLaVA;
use crate::vision_models::llava_inputs_processor::{self, LLaVAProcessor};
use crate::vision_models::llava_next::Model as LLaVANext;
use crate::vision_models::llava_next_inputs_processor::{self, LLaVANextProcessor};
use crate::vision_models::mistral3::{Mistral3Config, Mistral3Model, Mistral3Processor};
use crate::vision_models::mllama::{MLlamaConfig, MLlamaModel, MLlamaProcessor};
use crate::vision_models::phi3::{Config as Phi3Config, Model as Phi3, PHI3V_CLIP_CONFIG};
use crate::vision_models::phi3_inputs_processor::Phi3Processor;
use crate::vision_models::phi4::{Phi4MMConfig, Phi4MMModel, PHI4_MM_VISION_CFG};
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::qwen2_5_vl::{
    Config as Qwen2_5VLConfig, Qwen2_5VLModel, Qwen2_5VLProcessor,
};
use crate::vision_models::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel, Qwen2VLProcessor};
use crate::vision_models::qwen3_vl::{Config as Qwen3VLConfig, Qwen3VLModel, Qwen3VLProcessor};
use crate::vision_models::qwen3_vl_moe::{
    Config as Qwen3VLMoEConfig, Qwen3VLMoEModel, Qwen3VLMoEProcessor,
};
use crate::vision_models::voxtral::config::VoxtralConfig;
use crate::vision_models::voxtral::{VoxtralModel, VoxtralProcessor};
use crate::vision_models::{minicpmo, phi4};

pub trait VisionModel: IsqModel + AnyMoeBaseModelMixin {
    // pixel_values and pixel_attention_mask only specified for prompt seqs
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    /// For a prompt without images. Requires batch size of 1!
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any>;
    /// Return encoder cache hit/miss counters (hits, misses) if this model has an encoder cache.
    fn encoder_cache_counters(&self) -> Option<(Arc<AtomicUsize>, Arc<AtomicUsize>)> {
        None
    }
    /// Reset model-specific state (e.g. cached audio embeddings) between requests.
    /// Called when the pipeline's non-granular state is reset.
    fn reset_model_specific_state(&self) {}
}

pub trait VisionModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>>;
    fn is_gptx(&self, config: &str) -> bool;
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
    fn supports_paged_attention(&self, config: &str) -> bool;
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        // Default is false, specific model must override.
        false
    }
    fn modalities(&self, config: &str) -> Result<Modalities>;
    fn prefixer(&self, config: &str) -> Arc<dyn MultimodalPromptPrefixer>;
    /// Return a default chat template (Jinja string) for models that don't ship a
    /// `tokenizer_config.json` or `chat_template.jinja`. Returns `None` by default.
    /// The `config` parameter is the raw model config JSON, used by `AutoVisionLoader`
    /// to delegate to the correct concrete loader.
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        None
    }
    /// Return default (bos_token, eos_token) strings for models that don't ship a
    /// `tokenizer_config.json`. Used to populate the chat template context and
    /// EOS token detection. Returns `None` by default.
    fn default_bos_eos(&self, _config: &str) -> Option<(String, String)> {
        None
    }
    fn get_device_for_tensor(
        &self,
        config: &str,
        _mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        if loading_isq {
            Ok(Arc::new(|_| DeviceForLoadTensor::Base))
        } else {
            let re = Regex::new(r"\.layers\.(\d+)\.").unwrap();
            let num_layers = self.model_config(config)?.num_layers();
            let closure = move |name: String| {
                if let Some(captures) = re.captures(&name) {
                    captures
                        .get(1)
                        .and_then(|m| m.as_str().parse::<usize>().ok())
                        .map(|l| l.min(num_layers))
                        .map(DeviceForLoadTensor::Idx)
                        .unwrap_or(DeviceForLoadTensor::Base)
                } else {
                    DeviceForLoadTensor::Base
                }
            };

            Ok(Arc::new(closure))
        }
    }
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
/// The architecture to load the vision model as.
pub enum VisionLoaderType {
    #[serde(rename = "phi3v")]
    Phi3V,
    #[serde(rename = "idefics2")]
    Idefics2,
    #[serde(rename = "llava_next")]
    LLaVANext,
    #[serde(rename = "llava")]
    LLaVA,
    #[serde(rename = "vllama")]
    VLlama,
    #[serde(rename = "qwen2vl")]
    Qwen2VL,
    #[serde(rename = "idefics3")]
    Idefics3,
    #[serde(rename = "minicpmo")]
    MiniCpmO,
    #[serde(rename = "phi4mm")]
    Phi4MM,
    #[serde(rename = "qwen2_5vl")]
    Qwen2_5VL,
    #[serde(rename = "gemma3")]
    Gemma3,
    #[serde(rename = "mistral3")]
    Mistral3,
    #[serde(rename = "llama4")]
    Llama4,
    #[serde(rename = "gemma3n")]
    Gemma3n,
    #[serde(rename = "qwen3vl")]
    Qwen3VL,
    #[serde(rename = "qwen3vlmoe")]
    Qwen3VLMoE,
    #[serde(rename = "voxtral")]
    Voxtral,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl VisionLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "Phi3VForCausalLM" => Ok(Self::Phi3V),
            "Idefics2ForConditionalGeneration" => Ok(Self::Idefics2),
            "LlavaNextForConditionalGeneration" => Ok(Self::LLaVANext),
            "LlavaForConditionalGeneration" => Ok(Self::LLaVA),
            "MllamaForConditionalGeneration" => Ok(Self::VLlama),
            "Qwen2VLForConditionalGeneration" => Ok(Self::Qwen2VL),
            "Idefics3ForConditionalGeneration" => Ok(Self::Idefics3),
            "MiniCPMO" => Ok(Self::MiniCpmO),
            "Phi4MMForCausalLM" => Ok(Self::Phi4MM),
            "Qwen2_5_VLForConditionalGeneration" => Ok(Self::Qwen2_5VL),
            "Gemma3ForConditionalGeneration" | "Gemma3ForCausalLM" => Ok(Self::Gemma3),
            "Mistral3ForConditionalGeneration" => Ok(Self::Mistral3),
            "Llama4ForConditionalGeneration" => Ok(Self::Llama4),
            "Gemma3nForConditionalGeneration" => Ok(Self::Gemma3n),
            "Qwen3VLForConditionalGeneration" => Ok(Self::Qwen3VL),
            "Qwen3VLMoeForConditionalGeneration" => Ok(Self::Qwen3VLMoE),
            "VoxtralForConditionalGeneration"
            | "VoxtralRealtimeForConditionalGeneration" => Ok(Self::Voxtral),
            other => anyhow::bail!(
                "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
            ),
        }
    }
}

impl FromStr for VisionLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "phi3v" => Ok(Self::Phi3V),
            "idefics2" => Ok(Self::Idefics2),
            "llava_next" => Ok(Self::LLaVANext),
            "llava" => Ok(Self::LLaVA),
            "vllama" => Ok(Self::VLlama),
            "qwen2vl" => Ok(Self::Qwen2VL),
            "idefics3" => Ok(Self::Idefics3),
            "minicpmo" => Ok(Self::MiniCpmO),
            "phi4mm" => Ok(Self::Phi4MM),
            "qwen2_5vl" => Ok(Self::Qwen2_5VL),
            "gemma3" => Ok(Self::Gemma3),
            "mistral3" => Ok(Self::Mistral3),
            "llama4" => Ok(Self::Llama4),
            "gemma3n" => Ok(Self::Gemma3n),
            "qwen3vl" => Ok(Self::Qwen3VL),
            "qwen3vlmoe" => Ok(Self::Qwen3VLMoE),
            "voxtral" => Ok(Self::Voxtral),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `phi3v`, `idefics2`, `llava_next`, `llava`, `vllama`, `qwen2vl`, `idefics3`, `minicpmo`, `phi4mm`, `qwen2_5vl`, `gemma3`, `mistral3`, `llama4`, `gemma3n`, `qwen3vl`, `qwen3vlmoe`, `voxtral`.")),
        }
    }
}

impl std::fmt::Display for VisionLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            VisionLoaderType::Phi3V => "phi3v",
            VisionLoaderType::Idefics2 => "idefics2",
            VisionLoaderType::LLaVANext => "llava_next",
            VisionLoaderType::LLaVA => "llava",
            VisionLoaderType::VLlama => "vllama",
            VisionLoaderType::Qwen2VL => "qwen2vl",
            VisionLoaderType::Idefics3 => "idefics3",
            VisionLoaderType::MiniCpmO => "minicpmo",
            VisionLoaderType::Phi4MM => "phi4mm",
            VisionLoaderType::Qwen2_5VL => "qwen2_5vl",
            VisionLoaderType::Gemma3 => "gemma3",
            VisionLoaderType::Mistral3 => "mistral3",
            VisionLoaderType::Llama4 => "llama4",
            VisionLoaderType::Gemma3n => "gemma3n",
            VisionLoaderType::Qwen3VL => "qwen3vl",
            VisionLoaderType::Qwen3VLMoE => "qwen3vlmoe",
            VisionLoaderType::Voxtral => "voxtral",
        };
        write!(f, "{name}")
    }
}

#[derive(Deserialize)]
struct AutoVisionLoaderConfig {
    #[serde(default)]
    architectures: Vec<String>,
    /// Voxtral params.json uses a `multimodal` key instead of `architectures`.
    #[serde(default)]
    multimodal: Option<serde_json::Value>,
}

/// Automatically selects a VisionModelLoader implementation based on the JSON `architectures` field.
pub struct AutoVisionLoader;

impl AutoVisionLoader {
    fn get_loader(config: &str) -> Result<Box<dyn VisionModelLoader>> {
        let auto_cfg: AutoVisionLoaderConfig = serde_json::from_str(config)?;

        // Voxtral: params.json has `multimodal` but no `architectures`
        if auto_cfg.multimodal.is_some() && auto_cfg.architectures.is_empty() {
            once_log_info("Automatic loader type determined to be `voxtral`");
            return Ok(Box::new(VoxtralLoader));
        }

        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected exactly one architecture in config");
        }

        let name = &auto_cfg.architectures[0];
        let tp = VisionLoaderType::from_causal_lm_name(name)?;

        once_log_info(format!("Automatic loader type determined to be `{tp}`"));

        // Delegate to the concrete loader
        Ok(match tp {
            VisionLoaderType::Phi3V => Box::new(Phi3VLoader),
            VisionLoaderType::Idefics2 => Box::new(Idefics2Loader),
            VisionLoaderType::LLaVANext => Box::new(LLaVANextLoader),
            VisionLoaderType::LLaVA => Box::new(LLaVALoader),
            VisionLoaderType::VLlama => Box::new(VLlamaLoader),
            VisionLoaderType::Qwen2VL => Box::new(Qwen2VLLoader),
            VisionLoaderType::Idefics3 => Box::new(Idefics3Loader),
            VisionLoaderType::MiniCpmO => Box::new(MiniCpmOLoader),
            VisionLoaderType::Phi4MM => Box::new(Phi4MMLoader),
            VisionLoaderType::Qwen2_5VL => Box::new(Qwen2_5VLLoader),
            VisionLoaderType::Gemma3 => Box::new(Gemma3Loader),
            VisionLoaderType::Mistral3 => Box::new(Mistral3Loader),
            VisionLoaderType::Llama4 => Box::new(VLlama4Loader),
            VisionLoaderType::Gemma3n => Box::new(Gemma3nLoader),
            VisionLoaderType::Qwen3VL => Box::new(Qwen3VLLoader),
            VisionLoaderType::Qwen3VLMoE => Box::new(Qwen3VLMoELoader),
            VisionLoaderType::Voxtral => Box::new(VoxtralLoader),
        })
    }
}

impl VisionModelLoader for AutoVisionLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }

    fn is_gptx(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoVisionLoader get_loader")
            .is_gptx(config)
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }

    fn get_processor(
        &self,
        model_config: &str,
        proc_cfg: Option<ProcessorConfig>,
        preproc_cfg: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Self::get_loader(model_config)
            .expect("AutoVisionLoader get_loader")
            .get_processor(model_config, proc_cfg, preproc_cfg, max_edge)
    }

    fn supports_paged_attention(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoVisionLoader")
            .supports_paged_attention(config)
    }

    fn modalities(&self, config: &str) -> Result<Modalities> {
        Self::get_loader(config)?.modalities(config)
    }

    fn supports_prefix_cacher(&self, config: &str) -> bool {
        Self::get_loader(config)
            .expect("AutoVisionLoader")
            .supports_prefix_cacher(config)
    }

    fn prefixer(&self, config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Self::get_loader(config)
            .expect("AutoVisionLoader")
            .prefixer(config)
    }

    fn default_chat_template(&self, config: &str) -> Option<String> {
        Self::get_loader(config).ok()?.default_chat_template(config)
    }

    fn default_bos_eos(&self, config: &str) -> Option<(String, String)> {
        Self::get_loader(config).ok()?.default_bos_eos(config)
    }

    fn get_device_for_tensor(
        &self,
        config: &str,
        mapper: &dyn DeviceMapper,
        loading_isq: bool,
    ) -> Result<Arc<dyn Fn(String) -> DeviceForLoadTensor + Send + Sync + 'static>> {
        Self::get_loader(config)?.get_device_for_tensor(config, mapper, loading_isq)
    }
}

impl IsqModelLoader for AutoVisionLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes_moqe(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates_moqe(config)
    }
}

impl DeviceMappedModelLoader for AutoVisionLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_max_act_size_elems(config, params)
    }
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        Self::get_loader(config)?.layer_sizes_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            _matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Self::get_loader(config)?.model_config(config)
    }
}

macro_rules! bias_if {
    ($cond:expr, $size:expr) => {
        if $cond {
            $size
        } else {
            0
        }
    };
}

fn get_clip_vit_num_elems(cfg: &ClipConfig) -> usize {
    let pre_layer_norm = cfg.hidden_size;
    let final_layer_norm = cfg.hidden_size;

    let num_patches = (cfg.image_size / cfg.patch_size).pow(2);
    let num_positions = num_patches + 1;

    let class_embedding = cfg.hidden_size;

    let position_ids = num_positions;
    let position_embedding = num_positions * cfg.hidden_size;

    let conv2dconfig = Conv2dConfig {
        stride: cfg.patch_size,
        ..Default::default()
    };
    let patch_embedding =
        cfg.num_channels * cfg.hidden_size / conv2dconfig.groups * cfg.patch_size * cfg.patch_size;

    let encoder_layer_elems = {
        let layer_norm1 = cfg.hidden_size;
        let layer_norm2 = cfg.hidden_size;

        let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
        let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

        let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
        let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

        layer_norm1 + layer_norm2 + q_proj + k_proj + v_proj + o_proj + fc1 + fc2
    };

    pre_layer_norm
        + final_layer_norm
        + class_embedding
        + position_ids
        + position_embedding
        + patch_embedding
        + cfg.num_hidden_layers * encoder_layer_elems
}

// ======================== Phi 3 loader

/// [`VisionLoader`] for a Phi 3 Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Phi3VLoader;

pub struct Phi3VPrefixer;

impl MultimodalPromptPrefixer for Phi3VPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        // Image indexing starts at 0.
        format!(
            "{}{prompt}",
            image_indexes
                .into_iter()
                .map(|image_index| format!("<|image_{}|>", image_index + 1))
                .join("")
        )
    }
}

impl VisionModelLoader for Phi3VLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::phi3::Config = serde_json::from_str(config)?;
        Ok(Box::new(Phi3::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::phi3::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Phi3Processor::new_processor(processor_config, preprocessor_config)
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Phi3VPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Phi3VLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi3VLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        // NOTE: we ignore max_num_images although it can only be one...
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Phi3Config = serde_json::from_str(config)?;

        let vcfg = &PHI3V_CLIP_CONFIG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        // NOTE: we ignore max_num_images although it can only be one...
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Phi3Config = serde_json::from_str(config)?;

        let vcfg = &PHI3V_CLIP_CONFIG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_vision_attn = {
            (max_batch_size * max_num_images) * cfg.num_attention_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Phi3Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;

            let image_embed = {
                let projection_cls = cfg
                    .embd_layer
                    .projection_cls
                    .clone()
                    .unwrap_or("linear".to_string());
                let with_learnable_separator =
                    cfg.embd_layer.with_learnable_separator.unwrap_or(false);
                let use_hd_transform = cfg.embd_layer.use_hd_transform.unwrap_or(false);
                let image_dim_out = cfg.img_processor.image_dim_out;

                let proj = match (projection_cls.as_str(), use_hd_transform) {
                    ("linear", _) => image_dim_out * cfg.hidden_size + cfg.hidden_size,
                    ("mlp", true) => {
                        let a = (image_dim_out * 4) * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    ("mlp", false) => {
                        let a = image_dim_out * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    _ => {
                        anyhow::bail!("projection_cls=`{projection_cls}` not implemented.");
                    }
                };

                let (glb_gn, sub_gn) = if with_learnable_separator {
                    let glb_gn = image_dim_out * 4;
                    let sub_gn = image_dim_out * 4;
                    (glb_gn, sub_gn)
                } else {
                    (0, 0)
                };

                let clip_vit = get_clip_vit_num_elems(&PHI3V_CLIP_CONFIG);

                proj + glb_gn + sub_gn + clip_vit
            };

            embed_tokens + lm_head + norm + image_embed
        };

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Phi3Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj = (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_up_proj = h_size * (2 * i_size) / weight_pack_factor;
            let down_proj = h_size * i_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + qkv_proj
                + o_proj
                + gate_up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Phi3Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Phi3Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Idefics 2 loader

/// [`VisionLoader`] for an Idefics 2 Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Idefics2Loader;

pub struct Idefics2Prefixer;

impl MultimodalPromptPrefixer for Idefics2Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl VisionModelLoader for Idefics2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::idefics2::Config = serde_json::from_str(config)?;
        Ok(Box::new(Idefics2::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::idefics2::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Idefics2Processor::new(
            processor_config.unwrap(),
            preprocessor_config,
            max_edge,
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Idefics2Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Idefics2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Idefics2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Idefics2Config = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Idefics2Config = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_vision_attn = {
            // do_image_splitting = true
            let images_factor = 5;

            (max_batch_size * images_factor * max_num_images)
                * cfg.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Idefics2Config = serde_json::from_str(config)?;
        let text_elems = {
            let tie_word_embeddings = cfg.tie_word_embeddings;
            let cfg = &cfg.text_config;

            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let connector_elems = {
            let tcfg = &cfg.text_config;
            let vcfg = &cfg.vision_config;
            let gate_proj = vcfg.hidden_size * tcfg.intermediate_size;
            let up_proj = vcfg.hidden_size * tcfg.intermediate_size;
            let down_proj = tcfg.intermediate_size * tcfg.hidden_size;

            let perceiver_elems = {
                let tcfg = &cfg.text_config;
                let pcfg = &cfg.perceiver_config;

                let n_latents = pcfg.resampler_n_latents;
                let hidden_size = tcfg.hidden_size;
                let depth = pcfg.resampler_depth;

                let norm = tcfg.hidden_size;
                let latents = n_latents * hidden_size;

                let layer_elems = {
                    let input_latents_norm = hidden_size;
                    let input_context_norm = hidden_size;
                    let post_attn_norm = hidden_size;

                    let num_heads = pcfg.resampler_n_heads;
                    let head_dim = pcfg.resampler_head_dim;
                    let num_key_value_heads = pcfg.num_key_value_heads;

                    let q_proj = hidden_size * num_heads * head_dim;
                    let k_proj = hidden_size * num_key_value_heads * head_dim;
                    let v_proj = hidden_size * num_key_value_heads * head_dim;
                    let o_proj = num_heads * head_dim * hidden_size;

                    let gate_proj = hidden_size * hidden_size * 4;
                    let up_proj = hidden_size * hidden_size * 4;
                    let down_proj = hidden_size * 4 * hidden_size;

                    input_latents_norm
                        + input_context_norm
                        + post_attn_norm
                        + q_proj
                        + k_proj
                        + v_proj
                        + o_proj
                        + gate_proj
                        + up_proj
                        + down_proj
                };

                norm + latents + layer_elems * depth
            };

            gate_proj + up_proj + down_proj + perceiver_elems
        };

        let vision_transformer = {
            let cfg = &cfg.vision_config;

            let post_layernorm = cfg.hidden_size;

            let conv_config = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                * cfg.patch_size
                * cfg.patch_size;

            let num_patches_per_side = cfg.image_size / cfg.patch_size;
            let num_patches = num_patches_per_side.pow(2);
            let position_embedding = num_patches * cfg.hidden_size;

            let layer_elems = {
                let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
            };

            post_layernorm + patch_embedding + position_embedding + layer_elems
        };

        let elems = text_elems + connector_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Idefics2Config = serde_json::from_str(config)?;
        let cfg = cfg.text_config;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Idefics2Config = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Idefics2Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== LLaVANext Loader

/// [`VisionLoader`] for an LLaVANext Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct LLaVANextLoader;

pub struct LLaVANextPrefixer;

impl MultimodalPromptPrefixer for LLaVANextPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", "<image>".repeat(image_indexes.len()))
    }
}

impl VisionModelLoader for LLaVANextLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(LLaVANext::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(LLaVANextProcessor::new(model_config))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(LLaVANextPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for LLaVANextLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for LLaVANextLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: LLaVAConfig = serde_json::from_str(config)?;

        #[allow(clippy::cast_possible_truncation)]
        let img_seq_len =
            llava_next_inputs_processor::LLaVANextInputProcessor::get_num_image_tokens(
                &config,
                (max_image_shape.0 as u32, max_image_shape.1 as u32),
            );
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
            let cfg = &config.text_config;
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);

            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: LLaVAConfig = serde_json::from_str(config)?;

        #[allow(clippy::cast_possible_truncation)]
        let img_seq_len =
            llava_next_inputs_processor::LLaVANextInputProcessor::get_num_image_tokens(
                &config,
                (max_image_shape.0 as u32, max_image_shape.1 as u32),
            );

        let max_vision_attn = {
            (max_batch_size * max_num_images)
                * config.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let image_newline = cfg.text_config.hidden_size;
        let mmproj = {
            let linear_1 = cfg.vision_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;
            let linear_2 = cfg.text_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;

            linear_1 + linear_2
        };
        let vision_tower = get_clip_vit_num_elems(&cfg.to_clip_config());

        let elems = text_elems + image_newline + mmproj + vision_tower;
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let cfg = &cfg.text_config;
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.text_config.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== LLaVA Loader

/// [`VisionLoader`] for an LLaVA Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct LLaVALoader;

pub struct LLaVAPrefixer;

impl MultimodalPromptPrefixer for LLaVAPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", "<image>".repeat(image_indexes.len()))
    }
}

impl VisionModelLoader for LLaVALoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(LLaVA::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::llava::config::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(LLaVAProcessor::new(model_config))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(LLaVAPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for LLaVALoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for LLaVALoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: LLaVAConfig = serde_json::from_str(config)?;

        let img_seq_len =
            llava_inputs_processor::LLaVAInputProcessor::get_num_image_tokens(&config);
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
            let cfg = &config.text_config;
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);

            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: LLaVAConfig = serde_json::from_str(config)?;

        let img_seq_len =
            llava_inputs_processor::LLaVAInputProcessor::get_num_image_tokens(&config);

        let max_vision_attn = {
            (max_batch_size * max_num_images)
                * config.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let image_newline = cfg.text_config.hidden_size;
        let mmproj = {
            let linear_1 = cfg.vision_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;
            let linear_2 = cfg.text_config.hidden_size * cfg.text_config.hidden_size
                + cfg.text_config.hidden_size;

            linear_1 + linear_2
        };
        let vision_tower = get_clip_vit_num_elems(&cfg.to_clip_config());

        let elems = text_elems + image_newline + mmproj + vision_tower;
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let cfg = &cfg.text_config;
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.text_config.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: LLaVAConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== MLlama Loader

/// [`VisionLoader`] for an Llama Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct VLlamaLoader;

pub struct VLlamaPrefixer;

impl MultimodalPromptPrefixer for VLlamaPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!("{}{prompt}", "<|image|>".repeat(image_indexes.len()))
    }
}

impl VisionModelLoader for VLlamaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::mllama::MLlamaConfig = serde_json::from_str(config)?;
        Ok(Box::new(MLlamaModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::mllama::MLlamaConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(MLlamaProcessor::new())
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VLlamaPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for VLlamaLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let cross_attn_layers = &config.text_config.cross_attention_layers;
        let transformer_layers =
            (0..config.text_config.num_hidden_layers).filter(|i| !cross_attn_layers.contains(i));
        let mut text_regexes = Vec::new();
        for layer in transformer_layers {
            text_regexes.extend(vec![
                // Attention text
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.q_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.k_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.v_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.self_attn\.o_proj\.(weight|bias)$"
                ))?,
                // MLP text
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.gate_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.up_proj\.(weight|bias)$"
                ))?,
                Regex::new(&format!(
                    r"language_model.model.layers\.{layer}\.mlp\.down_proj\.(weight|bias)$"
                ))?,
            ]);
        }
        let vision_regexes = vec![
            // Vision attention (transformer)
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.transformer.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // Vision attention (global transforemr)
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"vision_model.global_transformer.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
            )?,
            // MLP vision
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ];

        Ok([text_regexes, vision_regexes].concat())
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for VLlamaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: MLlamaConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &config.vision_config;
            let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
            let num_padding_patches = (8 - (num_patches as isize % 8)) % 8;
            cfg.max_num_tiles * (num_patches as isize + num_padding_patches) as usize
        };
        let img_seq_len = img_seq_len * max_num_images;

        let max_cross_text_attn = {
            let cfg = &config.text_config;
            max_batch_size * cfg.num_attention_heads * img_seq_len * img_seq_len
        };

        let max_self_text_attn = {
            let cfg = &config.text_config;
            max_batch_size * cfg.num_attention_heads * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2)
        };

        Ok(max_self_text_attn.max(max_cross_text_attn))
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let config: MLlamaConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &config.vision_config;
            let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
            let num_padding_patches = (8 - (num_patches as isize % 8)) % 8;
            cfg.max_num_tiles * (num_patches as isize + num_padding_patches) as usize
        };
        let max_vision_attn = {
            let cfg = &config.vision_config;
            (max_batch_size * max_num_images) * cfg.num_attention_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &config.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_elems = {
            let cfg = &config.vision_config;

            let conv_cfg = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_cfg.groups
                * cfg.patch_size
                * cfg.patch_size;

            let class_embedding = cfg.hidden_size;

            let gated_positional_embedding = {
                let num_patches = (cfg.image_size / cfg.patch_size).pow(2) + 1;
                let embedding = num_patches * cfg.hidden_size;
                let tile_embedding = (cfg.max_aspect_ratio_id() + 1)
                    * (cfg.max_num_tiles * num_patches * cfg.hidden_size);

                embedding + tile_embedding
            };

            let pre_tile_positional_embedding =
                (cfg.max_aspect_ratio_id() + 1) * (cfg.max_num_tiles * cfg.hidden_size);
            let post_tile_positional_embedding =
                (cfg.max_aspect_ratio_id() + 1) * (cfg.max_num_tiles * cfg.hidden_size);

            let layernorm_pre = cfg.hidden_size;
            let layernorm_post = cfg.hidden_size;

            let encoder_layer = {
                let input_layernorm = cfg.hidden_size + cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;
                let q_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let k_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let v_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;
                let o_proj =
                    cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor;

                let fc1 = (cfg.hidden_size * cfg.intermediate_size) / weight_pack_factor
                    + cfg.intermediate_size;
                let fc2 = (cfg.intermediate_size * cfg.hidden_size) / weight_pack_factor
                    + cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + fc1
                    + fc2
            };

            patch_embedding
                + class_embedding
                + gated_positional_embedding
                + pre_tile_positional_embedding
                + post_tile_positional_embedding
                + layernorm_pre
                + layernorm_post
                + encoder_layer * (cfg.num_hidden_layers + cfg.num_global_layers)
        };

        let elems = text_elems + vision_elems;
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let cfg = &config.text_config;

        let mut layer_sizes = Vec::new();

        for i in 0..cfg.num_hidden_layers {
            let weight_pack_factor = if cfg.cross_attention_layers.contains(&i) {
                // No isq for cross attention
                1
            } else {
                weight_pack_factor
            };

            let per_layer_elems = {
                let input_layernorm = cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size;

                let size_in = cfg.hidden_size;
                let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
                let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
                let q_proj = size_in * size_q / weight_pack_factor;
                let k_proj = size_in * size_kv / weight_pack_factor;
                let v_proj = size_in * size_kv / weight_pack_factor;
                let o_proj = size_q * size_in / weight_pack_factor;

                let h_size = cfg.hidden_size;
                let i_size = cfg.intermediate_size;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;

                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + gate_proj
                    + up_proj
                    + down_proj
            };

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        Ok(config.text_config.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: MLlamaConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Qwen2VL Loader

/// [`VisionLoader`] for an Qwen2-VL model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Qwen2VLLoader;

pub struct Qwen2VLPrefixer;

impl MultimodalPromptPrefixer for Qwen2VLPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            format!(
                "{}{}{}",
                Qwen2VLProcessor::VISION_START,
                Qwen2VLProcessor::IMAGE_PAD,
                Qwen2VLProcessor::VISION_END
            )
            .repeat(image_indexes.len())
        )
    }
}

impl VisionModelLoader for Qwen2VLLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen2VLModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen2VLProcessor::new(max_edge))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Qwen2VLPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen2VLLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        // For images, grid_t=1. After spatial merging, grid_h and grid_w are reduced.
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images (temporal dimension is for video only)
            let grid_t = 1;
            // After patch embedding and spatial merge, the effective grid dimensions are reduced
            let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        // For the vision encoder, before spatial merging
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images
            let grid_t = 1;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };

        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        let text_elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let patch_merger = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.embed_dim * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.hidden_size + cfg.hidden_size;

            let ln_q = cfg.embed_dim + bias_if!(true, cfg.embed_dim);

            mlp0 + mlp2 + ln_q
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            cfg.in_channels * cfg.embed_dim / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2]
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.embed_dim + bias_if!(true, cfg.embed_dim);
            let norm2 = cfg.embed_dim + bias_if!(true, cfg.embed_dim);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let mlp_hidden_dim = (cfg.embed_dim as f64 * cfg.mlp_ratio) as usize;
            let fc1 = cfg.embed_dim * mlp_hidden_dim + mlp_hidden_dim;
            let fc2 = cfg.embed_dim * mlp_hidden_dim + cfg.embed_dim;

            let qkv = cfg.embed_dim * cfg.embed_dim * 3 + cfg.embed_dim * 3;
            let out = cfg.embed_dim * cfg.embed_dim + cfg.embed_dim;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems =
            text_elems + patch_merger + patch_embed + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Idefics 3 loader

/// [`VisionLoader`] for an Idefics 3 Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Idefics3Loader;

pub struct Idefics3Prefixer;

impl MultimodalPromptPrefixer for Idefics3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl VisionModelLoader for Idefics3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::idefics3::Idefics3Config = serde_json::from_str(config)?;
        Ok(Box::new(Idefics3Model::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::idefics3::Idefics3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Idefics3Processor::new(
            processor_config.unwrap_or_default(),
            preprocessor_config,
            max_edge,
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Idefics3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Idefics3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"model.text_model.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model.text_model.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model.text_model.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model.text_model.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"model.text_model.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model.text_model.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model.text_model.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.text_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // // Attention (vision)
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
            // )?,
            // Regex::new(
            //     r"model\.vision_model\.encoder\.layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$",
            // )?,
            // MLP (vision)
            // Regex::new(r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            // Regex::new(r"model\.vision_model\.encoder\.layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Idefics3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Idefics3Config = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Idefics3Config = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_vision_attn = {
            // do_image_splitting = true
            let images_factor = 5;

            (max_batch_size * images_factor * max_num_images)
                * cfg.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Idefics3Config = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;

            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let connector_elems = {
            let in_dim = cfg.vision_config.hidden_size * cfg.scale_factor.pow(2);
            let out_dim = cfg.text_config.hidden_size;

            in_dim * out_dim
        };

        let vision_transformer = {
            let cfg = &cfg.vision_config;

            let post_layernorm = cfg.hidden_size;

            let conv_config = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                * cfg.patch_size
                * cfg.patch_size;

            let num_patches_per_side = cfg.image_size / cfg.patch_size;
            let num_patches = num_patches_per_side.pow(2);
            let position_embedding = num_patches * cfg.hidden_size;

            let layer_elems = {
                let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
            };

            post_layernorm
                + patch_embedding
                + position_embedding
                + layer_elems * cfg.num_hidden_layers
        };

        let elems = text_elems + connector_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Idefics3Config = serde_json::from_str(config)?;
        let cfg = cfg.text_config;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Idefics3Config = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Idefics3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== MiniCpm-O loader

/// [`VisionLoader`] for an MiniCpm-O model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct MiniCpmOLoader;

pub struct MiniCpmOPrefixer;

impl MultimodalPromptPrefixer for MiniCpmOPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            "(<image>./</image>)".repeat(image_indexes.len())
        )
    }
}

impl VisionModelLoader for MiniCpmOLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::minicpmo::MiniCpmOConfig = serde_json::from_str(config)?;
        Ok(Box::new(MiniCpmOModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::minicpmo::MiniCpmOConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(MiniCpmOProcessor::new(
            processor_config.unwrap_or_default(),
            preprocessor_config,
            max_edge,
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(MiniCpmOPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for MiniCpmOLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"llm.lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"llm.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"llm.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for MiniCpmOLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;

        let num_patches = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_vision_attn = {
            // do_image_splitting = true
            let images_factor = 5;

            (max_batch_size * images_factor * max_num_images)
                * cfg.vision_config.num_attention_heads
                * img_seq_len
                * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &cfg.text_config;

            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_transformer = {
            let cfg = &cfg.vision_config;

            let post_layernorm = cfg.hidden_size;

            let conv_config = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                * cfg.patch_size
                * cfg.patch_size;

            let num_patches_per_side = cfg.image_size / cfg.patch_size;
            let num_patches = num_patches_per_side.pow(2);
            let position_embedding = num_patches * cfg.hidden_size;

            let layer_elems = {
                let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
            };

            post_layernorm
                + patch_embedding
                + position_embedding
                + layer_elems * cfg.num_hidden_layers
        };

        let elems = text_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;
        let cfg = cfg.text_config;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: MiniCpmOConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi 4MM loader

/// [`VisionLoader`] for a Phi 4MM Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Phi4MMLoader;

pub struct Phi4MMPrefixer;

impl MultimodalPromptPrefixer for Phi4MMPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        // Image indexing starts at 0.

        format!(
            "{}{prompt}",
            image_indexes
                .into_iter()
                .map(|image_index| format!("<|image_{}|>", image_index + 1))
                .join("")
        )
    }
    fn prefix_audio(&self, audio_indexes: Vec<usize>, prompt: &str) -> String {
        // Image indexing starts at 0.

        format!(
            "{}{prompt}",
            audio_indexes
                .into_iter()
                .map(|audio_index| format!("<|audio_{}|>", audio_index + 1))
                .join("")
        )
    }
}

impl VisionModelLoader for Phi4MMLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: crate::vision_models::phi4::Phi4MMConfig = serde_json::from_str(config)?;
        Ok(Box::new(Phi4MMModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::phi4::Phi4MMConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Phi4MMProcessor::new_processor(processor_config, preprocessor_config)
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Phi4MMPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Audio,
            ],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Phi4MMLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi4MMLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        // NOTE: we ignore max_num_images although it can only be one...
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Phi4MMConfig = serde_json::from_str(config)?;

        let vcfg = &PHI4_MM_VISION_CFG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = (num_patches + 1) * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let vcfg = &PHI4_MM_VISION_CFG;

        let num_patches = (vcfg.image_size / vcfg.patch_size).pow(2);
        let img_seq_len = num_patches + 1;

        let max_batch_size = max_batch_size
            * (max_image_shape
                .0
                .div_ceil(phi4::inputs_processor::DYHD_BASE_RESOLUTION)
                * max_image_shape
                    .1
                    .div_ceil(phi4::inputs_processor::DYHD_BASE_RESOLUTION)
                + 1);

        let max_vision_attn = (max_batch_size * max_num_images)
            * vcfg.num_attention_heads
            * img_seq_len
            * img_seq_len;
        let max_qkv = 3
            * (max_batch_size
                * vcfg.num_attention_heads
                * img_seq_len
                * (vcfg.hidden_size / vcfg.num_attention_heads));

        Ok(max_vision_attn + max_qkv)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;

            let image_embed = if let Some(img_embed) = &cfg.embd_layer.image_embd_layer {
                let projection_cls = img_embed
                    .projection_cls
                    .clone()
                    .unwrap_or("linear".to_string());
                let with_learnable_separator = img_embed.with_learnable_separator.unwrap_or(false);
                let use_hd_transform = img_embed.use_hd_transform.unwrap_or(false);
                let image_dim_out = PHI4_MM_VISION_CFG.hidden_size;

                let proj = match (projection_cls.as_str(), use_hd_transform) {
                    ("linear", _) => image_dim_out * cfg.hidden_size + cfg.hidden_size,
                    ("mlp", true) => {
                        let a = (image_dim_out * 4) * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    ("mlp", false) => {
                        let a = image_dim_out * cfg.hidden_size + cfg.hidden_size;
                        let b = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        a + b
                    }
                    _ => {
                        anyhow::bail!("projection_cls=`{projection_cls}` not implemented.");
                    }
                };

                let (glb_gn, sub_gn) = if with_learnable_separator {
                    let glb_gn = image_dim_out * 4;
                    let sub_gn = image_dim_out * 4;
                    (glb_gn, sub_gn)
                } else {
                    (0, 0)
                };

                let vision_transformer = {
                    let cfg = &PHI4_MM_VISION_CFG;

                    let post_layernorm = cfg.hidden_size;

                    let conv_config = Conv2dConfig {
                        stride: cfg.patch_size,
                        ..Default::default()
                    };
                    let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                        * cfg.patch_size
                        * cfg.patch_size;

                    let num_patches_per_side = cfg.image_size / cfg.patch_size;
                    let num_patches = num_patches_per_side.pow(2);
                    let position_embedding = num_patches * cfg.hidden_size;

                    let layer_elems = {
                        let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                        let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                        let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                        let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                        let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                        let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                        layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
                    };

                    post_layernorm
                        + patch_embedding
                        + position_embedding
                        + layer_elems * cfg.num_hidden_layers
                };

                proj + glb_gn + sub_gn + vision_transformer
            } else {
                0
            };

            embed_tokens + lm_head + norm + image_embed
        };

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads() * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj = (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_up_proj = h_size * (2 * i_size) / weight_pack_factor;
            let down_proj = h_size * i_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + qkv_proj
                + o_proj
                + gate_up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
}

// ======================== Qwen2_5VL Loader

/// [`VisionLoader`] for an Qwen2_5VL model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Qwen2_5VLLoader;

pub struct Qwen2_5VLPrefixer;

impl MultimodalPromptPrefixer for Qwen2_5VLPrefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            format!(
                "{}{}{}",
                Qwen2_5VLProcessor::VISION_START,
                Qwen2_5VLProcessor::IMAGE_PAD,
                Qwen2_5VLProcessor::VISION_END
            )
            .repeat(image_indexes.len())
        )
    }
}

impl VisionModelLoader for Qwen2_5VLLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen2_5VLModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Qwen2_5VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen2_5VLProcessor::new(max_edge))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Qwen2_5VLPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen2_5VLLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2_5VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };
        let img_seq_len = img_seq_len * max_num_images;

        let max_text_attn = {
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;

        let img_seq_len = {
            let cfg = &cfg.vision_config;
            let grid_t = max_num_images / cfg.temporal_patch_size;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };

        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;
        let text_elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let patch_merger = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.hidden_size + cfg.hidden_size;

            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            mlp0 + mlp2 + ln_q
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2]
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let norm2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
            let fc2 = cfg.hidden_size * cfg.intermediate_size + cfg.hidden_size;

            let qkv = cfg.hidden_size * cfg.hidden_size * 3 + cfg.hidden_size * 3;
            let out = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems =
            text_elems + patch_merger + patch_embed + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Gemma 3 Loader

/// [`VisionLoader`] for an Gemma 3 model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Gemma3Loader;

pub struct Gemma3Prefixer;

impl MultimodalPromptPrefixer for Gemma3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for Gemma3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;
        Ok(Box::new(Gemma3Model::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Gemma3Config = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let config: Gemma3Config = serde_json::from_str(config).unwrap();
        // Handle the Gemma 3 1b case here
        Arc::new(Gemma3Processor::new(
            processor_config.unwrap_or_default(),
            matches!(config, Gemma3Config::WithVision { .. }),
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Gemma3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3Config = serde_json::from_str(config)?;

        match cfg {
            Gemma3Config::Text(text_config) => Ok(max_batch_size
                * text_config.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2)),
            Gemma3Config::WithVision {
                text_config,
                vision_config,
                ..
            } => {
                let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
                let img_seq_len = (num_patches + 1) * max_num_images;

                let max_text_attn = {
                    // This model injects the vision information directly into the input embeddings
                    let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
                    max_batch_size * text_config.num_attention_heads * max_seq_len * max_seq_len
                };
                Ok(max_text_attn)
            }
        }
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3Config = serde_json::from_str(config)?;

        match cfg {
            Gemma3Config::WithVision { vision_config, .. } => {
                let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
                let img_seq_len = num_patches + 1;

                let max_vision_attn = {
                    (max_batch_size * max_num_images)
                        * vision_config.num_attention_heads
                        * img_seq_len
                        * img_seq_len
                };

                Ok(max_vision_attn)
            }
            Gemma3Config::Text(_) => Ok(0),
        }
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = match &cfg {
                Gemma3Config::Text(cfg) => cfg,
                Gemma3Config::WithVision { text_config, .. } => text_config,
            };
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_transformer = if let Gemma3Config::WithVision {
            vision_config: cfg, ..
        } = &cfg
        {
            let post_layernorm = cfg.hidden_size;

            let conv_config = Conv2dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let patch_embedding = cfg.num_channels * cfg.hidden_size / conv_config.groups
                * cfg.patch_size
                * cfg.patch_size;

            let num_patches_per_side = cfg.image_size / cfg.patch_size;
            let num_patches = num_patches_per_side.pow(2);
            let position_embedding = num_patches * cfg.hidden_size;

            let layer_elems = {
                let layer_norm_1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
                let layer_norm_2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

                let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
                let fc2 = cfg.intermediate_size * cfg.hidden_size + cfg.hidden_size;

                let q_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let k_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let v_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;
                let o_proj = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

                layer_norm_1 + layer_norm_2 + fc1 + fc2 + q_proj + k_proj + v_proj + o_proj
            };

            post_layernorm
                + patch_embedding
                + position_embedding
                + layer_elems * cfg.num_hidden_layers
        } else {
            0
        };

        let elems = text_elems + vision_transformer;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let txt_cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };
        let per_layer_elems = {
            let cfg = txt_cfg;

            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim * cfg.num_attention_heads;
            let size_kv = cfg.head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            txt_cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let txt_cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };

        Ok(txt_cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let cfg = match &cfg {
            Gemma3Config::Text(cfg) => cfg,
            Gemma3Config::WithVision { text_config, .. } => text_config,
        };

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None, // None to be more forgiving, some do not
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Mistral 3 Loader

/// [`VisionLoader`] for an Mistral 3 model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Mistral3Loader;

pub struct Mistral3Prefixer;

impl MultimodalPromptPrefixer for Mistral3Prefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for Mistral3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut cfg: crate::vision_models::mistral3::Mistral3Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(Mistral3Model::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::vision_models::mistral3::Mistral3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Mistral3Processor::new(processor_config.unwrap_or_default()))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Mistral3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Mistral3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl DeviceMappedModelLoader for Mistral3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let vcfg = &cfg.vision_config;
        let tcfg = &cfg.text_config;

        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: (mut height, mut width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let img_seq_len = {
            // Reshaping algorithm

            // https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/preprocessor_config.json#L29
            let (max_height, max_width) = (1540, 1540);
            let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
            if ratio > 1. {
                height = (height as f64 / ratio).floor() as usize;
                width = (width as f64 / ratio).floor() as usize;
            }

            let num_height_tokens = (height - 1) / vcfg.patch_size + 1;
            let num_width_tokens = (width - 1) / vcfg.patch_size + 1;

            height = num_height_tokens * vcfg.patch_size;
            width = num_width_tokens * vcfg.patch_size;

            let num_height_tokens = height / vcfg.patch_size;
            let num_width_tokens = width / vcfg.patch_size;

            (num_width_tokens + 1) * num_height_tokens
        };

        // This model injects the vision information directly into the input embeddings
        let max_seq_len = img_seq_len * max_num_images + *max_seq_len.min(&ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * tcfg.num_attention_heads * max_seq_len * max_seq_len)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.vision_config;

        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: (mut height, mut width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let img_seq_len = {
            // Reshaping algorithm

            // https://huggingface.co/mistralai/Mistral-Small-3.1-24B-Instruct-2503/blob/main/preprocessor_config.json#L29
            let (max_height, max_width) = (1540, 1540);
            let ratio = (height as f64 / max_height as f64).max(width as f64 / max_width as f64);
            if ratio > 1. {
                height = (height as f64 / ratio).floor() as usize;
                width = (width as f64 / ratio).floor() as usize;
            }

            let num_height_tokens = (height - 1) / cfg.patch_size + 1;
            let num_width_tokens = (width - 1) / cfg.patch_size + 1;

            height = num_height_tokens * cfg.patch_size;
            width = num_width_tokens * cfg.patch_size;

            let num_height_tokens = height / cfg.patch_size;
            let num_width_tokens = width / cfg.patch_size;

            (num_width_tokens + 1) * num_height_tokens
        };

        Ok((max_batch_size * max_num_images) * cfg.num_attention_heads * img_seq_len * img_seq_len)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = &cfg.text_config;

            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !cfg.tie_word_embeddings || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_elems = {
            let cfg = &cfg.vision_config;

            let patch_embed = {
                let conv_cfg = Conv2dConfig {
                    stride: cfg.patch_size,
                    ..Default::default()
                };
                cfg.num_channels * cfg.hidden_size / conv_cfg.groups
                    * cfg.patch_size
                    * cfg.patch_size
                    * cfg.patch_size
            };
            let ln_pre = cfg.hidden_size;
            let vision_layer = {
                let attn_norm = cfg.hidden_size;
                let ffn_norm = cfg.hidden_size;

                let gate = cfg.hidden_size * cfg.intermediate_size;
                let up = cfg.hidden_size * cfg.intermediate_size;
                let down = cfg.hidden_size * cfg.intermediate_size;

                let q = cfg.hidden_size * cfg.hidden_size;
                let k = cfg.hidden_size * cfg.hidden_size;
                let v = cfg.hidden_size * cfg.hidden_size;
                let o = cfg.hidden_size * cfg.hidden_size;

                attn_norm + ffn_norm + gate + up + down + q + k + v + o
            };

            patch_embed + ln_pre + vision_layer * cfg.num_hidden_layers
        };

        let elems = text_elems + vision_elems;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Llama 4 Loader

/// [`VisionLoader`] for an Llama Vision model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct VLlama4Loader;

pub struct VLlama4Prefixer;

impl MultimodalPromptPrefixer for VLlama4Prefixer {
    fn prefix_image(&self, image_indexes: Vec<usize>, prompt: &str) -> String {
        format!(
            "{}{prompt}",
            llama4::IMAGE_TOKEN.repeat(image_indexes.len())
        )
    }
}

impl VisionModelLoader for VLlama4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut cfg: crate::vision_models::llama4::Llama4Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(Llama4Model::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let mut cfg: crate::vision_models::llama4::Llama4Config = serde_json::from_str(config)?;
        cfg.propagate_quantization_config();
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Llama4Processor::new(&processor_config.unwrap()))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VLlama4Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for VLlama4Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // FF MoE
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.router\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$")?,
            // FF MLP
            Regex::new(r"layers\.(\d+)\.feed_forward\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"language_model\.model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // FF MoE
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.gate_up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.router\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.shared_expert\.(weight|bias)$",
            )?,
            // FF MLP
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"language_model\.model\.layers\.(\d+)\.feed_forward\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
}

impl VLlama4Loader {
    /// This incorporates the max batch size!
    /// Returns (pixels max batch size, num text image tokens)
    #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
    fn run_dummy_processing(
        &self,
        cfg: &Llama4Config,
        height: usize,
        width: usize,
        max_num_images: usize,
        max_batch_size: usize,
    ) -> Result<(usize, usize)> {
        let cfg = &cfg.vision_config;

        let img_processor =
            Llama4ImageProcessor::new(Some(cfg.patch_size), Some(cfg.pixel_shuffle_ratio));
        let image = DynamicImage::new(width as u32, height as u32, ColorType::Rgb8);
        let res = img_processor.preprocess(
            vec![image; max_num_images],
            vec![],
            &PreProcessorConfig::default(),
            &Device::Cpu,
            (max_batch_size, max_num_images),
        )?;

        let pixels_batch_size = res.pixel_values.dim(0)?;
        let pixels_max_batch_size = pixels_batch_size * max_batch_size;

        let (image_h, image_w) = (
            res.pixel_values.dim(D::Minus2).unwrap(),
            res.pixel_values.dim(D::Minus1).unwrap(),
        );
        let num_patches_per_chunk = (image_h / img_processor.patch_size)
            * (image_w / img_processor.patch_size)
            / img_processor.downsample_ratio;

        Ok((
            pixels_max_batch_size,
            num_patches_per_chunk * pixels_max_batch_size,
        ))
    }
}

impl DeviceMappedModelLoader for VLlama4Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: (height, width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Llama4Config = serde_json::from_str(config)?;

        let (_pixels_batch_size, num_text_image_toks) =
            self.run_dummy_processing(&cfg, *height, *width, *max_num_images, *max_batch_size)?;

        let max_seq_len = max_seq_len.min(&ATTENTION_CHUNK_SIZE) + num_text_image_toks;

        Ok(max_batch_size * cfg.text_config.num_attention_heads * max_seq_len * max_seq_len)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: (height, width),
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Llama4Config = serde_json::from_str(config)?;

        let (pixels_batch_size, _num_text_image_toks) =
            self.run_dummy_processing(&cfg, *height, *width, *max_num_images, *max_batch_size)?;
        let max_seq_len = cfg.vision_config.num_patches();

        Ok((max_batch_size * pixels_batch_size)
            * cfg.vision_config.num_attention_heads
            * max_seq_len
            * max_seq_len)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let tcfg = &cfg.text_config;

        let text_elems = {
            let embed_tokens = tcfg.hidden_size * tcfg.vocab_size / weight_pack_factor;
            let lm_head = if !tcfg.tie_word_embeddings {
                tcfg.hidden_size * tcfg.vocab_size
            } else {
                0
            };
            let norm = tcfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let vision_elems = {
            let cfg = &cfg.vision_config;

            let num_patches = cfg.num_patches();

            let unfold_elems =
                (cfg.num_channels * cfg.patch_size * cfg.patch_size) * cfg.hidden_size;
            let class_embeddng_elems = cfg.hidden_size;
            let positional_embedding_vlm_elems = num_patches * cfg.hidden_size;
            let layernorm_pre_elems = cfg.hidden_size;
            let layernorm_post_elems = cfg.hidden_size;

            let pixel_shuffle_elems = cfg.intermediate_size * cfg.projector_input_dim
                / weight_pack_factor
                + cfg.projector_input_dim * cfg.projector_output_dim / weight_pack_factor;

            let encoder_layer = {
                let input_layernorm = cfg.hidden_size + cfg.hidden_size;
                let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

                let head_dim = cfg.hidden_size / cfg.num_attention_heads;
                let q_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    / weight_pack_factor
                    + cfg.num_attention_heads * head_dim;
                let k_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    / weight_pack_factor
                    + cfg.num_attention_heads * head_dim;
                let v_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    / weight_pack_factor
                    + cfg.num_attention_heads * head_dim;
                let o_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim
                    / weight_pack_factor
                    + cfg.num_attention_heads * head_dim;

                let fc1 = (cfg.hidden_size * cfg.intermediate_size) / weight_pack_factor
                    + cfg.intermediate_size;
                let fc2 = (cfg.intermediate_size * cfg.hidden_size) / weight_pack_factor
                    + cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + fc1
                    + fc2
            };

            unfold_elems
                + class_embeddng_elems
                + positional_embedding_vlm_elems
                + layernorm_post_elems
                + layernorm_pre_elems
                + pixel_shuffle_elems
                + encoder_layer * cfg.num_hidden_layers
        };

        let elems = text_elems + vision_elems;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let tcfg = &cfg.text_config;

        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..tcfg.num_hidden_layers {
            let input_layernorm = tcfg.hidden_size;
            let post_attention_layernorm = tcfg.hidden_size;

            let size_in = tcfg.hidden_size;
            let size_q = (tcfg.hidden_size / tcfg.num_attention_heads) * tcfg.num_attention_heads;
            let size_kv = (tcfg.hidden_size / tcfg.num_attention_heads) * tcfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let use_moe = tcfg.moe_layers().contains(&layer_idx);
            let moe_block = if use_moe {
                let h_size = tcfg.hidden_size;
                let i_size = tcfg.intermediate_size;
                let gate_proj = tcfg.num_local_experts * h_size * i_size / weight_pack_factor;
                let up_proj = tcfg.num_local_experts * h_size * i_size / weight_pack_factor;
                let down_proj = tcfg.num_local_experts * i_size * h_size / weight_pack_factor;

                gate_proj + up_proj + down_proj
            } else {
                let h_size = tcfg.hidden_size;
                let i_size = tcfg.intermediate_size_mlp;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;

                gate_proj + up_proj + down_proj
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Llama4Config = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Gemma 3n Loader

/// [`VisionLoader`] for an Gemma 3n model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Gemma3nLoader;

#[allow(dead_code)]
pub struct Gemma3nPrefixer;

impl MultimodalPromptPrefixer for Gemma3nPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for Gemma3nLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(Box::new(Gemma3nModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _config: &str,
        processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        // Handle the Gemma 3 1b case here
        Arc::new(Gemma3nProcessor::new(
            processor_config.unwrap_or_default(),
            true,
        ))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Gemma3Prefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![
                SupportedModality::Text,
                SupportedModality::Vision,
                SupportedModality::Audio,
            ],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Gemma3nLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Language model attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Language model MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // Audio conformer attention layers
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.attention\.attn\.v_proj\.(weight|bias)$")?,
            Regex::new(
                r"conformer\.(\d+)\.attention\.attn\.relative_position_embedding\.pos_proj\.(weight|bias)$",
            )?,
            Regex::new(r"conformer\.(\d+)\.attention\.post\.(weight|bias)$")?,
            // Audio conformer FFW layers
            Regex::new(r"conformer\.(\d+)\.ffw_layer_start\.ffw_layer_1\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_start\.ffw_layer_2\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_end\.ffw_layer_1\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.ffw_layer_end\.ffw_layer_2\.(weight|bias)$")?,
            // Audio conformer conv1d layers
            Regex::new(r"conformer\.(\d+)\.lconv1d\.linear_start\.(weight|bias)$")?,
            Regex::new(r"conformer\.(\d+)\.lconv1d\.linear_end\.(weight|bias)$")?,
            // Audio subsample projection
            Regex::new(r"subsample_conv_projection\.input_proj_linear\.(weight|bias)$")?,
            // Multimodal embedders
            Regex::new(r"embed_vision\.embedding_projection\.(weight|bias)$")?,
            Regex::new(r"embed_audio\.embedding_projection\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Language model attention
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Language model MLP
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // Projections
            Regex::new(r"model\.language_model\.per_layer_model_projection\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.altup_projections\.(\d+)\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.altup_unembed_projections\.(\d+)\.(weight|bias)$")?,
            // Audio conformer attention layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.q_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.k_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.v_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.attention\.attn\.relative_position_embedding\.pos_proj\.(weight|bias)$",
            )?,
            Regex::new(r"model\.audio_tower\.conformer\.(\d+)\.attention\.post\.(weight|bias)$")?,
            // Audio conformer FFW layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_start\.ffw_layer_1\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_start\.ffw_layer_2\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_end\.ffw_layer_1\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.ffw_layer_end\.ffw_layer_2\.(weight|bias)$",
            )?,
            // Audio conformer conv1d layers
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.lconv1d\.linear_start\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.audio_tower\.conformer\.(\d+)\.lconv1d\.linear_end\.(weight|bias)$",
            )?,
            // Audio subsample projection
            Regex::new(
                r"model\.audio_tower\.subsample_conv_projection\.input_proj_linear\.(weight|bias)$",
            )?,
            // Multimodal embedders
            Regex::new(r"model\.embed_vision\.embedding_projection\.(weight|bias)$")?,
            Regex::new(r"model\.embed_audio\.embedding_projection\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma3nLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        let text_cfg = &cfg.text_config;

        // Gemma3n is an "inject into the prompt" model, similar to Gemma3
        // We need to account for vision and audio tokens in the sequence length

        let mut total_seq_len = *max_seq_len.min(&ATTENTION_CHUNK_SIZE);

        // Add vision tokens
        {
            // Vision tokens are injected into the prompt
            // MSFA outputs fixed 16x16 features regardless of input size
            let msfa_spatial_size = 16; // Fixed from vision.rs line 1115
            let vision_tokens_per_image = msfa_spatial_size * msfa_spatial_size; // 256 tokens
            total_seq_len += vision_tokens_per_image * max_num_images;
        }

        // Add audio tokens
        {
            // Audio tokens are injected into the prompt
            // From config field audio_soft_tokens_per_image (typically 188)
            let audio_tokens = cfg.audio_soft_tokens_per_image;
            total_seq_len += audio_tokens;
        }

        // Calculate max attention size for text model with all injected tokens
        let max_text_attn =
            max_batch_size * text_cfg.num_attention_heads * total_seq_len * total_seq_len;

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape: _,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Calculate max activation sizes for each modality
        let mut max_activation = 0;

        // Vision activation size
        {
            // Vision is Gemma3n's MobileNetV5 architecture with Multi-Query Attention
            // The peak activation is in the Multi-Query Attention layers

            // From the architecture: stages 3 and 4 have MMQA blocks
            // Input images are 768x768 (from inputs_processor.rs)
            // Stage 3: 640 channels at 48x48 (768/16 downsampling), MMQA with num_heads=12, kv_dim=64
            // Stage 4: 1280 channels at 24x24 (768/32 downsampling), MMQA with num_heads=16, kv_dim=96
            // MSFA output: 2048 channels at fixed 16x16

            let vision_tower_act = {
                // Peak is during MMQA attention computation in stage 4
                // Stage 4 has higher memory usage than Stage 3 due to more heads (16 vs 12)
                // From vision.rs: Stage 4 has num_heads=16, kv_dim=96, kv_stride=1
                let num_heads = 16; // Stage 4 configuration
                let spatial_size = 24; // 768 / 32 = 24 (input 768x768, stage 4 has 32x downsampling)
                let seq_len = spatial_size * spatial_size;

                // Attention scores: [B * num_images, num_heads, seq_len, seq_len]
                max_batch_size * max_num_images * num_heads * seq_len * seq_len
            };

            // Vision embedder activations
            let vision_embed_act = {
                // MSFA output: 2048 channels at fixed 16x16 spatial (from vision.rs line 1115)
                let msfa_channels = 2048; // MSFA_OUT_CHANNELS from vision.rs
                let spatial_size = 16; // Fixed output resolution from MSFA
                let vision_features =
                    max_batch_size * max_num_images * msfa_channels * spatial_size * spatial_size;

                // After embedding projection to text hidden size
                let projected = max_batch_size
                    * max_num_images
                    * spatial_size
                    * spatial_size
                    * cfg.text_config.hidden_size;

                vision_features.max(projected)
            };

            max_activation = max_activation.max(vision_tower_act).max(vision_embed_act);
        }

        // Audio activation size
        {
            let audio_cfg = &cfg.audio_config;

            // Calculate max audio sequence length based on config
            // Audio uses conformer with subsampling and reduction

            // A rough estimate of max_audio_frames
            let max_audio_frames = 1280;

            let subsample_factor: usize = audio_cfg
                .sscp_conv_stride_size
                .iter()
                .map(|stride| stride[0]) // Time dimension stride
                .product();
            let audio_seq_after_subsample = max_audio_frames / subsample_factor;

            // Audio encoder activations
            let audio_encoder_act = {
                // Conformer FFW layers have expansion factor from config
                let intermediate_size = audio_cfg.hidden_size * 4; // FFW expansion factor

                // Peak is in the FFW layers before reduction
                max_batch_size * audio_seq_after_subsample * intermediate_size
            };

            // Audio attention activations
            let audio_attn_act = {
                // Attention uses chunked processing with specific context sizes
                let chunk_size = audio_cfg.conf_attention_chunk_size;
                let context_size = chunk_size + audio_cfg.conf_attention_context_left - 1
                    + audio_cfg.conf_attention_context_right;

                // Peak is attention scores: [B, num_heads, num_chunks, chunk_size, context_size]
                let num_chunks = audio_seq_after_subsample.div_ceil(chunk_size);

                max_batch_size
                    * audio_cfg.conf_num_attention_heads
                    * num_chunks
                    * chunk_size
                    * context_size
            };

            max_activation = max_activation.max(audio_encoder_act).max(audio_attn_act);
        }

        Ok(max_activation)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Apply matformer slicing if configured
        let text_cfg = if let Some(matformer_cfg) = matformer_config {
            use crate::device_map::DummyDeviceMapper;
            use crate::vision_models::gemma3n::text::handle_matformer_slicing;

            let dummy_mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let (adjusted_cfg, _, _, _, _) = handle_matformer_slicing(
                &cfg.text_config,
                &Some(matformer_cfg.clone()),
                &dummy_mapper,
            )?;
            adjusted_cfg
        } else {
            cfg.text_config.clone()
        };

        let text_cfg = &text_cfg;

        // Text components that are not device-mapped
        let text_elems = {
            // Embeddings
            let embed_tokens = text_cfg.hidden_size * text_cfg.vocab_size;
            let embed_tokens_per_layer = text_cfg.num_hidden_layers
                * text_cfg.hidden_size_per_layer_input
                * text_cfg.vocab_size_per_layer_input;

            // LM head (if not tied)
            let lm_head = if !text_cfg.tie_word_embeddings || weight_pack_factor != 1 {
                text_cfg.hidden_size * text_cfg.vocab_size / weight_pack_factor
            } else {
                0
            };

            // Final layer norm
            let norm = text_cfg.hidden_size;

            // AltUp projections (not device-mapped)
            let altup_projections =
                (text_cfg.altup_num_inputs - 1) * text_cfg.hidden_size * text_cfg.hidden_size
                    / weight_pack_factor;
            let altup_unembed_projections =
                (text_cfg.altup_num_inputs - 1) * text_cfg.hidden_size * text_cfg.hidden_size
                    / weight_pack_factor;

            // Per-layer model projection
            let per_layer_model_projection = text_cfg.num_hidden_layers
                * text_cfg.hidden_size
                * text_cfg.hidden_size_per_layer_input
                / weight_pack_factor;
            let per_layer_projection_norm = text_cfg.hidden_size;

            embed_tokens
                + embed_tokens_per_layer
                + lm_head
                + norm
                + altup_projections
                + altup_unembed_projections
                + per_layer_model_projection
                + per_layer_projection_norm
        };

        // Vision components
        let vision_elems = {
            let vision_cfg = &cfg.vision_config;
            // Vision tower - calculated from actual Gemma3n architecture
            // NOTE: Vision tower uses only Conv2d layers, NOT Arc<dyn QuantMethod>,
            // so NONE of these should be divided by weight_pack_factor
            let vision_tower_elems = {
                use crate::vision_models::gemma3n::vision::{
                    gemma3n_mobilenet_def, make_divisible, BlockType, INPUT_CHANNELS,
                    MSFA_EXPANSION_RATIO, MSFA_IN_CHANNELS, MSFA_OUT_CHANNELS, STEM_KERNEL_SIZE,
                    STEM_OUT_CHANNELS,
                };

                // Stem: ConvNormAct (Conv2d + RMSNorm)
                let stem_conv =
                    INPUT_CHANNELS * STEM_OUT_CHANNELS * STEM_KERNEL_SIZE * STEM_KERNEL_SIZE;
                let stem_norm = STEM_OUT_CHANNELS; // RMSNorm weight

                // Track input channels through the network
                let mut in_chs = STEM_OUT_CHANNELS;
                let mut total_elems = stem_conv + stem_norm;

                // Process all stages from gemma3n_mobilenet_def
                let block_defs = gemma3n_mobilenet_def();

                for stage_blocks in block_defs.iter() {
                    for block_type in stage_blocks.iter() {
                        match block_type {
                            BlockType::EdgeResidual {
                                out_channels,
                                kernel_size,
                                stride: _,
                                expand_ratio,
                                ..
                            } => {
                                #[allow(clippy::cast_precision_loss)]
                                let mid_chs = make_divisible(in_chs as f64 * expand_ratio, 8);
                                // EdgeResidual: all Conv2d layers, not quantizable
                                total_elems += in_chs * mid_chs * kernel_size * kernel_size; // conv_exp (Conv2d)
                                total_elems += mid_chs; // bn1 weight
                                total_elems += mid_chs * out_channels; // conv_pwl (Conv2d)
                                total_elems += out_channels; // bn2 weight
                                in_chs = *out_channels;
                            }
                            BlockType::UniversalInvertedResidual {
                                out_channels,
                                start_kernel_size,
                                mid_kernel_size,
                                stride: _,
                                expand_ratio,
                                ..
                            } => {
                                #[allow(clippy::cast_precision_loss)]
                                let mid_chs = make_divisible(in_chs as f64 * expand_ratio, 8);
                                // UniversalInvertedResidual: all Conv2d layers, not quantizable
                                if *expand_ratio != 1.0 {
                                    total_elems += in_chs * mid_chs; // expand conv (Conv2d)
                                    total_elems += mid_chs; // expand norm
                                }
                                if *start_kernel_size > 0 {
                                    total_elems += mid_chs * start_kernel_size * start_kernel_size; // depthwise start (Conv2d)
                                    total_elems += mid_chs; // norm
                                }
                                if *mid_kernel_size > 0 {
                                    total_elems += mid_chs * mid_kernel_size * mid_kernel_size; // depthwise mid (Conv2d)
                                    total_elems += mid_chs; // norm
                                }
                                total_elems += mid_chs * out_channels; // project conv (Conv2d)
                                total_elems += out_channels; // project norm
                                total_elems += out_channels; // layer scale gamma
                                in_chs = *out_channels;
                            }
                            BlockType::MultiQueryAttention {
                                num_heads,
                                kv_dim,
                                kv_stride: _,
                                ..
                            } => {
                                // MMQA: all Conv2d layers, not quantizable
                                let dw_kernel_size = 3; // Default dw_kernel_size for MMQA
                                total_elems += in_chs; // norm weight
                                total_elems += in_chs * num_heads * kv_dim; // query_proj (Conv2d)
                                total_elems += in_chs * kv_dim; // key_proj (Conv2d)
                                total_elems += in_chs * dw_kernel_size * dw_kernel_size; // key_dw_conv (Conv2d)
                                total_elems += *kv_dim; // value_down_conv (Conv2d)
                                total_elems += 1; // value_norm weight
                                total_elems += *kv_dim; // value_proj (Conv2d)
                                total_elems += num_heads * kv_dim * in_chs; // output_proj (Conv2d)
                                total_elems += in_chs; // layer scale
                            }
                        }
                    }
                }

                // Multi-scale fusion adapter (msfa) - also uses Conv2d layers
                let msfa_in = MSFA_IN_CHANNELS.iter().sum::<usize>();
                let msfa_out = MSFA_OUT_CHANNELS;
                #[allow(clippy::cast_precision_loss)]
                let msfa_mid = make_divisible(msfa_in as f64 * MSFA_EXPANSION_RATIO, 8);

                // MSFA FFN (UIR with expansion_ratio) - Conv2d layers, not quantizable
                total_elems += msfa_in * msfa_mid; // expand (Conv2d)
                total_elems += msfa_mid; // expand norm
                total_elems += msfa_mid * msfa_out; // project (Conv2d)
                total_elems += msfa_out; // project norm
                total_elems += msfa_out; // final norm

                total_elems
            };

            // Vision multimodal embedder components
            let embed_vision_elems = {
                // Embedding layer (not quantizable)
                let embedding = vision_cfg.vocab_size * vision_cfg.hidden_size;

                // Normalization layers (not quantizable)
                let hard_norm = vision_cfg.hidden_size;
                let soft_norm = vision_cfg.hidden_size;

                // Projection from vision to text hidden size (IS Arc<dyn QuantMethod>, so quantizable)
                let projection = vision_cfg.hidden_size * text_cfg.hidden_size / weight_pack_factor;

                // Post-projection norm (not quantizable)
                let post_norm = text_cfg.hidden_size;

                embedding + hard_norm + soft_norm + projection + post_norm
            };

            vision_tower_elems + embed_vision_elems
        };

        // Audio components - based on actual audio.rs structure
        let audio_elems = {
            let audio_cfg = &cfg.audio_config;

            // SubSampleConvProjection components
            let subsample_conv_projection_elems = {
                // Conv blocks (Conv2d layers - NOT quantizable)
                let mut conv_elems = 0;

                // conv_0: Conv2d from 1 channel to first channel size
                let in_ch_0 = 1;
                let out_ch_0 = audio_cfg.sscp_conv_channel_size[0];
                let kernel_0 = &audio_cfg.sscp_conv_kernel_size[0];
                conv_elems += in_ch_0 * out_ch_0 * kernel_0[0] * kernel_0[1];

                // conv_1: Conv2d from first to second channel size
                let in_ch_1 = out_ch_0;
                let out_ch_1 = audio_cfg.sscp_conv_channel_size[1];
                let kernel_1 = &audio_cfg.sscp_conv_kernel_size[1];
                conv_elems += in_ch_1 * out_ch_1 * kernel_1[0] * kernel_1[1];

                // CumulativeGroupNorm for each conv block (weight only, no bias by default)
                let norm_0 = out_ch_0; // norm weight for conv_0
                let norm_1 = out_ch_1; // norm weight for conv_1

                // input_proj_linear (Arc<dyn QuantMethod> - IS quantizable)
                let mut f_out = audio_cfg.input_feat_size;
                for i in 0..2 {
                    let kernel_w = audio_cfg.sscp_conv_kernel_size[i][1];
                    let stride_w = audio_cfg.sscp_conv_stride_size[i][1];
                    let pad_left = 1;
                    let pad_right = 1;
                    f_out = (f_out + pad_left + pad_right + stride_w - kernel_w) / stride_w;
                }
                let input_proj_in_features = out_ch_1 * f_out;
                let input_proj_linear =
                    input_proj_in_features * audio_cfg.hidden_size / weight_pack_factor;

                conv_elems + norm_0 + norm_1 + input_proj_linear
            };

            // Conformer blocks
            let conformer_elems = {
                let mut total = 0;

                for _ in 0..audio_cfg.conf_num_hidden_layers {
                    // ConformerAttention
                    let attention_elems = {
                        // Norms (NOT quantizable)
                        let pre_attn_norm = audio_cfg.hidden_size;
                        let post_norm = audio_cfg.hidden_size;

                        // Attention projections (Arc<dyn QuantMethod> - IS quantizable)
                        let q_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let k_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let v_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let post =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;

                        // RelativePositionEmbedding
                        let pos_proj =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;
                        let per_dim_scale =
                            audio_cfg.hidden_size / audio_cfg.conf_num_attention_heads; // head_dim
                        let inv_timescales = audio_cfg.hidden_size / 2; // num_timescales
                        let pos_indices = audio_cfg.conf_attention_context_left
                            + audio_cfg.conf_attention_context_right
                            + 1;

                        // Local causal masks (precomputed tensors)
                        let chunk_size = audio_cfg.conf_attention_chunk_size;
                        let context_size = chunk_size + audio_cfg.conf_attention_context_left - 1
                            + audio_cfg.conf_attention_context_right;
                        let local_causal_valid_mask = chunk_size * context_size; // U8 tensor
                        let invalid_logits_tensor = 1; // single f32 value

                        pre_attn_norm
                            + post_norm
                            + q_proj
                            + k_proj
                            + v_proj
                            + post
                            + pos_proj
                            + per_dim_scale
                            + inv_timescales
                            + pos_indices
                            + local_causal_valid_mask
                            + invalid_logits_tensor
                    };

                    // ConformerFeedForward (start and end)
                    let ffw_elems = {
                        // Each FFW has:
                        // - pre_layer_norm (NOT quantizable)
                        // - ffw_layer_1 (Arc<dyn QuantMethod> - IS quantizable)
                        // - ffw_layer_2 (Arc<dyn QuantMethod> - IS quantizable)
                        // - post_layer_norm (NOT quantizable)
                        let intermediate_size = audio_cfg.hidden_size * 4;

                        let ffw_start = {
                            let pre_norm = audio_cfg.hidden_size;
                            let layer_1 =
                                audio_cfg.hidden_size * intermediate_size / weight_pack_factor;
                            let layer_2 =
                                intermediate_size * audio_cfg.hidden_size / weight_pack_factor;
                            let post_norm = audio_cfg.hidden_size;
                            pre_norm + layer_1 + layer_2 + post_norm
                        };

                        let ffw_end = ffw_start; // Same structure

                        ffw_start + ffw_end
                    };

                    // ConformerLightConv1d
                    let lconv1d_elems = {
                        // Norms (NOT quantizable)
                        let pre_layer_norm = audio_cfg.hidden_size;
                        let conv_norm = audio_cfg.hidden_size;

                        // Linear layers (Arc<dyn QuantMethod> - IS quantizable)
                        let linear_start = audio_cfg.hidden_size * (audio_cfg.hidden_size * 2)
                            / weight_pack_factor;
                        let linear_end =
                            audio_cfg.hidden_size * audio_cfg.hidden_size / weight_pack_factor;

                        // depthwise_conv1d (Conv1d - NOT quantizable)
                        let depthwise = audio_cfg.hidden_size * audio_cfg.conf_conv_kernel_size;

                        pre_layer_norm + conv_norm + linear_start + linear_end + depthwise
                    };

                    // Final norm for conformer block (NOT quantizable)
                    let block_norm = audio_cfg.hidden_size;

                    total += attention_elems + ffw_elems + lconv1d_elems + block_norm;
                }

                total
            };

            // Audio multimodal embedder (embed_audio)
            let embed_audio_elems = {
                // Embedding layer (ScaledEmbedding - NOT quantizable)
                let embedding = audio_cfg.vocab_size * audio_cfg.hidden_size;

                // RMS norms (NOT quantizable)
                let hard_embedding_norm = audio_cfg.hidden_size; // with scale
                let soft_embedding_norm = audio_cfg.hidden_size; // with scale
                let embedding_post_projection_norm = text_cfg.hidden_size; // without scale

                // Projection (Arc<dyn QuantMethod> - IS quantizable)
                let embedding_projection =
                    audio_cfg.hidden_size * text_cfg.hidden_size / weight_pack_factor;

                embedding
                    + hard_embedding_norm
                    + soft_embedding_norm
                    + embedding_post_projection_norm
                    + embedding_projection
            };

            subsample_conv_projection_elems + conformer_elems + embed_audio_elems
        };

        let vision_dtype = if dtype == DType::F16 {
            // f16 -> f32 for vision model in particular.
            DType::F32
        } else {
            dtype
        };

        let total_elems = text_elems * dtype.size_in_bytes()
            + vision_elems * vision_dtype.size_in_bytes()
            + audio_elems * dtype.size_in_bytes();

        Ok(total_elems)
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;

        // Apply matformer slicing if configured
        let (text_cfg, _layer_rename_map, _layers_skipped) = if let Some(matformer_cfg) =
            matformer_config
        {
            use crate::device_map::DummyDeviceMapper;
            use crate::vision_models::gemma3n::text::handle_matformer_slicing;

            let dummy_mapper = DummyDeviceMapper {
                nm_device: Device::Cpu,
            };
            let (adjusted_cfg, _, _, layer_rename_map, layers_skipped) = handle_matformer_slicing(
                &cfg.text_config,
                &Some(matformer_cfg.clone()),
                &dummy_mapper,
            )?;
            (adjusted_cfg, layer_rename_map, layers_skipped)
        } else {
            (cfg.text_config.clone(), None, None)
        };

        let text_cfg = &text_cfg;

        // When matformer slicing is applied, we only include the layers that are kept
        let mut layer_sizes = Vec::new();

        // Note: We don't need orig_intermediate_sizes anymore since the adjusted config
        // already has the correct intermediate sizes after matformer slicing

        for layer_idx in 0..text_cfg.num_hidden_layers {
            let per_layer_elems = {
                // Layer norms
                let input_layernorm = text_cfg.hidden_size;
                let post_attention_layernorm = text_cfg.hidden_size;
                let pre_feedforward_layernorm = text_cfg.hidden_size;
                let post_feedforward_layernorm = text_cfg.hidden_size;
                let post_per_layer_input_norm = text_cfg.hidden_size;

                // Attention components
                let size_in = text_cfg.hidden_size;
                let size_q = text_cfg.num_attention_heads * text_cfg.head_dim;
                let size_kv = text_cfg.num_key_value_heads * text_cfg.head_dim;

                let q_proj = size_in * size_q / weight_pack_factor;
                let k_proj = size_in * size_kv / weight_pack_factor;
                let v_proj = size_in * size_kv / weight_pack_factor;
                let o_proj = size_q * size_in / weight_pack_factor;

                // Q, K, V norms
                let q_norm = text_cfg.head_dim;
                let k_norm = text_cfg.head_dim;
                let v_norm = text_cfg.head_dim; // No bias for v_norm

                // MLP components - use the adjusted intermediate sizes from matformer
                let intermediate_size = match &text_cfg.intermediate_size {
                    IntermediateSize::Single(size) => *size,
                    IntermediateSize::PerLayer(sizes) => sizes[layer_idx],
                    IntermediateSize::Matformer(sizes, _) => sizes[layer_idx],
                };
                let gate_proj = text_cfg.hidden_size * intermediate_size / weight_pack_factor;
                let up_proj = text_cfg.hidden_size * intermediate_size / weight_pack_factor;
                let down_proj = intermediate_size * text_cfg.hidden_size / weight_pack_factor;

                // AltUp components (per layer)
                let altup_elems = {
                    let correct_output_scale = text_cfg.hidden_size;
                    let correction_coefs = text_cfg.altup_num_inputs * text_cfg.altup_num_inputs;
                    let prediction_coefs =
                        text_cfg.altup_num_inputs * text_cfg.altup_num_inputs.pow(2);
                    let modality_router = text_cfg.hidden_size * text_cfg.altup_num_inputs;
                    let router_norm = text_cfg.hidden_size;

                    correct_output_scale
                        + correction_coefs
                        + prediction_coefs
                        + modality_router
                        + router_norm
                };

                // Laurel block components
                let laurel_elems = {
                    let left = text_cfg.hidden_size * text_cfg.laurel_rank;
                    let right = text_cfg.laurel_rank * text_cfg.hidden_size;
                    let post_norm = text_cfg.hidden_size;

                    left + right + post_norm
                };

                // Per-layer input components
                let per_layer_input_gate =
                    text_cfg.hidden_size * text_cfg.hidden_size_per_layer_input;
                let per_layer_projection =
                    text_cfg.hidden_size_per_layer_input * text_cfg.hidden_size;

                input_layernorm
                    + post_attention_layernorm
                    + pre_feedforward_layernorm
                    + post_feedforward_layernorm
                    + post_per_layer_input_norm
                    + q_proj
                    + k_proj
                    + v_proj
                    + o_proj
                    + q_norm
                    + k_norm
                    + v_norm
                    + gate_proj
                    + up_proj
                    + down_proj
                    + altup_elems
                    + laurel_elems
                    + per_layer_input_gate
                    + per_layer_projection
            };

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        Ok(cfg.text_config.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Gemma3nConfig = serde_json::from_str(config)?;
        let cfg = cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None, // None to be more forgiving, some do not
            k_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            v_head_dim: cfg.hidden_size / cfg.num_attention_heads,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision, NonMappedSubModel::Audio])
    }
}

// ======================== Qwen3VL Loader

/// [`VisionLoader`] for an Qwen3VL model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Qwen3VLLoader;

pub struct Qwen3VLPrefixer;

impl MultimodalPromptPrefixer for Qwen3VLPrefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

impl VisionModelLoader for Qwen3VLLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen3VLModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Qwen3VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLProcessor::new(max_edge))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Qwen3VLPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen3VLLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;

        // For images, grid_t=1. After spatial merging, grid_h and grid_w are reduced.
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images (temporal dimension is for video only)
            let grid_t = 1;
            // After patch embedding and spatial merge, the effective grid dimensions are reduced
            let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        let max_text_attn = {
            let cfg = &cfg.text_config;
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;

        // For the vision encoder, before spatial merging
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images
            let grid_t = 1;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };

        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;
        let tie = cfg.tie_word_embeddings;
        let text_elems = {
            let cfg = &cfg.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !tie || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let (patch_merger, deepstack_mergers) = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.out_hidden_size + cfg.out_hidden_size;

            // Main merger: norm uses cfg.hidden_size
            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let merger = mlp0 + mlp2 + ln_q;

            // Deepstack mergers: norm uses merged hidden_size
            let ds_ln = hidden_size + bias_if!(true, hidden_size);
            let ds_merger = mlp0 + mlp2 + ds_ln;
            let deepstack = cfg.deepstack_visual_indexes.len() * ds_merger;

            (merger, deepstack)
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            let weight = cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2];
            let bias = cfg.hidden_size;
            weight + bias
        };

        let pos_embed = {
            let cfg = &cfg.vision_config;
            cfg.num_position_embeddings * cfg.hidden_size
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let norm2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
            let fc2 = cfg.hidden_size * cfg.intermediate_size + cfg.hidden_size;

            let qkv = cfg.hidden_size * cfg.hidden_size * 3 + cfg.hidden_size * 3;
            let out = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems = text_elems
            + patch_merger
            + deepstack_mergers
            + patch_embed
            + pos_embed
            + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let cfg = &cfg.text_config;
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim * cfg.num_attention_heads;
            let size_kv = cfg.head_dim * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let q_norm = cfg.head_dim;
            let k_norm = cfg.head_dim;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + q_norm
                + k_norm
                + gate_proj
                + up_proj
                + down_proj
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.text_config.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen3VLConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Qwen3VLMoE Loader

/// [`VisionLoader`] for a Qwen3VLMoE model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct Qwen3VLMoELoader;

pub struct Qwen3VLMoEPrefixer;

impl MultimodalPromptPrefixer for Qwen3VLMoEPrefixer {
    // No-op: With MessagesAction::Keep, the chat template handles image tokens
    // when it sees {"type": "image"} entries in the content.
}

impl VisionModelLoader for Qwen3VLMoELoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen3VLMoEModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
    fn get_processor(
        &self,
        _model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Qwen3VLMoEProcessor::new(max_edge))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        true
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        true
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(Qwen3VLMoEPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Vision],
            output: vec![SupportedModality::Text],
        })
    }
}

impl IsqModelLoader for Qwen3VLMoELoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP (dense layers)
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MoE router
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|bias)$")?,
            // MoE experts - now unpacked into individual experts
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // MLP (dense layers)
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MoE router
            Regex::new(r"model\.language_model\.layers\.(\d+)\.mlp\.gate\.(weight|bias)$")?,
            // MoE experts
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$",
            )?,
            Regex::new(
                r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Qwen3VLMoELoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;

        // For images, grid_t=1. After spatial merging, grid_h and grid_w are reduced.
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images (temporal dimension is for video only)
            let grid_t = 1;
            // After patch embedding and spatial merge, the effective grid dimensions are reduced
            let grid_h = (max_image_shape.0 / cfg.patch_size) / cfg.spatial_merge_size;
            let grid_w = (max_image_shape.1 / cfg.patch_size) / cfg.spatial_merge_size;
            grid_t * grid_h * grid_w * max_num_images
        };

        let max_text_attn = {
            let cfg = &cfg.text_config;
            // This model injects the vision information directly into the input embeddings
            let max_seq_len = img_seq_len + max_seq_len.min(&ATTENTION_CHUNK_SIZE);
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
        };

        Ok(max_text_attn)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len: _,
            max_batch_size,
            max_image_shape,
            max_num_images,
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;

        // For the vision encoder, before spatial merging
        let img_seq_len = {
            let cfg = &cfg.vision_config;
            // grid_t is 1 for images
            let grid_t = 1;
            let grid_h = max_image_shape.0 / cfg.patch_size;
            let grid_w = max_image_shape.1 / cfg.patch_size;
            grid_t * grid_h * grid_w
        };

        let max_vision_attn = {
            let cfg = &cfg.vision_config;
            (max_batch_size * max_num_images) * cfg.num_heads * img_seq_len * img_seq_len
        };

        Ok(max_vision_attn)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        let tie = cfg.tie_word_embeddings;
        let text_elems = {
            let cfg = &cfg.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            // If embeddings are tied and no packing, reuse weights -> no separate lm_head needed
            let lm_head = if !tie || weight_pack_factor != 1 {
                cfg.hidden_size * cfg.vocab_size / weight_pack_factor
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };

        let (patch_merger, deepstack_mergers) = {
            let cfg = &cfg.vision_config;
            let hidden_size = cfg.hidden_size * cfg.spatial_merge_size.pow(2);

            let mlp0 = hidden_size * hidden_size + hidden_size;
            let mlp2 = hidden_size * cfg.out_hidden_size + cfg.out_hidden_size;

            // Main merger: norm uses cfg.hidden_size
            let ln_q = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let merger = mlp0 + mlp2 + ln_q;

            // Deepstack mergers: norm uses merged hidden_size
            let ds_ln = hidden_size + bias_if!(true, hidden_size);
            let ds_merger = mlp0 + mlp2 + ds_ln;
            let deepstack = cfg.deepstack_visual_indexes.len() * ds_merger;

            (merger, deepstack)
        };

        let patch_embed = {
            let cfg = &cfg.vision_config;
            let conv_cfg = Conv3dConfig {
                stride: cfg.patch_size,
                ..Default::default()
            };
            let kernel_sizes = [cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size];
            let weight = cfg.in_chans * cfg.hidden_size / conv_cfg.groups
                * kernel_sizes[0]
                * kernel_sizes[1]
                * kernel_sizes[2];
            let bias = cfg.hidden_size;
            weight + bias
        };

        let pos_embed = {
            let cfg = &cfg.vision_config;
            cfg.num_position_embeddings * cfg.hidden_size
        };

        let encoder_layer = {
            let cfg = &cfg.vision_config;
            let norm1 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);
            let norm2 = cfg.hidden_size + bias_if!(true, cfg.hidden_size);

            #[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
            let fc1 = cfg.hidden_size * cfg.intermediate_size + cfg.intermediate_size;
            let fc2 = cfg.hidden_size * cfg.intermediate_size + cfg.hidden_size;

            let qkv = cfg.hidden_size * cfg.hidden_size * 3 + cfg.hidden_size * 3;
            let out = cfg.hidden_size * cfg.hidden_size + cfg.hidden_size;

            norm1 + norm2 + fc1 + fc2 + qkv + out
        };

        let elems = text_elems
            + patch_merger
            + deepstack_mergers
            + patch_embed
            + pos_embed
            + encoder_layer * cfg.vision_config.depth;

        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        let text_cfg = &cfg.text_config;

        let mut layer_sizes = Vec::with_capacity(text_cfg.num_hidden_layers);

        for layer_idx in 0..text_cfg.num_hidden_layers {
            let input_layernorm = text_cfg.hidden_size;
            let post_attention_layernorm = text_cfg.hidden_size;

            let size_in = text_cfg.hidden_size;
            let size_q = text_cfg.head_dim * text_cfg.num_attention_heads;
            let size_kv = text_cfg.head_dim * text_cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let q_norm = text_cfg.head_dim;
            let k_norm = text_cfg.head_dim;

            // Check if this is a MoE layer
            let is_moe = !text_cfg.mlp_only_layers.contains(&layer_idx)
                && (text_cfg.num_experts > 0
                    && (layer_idx + 1) % text_cfg.decoder_sparse_step == 0);

            let mlp_elems = if is_moe {
                // MoE layer: gate + experts
                let gate = text_cfg.hidden_size * text_cfg.num_experts;
                let per_expert = {
                    let h_size = text_cfg.hidden_size;
                    let i_size = text_cfg.moe_intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    gate_proj + up_proj + down_proj
                };
                gate + per_expert * text_cfg.num_experts
            } else {
                // Dense MLP layer
                let h_size = text_cfg.hidden_size;
                let i_size = text_cfg.intermediate_size;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;
                gate_proj + up_proj + down_proj
            };

            let per_layer_elems = input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + q_norm
                + k_norm
                + mlp_elems;

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen3VLMoEConfig = serde_json::from_str(config)?;
        let cfg = &cfg.text_config;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ─── Voxtral ────────────────────────────────────────────────────────────────

/// [`VisionLoader`] for a Voxtral model.
///
/// [`VisionLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.VisionLoader.html
pub struct VoxtralLoader;

pub struct VoxtralPrefixer;

impl MultimodalPromptPrefixer for VoxtralPrefixer {
    fn prefix_image(&self, _image_indexes: Vec<usize>, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for VoxtralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(Box::new(VoxtralModel::new(
            &cfg,
            vb,
            self.is_gptx(config),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _config: &str) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn get_processor(
        &self,
        model_config: &str,
        _processor_config: Option<ProcessorConfig>,
        _preprocessor_config: PreProcessorConfig,
        _max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync> {
        let cfg: VoxtralConfig =
            serde_json::from_str(model_config).expect("Failed to parse VoxtralConfig");
        Arc::new(VoxtralProcessor::new(&cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> bool {
        false
    }
    fn supports_prefix_cacher(&self, _config: &str) -> bool {
        false
    }
    fn prefixer(&self, _config: &str) -> Arc<dyn MultimodalPromptPrefixer> {
        Arc::new(VoxtralPrefixer)
    }
    fn modalities(&self, _config: &str) -> Result<Modalities> {
        Ok(Modalities {
            input: vec![SupportedModality::Text, SupportedModality::Audio],
            output: vec![SupportedModality::Text],
        })
    }
    fn default_chat_template(&self, _config: &str) -> Option<String> {
        // Mistral v7 instruct format using [INST]/[/INST] tokens
        Some("{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '[INST] ' + message['content'] + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ message['content'] + eos_token + ' ' }}{% else %}{{ raise_exception('Only user and assistant roles are supported!') }}{% endif %}{% endfor %}".to_string())
    }
    fn default_bos_eos(&self, _config: &str) -> Option<(String, String)> {
        // Mistral tekken tokenizer: <s> = ID 1, </s> = ID 2
        Some(("<s>".to_string(), "</s>".to_string()))
    }
}

impl IsqModelLoader for VoxtralLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // Output / lm_head (tied with tok_embeddings)
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Decoder attention (Mistral-native naming)
            Regex::new(r"layers\.(\d+)\.attention\.wq\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wk\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wv\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wo\.(weight|bias)$")?,
            // Decoder MLP (Mistral-native naming)
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"tok_embeddings\.(weight|bias)$")?,
            // Decoder attention
            Regex::new(r"layers\.(\d+)\.attention\.wq\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wk\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wv\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.attention\.wo\.(weight|bias)$")?,
            // Decoder MLP
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
        ])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl DeviceMappedModelLoader for VoxtralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision {
            max_seq_len,
            max_batch_size,
            ..
        } = params
        else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: VoxtralConfig = serde_json::from_str(config)?;

        // Audio tokens are prepended: max audio len + text seq len
        // Audio: ~30s at 16kHz = 480k samples, /160 hop = 3000 frames, /2 conv stride = 1500, /4 adapter = 375 tokens
        let max_audio_tokens = 375;
        let total_seq = max_audio_tokens + *max_seq_len.min(&ATTENTION_CHUNK_SIZE);
        Ok(max_batch_size * cfg.n_heads * total_seq * total_seq)
    }

    fn non_mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Vision { max_batch_size, .. } = params else {
            anyhow::bail!("Expected vision AutoDeviceMapParams for this model!")
        };

        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let enc = &cfg.multimodal.whisper_model_args.encoder_args;
        // Encoder max activation: attention matrix
        // ~3000 mel frames, encoder has 32 heads, seq_len^2
        let max_enc_seq = 3000usize;
        Ok(max_batch_size * enc.n_heads * max_enc_seq * max_enc_seq)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        _weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let enc = &cfg.multimodal.whisper_model_args.encoder_args;
        let ds = &cfg.multimodal.whisper_model_args.downsample_args;

        let elem = dtype.size_in_bytes();

        // Encoder conv layers
        let conv1 = enc.dim * enc.audio_encoding_args.num_mel_bins * 3 + enc.dim; // weight + bias
        let conv2 = enc.dim * enc.dim * 3 + enc.dim;

        // Encoder layers
        let enc_attn_per_layer = 4 * enc.dim * enc.dim; // wq, wk, wv, wo (full heads)
        let enc_mlp_per_layer = 3 * enc.dim * enc.hidden_dim; // w1, w2, w3
        let enc_norm_per_layer = 2 * enc.dim; // attention_norm, ffn_norm
        let enc_layers =
            enc.n_layers * (enc_attn_per_layer + enc_mlp_per_layer + enc_norm_per_layer);
        let enc_final_norm = enc.dim;

        // Adapter
        let adapter_in_features = enc.dim * ds.downsample_factor;
        let adapter = adapter_in_features * cfg.dim + cfg.dim + cfg.dim * cfg.dim + cfg.dim;

        let total_encoder = conv1 + conv2 + enc_layers + enc_final_norm + adapter;

        // Decoder embeddings
        let embeddings = cfg.vocab_size * cfg.dim;

        Ok((total_encoder + embeddings) * elem)
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        let elem = dtype.size_in_bytes();

        let attn = (cfg.dim * cfg.n_heads * cfg.head_dim
            + cfg.dim * cfg.n_kv_heads * cfg.head_dim
            + cfg.dim * cfg.n_kv_heads * cfg.head_dim
            + cfg.n_heads * cfg.head_dim * cfg.dim)
            / weight_pack_factor;
        let mlp = (cfg.dim * cfg.hidden_dim + cfg.hidden_dim * cfg.dim + cfg.dim * cfg.hidden_dim)
            / weight_pack_factor;
        let norms = 2 * cfg.dim; // attention_norm + ffn_norm

        let per_layer = (attn + mlp + norms) * elem;

        Ok(vec![per_layer; cfg.n_layers])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;
        Ok(cfg.n_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: VoxtralConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.model_max_length,
            num_layers: cfg.n_layers,
            hidden_size: cfg.dim,
            num_kv_heads: cfg.n_kv_heads,
            num_attn_heads: cfg.n_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}
