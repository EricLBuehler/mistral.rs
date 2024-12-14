use std::any::Any;
use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use super::NormalLoadingMetadata;
use crate::amoe::AnyMoeBaseModelMixin;
use crate::paged_attention::{AttentionImplementation, ModelConfigMetadata};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{EitherCache, IsqModel, Processor, ProcessorCreator, VisionPromptPrefixer};
use crate::vision_models::idefics2::{Config as Idefics2Config, Idefics2};
use crate::vision_models::idefics2_input_processor::Idefics2Processor;
use crate::vision_models::idefics3::{Idefics3Config, Idefics3Model, Idefics3Processor};
use crate::vision_models::llava::config::Config as LLaVAConfig;
use crate::vision_models::llava15::Model as LLaVA;
use crate::vision_models::llava_inputs_processor::LLaVAProcessor;
use crate::vision_models::llava_next::Model as LLaVANext;
use crate::vision_models::llava_next_inputs_processor::LLaVANextProcessor;
use crate::vision_models::mllama::{MLlamaConfig, MLlamaModel, MLlamaProcessor};
use crate::vision_models::phi3::{Config as Phi3Config, Model as Phi3};
use crate::vision_models::phi3_inputs_processor::Phi3Processor;
use crate::vision_models::preprocessor_config::PreProcessorConfig;
use crate::vision_models::processor_config::ProcessorConfig;
use crate::vision_models::qwen2vl::{Config as Qwen2VLConfig, Qwen2VLModel, Qwen2VLProcessor};

pub trait VisionModel: IsqModel + AnyMoeBaseModelMixin {
    // pixel_values and pixel_attention_mask only specified for prompt seqs
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<Tensor>,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        model_specific_args: Box<dyn Any>, // pixel attention mask, or image sizes, or anything else
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
    fn max_seq_len(&self) -> usize;
    fn has_conv2d(&self) -> bool;
    fn config(&self) -> &ModelConfigMetadata;
    /// For a prompt without images. Requires batch size of 1!
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any>;
}

pub trait VisionModelLoader: IsqModelLoader + Send + Sync {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
    /// Get total num_hidden_layers for the layers which will be device mapped.
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize>;
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
    fn supports_paged_attention(&self) -> bool;
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, PartialEq)]
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
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `phi3v`, `idefics2`, `llava_next`, `llava`, `vllama`, `qwen2vl`, `idefics3`.")),
        }
    }
}

// ======================== Phi 3 loader

/// [`VisionLoader`] for a Phi 3 Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Phi3VLoader;

pub struct Phi3VPrefixer;

impl VisionPromptPrefixer for Phi3VPrefixer {
    fn prefix_image(&self, image_index: usize, prompt: &str) -> String {
        // Image indexing starts at 0.
        format!("<|image_{}|>{prompt}", image_index + 1)
    }
}

impl VisionModelLoader for Phi3VLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Phi3Config = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Phi3::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: Phi3Config = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: Phi3Config = serde_json::from_str(config)?;
        Ok(config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Phi3VPrefixer)
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
            Regex::new(r"layers\.(\d+)\.mlp\.gate__up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

// ======================== Idefics 2 loader

/// [`VisionLoader`] for an Idefics 2 Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Idefics2Loader;

pub struct Idefics2Prefixer;

impl VisionPromptPrefixer for Idefics2Prefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl VisionModelLoader for Idefics2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Idefics2Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Idefics2::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: Idefics2Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: Idefics2Config = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.text_config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Idefics2Prefixer)
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
}

// ======================== LLaVANext Loader

/// [`VisionLoader`] for an LLaVANext Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct LLaVANextLoader;

pub struct LLaVANextPrefixer;

impl VisionPromptPrefixer for LLaVANextPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!("<image>{prompt}")
    }
}

impl VisionModelLoader for LLaVANextLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: LLaVAConfig = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(LLaVANext::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: LLaVAConfig = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: LLaVAConfig = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.text_config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(LLaVANextPrefixer)
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
}

// ======================== LLaVA Loader

/// [`VisionLoader`] for an LLaVA Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct LLaVALoader;

pub struct LLaVAPrefixer;

impl VisionPromptPrefixer for LLaVAPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!("<image>{prompt}")
    }
}

impl VisionModelLoader for LLaVALoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: LLaVAConfig = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(LLaVA::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        false
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: LLaVAConfig = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: LLaVAConfig = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.text_config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(LLaVAPrefixer)
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
}

// ======================== MLlama Loader

/// [`VisionLoader`] for an Llama Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct VLlamaLoader;

pub struct VLlamaPrefixer;

impl VisionPromptPrefixer for VLlamaPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!("<|image|>{prompt}")
    }
}

impl VisionModelLoader for VLlamaLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: MLlamaConfig = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(MLlamaModel::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: MLlamaConfig = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.text_config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        false
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(VLlamaPrefixer)
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
}

// ======================== Qwen2VL Loader

/// [`VisionLoader`] for an Qwen2-VL model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Qwen2VLLoader;

pub struct Qwen2VLPrefixer;

impl VisionPromptPrefixer for Qwen2VLPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!(
            "{}{}{}{prompt}",
            Qwen2VLProcessor::VISION_START,
            Qwen2VLProcessor::IMAGE_PAD,
            Qwen2VLProcessor::VISION_END
        )
    }
}

impl VisionModelLoader for Qwen2VLLoader {
    fn load(
        &self,
        config: &str,
        _use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen2VLModel::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, _use_flash_attn: bool) -> Result<Box<dyn Debug>> {
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: Qwen2VLConfig = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        false
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Qwen2VLPrefixer)
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
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

// ======================== Idefics 3 loader

/// [`VisionLoader`] for an Idefics 3 Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Idefics3Loader;

pub struct Idefics3Prefixer;

impl VisionPromptPrefixer for Idefics3Prefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        // Chat template does it
        prompt.to_string()
    }
}

impl VisionModelLoader for Idefics3Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Idefics3Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Idefics3Model::new(
            &config,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: Idefics3Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        let config: Idefics3Config = serde_json::from_str(config)?;
        // We only apply device mapping to text model
        Ok(config.text_config.num_hidden_layers)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Idefics3Prefixer)
    }
}

impl IsqModelLoader for Idefics3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        todo!()
    }
}
