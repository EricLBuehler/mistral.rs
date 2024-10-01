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
use crate::pipeline::isq::{IsqModelLoader, WordEmbeddingsShim};
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{Cache, IsqModel, Processor, ProcessorCreator};
use crate::vision_models::idefics2::{Config as Idefics2Config, Idefics2};
use crate::vision_models::idefics2_input_processor::Idefics2Processor;
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
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
    fn has_conv2d(&self) -> bool;
    fn config(&self) -> &ModelConfigMetadata;
}

pub trait VisionModelLoader: IsqModelLoader {
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
    ) -> Arc<dyn Processor + Send + Sync>;
    fn supports_paged_attention(&self) -> bool;
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
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `phi3v`, `idefics2`, `llava_next`, `llava`, `vsllama`.")),
        }
    }
}

// ======================== Phi 3 loader

/// [`VisionLoader`] for a Phi 3 Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Phi3VLoader;

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
}

impl IsqModelLoader for Phi3VLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut regexes = Vec::new();
        if serde_json::from_str::<WordEmbeddingsShim>(config)?.tie_word_embeddings {
            regexes.push(Regex::new(r"(embed_tokens|lm_head)\.(weight|bias)$")?);
        } else {
            regexes.push(Regex::new(r"lm_head\.(weight|bias)$")?);
        }
        // Attention
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
        )?);
        // MLP
        regexes.push(Regex::new(
            r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        )?);
        Ok(regexes)
    }
}

// ======================== Idefics 2 loader

/// [`VisionLoader`] for an Idefics 2 Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Idefics2Loader;

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
    ) -> Arc<dyn Processor + Send + Sync> {
        Arc::new(Idefics2Processor::new(
            processor_config.unwrap(),
            preprocessor_config,
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
}

impl IsqModelLoader for Idefics2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // Tie weights is unsupported for this model
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
}

impl IsqModelLoader for LLaVANextLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // Tie weights is unsupported for this model
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
}

impl IsqModelLoader for LLaVALoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // Tie weights is unsupported for this model
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
}

impl IsqModelLoader for VLlamaLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut regexes = Vec::new();
        if serde_json::from_str::<MLlamaConfig>(config)?
            .text_config
            .tie_word_embeddings
        {
            regexes.push(Regex::new(r"(embed_tokens|lm_head)\.(weight|bias)$")?);
        } else {
            regexes.push(Regex::new(r"lm_head\.(weight|bias)$")?);
        }
        // Attention
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$",
        )?);
        // MLP
        regexes.push(Regex::new(
            r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$",
        )?);
        regexes.push(Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?);
        regexes.push(Regex::new(
            r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$",
        )?);
        Ok(regexes)
    }
}
