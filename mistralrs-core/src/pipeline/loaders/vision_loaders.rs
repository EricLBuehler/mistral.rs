use std::any::Any;
use std::sync::Arc;
use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};
use candle_nn::Conv2dConfig;
use image::{ColorType, DynamicImage};
use mistralrs_quant::ShardedVarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use self::minicpmo::{MiniCpmOConfig, MiniCpmOModel, MiniCpmOProcessor};

use super::{DeviceMappedModelLoader, NonMappedSubModel, NormalLoadingMetadata};
use crate::amoe::AnyMoeBaseModelMixin;
use crate::device_map::DeviceMapper;
use crate::layers::Conv3dConfig;
use crate::paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata};
use crate::pipeline::isq::IsqModelLoader;
use crate::pipeline::loaders::AutoDeviceMapParams;
use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};
use crate::pipeline::{EitherCache, IsqModel, Processor, ProcessorCreator, VisionPromptPrefixer};
use crate::utils::varbuilder_utils::DeviceForLoadTensor;
use crate::vision_models::clip::ClipConfig;
use crate::vision_models::gemma3::config::Gemma3Config;
use crate::vision_models::gemma3::{Gemma3Model, Gemma3Processor};
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
    fn has_conv2d(&self) -> bool;
    fn config(&self) -> &ModelConfigMetadata;
    /// For a prompt without images. Requires batch size of 1!
    fn default_model_specific_args(&self, input_ids: &Tensor) -> Box<dyn Any>;
}

pub trait VisionModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
    fn get_processor(
        &self,
        model_config: &str,
        processor_config: Option<ProcessorConfig>,
        preprocessor_config: PreProcessorConfig,
        max_edge: Option<u32>,
    ) -> Arc<dyn Processor + Send + Sync>;
    fn supports_paged_attention(&self) -> bool;
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer>;
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
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `phi3v`, `idefics2`, `llava_next`, `llava`, `vllama`, `qwen2vl`, `idefics3`, `minicpmo`, `phi4mm`, `qwen2_5vl`, `gemma3`, `mistral3`, `llama4`.")),
        }
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
        vb: ShardedVarBuilder,
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
            Regex::new(r"layers\.(\d+)\.mlp\.gate_up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Phi3VLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Phi3Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
    ) -> Result<Vec<usize>> {
        let cfg: Phi3Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size = head_dim * head_dim + 2 * cfg.num_key_value_heads * head_dim;
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
        vb: ShardedVarBuilder,
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

impl DeviceMappedModelLoader for Idefics2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
        vb: ShardedVarBuilder,
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

impl DeviceMappedModelLoader for LLaVANextLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;

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
        vb: ShardedVarBuilder,
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

impl DeviceMappedModelLoader for LLaVALoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;

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
        vb: ShardedVarBuilder,
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

impl DeviceMappedModelLoader for VLlamaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            max_batch_size * cfg.num_attention_heads * max_seq_len * max_seq_len
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
    ) -> Result<usize> {
        let config: MLlamaConfig = serde_json::from_str(config)?;
        let text_elems = {
            let cfg = &config.text_config;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
        vb: ShardedVarBuilder,
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

impl DeviceMappedModelLoader for Qwen2VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Qwen2VLConfig = serde_json::from_str(config)?;
        let text_elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
        vb: ShardedVarBuilder,
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
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Idefics3Prefixer)
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
}

impl DeviceMappedModelLoader for Idefics3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct MiniCpmOLoader;

pub struct MiniCpmOPrefixer;

impl VisionPromptPrefixer for MiniCpmOPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!("(<image>./</image>){prompt}")
    }
}

impl VisionModelLoader for MiniCpmOLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: MiniCpmOConfig = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(MiniCpmOModel::new(
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
        let mut config: MiniCpmOConfig = serde_json::from_str(config)?;
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
        Arc::new(MiniCpmOProcessor::new(
            processor_config.unwrap_or_default(),
            preprocessor_config,
            max_edge,
        ))
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(MiniCpmOPrefixer)
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
            Regex::new(r"llm.layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"llm.layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"llm.layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for MiniCpmOLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi 4MM loader

/// [`VisionLoader`] for a Phi 4MM Vision model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Phi4MMLoader;

pub struct Phi4MMPrefixer;

impl VisionPromptPrefixer for Phi4MMPrefixer {
    fn prefix_image(&self, image_index: usize, prompt: &str) -> String {
        // Image indexing starts at 0.
        format!("<|image_{}|>{prompt}", image_index + 1)
    }
}

impl VisionModelLoader for Phi4MMLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Phi4MMConfig = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Phi4MMModel::new(
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
        let mut config: Phi4MMConfig = serde_json::from_str(config)?;
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
        Phi4MMProcessor::new_processor(processor_config, preprocessor_config)
    }
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Phi4MMPrefixer)
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
}

impl DeviceMappedModelLoader for Phi4MMLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
    ) -> Result<Vec<usize>> {
        let cfg: Phi4MMConfig = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size = head_dim * head_dim + 2 * cfg.num_key_value_heads() * head_dim;
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
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}

// ======================== Qwen2_5VL Loader

/// [`VisionLoader`] for an Qwen2_5VL model.
///
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Qwen2_5VLLoader;

pub struct Qwen2_5VLPrefixer;

impl VisionPromptPrefixer for Qwen2_5VLPrefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!(
            "{}{}{}{prompt}",
            Qwen2_5VLProcessor::VISION_START,
            Qwen2_5VLProcessor::IMAGE_PAD,
            Qwen2_5VLProcessor::VISION_END
        )
    }
}

impl VisionModelLoader for Qwen2_5VLLoader {
    fn load(
        &self,
        config: &str,
        _use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let config: Qwen2_5VLConfig = serde_json::from_str(config)?;
        Ok(Box::new(Qwen2_5VLModel::new(
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
    fn supports_paged_attention(&self) -> bool {
        false
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Qwen2_5VLPrefixer)
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
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Qwen2_5VLLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
            let max_seq_len = img_seq_len + max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Qwen2_5VLConfig = serde_json::from_str(config)?;
        let text_elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Gemma3Loader;

pub struct Gemma3Prefixer;

impl VisionPromptPrefixer for Gemma3Prefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for Gemma3Loader {
    fn load(
        &self,
        config: &str,
        _use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let config: Gemma3Config = serde_json::from_str(config)?;
        Ok(Box::new(Gemma3Model::new(
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
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Gemma3Prefixer)
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
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Gemma3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
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
                * prompt_chunksize
                * prompt_chunksize),
            Gemma3Config::WithVision {
                text_config,
                vision_config,
                ..
            } => {
                let num_patches = (vision_config.image_size / vision_config.patch_size).pow(2);
                let img_seq_len = (num_patches + 1) * max_num_images;

                let max_text_attn = {
                    // This model injects the vision information directly into the input embeddings
                    let max_seq_len = img_seq_len + *max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Gemma3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = match &cfg {
                Gemma3Config::Text(cfg) => cfg,
                Gemma3Config::WithVision { text_config, .. } => text_config,
            };
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct Mistral3Loader;

pub struct Mistral3Prefixer;

impl VisionPromptPrefixer for Mistral3Prefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        prompt.to_string()
    }
}

impl VisionModelLoader for Mistral3Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Mistral3Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Mistral3Model::new(
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
        let config: Mistral3Config = serde_json::from_str(config)?;
        Ok(Box::new(config))
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
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(Mistral3Prefixer)
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
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

#[allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]
impl DeviceMappedModelLoader for Mistral3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        _prompt_chunksize: usize,
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
        let max_seq_len = img_seq_len * max_num_images + *max_seq_len;
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
    ) -> Result<usize> {
        let cfg: Mistral3Config = serde_json::from_str(config)?;

        let text_elems = {
            let cfg = &cfg.text_config;

            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
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
/// [`VisionLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.VisionLoader.html
pub struct VLlama4Loader;

pub struct VLlama4Prefixer;

impl VisionPromptPrefixer for VLlama4Prefixer {
    fn prefix_image(&self, _image_index: usize, prompt: &str) -> String {
        format!("{}{prompt}", llama4::IMAGE_TOKEN)
    }
}

impl VisionModelLoader for VLlama4Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn VisionModel + Send + Sync>> {
        let mut config: Llama4Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(Llama4Model::new(
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
        let mut config: Llama4Config = serde_json::from_str(config)?;
        config.text_config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
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
    fn supports_paged_attention(&self) -> bool {
        true
    }
    fn prefixer(&self) -> Arc<dyn VisionPromptPrefixer> {
        Arc::new(VLlama4Prefixer)
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
        _prompt_chunksize: usize,
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

        let max_seq_len = max_seq_len + num_text_image_toks;

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
        };

        Ok(Box::new(cfg))
    }

    fn non_mapped_sub_models(&self) -> Option<Vec<NonMappedSubModel>> {
        Some(vec![NonMappedSubModel::Vision])
    }
}
