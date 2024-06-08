use std::{collections::HashMap, fmt::Debug, str::FromStr};

use crate::lora::{LoraConfig, Ordering};
use anyhow::Result;
use candle_core::Device;
use candle_nn::{Activation, VarBuilder};
use either::Either;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

/// Metadata for loading a model with ISQ or device mapping.
pub struct NormalLoadingMetadata {
    // Device mapping metadata which can be used to construct a concrete device mapper
    pub mapper: DeviceMapMetadata,
    // Flag to check if loading in ISQ
    pub loading_isq: bool,
    // Device mapping target device (the one that is not the cpu)
    pub real_device: Device,
}

pub trait NormalModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
}

use super::NormalModel;
use crate::{
    models,
    xlora_models::{self, XLoraConfig},
    DeviceMapMetadata,
};

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Clone, Debug, Deserialize)]
/// The architecture to load the normal model as.
pub enum NormalLoaderType {
    #[serde(rename = "mistral")]
    Mistral,
    #[serde(rename = "gemma")]
    Gemma,
    #[serde(rename = "mixtral")]
    Mixtral,
    #[serde(rename = "llama")]
    Llama,
    #[serde(rename = "phi2")]
    Phi2,
    #[serde(rename = "phi3")]
    Phi3,
    #[serde(rename = "qwen2")]
    Qwen2,
}

impl FromStr for NormalLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mistral" => Ok(Self::Mistral),
            "gemma" => Ok(Self::Gemma),
            "mixtral" => Ok(Self::Mixtral),
            "llama" => Ok(Self::Llama),
            "phi2" => Ok(Self::Phi2),
            "phi3" => Ok(Self::Phi3),
            "qwen2" => Ok(Self::Qwen2),
            a => Err(format!("Unknown architecture `{a}`")),
        }
    }
}

// ======================== Mistral loader

#[derive(Deserialize, Debug)]
struct MistralBasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: Option<usize>,
}

impl MistralBasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::mistral::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::mistral::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            sliding_window: basic_config.sliding_window,
            use_flash_attn,
        })
    }
}

pub struct MistralLoader;

impl NormalModelLoader for MistralLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mistral::Model::new(
            &MistralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraMistral::new(
            &MistralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(MistralBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Gemma loader

fn default_max_position_embeddings() -> usize {
    4096
}

#[derive(Deserialize)]
struct GemmaBasicConfig {
    attention_bias: bool,
    head_dim: usize,
    // The code gemma configs include both hidden_act and hidden_activation.
    hidden_act: Option<Activation>,
    hidden_activation: Option<Activation>,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
}

impl GemmaBasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::gemma::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::gemma::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            hidden_activation: basic_config.hidden_activation,
            max_position_embeddings: basic_config.max_position_embeddings,
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            attention_bias: basic_config.attention_bias,
            head_dim: basic_config.head_dim,
            use_flash_attn,
        })
    }
}

/// [`NormalLoader`] for a Gemma model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct GemmaLoader;

impl NormalModelLoader for GemmaLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::gemma::Model::new(
            &GemmaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraGemma::new(
            &GemmaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(GemmaBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Llama loader

#[derive(Deserialize)]
struct LlamaBasicConfig {
    hidden_size: usize,
    intermediate_size: usize,
    vocab_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    rms_norm_eps: f64,
    #[serde(default = "default_rope")]
    rope_theta: f32,
    max_position_embeddings: usize,
}

fn default_rope() -> f32 {
    10_000.0
}

impl LlamaBasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::llama::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::llama::Config {
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            vocab_size: basic_config.vocab_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config
                .num_key_value_heads
                .unwrap_or(basic_config.num_attention_heads),
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            use_flash_attn,
            max_position_embeddings: basic_config.max_position_embeddings,
        })
    }
}

/// [`NormalLoader`] for a Llama model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct LlamaLoader;

impl NormalModelLoader for LlamaLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::llama::Llama::new(
            &LlamaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraLlama::new(
            &LlamaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(LlamaBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Mixtral loader

#[derive(Deserialize)]
struct MixtralBasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: Activation,
    max_position_embeddings: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    sliding_window: usize,
    num_experts_per_tok: usize,
    num_local_experts: usize,
}

impl MixtralBasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::mixtral::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::mixtral::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rms_norm_eps: basic_config.rms_norm_eps,
            rope_theta: basic_config.rope_theta,
            sliding_window: basic_config.sliding_window,
            use_flash_attn,
            num_experts_per_tok: basic_config.num_experts_per_tok,
            num_local_experts: basic_config.num_local_experts,
        })
    }
}

pub struct MixtralLoader;

impl NormalModelLoader for MixtralLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mixtral::Model::new(
            &MixtralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraMixtral::new(
            &MixtralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(MixtralBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Phi2 loader

#[derive(Deserialize)]
struct Phi2BasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: Option<usize>,
    hidden_act: Activation,
    max_position_embeddings: usize,
    layer_norm_eps: f64,
    tie_word_embeddings: bool,
    rope_theta: f32,
    partial_rotary_factor: f64,
    qk_layernorm: bool,
}

impl Phi2BasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::phi2::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::phi2::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rope_theta: basic_config.rope_theta,
            layer_norm_eps: basic_config.layer_norm_eps,
            tie_word_embeddings: basic_config.tie_word_embeddings,
            partial_rotary_factor: basic_config.partial_rotary_factor,
            qk_layernorm: basic_config.qk_layernorm,
            use_flash_attn,
        })
    }
}

/// [`NormalLoader`] for a Phi 2 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Phi2Loader;

impl NormalModelLoader for Phi2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi2::Model::new(
            &Phi2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraPhi2::new(
            &Phi2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Phi2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Phi3 loader

#[derive(Deserialize, Debug, Clone)]
pub struct Phi3RopeScaling(#[serde(with = "either::serde_untagged")] pub Either<Vec<f32>, String>);

#[derive(Deserialize)]
struct Phi3BasicConfig {
    vocab_size: usize,
    hidden_act: candle_nn::Activation,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    bos_token_id: Option<u32>,
    eos_token_id: Option<u32>,
    rope_scaling: Option<HashMap<String, Phi3RopeScaling>>,
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
    sliding_window: Option<usize>,
}

impl Phi3BasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::phi3::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::phi3::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rope_theta: basic_config.rope_theta,
            rms_norm_eps: basic_config.rms_norm_eps,
            eos_token_id: basic_config.eos_token_id,
            bos_token_id: basic_config.bos_token_id,
            rope_scaling: basic_config.rope_scaling,
            original_max_position_embeddings: basic_config.original_max_position_embeddings,
            use_flash_attn,
            sliding_window: basic_config.sliding_window,
        })
    }
}

/// [`NormalLoader`] for a Phi 3 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Phi3Loader;

impl NormalModelLoader for Phi3Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi3::Model::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(),
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Phi3BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

// ======================== Qwen2 loader

#[derive(Deserialize)]
struct Qwen2BasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    sliding_window: usize,
    max_window_layers: usize,
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    use_sliding_window: bool,
    hidden_act: Activation,
}

impl Qwen2BasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::qwen2::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::qwen2::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rope_theta: basic_config.rope_theta,
            rms_norm_eps: basic_config.rms_norm_eps,
            sliding_window: basic_config.sliding_window,
            max_window_layers: basic_config.max_window_layers,
            tie_word_embeddings: basic_config.tie_word_embeddings,
            use_sliding_window: basic_config.use_sliding_window,
            use_flash_attn,
        })
    }
}

/// [`NormalLoader`] for a Qwen 2 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Qwen2Loader;

impl NormalModelLoader for Qwen2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::qwen2::Model::new(
            &Qwen2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _use_flash_attn: bool,
        _vb: VarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Qwen2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}
