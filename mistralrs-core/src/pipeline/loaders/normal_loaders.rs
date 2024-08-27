use std::{collections::HashMap, fmt::Debug, str::FromStr};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{Llama3RopeConfig, PhiRopeScalingConfig},
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigMetadata},
    pipeline::{
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        Cache, IsqModel,
    },
    xlora_models::NonGranularState,
};
use anyhow::Result;
use candle_core::{Device, Tensor};
use candle_nn::{Activation, VarBuilder};

use mistralrs_quant::QuantizedConfig;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

use crate::{
    models,
    xlora_models::{self, XLoraConfig},
};

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        start_offsets_kernel: Tensor,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &mut PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        start_offsets_kernel: Tensor,
        start_offsets_kernel_full: Tensor,
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &Cache;
    fn max_seq_len(&self) -> usize;
    fn activate_adapters(&mut self, _: Vec<String>) -> candle_core::Result<usize> {
        // NOTE: While X-LoRA shares a similar name, it is not equivalent. Its adapter set must remain the same.
        candle_core::bail!(
            "Activating adapters is only supported for models fine-tuned with LoRA."
        );
    }
    fn config(&self) -> &ModelConfigMetadata;
}

/// Metadata for loading a model with ISQ or device mapping.
pub struct NormalLoadingMetadata {
    // Device mapping metadata which can be used to construct a concrete device mapper
    pub mapper: Box<dyn DeviceMapper + Send + Sync>,
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
        attention_mechanism: AttentionImplementation,
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
    /// Get total num_hidden_layers for the layers which will be device mapped.
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, PartialEq)]
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
    #[serde(rename = "gemma2")]
    Gemma2,
    #[serde(rename = "starcoder2")]
    Starcoder2,
    #[serde(rename = "phi3.5moe")]
    Phi3_5MoE,
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
            "gemma2" => Ok(Self::Gemma2),
            "starcoder2" => Ok(Self::Starcoder2),
            "phi3.5moe" => Ok(Self::Phi3_5MoE),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `mistral`, `gemma`, `mixtral`, `llama`, `phi2`, `phi3`, `qwen2`, `gemma2`, `starcoder2`, `phi3.5moe`.")),
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
    head_dim: Option<usize>,
    quantization_config: Option<QuantizedConfig>,
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
            head_dim: basic_config.head_dim,
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mistral::Model::new(
            &MistralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(MistralBasicConfig::deserialize(config, false)?.num_hidden_layers)
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
    quantization_config: Option<QuantizedConfig>,
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
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::gemma::Model::new(
            &GemmaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(GemmaBasicConfig::deserialize(config, false)?.num_hidden_layers)
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
    rope_scaling: Option<Llama3RopeConfig>,
    quantization_config: Option<QuantizedConfig>,
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
            rope_scaling: basic_config.rope_scaling,
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::llama::Llama::new(
            &LlamaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(LlamaBasicConfig::deserialize(config, false)?.num_hidden_layers)
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
    sliding_window: Option<usize>,
    num_experts_per_tok: usize,
    num_local_experts: usize,
    quantization_config: Option<QuantizedConfig>,
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
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mixtral::Model::new(
            &MixtralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(MixtralBasicConfig::deserialize(config, false)?.num_hidden_layers)
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
    rope_theta: f32,
    partial_rotary_factor: f64,
    qk_layernorm: bool,
    quantization_config: Option<QuantizedConfig>,
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
            partial_rotary_factor: basic_config.partial_rotary_factor,
            qk_layernorm: basic_config.qk_layernorm,
            use_flash_attn,
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi2::Model::new(
            &Phi2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(Phi2BasicConfig::deserialize(config, false)?.num_hidden_layers)
    }
}

// ======================== Phi3 loader

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
    rope_scaling: Option<PhiRopeScalingConfig>,
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
    sliding_window: Option<usize>,
    quantization_config: Option<QuantizedConfig>,
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
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi3::Model::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(Phi3BasicConfig::deserialize(config, false)?.num_hidden_layers)
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
    rope_theta: f64,
    rms_norm_eps: f64,
    hidden_act: Activation,
    quantization_config: Option<QuantizedConfig>,
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
            use_flash_attn,
            quantization_config: basic_config.quantization_config,
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
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::qwen2::Model::new(
            &Qwen2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(Qwen2BasicConfig::deserialize(config, false)?.num_hidden_layers)
    }
}

// ======================== Gemma2 loader

#[derive(Deserialize)]
struct Gemma2BasicConfig {
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
    sliding_window: usize,
    attn_logit_softcapping: Option<f64>,
    final_logit_softcapping: Option<f64>,
    query_pre_attn_scalar: usize,

    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    quantization_config: Option<QuantizedConfig>,
}

impl Gemma2BasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::gemma2::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::gemma2::Config {
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
            quantization_config: basic_config.quantization_config,
            sliding_window: basic_config.sliding_window,
            attn_logit_softcapping: basic_config.attn_logit_softcapping,
            final_logit_softcapping: basic_config.final_logit_softcapping,
            query_pre_attn_scalar: basic_config.query_pre_attn_scalar,
        })
    }
}

/// [`NormalLoader`] for a Gemma2 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Gemma2Loader;

impl NormalModelLoader for Gemma2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::gemma2::Model::new(
            &Gemma2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
        Ok(Box::new(xlora_models::XLoraGemma2::new(
            &Gemma2BasicConfig::deserialize(config, use_flash_attn)?,
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
        // Already will warn about it
        Ok(Box::new(Gemma2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(Gemma2BasicConfig::deserialize(config, false)?.num_hidden_layers)
    }
}

// ======================== Starcoder2 loader

#[derive(Deserialize, Debug)]
struct Starcoder2BasicConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    hidden_act: candle_nn::Activation,
    max_position_embeddings: usize,
    norm_epsilon: f64,
    rope_theta: f64,
    use_bias: bool,
    sliding_window: Option<usize>,
    quantization_config: Option<QuantizedConfig>,
}

impl Starcoder2BasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::starcoder2::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::starcoder2::Config {
            vocab_size: basic_config.vocab_size,
            hidden_size: basic_config.hidden_size,
            intermediate_size: basic_config.intermediate_size,
            num_hidden_layers: basic_config.num_hidden_layers,
            num_attention_heads: basic_config.num_attention_heads,
            num_key_value_heads: basic_config.num_key_value_heads,
            hidden_act: basic_config.hidden_act,
            max_position_embeddings: basic_config.max_position_embeddings,
            rope_theta: basic_config.rope_theta,
            sliding_window: basic_config.sliding_window,
            use_flash_attn,
            norm_epsilon: basic_config.norm_epsilon,
            use_bias: basic_config.use_bias,
            quantization_config: basic_config.quantization_config,
        })
    }
}

/// [`NormalLoader`] for a Starcoder2 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Starcoder2Loader;

impl NormalModelLoader for Starcoder2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::starcoder2::Model::new(
            &Starcoder2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
        Ok(Box::new(xlora_models::XLoraStarcoder2::new(
            &Starcoder2BasicConfig::deserialize(config, use_flash_attn)?,
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
    fn get_config_repr(&self, config: &str, _use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(serde_json::from_str::<Starcoder2BasicConfig>(
            config,
        )?))
    }
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(serde_json::from_str::<Starcoder2BasicConfig>(config)?.num_hidden_layers)
    }
}

// ======================== Phi3 loader

#[derive(Deserialize)]
struct Phi3_5MoEBasicConfig {
    vocab_size: usize,
    hidden_act: candle_nn::Activation,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    rope_scaling: Option<PhiRopeScalingConfig>,
    max_position_embeddings: usize,
    original_max_position_embeddings: usize,
    sliding_window: Option<usize>,
    quantization_config: Option<QuantizedConfig>,
    lm_head_bias: bool,
    attention_bias: bool,
    num_local_experts: usize,
    router_jitter_noise: f64,
}

impl Phi3_5MoEBasicConfig {
    fn deserialize(slice: &str, use_flash_attn: bool) -> Result<models::phi3_5_moe::Config> {
        let basic_config: Self = serde_json::from_str(slice)?;
        Ok(models::phi3_5_moe::Config {
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
            rope_scaling: basic_config.rope_scaling,
            original_max_position_embeddings: basic_config.original_max_position_embeddings,
            use_flash_attn,
            sliding_window: basic_config.sliding_window,
            quantization_config: basic_config.quantization_config,
            lm_head_bias: basic_config.lm_head_bias,
            attention_bias: basic_config.attention_bias,
            num_local_experts: basic_config.num_local_experts,
            router_jitter_noise: basic_config.router_jitter_noise,
        })
    }
}

/// [`NormalLoader`] for a Phi 3.5 MoE model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct Phi3_5MoELoader;

impl NormalModelLoader for Phi3_5MoELoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi3_5_moe::Model::new(
            &Phi3_5MoEBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(),
            normal_loading_metadata,
            attention_mechanism,
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
        Ok(Box::new(Phi3_5MoEBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
    fn get_total_device_mapping_num_layers(&self, config: &str) -> Result<usize> {
        Ok(Phi3_5MoEBasicConfig::deserialize(config, false)?.num_hidden_layers)
    }
}
