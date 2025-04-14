use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    str::FromStr,
    sync::Arc,
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    layers::{Activation, Llama3RopeConfig, PhiRopeScalingConfig},
    lora::{LoraConfig, Ordering},
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::{
        isq::IsqModelLoader,
        text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
        EitherCache, IsqModel,
    },
    serde_default_fn,
    utils::{log::once_log_info, varbuilder_utils::DeviceForLoadTensor},
    xlora_models::NonGranularState,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};

use indicatif::MultiProgress;
use mistralrs_quant::{QuantizedConfig, ShardedVarBuilder};
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use crate::{
    models,
    xlora_models::{self, XLoraConfig},
};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    #[allow(clippy::too_many_arguments)]
    fn xlora_forward(
        &self,
        input_ids: &Tensor,
        input_ids_full: &Tensor,
        seqlen_offsets: &[usize],
        seqlen_offsets_full: &[usize],
        no_kv_cache: bool,
        non_granular_state: &Option<NonGranularState>,
        context_lens: Vec<(usize, usize)>,
        position_ids: Vec<usize>,
        flash_params: &FlashParams,
        flash_params_full: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn is_xlora(&self) -> bool;
    fn device(&self) -> &Device;
    fn cache(&self) -> &EitherCache;
    fn cache_mut(&mut self) -> &mut EitherCache;
    fn max_seq_len(&self) -> usize;
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
    // MultiProgress support for parallelized loading
    pub multi_progress: Arc<MultiProgress>,
}

pub trait NormalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    fn is_gptx(&self, config: &str) -> Result<bool>;
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>>;
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
    #[serde(rename = "deepseekv2")]
    DeepSeekV2,
    #[serde(rename = "deepseekv3")]
    DeepSeekV3,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448

impl NormalLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "MistralForCausalLM" => Ok(Self::Mistral),
            "MixtralForCausalLM" => Ok(Self::Mixtral),
            "GemmaForCausalLM" => Ok(Self::Gemma),
            "Gemma2ForCausalLM" => Ok(Self::Gemma2),
            "PhiForCausalLM" => Ok(Self::Phi2),
            "Phi3ForCausalLM" => Ok(Self::Phi3),
            "LlamaForCausalLM" => Ok(Self::Llama),
            "Qwen2ForCausalLM" => Ok(Self::Qwen2),
            "Starcoder2ForCausalLM" => Ok(Self::Starcoder2),
            "PhiMoEForCausalLM" => Ok(Self::Phi3_5MoE),
            "DeepseekV2ForCausalLM" => Ok(Self::DeepSeekV2),
            "DeepseekV3ForCausalLM" => Ok(Self::DeepSeekV3),
            other => anyhow::bail!(
                "Unsupported Huggging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
            ),
        }
    }
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
            "deepseekv2" => Ok(Self::DeepSeekV2),
            "deepseekv3" => Ok(Self::DeepSeekV3),
            a => Err(format!("Unknown architecture `{a}`. Possible architectures: `mistral`, `gemma`, `mixtral`, `llama`, `phi2`, `phi3`, `qwen2`, `gemma2`, `starcoder2`, `phi3.5moe`, `deepseekv2`, `deepseekv3`.")),
        }
    }
}

impl Display for NormalLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gemma => write!(f, "gemma"),
            Self::Gemma2 => write!(f, "gemma2"),
            Self::Llama => write!(f, "llama"),
            Self::Mistral => write!(f, "mistral"),
            Self::Mixtral => write!(f, "mixtral"),
            Self::Phi2 => write!(f, "phi2"),
            Self::Phi3 => write!(f, "phi3"),
            Self::Phi3_5MoE => write!(f, "phi3.5moe"),
            Self::Qwen2 => write!(f, "qwen2"),
            Self::Starcoder2 => write!(f, "starcoder2"),
            Self::DeepSeekV2 => write!(f, "deepseekv2"),
            Self::DeepSeekV3 => write!(f, "deepseekv3"),
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

/// Load a model based on the Huggging Face Transformers -CausalLM model class
pub struct AutoLoader;

#[derive(Deserialize)]
struct AutoLoaderConfig {
    architectures: Vec<String>,
}

impl AutoLoader {
    fn get_loader(config: &str) -> Result<Box<dyn NormalModelLoader>> {
        let auto_cfg: AutoLoaderConfig = serde_json::from_str(config)?;
        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected to have one name for `architectures` config field.")
        }

        let name = &auto_cfg.architectures[0];

        let tp = NormalLoaderType::from_causal_lm_name(name)?;

        once_log_info(format!("Automatic loader type determined to be `{tp}`"));

        match tp {
            NormalLoaderType::Mistral => Ok(Box::new(MistralLoader)),
            NormalLoaderType::Gemma => Ok(Box::new(GemmaLoader)),
            NormalLoaderType::Llama => Ok(Box::new(LlamaLoader)),
            NormalLoaderType::Mixtral => Ok(Box::new(MixtralLoader)),
            NormalLoaderType::Phi2 => Ok(Box::new(Phi2Loader)),
            NormalLoaderType::Phi3 => Ok(Box::new(Phi3Loader)),
            NormalLoaderType::Qwen2 => Ok(Box::new(Qwen2Loader)),
            NormalLoaderType::Gemma2 => Ok(Box::new(Gemma2Loader)),
            NormalLoaderType::Starcoder2 => Ok(Box::new(Starcoder2Loader)),
            NormalLoaderType::Phi3_5MoE => Ok(Box::new(Phi3_5MoELoader)),
            NormalLoaderType::DeepSeekV2 => Ok(Box::new(DeepSeekV2Loader)),
            NormalLoaderType::DeepSeekV3 => Ok(Box::new(DeepSeekV3Loader)),
        }
    }
}

impl NormalModelLoader for AutoLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load(
            config,
            use_flash_attn,
            vb,
            normal_loading_metadata,
            attention_mechanism,
        )
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load_xlora(
            config,
            use_flash_attn,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            normal_loading_metadata,
            preload_adapters,
        )
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config, use_flash_attn)
    }
    fn supports_paged_attention(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.supports_paged_attention(config)
    }
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_gptx(config)
    }
}

impl IsqModelLoader for AutoLoader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for AutoLoader {
    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(config, dtype, weight_pack_factor)
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
    }
    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        Self::get_loader(config)?.layer_sizes_in_bytes(config, dtype, weight_pack_factor)
    }
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &super::AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params, prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        Self::get_loader(config)?.model_config(config)
    }
}

serde_default_fn!(bool, word_emb_default, false);

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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
        })
    }
}

pub struct MistralLoader;

impl NormalModelLoader for MistralLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mistral::Model::new(
            &MistralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraMistral::new(
            &MistralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(MistralBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for MistralLoader {
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

impl DeviceMappedModelLoader for MistralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = MistralBasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = MistralBasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = MistralBasicConfig::deserialize(config, false)?;
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
        let cfg = MistralBasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = MistralBasicConfig::deserialize(config, false)?;

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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::gemma::Model::new(
            &GemmaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraGemma::new(
            &GemmaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(GemmaBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for GemmaLoader {
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

impl DeviceMappedModelLoader for GemmaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = GemmaBasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = GemmaBasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = GemmaBasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
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
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = GemmaBasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = GemmaBasicConfig::deserialize(config, false)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Llama loader

#[derive(Deserialize)]
struct LlamaBasicConfig {
    hidden_act: Activation,
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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
            hidden_act: basic_config.hidden_act,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::llama::Llama::new(
            &LlamaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraLlama::new(
            &LlamaBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(LlamaBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for LlamaLoader {
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

impl DeviceMappedModelLoader for LlamaLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = LlamaBasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = LlamaBasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = LlamaBasicConfig::deserialize(config, false)?;
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
        let cfg = LlamaBasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = LlamaBasicConfig::deserialize(config, false)?;

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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
        })
    }
}

pub struct MixtralLoader;

impl NormalModelLoader for MixtralLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::mixtral::Model::new(
            &MixtralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraMixtral::new(
            &MixtralBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(MixtralBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for MixtralLoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Experts
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.gate\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for MixtralLoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = MixtralBasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = MixtralBasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = MixtralBasicConfig::deserialize(config, false)?;
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

            let moe_block = {
                let gate = cfg.hidden_size * cfg.num_local_experts;
                // Assume quantizing weight pack factor
                let w1 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w2 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w3 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                gate + cfg.num_local_experts * w1
                    + cfg.num_local_experts * w2
                    + cfg.num_local_experts * w3
            };

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + moe_block
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = MixtralBasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = MixtralBasicConfig::deserialize(config, false)?;

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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi2::Model::new(
            &Phi2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraPhi2::new(
            &Phi2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Phi2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for Phi2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Phi2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Phi2BasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Phi2BasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Phi2BasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size + cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor + size_in;
            let (q_norm, k_norm) = if cfg.qk_layernorm {
                (cfg.head_dim(), cfg.head_dim())
            } else {
                (0, 0)
            };

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let fc1 = h_size * i_size / weight_pack_factor;
            let fc2 = h_size * i_size / weight_pack_factor;

            input_layernorm + q_proj + k_proj + v_proj + o_proj + q_norm + k_norm + fc1 + fc2
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Phi2BasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Phi2BasicConfig::deserialize(config, false)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi3 loader

#[derive(Deserialize)]
struct Phi3BasicConfig {
    vocab_size: usize,
    hidden_act: Activation,
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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
    partial_rotary_factor: Option<f64>,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
            partial_rotary_factor: basic_config.partial_rotary_factor,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi3::Model::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Phi3BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for Phi3Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.qkv_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Phi3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Phi3BasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Phi3BasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Phi3BasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size = head_dim * head_dim + 2 * cfg.num_key_value_heads * head_dim;
            let qkv_proj = size_in * op_size / weight_pack_factor;
            let o_proj =
                (cfg.num_attention_heads * head_dim) * size_in / weight_pack_factor + size_in;

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
        let cfg = Phi3BasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Phi3BasicConfig::deserialize(config, false)?;

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
    sliding_window: Option<usize>,
    rope_theta: f64,
    rms_norm_eps: f64,
    hidden_act: Activation,
    quantization_config: Option<QuantizedConfig>,
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::qwen2::Model::new(
            &Qwen2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _use_flash_attn: bool,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Qwen2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for Qwen2Loader {
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

impl DeviceMappedModelLoader for Qwen2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Qwen2BasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Qwen2BasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Qwen2BasicConfig::deserialize(config, false)?;
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
        let cfg = Qwen2BasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Qwen2BasicConfig::deserialize(config, false)?;

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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::gemma2::Model::new(
            &Gemma2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraGemma2::new(
            &Gemma2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        // Already will warn about it
        Ok(Box::new(Gemma2BasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for Gemma2Loader {
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

impl DeviceMappedModelLoader for Gemma2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Gemma2BasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Gemma2BasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Gemma2BasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
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
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Gemma2BasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Gemma2BasicConfig::deserialize(config, false)?;

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
    hidden_act: Activation,
    max_position_embeddings: usize,
    norm_epsilon: f64,
    rope_theta: f64,
    use_bias: bool,
    sliding_window: Option<usize>,
    quantization_config: Option<QuantizedConfig>,
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::starcoder2::Model::new(
            &Starcoder2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraStarcoder2::new(
            &Starcoder2BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, _use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(serde_json::from_str::<Starcoder2BasicConfig>(
            config,
        )?))
    }
}

impl IsqModelLoader for Starcoder2Loader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.dense\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.c_proj\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Starcoder2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Starcoder2BasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Starcoder2BasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size + cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Starcoder2BasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size + cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size + cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_attention_heads;
            let size_kv = (cfg.hidden_size / cfg.num_attention_heads) * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + bias_if!(cfg.use_bias, size_q);
            let k_proj = size_in * size_kv / weight_pack_factor + bias_if!(cfg.use_bias, size_kv);
            let v_proj = size_in * size_kv / weight_pack_factor + bias_if!(cfg.use_bias, size_kv);
            let o_proj = size_q * size_in / weight_pack_factor + bias_if!(cfg.use_bias, size_in);

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let fc1 = h_size * i_size / weight_pack_factor + bias_if!(cfg.use_bias, i_size);
            let fc2 = h_size * i_size / weight_pack_factor + bias_if!(cfg.use_bias, h_size);

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + fc1
                + fc2
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Starcoder2BasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Starcoder2BasicConfig::deserialize(config, false)?;

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
}

// ======================== Phi3 loader

#[derive(Deserialize)]
struct Phi3_5MoEBasicConfig {
    vocab_size: usize,
    hidden_act: Activation,
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
    #[serde(default = "word_emb_default")]
    tie_word_embeddings: bool,
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
            tie_word_embeddings: basic_config.tie_word_embeddings,
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
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(models::phi3_5_moe::Model::new(
            &Phi3_5MoEBasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &Phi3BasicConfig::deserialize(config, use_flash_attn)?,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx(config)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        Ok(Box::new(Phi3_5MoEBasicConfig::deserialize(
            config,
            use_flash_attn,
        )?))
    }
}

impl IsqModelLoader for Phi3_5MoELoader {
    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Phi3_5MoELoader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg = Phi3_5MoEBasicConfig::deserialize(config, false)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg = Phi3_5MoEBasicConfig::deserialize(config, false)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg = Phi3_5MoEBasicConfig::deserialize(config, false)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let moe_block = {
                let gate = cfg.hidden_size * cfg.num_local_experts;
                // Assume quantizing weight pack factor
                let w1 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w2 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                let w3 = cfg.hidden_size * cfg.intermediate_size / weight_pack_factor;
                gate + cfg.num_local_experts * w1
                    + cfg.num_local_experts * w2
                    + cfg.num_local_experts * w3
            };

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + moe_block
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = Phi3_5MoEBasicConfig::deserialize(config, false)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = Phi3_5MoEBasicConfig::deserialize(config, false)?;

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
}

/// [`NormalLoader`] for a DeepSeekV2 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV2Loader;

impl NormalModelLoader for DeepSeekV2Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let mut cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        cfg.use_flash_attn = use_flash_attn;
        Ok(Box::new(models::deepseek2::DeepSeekV2::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _use_flash_attn: bool,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
    }
}

impl IsqModelLoader for DeepSeekV2Loader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        if cfg.q_lora_rank.is_some() {
            data.extend(vec![
                Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            ]);
        } else {
            data.push(Regex::new(
                r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?);
        }
        for layer_idx in 0..cfg.num_hidden_layers {
            if cfg.n_routed_experts.is_some()
                && layer_idx >= cfg.first_k_dense_replace
                && layer_idx % cfg.moe_layer_freq == 0
            {
                for i in 0..cfg.n_routed_experts.unwrap() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if cfg.n_routed_experts.is_some()
                && layer_idx >= cfg.first_k_dense_replace
                && layer_idx % cfg.moe_layer_freq == 0
            {
                for i in 0..cfg.n_routed_experts.unwrap() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
}

impl DeviceMappedModelLoader for DeepSeekV2Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * cfg.q_head_dim()) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.hidden_size,
            };
            let kv_a_proj_with_mqa = cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.kv_lora_rank + cfg.qk_rope_head_dim);
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.hidden_size);

            let moe_block = {
                let mut sum = 0;
                if cfg.n_routed_experts.is_some()
                    && layer_idx >= cfg.first_k_dense_replace
                    && layer_idx % cfg.moe_layer_freq == 0
                {
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let shared_experts = if let Some(n_shared_experts) = cfg.n_shared_experts {
                        let gate_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let up_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let down_proj = (cfg.intermediate_size * n_shared_experts) * h_size
                            / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts.unwrap() * cfg.hidden_size;
                    sum += gate_proj + up_proj + down_proj + shared_experts + gate_weight;
                } else {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
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
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a DeepSeekV3 model.
///
/// [`NormalLoader`]: https://ericlbuehler.github.io/mistral.rs/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV3Loader;

impl NormalModelLoader for DeepSeekV3Loader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let mut cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        cfg.use_flash_attn = use_flash_attn;
        Ok(Box::new(models::deepseek3::DeepSeekV3::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _use_flash_attn: bool,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        todo!()
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str, use_flash_attn: bool) -> Result<Box<dyn Debug>> {
        let mut config: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        config.use_flash_attn = use_flash_attn;
        Ok(Box::new(config))
    }
}

impl IsqModelLoader for DeepSeekV3Loader {
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
        ];
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        if cfg.q_lora_rank.is_some() {
            data.extend(vec![
                Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
                Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            ]);
        } else {
            data.push(Regex::new(
                r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$",
            )?);
        }
        for layer_idx in 0..cfg.num_hidden_layers {
            if cfg.n_routed_experts.is_some()
                && layer_idx >= cfg.first_k_dense_replace
                && layer_idx % cfg.moe_layer_freq == 0
            {
                for i in 0..cfg.n_routed_experts.unwrap() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }

    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![Regex::new(r"lm_head\.(weight|bias)$")?];
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if cfg.n_routed_experts.is_some()
                && layer_idx >= cfg.first_k_dense_replace
                && layer_idx % cfg.moe_layer_freq == 0
            {
                for i in 0..cfg.n_routed_experts.unwrap() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.experts\.{i}\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
                if cfg.n_shared_experts.is_some() {
                    data.extend(vec![
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.gate_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.up_proj\.(weight|bias)$"
                        ))?,
                        Regex::new(&format!(
                            r"layers\.{layer_idx}\.mlp\.shared_experts\.down_proj\.(weight|bias)$"
                        ))?,
                    ]);
                }
            } else {
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(r"layers.{layer_idx}.mlp\.up_proj\.(weight|bias)$"))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
}

impl DeviceMappedModelLoader for DeepSeekV3Loader {
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
        prompt_chunksize: usize,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len: _,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        Ok(max_batch_size * cfg.num_attention_heads * prompt_chunksize * prompt_chunksize)
    }
    fn non_mapped_max_act_size_elems(
        &self,
        _config: &str,
        _params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        Ok(0)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size
            } else {
                0
            };
            let norm = cfg.hidden_size;
            embed_tokens + lm_head + norm
        };
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let q_proj = match cfg.q_lora_rank {
                Some(lora_rank) => {
                    let a = cfg.hidden_size * lora_rank;
                    let norm = lora_rank;
                    let b = (cfg.num_attention_heads * cfg.q_head_dim()) * lora_rank;
                    a + norm + b
                }
                None => (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.hidden_size,
            };
            let kv_a_proj_with_mqa = cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim)
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.kv_lora_rank + cfg.qk_rope_head_dim);
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj = cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size
                / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.hidden_size);

            let moe_block = {
                let mut sum = 0;
                if cfg.n_routed_experts.is_some()
                    && layer_idx >= cfg.first_k_dense_replace
                    && layer_idx % cfg.moe_layer_freq == 0
                {
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts.unwrap();
                    let shared_experts = if let Some(n_shared_experts) = cfg.n_shared_experts {
                        let gate_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let up_proj = h_size * (cfg.intermediate_size * n_shared_experts)
                            / weight_pack_factor;
                        let down_proj = (cfg.intermediate_size * n_shared_experts) * h_size
                            / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts.unwrap() * cfg.hidden_size;
                    sum += gate_proj + up_proj + down_proj + shared_experts + gate_weight;
                } else {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    sum += gate_proj + up_proj + down_proj;
                }
                sum
            };

            per_layer_elems.push(
                input_layernorm
                    + post_attention_layernorm
                    + q_proj
                    + kv_a_layernorm
                    + kv_a_proj_with_mqa
                    + kv_b_proj
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
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
        };

        Ok(Box::new(cfg))
    }
}
