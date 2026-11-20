use std::{
    borrow::Cow,
    collections::HashMap,
    fmt::{Debug, Display},
    str::FromStr,
    sync::Arc,
};

use crate::{attention::ATTENTION_CHUNK_SIZE, matformer::MatformerSliceConfig};

use crate::speculative::SpeculativeTargetMixin;
use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    lora::{LoraConfig, Ordering},
    paged_attention::{
        AttentionImplementation, ModelConfigLike, ModelConfigMetadata,
        PrefixPrefillAttentionFeatures,
    },
    pipeline::{
        isq::IsqModelLoader, text_models_inputs_processor::FlashParams, EitherCache, IsqModel,
        ModelForwardContext,
    },
    utils::varbuilder_utils::DeviceForLoadTensor,
    xlora_models::NonGranularState,
};
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::log::once_log_debug;

use indicatif::MultiProgress;
use mistralrs_quant::ShardedVarBuilder;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::Deserialize;

use crate::{
    models,
    xlora_models::{self, XLoraConfig},
};

use super::{ActivationModelConfig, AutoDeviceMapParams, DeviceMappedModelLoader, Lfm3Loader};
use crate::gguf::normal_registry::RopePairing;

const PHI35_MOE_EXPERTS_PER_TOKEN: usize = 2;

fn activation_width(parts: &[(usize, usize)]) -> Result<usize> {
    parts.iter().try_fold(0usize, |total, &(count, width)| {
        count
            .checked_mul(width)
            .and_then(|part| total.checked_add(part))
            .context("Model activation width overflow")
    })
}

fn uniform_attention_features(
    num_layers: usize,
    sliding_layers: usize,
    mut features: PrefixPrefillAttentionFeatures,
) -> Option<PrefixPrefillAttentionFeatures> {
    if sliding_layers > 0 && sliding_layers < num_layers {
        return None;
    }
    features.has_sliding_window = sliding_layers > 0;
    Some(features)
}

fn per_layer_activation_value(
    value: &models::hunyuan_v1_moe::PerLayerValue,
    layer: usize,
) -> Result<usize> {
    match value {
        models::hunyuan_v1_moe::PerLayerValue::Scalar(value) => Ok(*value),
        models::hunyuan_v1_moe::PerLayerValue::Array(values) => values
            .get(layer)
            .copied()
            .context("Missing per-layer MoE activation configuration"),
    }
}

pub trait NormalModel: IsqModel + AnyMoeBaseModelMixin + SpeculativeTargetMixin {
    fn forward(
        &self,
        input_ids: &Tensor,
        ctx: &mut ModelForwardContext<'_>,
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
    fn max_seq_len(&self) -> usize;
    fn config(&self) -> &ModelConfigMetadata;
    /// True only when the full forward handles packed prompts and never treats physical rows as logical requests.
    fn supports_packed_prefill(&self) -> bool {
        false
    }
    #[cfg(feature = "cuda")]
    fn supports_cuda_decode_graphs(&self) -> bool {
        false
    }
    fn model_config(&self) -> Arc<dyn ModelConfigLike + Send + Sync> {
        Arc::new(self.config().clone())
    }
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
    // Optional Matryoshka Transformer slicing configuration
    pub matformer_slicing_config: Option<MatformerSliceConfig>,
    pub(crate) rope_pairing: Option<RopePairing>,
}

pub trait NormalModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn new_lora_registry(&self, config: &str) -> Result<mistralrs_quant::LoraLayerRegistry> {
        serde_json::from_str::<serde_json::Value>(config)?;
        Ok(mistralrs_quant::LoraLayerRegistry::new())
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    #[allow(clippy::too_many_arguments)]
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>>;
    /// Overridden by families that size RoPE from `max_model_len`; otherwise the engine enforces it alone.
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        _max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        Ok(Cow::Borrowed(config))
    }
    fn is_gptx(&self, config: &str) -> Result<bool>;
    fn is_gptx_for(
        &self,
        config: &str,
        normal_loading_metadata: &NormalLoadingMetadata,
    ) -> Result<bool> {
        match normal_loading_metadata.rope_pairing {
            Some(RopePairing::Adjacent) => Ok(false),
            Some(RopePairing::HalfSplit) => Ok(true),
            None => match super::qk_rope_layout_from_config(config)? {
                Some(RopePairing::Adjacent) => Ok(false),
                Some(RopePairing::HalfSplit) => Ok(true),
                None => self.is_gptx(config),
            },
        }
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
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

// One row per architecture: variant, CLI/serde name, HF -CausalLM class, HF model_type, loader.
macro_rules! normal_architectures {
    ($($variant:ident: $name:literal, $class:literal, $model_type:literal, $loader:expr;)*) => {
        #[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
        #[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq, strum::EnumIter)]
        /// The architecture to load the normal model as.
        pub enum NormalLoaderType {
            $(#[serde(rename = $name)] $variant,)*
        }

        impl NormalLoaderType {
            const NAMES: &[&str] = &[$($name),*];

            fn name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $name,)*
                }
            }

            pub(crate) fn causal_lm_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $class,)*
                }
            }

            pub(crate) fn model_type_name(&self) -> &'static str {
                match self {
                    $(Self::$variant => $model_type,)*
                }
            }

            pub fn from_causal_lm_name(name: &str) -> Result<Self> {
                match name {
                    $($class => Ok(Self::$variant),)*
                    other => anyhow::bail!(
                        "Unsupported Hugging Face Transformers -CausalLM model class `{other}`. Please raise an issue."
                    ),
                }
            }

            pub(crate) fn loader(&self) -> Box<dyn NormalModelLoader> {
                match self {
                    $(Self::$variant => Box::new($loader),)*
                }
            }
        }

        impl FromStr for NormalLoaderType {
            type Err = String;
            fn from_str(s: &str) -> Result<Self, Self::Err> {
                match s {
                    $($name => Ok(Self::$variant),)*
                    a => Err(format!(
                        "Unknown architecture `{a}`. Possible architectures: `{}`.",
                        Self::NAMES.join("`, `")
                    )),
                }
            }
        }
    };
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
normal_architectures! {
    Mistral: "mistral", "MistralForCausalLM", "mistral", MistralLoader;
    Gemma: "gemma", "GemmaForCausalLM", "gemma", GemmaLoader;
    Mixtral: "mixtral", "MixtralForCausalLM", "mixtral", MixtralLoader;
    Llama: "llama", "LlamaForCausalLM", "llama", LlamaLoader;
    Phi2: "phi2", "PhiForCausalLM", "phi", Phi2Loader;
    Phi3: "phi3", "Phi3ForCausalLM", "phi3", Phi3Loader;
    Qwen2: "qwen2", "Qwen2ForCausalLM", "qwen2", Qwen2Loader;
    Gemma2: "gemma2", "Gemma2ForCausalLM", "gemma2", Gemma2Loader;
    Starcoder2: "starcoder2", "Starcoder2ForCausalLM", "starcoder2", Starcoder2Loader;
    Phi3_5MoE: "phi3.5moe", "PhiMoEForCausalLM", "phimoe", Phi3_5MoELoader;
    DeepSeekV2: "deepseekv2", "DeepseekV2ForCausalLM", "deepseek_v2", DeepSeekV2Loader;
    DeepSeekV3: "deepseekv3", "DeepseekV3ForCausalLM", "deepseek_v3", DeepSeekV3Loader;
    Qwen3: "qwen3", "Qwen3ForCausalLM", "qwen3", Qwen3Loader;
    GLM4: "glm4", "Glm4ForCausalLM", "glm4", GLM4Loader;
    GLM4MoeLite: "glm4moelite", "Glm4MoeLiteForCausalLM", "glm4_moe_lite", GLM4MoeLiteLoader;
    GLM4Moe: "glm4moe", "Glm4MoeForCausalLM", "glm4_moe", GLM4MoeLoader;
    Qwen3Moe: "qwen3moe", "Qwen3MoeForCausalLM", "qwen3_moe", Qwen3MoELoader;
    Flint: "flint", "FlintForCausalLM", "flint", FlintLoader;
    SmolLm3: "smollm3", "SmolLM3ForCausalLM", "smollm3", SmolLm3Loader;
    GraniteMoeHybrid: "granitemoehybrid", "GraniteMoeHybridForCausalLM", "granitemoehybrid", GraniteMoeHybridLoader;
    GptOss: "gpt_oss", "GptOssForCausalLM", "gpt_oss", GptOssLoader;
    HunYuanDenseV1: "hunyuanv1dense", "HunYuanDenseV1ForCausalLM", "hunyuan_v1_dense", HunYuanDenseV1Loader;
    HunYuanMoEV1: "hunyuanv1moe", "HunYuanMoEV1ForCausalLM", "hunyuan_v1_moe", HunYuanMoEV1Loader;
    Qwen3Next: "qwen3next", "Qwen3NextForCausalLM", "qwen3_next", Qwen3NextLoader;
    Qwen3_5: "qwen3_5", "Qwen3_5ForCausalLM", "qwen3_5_text", Qwen3_5TextLoader;
    Lfm2: "lfm2", "Lfm2ForCausalLM", "lfm2", Lfm2Loader;
    Lfm3: "lfm3", "Lfm3ForCausalLM", "lfm3", Lfm3Loader;
    Lfm2Moe: "lfm2_moe", "Lfm2MoeForCausalLM", "lfm2_moe", Lfm2Loader;
}

impl Display for NormalLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
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

/// Load a model based on the Hugging Face Transformers -CausalLM model class
pub struct AutoNormalLoader;

#[derive(Deserialize)]
struct AutoNormalLoaderConfig {
    architectures: Vec<String>,
}

impl AutoNormalLoader {
    fn get_loader(config: &str) -> Result<Box<dyn NormalModelLoader>> {
        let auto_cfg: AutoNormalLoaderConfig = serde_json::from_str(config)?;
        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected to have one name for `architectures` config field.")
        }

        let name = &auto_cfg.architectures[0];

        let tp = NormalLoaderType::from_causal_lm_name(name)?;

        once_log_debug(format!("Automatic loader type determined to be `{tp}`"));

        Ok(tp.loader())
    }
}

impl NormalModelLoader for AutoNormalLoader {
    fn new_lora_registry(&self, config: &str) -> Result<mistralrs_quant::LoraLayerRegistry> {
        Self::get_loader(config)?.new_lora_registry(config)
    }

    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        Self::get_loader(config)?.runtime_config(config, max_model_len)
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        Self::get_loader(config)?.load_xlora(
            config,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            normal_loading_metadata,
            preload_adapters,
        )
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }
    fn supports_paged_attention(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.supports_paged_attention(config)
    }
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_gptx(config)
    }
}

impl IsqModelLoader for AutoNormalLoader {
    fn promoted_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.promoted_isq_predicates(config)
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.immediate_isq_predicates_moqe(config)
    }
    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        Self::get_loader(config)?.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for AutoNormalLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        Self::get_loader(config)?.activation_model_config(config)
    }

    fn non_mapped_size_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        Self::get_loader(config)?.non_mapped_size_in_bytes(
            config,
            dtype,
            weight_pack_factor,
            quantization,
            _matformer_config,
        )
    }
    fn num_layers(&self, config: &str) -> Result<usize> {
        Self::get_loader(config)?.num_layers(config)
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
    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &super::AutoDeviceMapParams,
    ) -> Result<usize> {
        Self::get_loader(config)?.mapped_max_act_size_elems(config, params)
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

// ======================== Mistral loader

pub struct MistralLoader;

impl NormalModelLoader for MistralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(models::mistral::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(xlora_models::XLoraMistral::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for MistralLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for MistralLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::mistral::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::mistral::Config = serde_json::from_str(config)?;

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
}

// ======================== Gemma loader

/// [`NormalLoader`] for a Gemma model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GemmaLoader;

impl NormalModelLoader for GemmaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gemma::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraGemma::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GemmaLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for GemmaLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::gemma::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens_pack_factor = super::tied_promoted_tensor_pack_factor(
                _quantization,
                "model.embed_tokens.weight",
                "lm_head.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = 0;
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gemma::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Llama loader

/// [`NormalLoader`] for a Llama model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct LlamaLoader;

impl NormalModelLoader for LlamaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::llama::Llama::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraLlama::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for LlamaLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for LlamaLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::llama::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::llama::Config = serde_json::from_str(config)?;

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

// ======================== Mixtral loader

pub struct MixtralLoader;

impl NormalModelLoader for MixtralLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::mixtral::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraMixtral::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for MixtralLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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
            Regex::new(
                r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for MixtralLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::mixtral::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: activation_width(&[(
                cfg.num_experts_per_tok,
                cfg.intermediate_size,
            )])?,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::mixtral::Config = serde_json::from_str(config)?;

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
}

// ======================== Phi2 loader

/// [`NormalLoader`] for a Phi 2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi2Loader;

impl NormalModelLoader for Phi2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Phi2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::phi2::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi2::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Phi3 loader

/// [`NormalLoader`] for a Phi 3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi3Loader;

impl NormalModelLoader for Phi3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi3::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for Phi3Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::phi3::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let op_size =
                cfg.num_attention_heads * head_dim + 2 * cfg.num_key_value_heads * head_dim;
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
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

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
}

// ======================== Qwen2 loader

/// [`NormalLoader`] for a Qwen 2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen2Loader;

impl NormalModelLoader for Qwen2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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
            // MLP MoE
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::qwen2::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                if cfg.use_sliding_window && cfg.sliding_window.is_some() {
                    cfg.layer_types.as_ref().map_or_else(
                        || cfg.num_hidden_layers.saturating_sub(cfg.max_window_layers),
                        |layers| {
                            layers
                                .iter()
                                .take(cfg.num_hidden_layers)
                                .filter(|layer| {
                                    matches!(
                                        layer,
                                        models::qwen2::Qwen2AttentionType::SlidingAttention
                                    )
                                })
                                .count()
                        },
                    )
                } else {
                    0
                },
                PrefixPrefillAttentionFeatures::default(),
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::qwen2::Config = serde_json::from_str(config)?;

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
}

// ======================== Gemma2 loader

/// [`NormalLoader`] for a Gemma2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Gemma2Loader;

impl NormalModelLoader for Gemma2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gemma2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraGemma2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Gemma2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for Gemma2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::gemma2::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                cfg.num_hidden_layers.div_ceil(2),
                PrefixPrefillAttentionFeatures {
                    has_softcap: cfg.attn_logit_softcapping.is_some(),
                    ..Default::default()
                },
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens_pack_factor = super::tied_promoted_tensor_pack_factor(
                _quantization,
                "model.embed_tokens.weight",
                "lm_head.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = 0;
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gemma2::Config = serde_json::from_str(config)?;

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
}

// ======================== Starcoder2 loader

/// [`NormalLoader`] for a Starcoder2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Starcoder2Loader;

impl NormalModelLoader for Starcoder2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::starcoder2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraStarcoder2::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Starcoder2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // MLP
            Regex::new(r"layers\.(\d+)\.mlp\.c_fc\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.c_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Starcoder2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::starcoder2::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens_pack_factor = super::tied_promoted_tensor_pack_factor(
                _quantization,
                "model.embed_tokens.weight",
                "lm_head.weight",
                dtype,
                weight_pack_factor,
            )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = 0;
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::starcoder2::Config = serde_json::from_str(config)?;

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
}

// ======================== Phi3 loader

/// [`NormalLoader`] for a Phi 3.5 MoE model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Phi3_5MoELoader;

impl NormalModelLoader for Phi3_5MoELoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::phi3_5_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        lora_config: &[((String, String), LoraConfig)],
        xlora_config: Option<XLoraConfig>,
        xlora_ordering: Ordering,
        normal_loading_metadata: NormalLoadingMetadata,
        preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::phi3::Config = serde_json::from_str(config)?;

        Ok(Box::new(xlora_models::XLoraPhi3::new(
            &cfg,
            vb,
            lora_config,
            xlora_config,
            xlora_ordering,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            preload_adapters,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Phi3_5MoELoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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
            Regex::new(
                r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            // MLP
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.w3\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.block_sparse_moe\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Phi3_5MoELoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::phi3_5_moe::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: activation_width(&[(
                PHI35_MOE_EXPERTS_PER_TOKEN,
                cfg.intermediate_size,
            )])?,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::phi3_5_moe::Config = serde_json::from_str(config)?;

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
}

/// [`NormalLoader`] for a DeepSeekV2 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV2Loader;

impl NormalModelLoader for DeepSeekV2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        Ok(Box::new(models::deepseek2::DeepSeekV2::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.(kv_b|k_b|v_b)_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
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
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
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
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for DeepSeekV2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let routed_width = if cfg.n_routed_experts.is_some() {
            activation_width(&[
                (
                    cfg.num_experts_per_tok
                        .context("Missing MoE experts per token")?,
                    cfg.moe_intermediate_size,
                ),
                (cfg.n_shared_experts.unwrap_or(0), cfg.moe_intermediate_size),
            ])?
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(routed_width),
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token: activation_width(&[
                (1, cfg.q_lora_rank.unwrap_or(0)),
                (1, cfg.kv_lora_rank),
                (1, cfg.qk_rope_head_dim),
            ])?,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek2::DeepSeekV2Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
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
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h_size = cfg.hidden_size;
                    let gate_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let up_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let down_proj =
                        cfg.moe_intermediate_size * h_size / weight_pack_factor * n_routed_experts;
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
                    let gate_weight = n_routed_experts * cfg.hidden_size;
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
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a DeepSeekV3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct DeepSeekV3Loader;

impl NormalModelLoader for DeepSeekV3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(models::deepseek3::DeepSeekV3::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for DeepSeekV3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.(kv_b|k_b|v_b)_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
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
            if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
            }) {
                for i in 0..n_routed_experts {
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
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for DeepSeekV3Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let routed_width = if cfg.n_routed_experts.is_some() {
            activation_width(&[
                (
                    cfg.num_experts_per_tok
                        .context("Missing MoE experts per token")?,
                    cfg.moe_intermediate_size,
                ),
                (cfg.n_shared_experts.unwrap_or(0), cfg.moe_intermediate_size),
            ])?
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(routed_width),
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token: activation_width(&[
                (1, cfg.q_lora_rank.unwrap_or(0)),
                (1, cfg.kv_lora_rank),
                (1, cfg.qk_rope_head_dim),
            ])?,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::deepseek3::DeepSeekV3Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
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
                if let Some(n_routed_experts) = cfg.n_routed_experts.filter(|_| {
                    layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0
                }) {
                    let h_size = cfg.hidden_size;
                    let gate_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let up_proj =
                        h_size * cfg.moe_intermediate_size / weight_pack_factor * n_routed_experts;
                    let down_proj =
                        cfg.moe_intermediate_size * h_size / weight_pack_factor * n_routed_experts;
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
                    let gate_weight = n_routed_experts * cfg.hidden_size;
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
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a Qwen 3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3Loader;

impl NormalModelLoader for Qwen3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for Qwen3Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                if cfg.use_sliding_window && cfg.sliding_window.is_some() {
                    cfg.num_hidden_layers.saturating_sub(cfg.max_window_layers)
                } else {
                    0
                },
                PrefixPrefillAttentionFeatures::default(),
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::qwen3::Config = serde_json::from_str(config)?;

        Ok(max_batch_size
            * cfg.num_attention_heads
            * max_seq_len.min(&ATTENTION_CHUNK_SIZE)
            * max_seq_len)
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor + size_q;
            let k_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let v_proj = size_in * size_kv / weight_pack_factor + size_kv;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            let q_norm = cfg.head_dim();
            let k_norm = cfg.head_dim();

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
                + q_norm
                + k_norm
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::qwen3::Config = serde_json::from_str(config)?;

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
}

/// [`NormalLoader`] for a HunYuanDenseV1 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct HunYuanDenseV1Loader;

impl NormalModelLoader for HunYuanDenseV1Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::hunyuan_v1_dense::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for HunYuanDenseV1Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for HunYuanDenseV1Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = head_dim * cfg.num_attention_heads;
            let size_kv = head_dim * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let i_size = cfg.intermediate_size;
            let gate_proj = h_size * i_size / weight_pack_factor;
            let up_proj = h_size * i_size / weight_pack_factor;
            let down_proj = i_size * h_size / weight_pack_factor;

            let q_norm = head_dim;
            let k_norm = head_dim;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + gate_proj
                + up_proj
                + down_proj
                + q_norm
                + k_norm
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::hunyuan_v1_dense::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a HunYuanMoEV1 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct HunYuanMoEV1Loader;

impl NormalModelLoader for HunYuanMoEV1Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::hunyuan_v1_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for HunYuanMoEV1Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Dense MLP
            Regex::new(r"layers\.(\d+)\.mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.down_proj\.(weight|bias)$")?,
            // MoE experts
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.shared_mlp\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for HunYuanMoEV1Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        let mut intermediate_size = cfg.intermediate_size;
        if cfg.uses_moe() {
            for layer in 0..cfg.num_hidden_layers {
                let routed = per_layer_activation_value(&cfg.moe_topk, layer)?;
                let shared = if cfg.use_mixed_mlp_moe {
                    per_layer_activation_value(&cfg.num_shared_expert, layer)?
                } else {
                    0
                };
                intermediate_size = intermediate_size.max(activation_width(&[
                    (routed, cfg.intermediate_size),
                    (shared, cfg.intermediate_size),
                ])?);
            }
        }
        Ok(ActivationModelConfig {
            intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();

        let mut layer_sizes = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = head_dim * cfg.num_attention_heads;
            let size_kv = head_dim * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let h_size = cfg.hidden_size;
            let expert_size = {
                let expert_gate = h_size * cfg.intermediate_size / weight_pack_factor;
                let expert_up = h_size * cfg.intermediate_size / weight_pack_factor;
                let expert_down = cfg.intermediate_size * h_size / weight_pack_factor;
                expert_gate + expert_up + expert_down
            };
            let (router_size, mlp_size) = if cfg.uses_moe() {
                let shared_expert_size = if cfg.use_mixed_mlp_moe {
                    expert_size * cfg.num_shared_expert.get(layer_idx)
                } else {
                    0
                };
                (
                    h_size * cfg.num_experts,
                    shared_expert_size + expert_size * cfg.num_experts,
                )
            } else {
                (0, expert_size)
            };
            let qk_norm = if cfg.use_qk_norm { head_dim * 2 } else { 0 };

            let non_router_elems = input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + mlp_size
                + qk_norm;

            layer_sizes.push(
                non_router_elems * dtype.size_in_bytes() + router_size * DType::F32.size_in_bytes(),
            );
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::hunyuan_v1_moe::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a GLM 4 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4Loader;

impl NormalModelLoader for GLM4Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::glm4::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
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

impl DeviceMappedModelLoader for GLM4Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures {
                has_sliding_window: cfg.sliding_window.is_some(),
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::glm4::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size * 3; //+post_self_attn_layernorm and post_mlp_layernorm

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
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
        let cfg: models::glm4::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::glm4::Config = serde_json::from_str(config)?;

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
}

/// [`NormalLoader`] for a GLM 4 MoE Lite model (GLM-4.7-Flash).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4MoeLiteLoader;

impl NormalModelLoader for GLM4MoeLiteLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(Box::new(models::glm4_moe_lite::Glm4MoeLite::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4MoeLiteLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention (MLA)
            Regex::new(r"layers\.(\d+)\.self_attn\.kv_a_proj_with_mqa\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.(kv_b|k_b|v_b)_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            // Q LoRA projections
            Regex::new(r"layers\.(\d+)\.self_attn\.q_a_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_b_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ];
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0 {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
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
                if cfg.n_shared_experts > 0 {
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
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GLM4MoeLiteLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(activation_width(&[
                (cfg.num_experts_per_tok, cfg.moe_intermediate_size),
                (cfg.n_shared_experts, cfg.moe_intermediate_size),
            ])?),
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token: activation_width(&[
                (1, cfg.q_lora_rank),
                (1, cfg.kv_lora_rank),
                (1, cfg.qk_rope_head_dim),
            ])?,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            // Q LoRA projection
            let q_proj = {
                let a = cfg.hidden_size * cfg.q_lora_rank / weight_pack_factor;
                let norm = cfg.q_lora_rank;
                let b = (cfg.num_attention_heads * cfg.q_head_dim()) * cfg.q_lora_rank
                    / weight_pack_factor;
                a + norm + b
            };
            let kv_a_proj_with_mqa =
                cfg.hidden_size * (cfg.kv_lora_rank + cfg.qk_rope_head_dim) / weight_pack_factor;
            let kv_a_layernorm = cfg.kv_lora_rank;
            let kv_b_proj = cfg.kv_lora_rank
                * cfg.num_attention_heads
                * (cfg.q_head_dim() - cfg.qk_rope_head_dim + cfg.v_head_dim)
                / weight_pack_factor;
            let o_proj =
                cfg.num_attention_heads * cfg.v_head_dim * cfg.hidden_size / weight_pack_factor;

            let moe_block = {
                let mut sum = 0;
                if layer_idx >= cfg.first_k_dense_replace && layer_idx % cfg.moe_layer_freq == 0 {
                    // MoE layer
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let shared_experts = if cfg.n_shared_experts > 0 {
                        let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts * cfg.hidden_size;
                    let e_score_correction_bias = cfg.n_routed_experts;
                    sum += gate_proj
                        + up_proj
                        + down_proj
                        + shared_experts
                        + gate_weight
                        + e_score_correction_bias;
                } else {
                    // Dense MLP layer
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
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::glm4_moe_lite::Glm4MoeLiteConfig = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_attention_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.qk_rope_head_dim + cfg.qk_nope_head_dim,
            v_head_dim: cfg.v_head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a GLM 4 MoE model (GLM-4.5).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GLM4MoeLoader;

impl NormalModelLoader for GLM4MoeLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(models::glm4_moe::Glm4Moe::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GLM4MoeLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut data = vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention (standard GQA)
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ];
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        for layer_idx in 0..cfg.num_hidden_layers {
            if layer_idx >= cfg.first_k_dense_replace {
                // MoE layer
                for i in 0..cfg.n_routed_experts {
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
                if cfg.n_shared_experts > 0 {
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
                // Dense MLP layer
                data.extend(vec![
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.gate_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.up_proj\.(weight|bias)$"
                    ))?,
                    Regex::new(&format!(
                        r"layers\.{layer_idx}\.mlp\.down_proj\.(weight|bias)$"
                    ))?,
                ]);
            };
        }
        Ok(data)
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GLM4MoeLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(activation_width(&[
                (cfg.num_experts_per_tok, cfg.moe_intermediate_size),
                (cfg.n_shared_experts, cfg.moe_intermediate_size),
            ])?),
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        let mut per_layer_elems = Vec::new();

        let head_dim = cfg.head_dim();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            // Standard GQA attention
            let q_proj = cfg.hidden_size * cfg.num_attention_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_attention_heads * head_dim);
            let k_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let v_proj = cfg.hidden_size * cfg.num_key_value_heads * head_dim / weight_pack_factor
                + bias_if!(cfg.attention_bias, cfg.num_key_value_heads * head_dim);
            let o_proj = cfg.num_attention_heads * head_dim * cfg.hidden_size / weight_pack_factor;

            // QK norm if enabled
            let qk_norm = if cfg.use_qk_norm {
                head_dim * 2 // q_norm + k_norm
            } else {
                0
            };

            let moe_block = {
                let mut sum = 0;
                if layer_idx >= cfg.first_k_dense_replace {
                    // MoE layer
                    let h_size = cfg.hidden_size;
                    let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor
                        * cfg.n_routed_experts;
                    let shared_experts = if cfg.n_shared_experts > 0 {
                        let gate_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let up_proj = h_size * cfg.moe_intermediate_size / weight_pack_factor;
                        let down_proj = cfg.moe_intermediate_size * h_size / weight_pack_factor;
                        gate_proj + up_proj + down_proj
                    } else {
                        0
                    };
                    let gate_weight = cfg.n_routed_experts * cfg.hidden_size;
                    let e_score_correction_bias = cfg.n_routed_experts;
                    sum += gate_proj
                        + up_proj
                        + down_proj
                        + shared_experts
                        + gate_weight
                        + e_score_correction_bias;
                } else {
                    // Dense MLP layer
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
                    + k_proj
                    + v_proj
                    + o_proj
                    + qk_norm
                    + moe_block,
            );
        }

        Ok(per_layer_elems
            .into_iter()
            .map(|x| x * dtype.size_in_bytes())
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::glm4_moe::Glm4MoeConfig = serde_json::from_str(config)?;

        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for a Qwen 3 MoE model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3MoELoader;

impl NormalModelLoader for Qwen3MoELoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3_moe::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3MoELoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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
            // MLP MoE
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.gate_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.up_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
}

impl DeviceMappedModelLoader for Qwen3MoELoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(activation_width(&[(
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
            )])?),
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                if cfg.use_sliding_window && cfg.sliding_window.is_some() {
                    cfg.num_hidden_layers.saturating_sub(cfg.max_window_layers)
                } else {
                    0
                },
                PrefixPrefillAttentionFeatures::default(),
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;
        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

        let mut layer_sizes_in_bytes = Vec::new();
        for layer_idx in 0..cfg.num_hidden_layers {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads;
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;

            let mlp_size = if !cfg.mlp_only_layers.contains(&layer_idx)
                && (cfg.num_experts > 0 && (layer_idx + 1) % cfg.decoder_sparse_step == 0)
            {
                let gate_size = cfg.hidden_size * cfg.num_experts;
                let expert_size = {
                    let h_size = cfg.hidden_size;
                    let i_size = cfg.moe_intermediate_size;
                    let gate_proj = h_size * i_size / weight_pack_factor;
                    let up_proj = h_size * i_size / weight_pack_factor;
                    let down_proj = i_size * h_size / weight_pack_factor;
                    gate_proj + up_proj + down_proj
                };
                expert_size * cfg.num_experts + gate_size
            } else {
                let h_size = cfg.hidden_size;
                let i_size = cfg.intermediate_size;
                let gate_proj = h_size * i_size / weight_pack_factor;
                let up_proj = h_size * i_size / weight_pack_factor;
                let down_proj = i_size * h_size / weight_pack_factor;
                gate_proj + up_proj + down_proj
            };

            let q_norm = cfg.head_dim();
            let k_norm = cfg.head_dim();

            let size_elems = input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + mlp_size
                + q_norm
                + k_norm;

            let size_in_bytes = size_elems * dtype.size_in_bytes();
            layer_sizes_in_bytes.push(size_in_bytes);
        }

        Ok(layer_sizes_in_bytes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::qwen3_moe::Config = serde_json::from_str(config)?;

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
}

pub struct FlintLoader;

impl NormalModelLoader for FlintLoader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        let Some(max_model_len) = max_model_len else {
            return Ok(Cow::Borrowed(config));
        };
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        anyhow::ensure!(
            max_model_len > 0 && max_model_len <= cfg.max_position_embeddings,
            "Flint max_model_len must be between 1 and {}",
            cfg.max_position_embeddings
        );
        let mut config: serde_json::Value = serde_json::from_str(config)?;
        config["max_position_embeddings"] = max_model_len.into();
        Ok(Cow::Owned(serde_json::to_string(&config)?))
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        Ok(Box::new(models::flint::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }

    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("Flint does not support X-LoRA")
    }

    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for FlintLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, config: &str) -> Result<Vec<Regex>> {
        let mut regexes = self.isq_layer_regexes_moqe(config)?;
        regexes.extend([
            Regex::new(r"^lm_head\.weight$")?,
            Regex::new(r"^model\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$")?,
            Regex::new(r"^model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")?,
            Regex::new(r"^model\.layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ]);
        Ok(regexes)
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![Regex::new(
            r"^model\.layers\.(\d+)\.mlp\.experts\.(?:(\d+)\.)?(gate_proj|up_proj|down_proj)\.weight$",
        )?])
    }
}

impl DeviceMappedModelLoader for FlintLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        let sliding_layers = (0..cfg.num_hidden_layers)
            .filter(|&layer| cfg.sliding_window(layer).is_some())
            .count();
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(activation_width(&[
                (cfg.num_experts_per_tok, cfg.moe_intermediate_size),
                (1, cfg.shared_expert_intermediate_size),
            ])?),
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                sliding_layers,
                PrefixPrefillAttentionFeatures {
                    has_sinks: cfg.has_attention_sinks(),
                    ..Default::default()
                },
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for Flint")
        };
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        let (embedding_pack_factor, lm_head_pack_factor) = super::language_model_pack_factors(
            quantization,
            "model.embed_tokens.weight",
            "lm_head.weight",
            cfg.tie_word_embeddings,
            dtype,
            weight_pack_factor,
        )?;
        let embedding = cfg.hidden_size * cfg.vocab_size / embedding_pack_factor;
        let lm_head = if cfg.tie_word_embeddings {
            0
        } else {
            cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
        };
        Ok((embedding + lm_head + cfg.hidden_size) * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        let hidden = cfg.hidden_size;
        let query = cfg.num_attention_heads * cfg.head_dim();
        let kv = cfg.num_key_value_heads * cfg.head_dim();
        let attention = 2 * hidden * (query + kv) / weight_pack_factor
            + bias_if!(cfg.attention_bias, query + 2 * kv + hidden);
        let norms = 2 * hidden + 2 * cfg.head_dim();
        let sinks = bias_if!(cfg.has_attention_sinks(), cfg.num_attention_heads);
        Ok((0..cfg.num_hidden_layers)
            .map(|layer| {
                let mlp = if cfg.is_moe_layer(layer) {
                    let routed = cfg.num_experts * 3 * hidden * cfg.moe_intermediate_size
                        / weight_pack_factor;
                    let shared =
                        3 * hidden * cfg.shared_expert_intermediate_size / weight_pack_factor;
                    let router = hidden * cfg.num_experts;
                    routed + shared + router
                } else {
                    3 * hidden * cfg.intermediate_size / weight_pack_factor
                };
                (attention + norms + sinks + mlp) * dtype.size_in_bytes()
            })
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: models::flint::Config = serde_json::from_str(config)?;
        Ok(Box::new(models::flint::ModelConfig::new(
            ModelConfigMetadata {
                max_seq_len: cfg.max_position_embeddings,
                num_layers: cfg.num_hidden_layers,
                hidden_size: cfg.hidden_size,
                num_kv_heads: cfg.num_key_value_heads,
                num_attn_heads: cfg.num_attention_heads,
                sliding_window: None,
                k_head_dim: cfg.head_dim(),
                v_head_dim: cfg.head_dim(),
                kv_cache_layout: crate::paged_attention::KvCacheLayout::StandardNoFlashInfer,
            },
            (0..cfg.num_hidden_layers)
                .map(|layer| cfg.sliding_window(layer))
                .collect(),
            cfg.has_attention_sinks(),
        )))
    }
}

// ======================== SmolLm3 loader

/// [`NormalLoader`] for a SmolLm3 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct SmolLm3Loader;

impl NormalModelLoader for SmolLm3Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::smollm3::SmolLm3::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
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
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for SmolLm3Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

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

impl DeviceMappedModelLoader for SmolLm3Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::smollm3::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

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
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::smollm3::Config = serde_json::from_str(config)?;

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

// ======================== GraniteMoeHybrid loader

/// [`NormalLoader`] for a GraniteMoeHybrid model (IBM Granite 4.0).
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GraniteMoeHybridLoader;

impl NormalModelLoader for GraniteMoeHybridLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::granite::GraniteMoeHybrid::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("GraniteMoeHybrid does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for GraniteMoeHybridLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.mamba\.(in_proj|out_proj)\.(weight|bias)$")?,
            // MLP (GraniteMLP uses shared_mlp.input_linear and shared_mlp.output_linear)
            Regex::new(r"layers\.(\d+)\.shared_mlp\.input_linear\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.shared_mlp\.output_linear\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![Regex::new(
            r"layers\.(\d+)\.block_sparse_moe\.(input_linear|output_linear)\.weight$",
        )?])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GraniteMoeHybridLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::granite::Config = serde_json::from_str(config)?;
        let extra_hidden_elements_per_token = if cfg
            .layer_types()
            .iter()
            .any(|layer| matches!(layer, models::granite::GraniteLayerType::Mamba))
        {
            let intermediate = activation_width(&[(cfg.mamba_expand, cfg.hidden_size)])?;
            let state = activation_width(&[(cfg.mamba_n_groups, cfg.mamba_d_state)])?;
            activation_width(&[(2, intermediate), (2, state), (1, cfg.mamba_n_heads())])?
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: activation_width(&[
                (1, cfg.shared_intermediate_size()),
                (
                    if cfg.num_local_experts > 0 {
                        cfg.num_experts_per_tok
                    } else {
                        0
                    },
                    cfg.intermediate_size,
                ),
            ])?,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let attention_elems = {
            let size_in = cfg.hidden_size;
            let size_q = cfg.head_dim() * cfg.num_attention_heads;
            let size_kv = cfg.head_dim() * cfg.num_key_value_heads();
            let q_proj = size_in * size_q / weight_pack_factor;
            let k_proj = size_in * size_kv / weight_pack_factor;
            let v_proj = size_in * size_kv / weight_pack_factor;
            let o_proj = size_q * size_in / weight_pack_factor;
            q_proj + k_proj + v_proj + o_proj
        };

        let mamba_elems = {
            let intermediate_size = cfg.mamba_intermediate_size();
            let conv_dim = cfg.mamba_conv_dim();
            let num_heads = cfg.mamba_n_heads();
            let projection_size = intermediate_size + conv_dim + num_heads;
            let in_proj =
                projection_size * cfg.hidden_size + bias_if!(cfg.mamba_proj_bias, projection_size);
            let conv1d = conv_dim * cfg.mamba_d_conv + bias_if!(cfg.mamba_conv_bias, conv_dim);
            let state = num_heads * 3;
            let norm = intermediate_size;
            let out_proj = cfg.hidden_size * intermediate_size
                + bias_if!(cfg.mamba_proj_bias, cfg.hidden_size);
            in_proj + conv1d + state + norm + out_proj
        };

        let shared_mlp_elems = {
            let shared_intermediate_size = if cfg.num_local_experts == 0 {
                cfg.shared_intermediate_size()
            } else {
                cfg.shared_intermediate_size.unwrap_or(0)
            };
            cfg.hidden_size * shared_intermediate_size * 2 / weight_pack_factor
                + shared_intermediate_size * cfg.hidden_size / weight_pack_factor
        };
        let routed_moe_elems = if cfg.num_local_experts > 0 {
            let router = cfg.num_local_experts * cfg.hidden_size;
            let input_linear = cfg.num_local_experts * cfg.intermediate_size * 2 * cfg.hidden_size;
            let output_linear = cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size;
            router + input_linear + output_linear
        } else {
            0
        };
        let common_elems = cfg.hidden_size * 2 + shared_mlp_elems + routed_moe_elems;

        Ok(cfg
            .layer_types()
            .into_iter()
            .map(|layer_type| {
                let operator_elems = match layer_type {
                    crate::models::granite::GraniteLayerType::Attention => attention_elems,
                    crate::models::granite::GraniteLayerType::Mamba => mamba_elems,
                };
                (common_elems + operator_elems) * dtype.size_in_bytes()
            })
            .collect())
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::granite::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads(),
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim(),
            v_head_dim: cfg.head_dim(),
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== GPT-OSS loader

/// [`NormalLoader`] for a GPT-OSS model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct GptOssLoader;

impl NormalModelLoader for GptOssLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::gpt_oss::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("GPT-OSS does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for GptOssLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            // Attention
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![Regex::new(
            r"layers\.(\d+)\.mlp\.experts\.(gate_up_proj|gate_proj|up_proj|down_proj)\.weight$",
        )?])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for GptOssLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::gpt_oss::Config = serde_json::from_str(config)?;
        Ok(ActivationModelConfig {
            intermediate_size: activation_width(&[(
                cfg.num_experts_per_tok,
                cfg.intermediate_size,
            )])?,
            vocab_size: cfg.vocab_size,
            attention_features: uniform_attention_features(
                cfg.num_hidden_layers,
                if cfg.sliding_window.is_some() {
                    cfg.layer_types
                        .iter()
                        .take(cfg.num_hidden_layers)
                        .filter(|layer| {
                            matches!(layer, models::gpt_oss::LayerType::SlidingAttention)
                        })
                        .count()
                } else {
                    0
                },
                PrefixPrefillAttentionFeatures {
                    has_sinks: true,
                    ..Default::default()
                },
            ),
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let per_layer_elems = {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let size_in = cfg.hidden_size;
            let head_dim = cfg.head_dim();
            let size_q = head_dim * cfg.num_attention_heads;
            let size_kv = head_dim * cfg.num_key_value_heads;
            let q_proj =
                size_in * size_q / weight_pack_factor + bias_if!(cfg.attention_bias, size_q);
            let k_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let v_proj =
                size_in * size_kv / weight_pack_factor + bias_if!(cfg.attention_bias, size_kv);
            let o_proj =
                size_q * size_in / weight_pack_factor + bias_if!(cfg.attention_bias, size_in);

            let expert_weights = if matches!(
                cfg.quantization_config.as_ref(),
                Some(mistralrs_quant::QuantizedConfig::MXFP4 {})
            ) {
                let gate_up = cfg.num_local_experts * cfg.intermediate_size * 2 * cfg.hidden_size;
                let down = cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size;
                gate_up / 2 + down / 2 + gate_up / 32 + down / 32
            } else {
                let projection = cfg.num_local_experts * cfg.hidden_size * cfg.intermediate_size
                    / weight_pack_factor;
                projection * 3
            };
            let gate_up_bias = cfg.num_local_experts * cfg.intermediate_size * 2;
            let down_bias = cfg.num_local_experts * cfg.hidden_size;
            let router = cfg.hidden_size * cfg.num_local_experts + cfg.num_local_experts;
            let sinks = cfg.num_attention_heads;

            input_layernorm
                + post_attention_layernorm
                + q_proj
                + k_proj
                + v_proj
                + o_proj
                + expert_weights
                + gate_up_bias
                + down_bias
                + router
                + sinks
        };
        Ok(vec![
            per_layer_elems * dtype.size_in_bytes();
            cfg.num_hidden_layers
        ])
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::gpt_oss::Config = serde_json::from_str(config)?;

        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: cfg.sliding_window,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

// ======================== Qwen3Next loader

/// [`NormalLoader`] for a Qwen3Next (Qwen3-Coder-Next) model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3NextLoader;

impl NormalModelLoader for Qwen3NextLoader {
    fn new_lora_registry(&self, config: &str) -> Result<mistralrs_quant::LoraLayerRegistry> {
        let config = serde_json::from_str::<serde_json::Value>(config)?;
        let qwen35_moe_identity = config
            .get("architectures")
            .and_then(serde_json::Value::as_array)
            .and_then(|architectures| architectures.first())
            .and_then(serde_json::Value::as_str)
            == Some(NormalLoaderType::Qwen3Next.causal_lm_name())
            && config
                .get(crate::gdn::GDN_V_HEAD_LAYOUT_CONFIG_KEY)
                .and_then(serde_json::Value::as_str)
                == Some("tiled");
        Ok(if qwen35_moe_identity {
            mistralrs_quant::LoraLayerRegistry::new_with_site_prefix_alias(
                "model",
                "model.language_model",
            )?
        } else {
            mistralrs_quant::LoraLayerRegistry::new()
        })
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::qwen3_next::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("Qwen3Next does not support X-LoRA")
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Qwen3NextLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.o_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.linear_attn\.(in_proj_qkvz|in_proj_qkv|in_proj_z|in_proj_ba|in_proj_b|in_proj_a)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.linear_attn\.out_proj\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
            Regex::new(
                r"layers\.(\d+)\.mlp\.shared_expert\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
        ])
    }
    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
            Regex::new(r"layers\.(\d+)\.mlp\.experts\.(gate_proj|up_proj|down_proj)\.weight$")?,
        ])
    }
    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Qwen3NextLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::qwen3_next::Config = serde_json::from_str(config)?;
        let extra_hidden_elements_per_token = if cfg
            .layer_types()
            .iter()
            .any(|layer| matches!(layer, models::qwen3_next::LayerType::LinearAttention))
        {
            let keys = activation_width(&[(cfg.linear_num_key_heads, cfg.linear_key_head_dim)])?;
            let values =
                activation_width(&[(cfg.linear_num_value_heads, cfg.linear_value_head_dim)])?;
            activation_width(&[(2, keys), (2, values), (2, cfg.linear_num_value_heads)])?
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size.max(activation_width(&[
                (cfg.num_experts_per_tok, cfg.moe_intermediate_size),
                (1, cfg.shared_expert_intermediate_size),
            ])?),
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        let elems = {
            let (embed_tokens_pack_factor, lm_head_pack_factor) =
                super::language_model_pack_factors(
                    _quantization,
                    "model.embed_tokens.weight",
                    "lm_head.weight",
                    cfg.tie_word_embeddings,
                    dtype,
                    weight_pack_factor,
                )?;
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
            let lm_head = if !cfg.tie_word_embeddings {
                cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        let layer_types = cfg.layer_types();
        let mut layer_sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_type in &layer_types {
            let input_layernorm = cfg.hidden_size;
            let post_attention_layernorm = cfg.hidden_size;

            let attn_elems = match layer_type {
                crate::models::qwen3_next::LayerType::FullAttention => {
                    let hidden = cfg.hidden_size;
                    let q_dim = cfg.head_dim * cfg.num_attention_heads;
                    let kv_dim = cfg.head_dim * cfg.num_key_value_heads;
                    let q_proj = hidden * q_dim * 2 / weight_pack_factor;
                    let k_proj = hidden * kv_dim / weight_pack_factor;
                    let v_proj = hidden * kv_dim / weight_pack_factor;
                    let o_proj = q_dim * hidden / weight_pack_factor;
                    let q_norm = cfg.head_dim;
                    let k_norm = cfg.head_dim;
                    q_proj + k_proj + v_proj + o_proj + q_norm + k_norm
                }
                crate::models::qwen3_next::LayerType::LinearAttention => {
                    let hidden = cfg.hidden_size;
                    let key_dim = cfg.linear_key_dim();
                    let value_dim = cfg.linear_value_dim();
                    let conv_dim = cfg.linear_conv_dim();
                    // in_proj_qkvz: (2 * key_dim + 2 * value_dim, hidden)
                    let in_proj_qkvz = hidden * (key_dim * 2 + value_dim * 2) / weight_pack_factor;
                    // in_proj_ba: (2 * num_v_heads, hidden)
                    let in_proj_ba = hidden * (cfg.linear_num_value_heads * 2) / weight_pack_factor;
                    let out_proj = value_dim * hidden / weight_pack_factor;
                    let conv1d = conv_dim * cfg.linear_conv_kernel_dim;
                    let dt_bias = cfg.linear_num_value_heads;
                    let a_log = cfg.linear_num_value_heads;
                    let norm = cfg.linear_value_head_dim;
                    in_proj_qkvz + in_proj_ba + out_proj + conv1d + dt_bias + a_log + norm
                }
            };

            let moe_gate = cfg.hidden_size * cfg.num_experts;
            let shared_expert =
                3 * cfg.hidden_size * cfg.shared_expert_intermediate_size / weight_pack_factor;
            let routed_experts = cfg.num_experts * 3 * cfg.hidden_size * cfg.moe_intermediate_size
                / weight_pack_factor;

            let per_layer_elems = input_layernorm
                + post_attention_layernorm
                + attn_elems
                + moe_gate
                + shared_expert
                + routed_experts;

            layer_sizes.push(per_layer_elems * dtype.size_in_bytes());
        }

        Ok(layer_sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }
    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::qwen3_next::Config = serde_json::from_str(config)?;

        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

/// [`NormalLoader`] for the text backbone of a dense Qwen3.5 model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Qwen3_5TextLoader;

fn parse_qwen35_text_config(config: &str) -> Result<crate::vision_models::qwen3_5::TextConfig> {
    let cfg: crate::vision_models::qwen3_5::TextConfig = serde_json::from_str(config)?;
    cfg.validate()?;
    Ok(cfg)
}

impl NormalModelLoader for Qwen3_5TextLoader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        match max_model_len {
            Some(max_model_len) => Ok(Cow::Owned(
                crate::vision_models::qwen3_5::config::apply_max_model_len(config, max_model_len)?,
            )),
            None => Ok(Cow::Borrowed(config)),
        }
    }

    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg = parse_qwen35_text_config(config)?;
        Ok(Box::new(
            crate::vision_models::qwen3_5::Qwen3_5TextModel::new(
                &cfg,
                vb,
                cfg.tie_word_embeddings,
                false,
                normal_loading_metadata,
                attention_mechanism,
            )?,
        ))
    }

    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("Qwen3.5 does not support X-LoRA")
    }

    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg = parse_qwen35_text_config(config)?;
        Ok(Box::new(cfg))
    }

    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Qwen3_5TextLoader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(
                r"^(model\.language_model|language_model\.model|model)\.embed_tokens\.weight$",
            )?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^lm_head\.(weight|bias)$")?,
            Regex::new(
                r"^(model\.language_model|language_model\.model|model)\.layers\.(\d+)\.self_attn\.(q_proj|k_proj|v_proj|o_proj)\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(model\.language_model|language_model\.model|model)\.layers\.(\d+)\.linear_attn\.(in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|out_proj)\.(weight|bias)$",
            )?,
            Regex::new(
                r"^(model\.language_model|language_model\.model|model)\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.(weight|bias)$",
            )?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Qwen3_5TextLoader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg = parse_qwen35_text_config(config)?;
        let extra_hidden_elements_per_token = if cfg.layer_types().iter().any(|layer| {
            matches!(
                layer,
                crate::vision_models::qwen3_5::config::LayerType::LinearAttention
            )
        }) {
            let keys = activation_width(&[(cfg.linear_num_key_heads, cfg.linear_key_head_dim)])?;
            let values =
                activation_width(&[(cfg.linear_num_value_heads, cfg.linear_value_head_dim)])?;
            activation_width(&[(2, keys), (2, values), (2, cfg.linear_num_value_heads)])?
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };
        let cfg = parse_qwen35_text_config(config)?;
        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg = parse_qwen35_text_config(config)?;
        let (embed_tokens_pack_factor, lm_head_pack_factor) =
            super::language_model_pack_factors_with_aliases(
                quantization,
                &[
                    "model.language_model.embed_tokens.weight",
                    "language_model.model.embed_tokens.weight",
                    "model.embed_tokens.weight",
                ],
                &["lm_head.weight"],
                cfg.tie_word_embeddings,
                dtype,
                weight_pack_factor,
            )?;
        let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
        let lm_head = if cfg.tie_word_embeddings {
            0
        } else {
            cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
        };
        Ok((embed_tokens + lm_head + cfg.hidden_size) * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg = parse_qwen35_text_config(config)?;
        let mut sizes = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_type in cfg.layer_types() {
            let attention = match layer_type {
                crate::vision_models::qwen3_5::config::LayerType::FullAttention => {
                    let q_dim = cfg.head_dim * cfg.num_attention_heads;
                    let kv_dim = cfg.head_dim * cfg.num_key_value_heads;
                    (cfg.hidden_size * (q_dim * 2 + kv_dim * 2) + q_dim * cfg.hidden_size)
                        / weight_pack_factor
                        + cfg.head_dim * 2
                }
                crate::vision_models::qwen3_5::config::LayerType::LinearAttention => {
                    let value_dim = cfg.linear_value_dim();
                    let projections = cfg.hidden_size
                        * (cfg.linear_conv_dim() + value_dim + cfg.linear_num_value_heads * 2)
                        / weight_pack_factor;
                    let out_proj = value_dim * cfg.hidden_size / weight_pack_factor;
                    let residual = cfg.linear_conv_dim() * cfg.linear_conv_kernel_dim
                        + cfg.linear_num_value_heads * 2
                        + cfg.linear_value_head_dim;
                    projections + out_proj + residual
                }
            };
            let mlp = cfg.hidden_size * cfg.intermediate_size * 3 / weight_pack_factor;
            sizes.push((cfg.hidden_size * 2 + attention + mlp) * dtype.size_in_bytes());
        }
        Ok(sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg = parse_qwen35_text_config(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg = parse_qwen35_text_config(config)?;
        Ok(Box::new(ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: cfg.head_dim,
            v_head_dim: cfg.head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        }))
    }
}

// ======================== LFM2 loader

/// [`NormalLoader`] for an LFM2 hybrid attention/short-conv model.
///
/// [`NormalLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.NormalLoader.html
pub struct Lfm2Loader;

impl NormalModelLoader for Lfm2Loader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;

        Ok(Box::new(models::lfm2::Model::new(
            &cfg,
            vb,
            self.is_gptx_for(config, &normal_loading_metadata)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }

    fn load_xlora(
        &self,
        _config: &str,
        _vb: ShardedVarBuilder,
        _lora_config: &[((String, String), LoraConfig)],
        _xlora_config: Option<XLoraConfig>,
        _xlora_ordering: Ordering,
        _normal_loading_metadata: NormalLoadingMetadata,
        _preload_adapters: &Option<HashMap<String, (ShardedVarBuilder, LoraConfig)>>,
    ) -> Result<Box<dyn NormalModel + Send + Sync>> {
        anyhow::bail!("LFM2 does not support X-LoRA")
    }

    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }

    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Lfm2Loader {
    fn promoted_isq_predicates(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"^model\.embed_tokens\.weight$")?,
            Regex::new(r"^lm_head\.(weight|bias)$")?,
        ])
    }

    fn isq_layer_regexes(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"lm_head\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.q_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.k_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.v_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.out_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.conv\.in_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.conv\.out_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w3\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.feed_forward\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }

    fn isq_layer_regexes_moqe(&self, _config: &str) -> Result<Vec<Regex>> {
        Ok(vec![
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.experts\.(\d+)\.w3\.(weight|bias)$")?,
            Regex::new(
                r"layers\.(\d+)\.feed_forward\.experts\.(gate_proj|up_proj|down_proj)\.weight$",
            )?,
        ])
    }

    fn immediate_isq_predicates_moqe(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes_moqe(config)
    }
}

impl DeviceMappedModelLoader for Lfm2Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: models::lfm2::Config = serde_json::from_str(config)?;
        let intermediate_size = if cfg.is_moe() {
            cfg.intermediate_size().max(activation_width(&[(
                cfg.num_experts_per_tok,
                cfg.moe_intermediate_size,
            )])?)
        } else {
            cfg.intermediate_size()
        };
        let extra_hidden_elements_per_token =
            if cfg.layer_types().contains(&models::lfm2::LayerType::Conv) {
                activation_width(&[(3, cfg.hidden_size)])?
            } else {
                0
            };
        Ok(ActivationModelConfig {
            intermediate_size,
            vocab_size: cfg.vocab_size,
            attention_features: Some(PrefixPrefillAttentionFeatures::default()),
            extra_hidden_elements_per_token,
            ..Default::default()
        })
    }

    fn mapped_max_act_size_elems(
        &self,
        config: &str,
        params: &AutoDeviceMapParams,
    ) -> Result<usize> {
        let AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        } = params
        else {
            anyhow::bail!("Expected text AutoDeviceMapParams for this model!")
        };

        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;

        Ok(
            max_batch_size
                * cfg.num_attention_heads
                * max_seq_len.min(&ATTENTION_CHUNK_SIZE).pow(2),
        )
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
        _quantization: Option<&super::AutoDeviceMapQuantization<'_>>,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;
        let tied = cfg.tie_word_embeddings();
        let (embed_tokens_pack_factor, lm_head_pack_factor) = super::language_model_pack_factors(
            _quantization,
            "model.embed_tokens.weight",
            "lm_head.weight",
            tied,
            dtype,
            weight_pack_factor,
        )?;
        let embed_tokens = cfg.hidden_size * cfg.vocab_size / embed_tokens_pack_factor;
        let lm_head = if tied {
            0
        } else {
            cfg.hidden_size * cfg.vocab_size / lm_head_pack_factor
        };
        let norm = cfg.hidden_size;
        Ok((embed_tokens + lm_head + norm) * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size();
        let mut sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for (layer_idx, layer_type) in cfg.layer_types().into_iter().enumerate() {
            let operator_norm = hidden;
            let ffn_norm = hidden;
            let feed_forward = match cfg.feed_forward_type(layer_idx) {
                crate::models::lfm2::FeedForwardType::Dense => {
                    3 * hidden * intermediate / weight_pack_factor
                }
                crate::models::lfm2::FeedForwardType::Moe => {
                    let gate = hidden * cfg.num_experts;
                    let expert_bias = if cfg.use_expert_bias {
                        cfg.num_experts
                    } else {
                        0
                    };
                    let experts = 3 * cfg.num_experts * hidden * cfg.moe_intermediate_size
                        / weight_pack_factor;
                    gate + expert_bias + experts
                }
            };
            let operator = match layer_type {
                crate::models::lfm2::LayerType::Attention => {
                    let q_dim = cfg.num_attention_heads * head_dim;
                    let kv_dim = cfg.num_key_value_heads * head_dim;
                    let projections = (hidden * q_dim + hidden * kv_dim * 2 + q_dim * hidden)
                        / weight_pack_factor;
                    projections + 2 * head_dim
                }
                crate::models::lfm2::LayerType::Conv => {
                    let projections = (hidden * 3 * hidden + hidden * hidden) / weight_pack_factor;
                    let conv = hidden * cfg.conv_l_cache;
                    let bias = if cfg.conv_bias { 5 * hidden } else { 0 };
                    projections + conv + bias
                }
            };

            sizes
                .push((operator_norm + ffn_norm + operator + feed_forward) * dtype.size_in_bytes());
        }

        Ok(sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: crate::models::lfm2::Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();
        let cfg = ModelConfigMetadata {
            max_seq_len: cfg.max_position_embeddings,
            num_layers: cfg.num_hidden_layers,
            hidden_size: cfg.hidden_size,
            num_kv_heads: cfg.num_key_value_heads,
            num_attn_heads: cfg.num_attention_heads,
            sliding_window: None,
            k_head_dim: head_dim,
            v_head_dim: head_dim,
            kv_cache_layout: crate::paged_attention::KvCacheLayout::Standard,
        };

        Ok(Box::new(cfg))
    }
}

#[cfg(test)]
#[path = "normal_loaders_tests.rs"]
mod tests;
