use std::{
    fmt::{self, Debug, Display},
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use crate::{
    attention::ATTENTION_CHUNK_SIZE,
    embedding_models::{
        embedding_gemma::{EmbeddingGemma, EmbeddingGemmaConfig},
        qwen3_embedding::{Config as Qwen3EmbeddingConfig, Model as Qwen3EmbeddingModel},
    },
    matformer::MatformerSliceConfig,
    pipeline::{loaders::auto_device_map::NonMappedSubModel, NormalLoadingMetadata},
};

use crate::{
    amoe::AnyMoeBaseModelMixin,
    device_map::DeviceMapper,
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::{isq::IsqModelLoader, text_models_inputs_processor::FlashParams, IsqModel},
    utils::varbuilder_utils::DeviceForLoadTensor,
};
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use mistralrs_quant::log::once_log_info;

use mistralrs_quant::ShardedVarBuilder;
#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use regex::Regex;
use serde::{de::Visitor, Deserialize, Deserializer, Serialize};

use super::{AutoDeviceMapParams, DeviceMappedModelLoader};

pub trait EmbeddingModel: IsqModel + AnyMoeBaseModelMixin {
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        input_ids: &Tensor,
        flash_params: &FlashParams,
    ) -> candle_core::Result<Tensor>;
    fn device(&self) -> &Device;
}

pub trait EmbeddingModelLoader: IsqModelLoader + Send + Sync + DeviceMappedModelLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>>;
    fn is_gptx(&self, config: &str) -> Result<bool>;
    fn has_causal_attention(&self, config: &str) -> Result<bool>;
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

#[cfg_attr(feature = "pyo3_macros", pyclass(eq, eq_int))]
#[derive(Clone, Debug, Deserialize, serde::Serialize, PartialEq)]
/// The architecture to load the embedding model as.
pub enum EmbeddingLoaderType {
    #[serde(rename = "embeddinggemma")]
    EmbeddingGemma,
    #[serde(rename = "qwen3embedding")]
    Qwen3Embedding,
}

// https://github.com/huggingface/transformers/blob/cff06aac6fad28019930be03f5d467055bf62177/src/transformers/models/auto/modeling_auto.py#L448
impl EmbeddingLoaderType {
    pub fn from_causal_lm_name(name: &str) -> Result<Self> {
        match name {
            "Gemma3TextModel" => Ok(Self::EmbeddingGemma),
            "Qwen3ForCausalLM" => Ok(Self::Qwen3Embedding),
            other => anyhow::bail!(
                "Unsupported Hugging Face Transformers model class `{other}`. Please raise an issue."
            ),
        }
    }
}

impl FromStr for EmbeddingLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "embeddinggemma" => Ok(Self::EmbeddingGemma),
            "qwen3embedding" => Ok(Self::Qwen3Embedding),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `embeddinggemma`, `qwen3embedding`."
            )),
        }
    }
}

impl Display for EmbeddingLoaderType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmbeddingGemma => write!(f, "embeddinggemma"),
            Self::Qwen3Embedding => write!(f, "qwen3embedding"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub enum EmbeddingModulePaths {
    Transformer {
        path: String,
    },
    Pooling {
        path: String,
        config: PathBuf,
    },
    Dense {
        path: String,
        config: PathBuf,
        model: PathBuf,
    },
    Normalize {
        path: String,
    },
}

impl EmbeddingModulePaths {
    pub fn serialize_modules(modules: &[EmbeddingModulePaths]) -> String {
        #[derive(Serialize)]
        struct OutputModule {
            idx: usize,
            name: String,
            path: String,
            #[serde(rename = "type")]
            ty: String,
        }

        let mapped: Vec<OutputModule> = modules
            .iter()
            .enumerate()
            .map(|(i, m)| {
                let (path, ty) = match m {
                    EmbeddingModulePaths::Transformer { path } => (
                        path.clone(),
                        "sentence_transformers.models.Transformer".to_string(),
                    ),
                    EmbeddingModulePaths::Pooling { path, .. } => (
                        path.clone(),
                        "sentence_transformers.models.Pooling".to_string(),
                    ),
                    EmbeddingModulePaths::Dense { path, .. } => (
                        path.clone(),
                        "sentence_transformers.models.Dense".to_string(),
                    ),
                    EmbeddingModulePaths::Normalize { path } => (
                        path.clone(),
                        "sentence_transformers.models.Normalize".to_string(),
                    ),
                };

                OutputModule {
                    idx: i,
                    name: i.to_string(),
                    path,
                    ty,
                }
            })
            .collect();

        serde_json::to_string_pretty(&mapped).unwrap()
    }
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingModule {
    pub path: String,
    #[serde(rename = "type", deserialize_with = "deserialize_module_type")]
    pub ty: EmbeddingModuleType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingModuleType {
    Transformer,
    Pooling,
    Dense,
    Normalize,
}

fn deserialize_module_type<'de, D>(deserializer: D) -> Result<EmbeddingModuleType, D::Error>
where
    D: Deserializer<'de>,
{
    struct ModuleTypeVisitor;

    impl<'de> Visitor<'de> for ModuleTypeVisitor {
        type Value = EmbeddingModuleType;

        fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
            f.write_str("a sentence-transformers module type string")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            // Accept fully-qualified ("sentence_transformers.models.X") or just "X".
            let last = v.rsplit('.').next().unwrap_or(v).to_ascii_lowercase();
            match last.as_str() {
                "transformer" => Ok(EmbeddingModuleType::Transformer),
                "pooling" => Ok(EmbeddingModuleType::Pooling),
                "dense" => Ok(EmbeddingModuleType::Dense),
                "normalize" => Ok(EmbeddingModuleType::Normalize),
                _ => Err(E::invalid_value(
                    serde::de::Unexpected::Str(v),
                    &"Transformer/Pooling/Dense/Normalize",
                )),
            }
        }
    }

    deserializer.deserialize_str(ModuleTypeVisitor)
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
pub struct AutoEmbeddingLoader;

#[derive(Deserialize)]
struct AutoEmbeddingLoaderConfig {
    architectures: Vec<String>,
}

impl AutoEmbeddingLoader {
    fn get_loader(config: &str) -> Result<Box<dyn EmbeddingModelLoader>> {
        let auto_cfg: AutoEmbeddingLoaderConfig = serde_json::from_str(config)?;
        if auto_cfg.architectures.len() != 1 {
            anyhow::bail!("Expected to have one name for `architectures` config field.")
        }

        let name = &auto_cfg.architectures[0];

        let tp = EmbeddingLoaderType::from_causal_lm_name(name)?;

        once_log_info(format!("Automatic loader type determined to be `{tp}`"));

        match tp {
            EmbeddingLoaderType::EmbeddingGemma => Ok(Box::new(EmbeddingGemmaLoader)),
            EmbeddingLoaderType::Qwen3Embedding => Ok(Box::new(Qwen3EmbeddingLoader)),
        }
    }
}

impl EmbeddingModelLoader for AutoEmbeddingLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        Self::get_loader(config)?.load(config, vb, normal_loading_metadata, attention_mechanism)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        Self::get_loader(config)?.get_config_repr(config)
    }
    fn has_causal_attention(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.has_causal_attention(config)
    }
    fn is_gptx(&self, config: &str) -> Result<bool> {
        Self::get_loader(config)?.is_gptx(config)
    }
}

impl IsqModelLoader for AutoEmbeddingLoader {
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

impl DeviceMappedModelLoader for AutoEmbeddingLoader {
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

/// [`EmbeddingModelLoader`] for an Embedding Gemma model.
///
/// [`EmbeddingModelLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.EmbeddingModelLoader.html
pub struct EmbeddingGemmaLoader;

impl EmbeddingModelLoader for EmbeddingGemmaLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

        Ok(Box::new(EmbeddingGemma::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(false)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for EmbeddingGemmaLoader {
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

impl DeviceMappedModelLoader for EmbeddingGemmaLoader {
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

        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

        let elems = {
            let embed_tokens = cfg.hidden_size * cfg.vocab_size / weight_pack_factor;
            let norm = cfg.hidden_size;
            embed_tokens + norm
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
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

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
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: EmbeddingGemmaConfig = serde_json::from_str(config)?;

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
        None // todo
    }
}

/// [`EmbeddingModelLoader`] for a Qwen 3 model.
///
/// [`EmbeddingModelLoader`]: https://docs.rs/mistralrs/latest/mistralrs/struct.EmbeddingModelLoader.html
pub struct Qwen3EmbeddingLoader;

impl EmbeddingModelLoader for Qwen3EmbeddingLoader {
    fn load(
        &self,
        config: &str,
        vb: ShardedVarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
        attention_mechanism: AttentionImplementation,
    ) -> Result<Box<dyn EmbeddingModel + Send + Sync>> {
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;

        Ok(Box::new(Qwen3EmbeddingModel::new(
            &cfg,
            vb,
            self.is_gptx(config)?,
            normal_loading_metadata,
            attention_mechanism,
        )?))
    }
    fn has_causal_attention(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn is_gptx(&self, _: &str) -> Result<bool> {
        Ok(true)
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;

        Ok(Box::new(cfg))
    }
}

impl IsqModelLoader for Qwen3EmbeddingLoader {
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

impl DeviceMappedModelLoader for Qwen3EmbeddingLoader {
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

        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;

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
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<usize> {
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;
        let elems = {
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
        Ok(elems * dtype.size_in_bytes())
    }

    fn layer_sizes_in_bytes(
        &self,
        config: &str,
        dtype: DType,
        weight_pack_factor: usize,
        _matformer_config: Option<&MatformerSliceConfig>,
    ) -> Result<Vec<usize>> {
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;
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
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Qwen3EmbeddingConfig = serde_json::from_str(config)?;

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
