use super::{
    normal_loaders::{NormalLoadingMetadata, NormalModel, NormalModelLoader},
    ActivationModelConfig, AutoDeviceMapParams, DeviceMappedModelLoader,
};
use crate::{
    attention::ATTENTION_CHUNK_SIZE,
    lora::{LoraConfig, Ordering},
    matformer::MatformerSliceConfig,
    models::lfm3::{Config, LayerType, Model, ModelConfig},
    paged_attention::{AttentionImplementation, ModelConfigLike, ModelConfigMetadata},
    pipeline::isq::IsqModelLoader,
    xlora_models::XLoraConfig,
};
use anyhow::Result;
use candle_core::DType;
use mistralrs_quant::ShardedVarBuilder;
use regex::Regex;
use std::{borrow::Cow, collections::HashMap, fmt::Debug};

pub struct Lfm3Loader;

impl NormalModelLoader for Lfm3Loader {
    fn runtime_config<'a>(
        &self,
        config: &'a str,
        max_model_len: Option<usize>,
    ) -> Result<Cow<'a, str>> {
        let Some(max_model_len) = max_model_len else {
            return Ok(Cow::Borrowed(config));
        };
        let cfg: Config = serde_json::from_str(config)?;
        anyhow::ensure!(
            max_model_len > 0 && max_model_len <= cfg.max_position_embeddings,
            "LFM3 max_model_len must be between 1 and {}",
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
        let cfg: Config = serde_json::from_str(config)?;

        Ok(Box::new(Model::new(
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
        anyhow::bail!("LFM3 does not support X-LoRA")
    }

    fn is_gptx(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }

    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let cfg: Config = serde_json::from_str(config)?;
        Ok(Box::new(cfg))
    }

    fn supports_paged_attention(&self, _config: &str) -> Result<bool> {
        Ok(true)
    }
}

impl IsqModelLoader for Lfm3Loader {
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
            Regex::new(r"layers\.(\d+)\.conv\.in_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.conv\.out_proj\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w1\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w2\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.feed_forward\.w3\.(weight|bias)$")?,
            Regex::new(r"layers\.(\d+)\.self_attn\.gate_proj\.(weight|bias)$")?,
        ])
    }

    fn immediate_isq_predicates(&self, config: &str) -> Result<Vec<Regex>> {
        self.isq_layer_regexes(config)
    }
}

impl DeviceMappedModelLoader for Lfm3Loader {
    fn activation_model_config(&self, config: &str) -> Result<ActivationModelConfig> {
        let cfg: Config = serde_json::from_str(config)?;
        let extra_hidden_elements_per_token = if cfg.layer_types().contains(&LayerType::Conv) {
            3 * cfg.hidden_size
        } else {
            0
        };
        Ok(ActivationModelConfig {
            intermediate_size: cfg.intermediate_size,
            vocab_size: cfg.vocab_size,
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

        let cfg: Config = serde_json::from_str(config)?;

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
        let cfg: Config = serde_json::from_str(config)?;
        let tied = cfg.tie_word_embeddings();
        let (embed_tokens_pack_factor, lm_head_pack_factor) = super::language_model_pack_factors(
            quantization,
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
        let cfg: Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();
        let hidden = cfg.hidden_size;
        let intermediate = cfg.intermediate_size;
        let mut sizes = Vec::with_capacity(cfg.num_hidden_layers);

        for layer_type in cfg.layer_types() {
            let operator_norm = hidden;
            let ffn_norm = hidden;
            let feed_forward = 3 * hidden * intermediate / weight_pack_factor;
            let sandwich_norms = if cfg.use_sandwich_norm { 2 * hidden } else { 0 };
            let operator = match layer_type {
                LayerType::Attention | LayerType::SlidingAttention => {
                    let q_dim = cfg.num_attention_heads * head_dim;
                    let kv_dim = cfg.num_key_value_heads * head_dim;
                    let projections = (hidden * q_dim
                        + hidden * kv_dim * 2
                        + q_dim * hidden
                        + hidden * cfg.num_attention_heads)
                        / weight_pack_factor;
                    projections + 2 * head_dim
                }
                LayerType::Conv => {
                    let projections = (hidden * 3 * hidden + hidden * hidden) / weight_pack_factor;
                    let conv = hidden * cfg.conv_l_cache;
                    let bias = if cfg.conv_bias { 5 * hidden } else { 0 };
                    projections + conv + bias
                }
            };

            sizes.push(
                (operator_norm + ffn_norm + sandwich_norms + operator + feed_forward)
                    * dtype.size_in_bytes(),
            );
        }

        Ok(sizes)
    }

    fn num_layers(&self, config: &str) -> Result<usize> {
        let cfg: Config = serde_json::from_str(config)?;
        Ok(cfg.num_hidden_layers)
    }

    fn model_config(&self, config: &str) -> Result<Box<dyn ModelConfigLike>> {
        let cfg: Config = serde_json::from_str(config)?;
        let head_dim = cfg.head_dim();
        let metadata = ModelConfigMetadata {
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

        Ok(Box::new(ModelConfig::new(
            metadata,
            cfg.layer_types(),
            cfg.sliding_window,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHECKPOINT_PARAMETERS: usize = 1_728_801_536;

    #[test]
    fn checkpoint_storage_matches_lfm3_1_7b() -> Result<()> {
        let config = serde_json::json!({
            "max_position_embeddings": 262144,
            "head_dim": 128,
        })
        .to_string();
        let layers = Lfm3Loader.layer_sizes_in_bytes(&config, DType::BF16, 1, None)?;
        let shared = Lfm3Loader.non_mapped_size_in_bytes(&config, DType::BF16, 1, None, None)?;
        assert_eq!(
            layers.iter().sum::<usize>() + shared,
            CHECKPOINT_PARAMETERS * DType::BF16.size_in_bytes()
        );
        let cfg = Lfm3Loader.model_config(&config)?;
        assert_eq!(cfg.k_head_dim(), 128);
        assert_eq!(cfg.v_head_dim(), 128);
        assert_eq!(cfg.total_kv_cache_elements_per_token(), 11 * 4 * 256);
        assert_eq!(
            (0..cfg.num_layers())
                .filter(|layer| cfg.layer_has_paged_kv_cache(*layer))
                .count(),
            11
        );
        assert_eq!(
            (0..cfg.num_layers())
                .filter(|layer| cfg.sliding_window_for_layer(*layer).is_some())
                .count(),
            5
        );
        assert!(!cfg.layer_has_paged_kv_cache(0));
        assert!(cfg.layer_has_paged_kv_cache(2));
        assert_eq!(cfg.sliding_window_for_layer(2), None);
        assert_eq!(cfg.sliding_window_for_layer(4), Some(256));
        assert_eq!(cfg.prefix_prefill_attention_features(0), None);
        assert!(
            cfg.prefix_prefill_attention_features(4)
                .unwrap()
                .has_sliding_window
        );
        let runtime = Lfm3Loader.runtime_config(&config, Some(4096))?;
        let runtime: Config = serde_json::from_str(&runtime)?;
        assert_eq!(runtime.max_position_embeddings, 4096);
        assert!(Lfm3Loader.runtime_config(&config, Some(0)).is_err());
        assert!(Lfm3Loader.runtime_config(&config, Some(262145)).is_err());
        Ok(())
    }

    #[test]
    fn quantization_covers_gate_and_checkpoint_projection_names() -> Result<()> {
        let regexes = Lfm3Loader.isq_layer_regexes("{}")?;
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"] {
            let key = format!("model.layers.2.self_attn.{name}.weight");
            assert!(regexes.iter().any(|regex| regex.is_match(&key)), "{key}");
        }
        for name in ["q_norm", "k_norm"] {
            let key = format!("model.layers.2.self_attn.{name}.weight");
            assert!(!regexes.iter().any(|regex| regex.is_match(&key)), "{key}");
        }
        Ok(())
    }
}
