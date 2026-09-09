#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use mistralrs_quant::QuantizedConfig;
use serde::Deserialize;

use crate::gdn::{GdnConfig, GdnOutputGate, GdnStateDType, GdnVHeadLayout};
use crate::layers::Activation;
use crate::vision_models::qwen3_5::config::RopeParameters;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum LayerType {
    FullAttention,
    LinearAttention,
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub(crate) struct Config {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub norm_topk_prob: bool,
    pub full_attention_interval: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
    pub hc_count: usize,
    pub hc_lowrank: usize,
    pub indexer_n_heads: usize,
    pub indexer_kv_heads: usize,
    pub indexer_head_dim: usize,
    pub indexer_budget: usize,
    pub indexer_compress_ratio: usize,
    pub output_gate_type: GdnOutputGate,
    pub ple_layer_ids: Vec<usize>,
    pub ple_embed_dim: usize,
    pub ple_conv_kernel_size: usize,
    pub ngram_size: usize,
    pub heads_per_ngram: usize,
    pub ple_layer_multipliers: Vec<u64>,
    pub ple_head_offsets: Vec<u64>,
    pub ple_head_vocab_sizes: Vec<u64>,
    pub eos_token_id: u32,
    #[serde(default)]
    pub image_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_gdn_v_head_layout")]
    gdn_v_head_layout: GdnVHeadLayout,
}

#[allow(dead_code)]
impl Config {
    pub(crate) fn validate(&self) -> candle_core::Result<()> {
        if self.output_gate_type != GdnOutputGate::Sigmoid {
            candle_core::bail!("Qwen4Exp output_gate_type must be sigmoid");
        }
        if self.num_hidden_layers == 0 || self.layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Qwen4Exp layer_types must list exactly {} layers",
                self.num_hidden_layers
            );
        }
        if !self.layer_types.contains(&LayerType::FullAttention)
            || !self.layer_types.contains(&LayerType::LinearAttention)
        {
            candle_core::bail!("Qwen4Exp requires both full_attention and linear_attention layers");
        }
        if self.full_attention_interval == 0 {
            candle_core::bail!("Qwen4Exp full_attention_interval must be positive");
        }
        if self.hc_count <= 1 || self.hc_lowrank == 0 {
            candle_core::bail!(
                "Qwen4Exp hyper-connections require hc_count > 1 and hc_lowrank > 0"
            );
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            candle_core::bail!(
                "Qwen4Exp has incompatible attention head counts: {} query and {} KV",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
            || self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_conv_kernel_dim == 0
        {
            candle_core::bail!("Qwen4Exp GDN dimensions and head counts are invalid");
        }
        if self.indexer_n_heads == 0
            || self.indexer_kv_heads != 1
            || self.indexer_head_dim == 0
            || self.indexer_budget == 0
            || self.indexer_compress_ratio == 0
            || !self
                .indexer_budget
                .is_multiple_of(self.indexer_compress_ratio)
        {
            candle_core::bail!(
                "Qwen4Exp QSA requires positive dimensions, one indexer KV head, and a budget divisible by the compression ratio"
            );
        }
        if self.max_position_embeddings == 0
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
            || !self.rope_parameters.mrope_interleaved
        {
            candle_core::bail!("Qwen4Exp requires valid interleaved MRoPE parameters");
        }
        let rotary_dim =
            (self.head_dim as f64 * self.rope_parameters.partial_rotary_factor) as usize;
        let section_width = self
            .rope_parameters
            .mrope_section
            .iter()
            .try_fold(0usize, |sum, width| sum.checked_add(*width))
            .ok_or_else(|| candle_core::Error::msg("Qwen4Exp MRoPE section width overflow"))?;
        if self.rope_parameters.mrope_section.len() != 3
            || self.rope_parameters.mrope_section.contains(&0)
            || rotary_dim == 0
            || rotary_dim > self.head_dim
            || !rotary_dim.is_multiple_of(2)
            || section_width.checked_mul(2) != Some(rotary_dim)
        {
            candle_core::bail!("Qwen4Exp has inconsistent MRoPE dimensions");
        }
        if self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
            || self.moe_intermediate_size == 0
            || self.shared_expert_intermediate_size == 0
        {
            candle_core::bail!("Qwen4Exp has invalid MoE dimensions or routing");
        }
        if self.ple_layer_ids.len() > 1 {
            candle_core::bail!("Qwen4Exp currently supports at most one PLE layer");
        }
        if let Some(&layer) = self.ple_layer_ids.first() {
            if layer >= self.num_hidden_layers {
                candle_core::bail!("Qwen4Exp PLE layer {layer} is out of range");
            }
            if self.layer_types[layer] != LayerType::LinearAttention {
                candle_core::bail!("Qwen4Exp PLE layer {layer} must use linear attention");
            }
            if self.ple_embed_dim == 0
                || self.ple_conv_kernel_size == 0
                || !(2..=4).contains(&self.ngram_size)
                || self.heads_per_ngram == 0
            {
                candle_core::bail!("Qwen4Exp PLE dimensions are invalid");
            }
            let head_count = (self.ngram_size - 1)
                .checked_mul(self.heads_per_ngram)
                .ok_or_else(|| candle_core::Error::msg("Qwen4Exp PLE head count overflow"))?;
            if !self.ple_embed_dim.is_multiple_of(head_count) {
                candle_core::bail!(
                    "Qwen4Exp ple_embed_dim {} must be divisible by {head_count} PLE heads",
                    self.ple_embed_dim
                );
            }
            if self.ple_layer_multipliers.len() < self.ngram_size
                || self.ple_head_offsets.len() != head_count
                || self.ple_head_vocab_sizes.len() != head_count
            {
                candle_core::bail!(
                    "Qwen4Exp PLE hash arrays must contain at least {} multipliers and exactly {head_count} head ranges",
                    self.ngram_size
                );
            }
            for (head, (&offset, &vocab_size)) in self
                .ple_head_offsets
                .iter()
                .zip(&self.ple_head_vocab_sizes)
                .enumerate()
            {
                if vocab_size == 0 || offset.checked_add(vocab_size).is_none() {
                    candle_core::bail!("Qwen4Exp PLE head {head} has an invalid row range");
                }
            }
        }
        Ok(())
    }
}

impl GdnConfig for Config {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn rms_norm_eps(&self) -> f64 {
        self.rms_norm_eps
    }

    fn linear_conv_kernel_dim(&self) -> usize {
        self.linear_conv_kernel_dim
    }

    fn linear_key_head_dim(&self) -> usize {
        self.linear_key_head_dim
    }

    fn linear_value_head_dim(&self) -> usize {
        self.linear_value_head_dim
    }

    fn linear_num_key_heads(&self) -> usize {
        self.linear_num_key_heads
    }

    fn linear_num_value_heads(&self) -> usize {
        self.linear_num_value_heads
    }

    fn quantization_config(&self) -> &Option<QuantizedConfig> {
        &self.quantization_config
    }

    fn output_gate(&self) -> GdnOutputGate {
        self.output_gate_type
    }

    fn v_head_layout(&self) -> GdnVHeadLayout {
        self.gdn_v_head_layout
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const OFFICIAL_TEXT_CONFIG: &str = r#"{
        "head_dim": 256,
        "vocab_size": 248320,
        "hidden_size": 2560,
        "num_hidden_layers": 4,
        "num_attention_heads": 24,
        "num_key_value_heads": 2,
        "hidden_act": "silu",
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "mrope_section": [11, 11, 10],
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": true
        },
        "moe_intermediate_size": 640,
        "shared_expert_intermediate_size": 640,
        "num_experts": 512,
        "num_experts_per_tok": 10,
        "norm_topk_prob": true,
        "full_attention_interval": 4,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 48,
        "mamba_ssm_dtype": "float32",
        "hc_count": 4,
        "hc_lowrank": 320,
        "indexer_n_heads": 4,
        "indexer_kv_heads": 1,
        "indexer_head_dim": 128,
        "indexer_budget": 2048,
        "indexer_compress_ratio": 4,
        "output_gate_type": "sigmoid",
        "ple_layer_ids": [2],
        "ple_embed_dim": 2560,
        "ple_conv_kernel_size": 4,
        "ngram_size": 3,
        "heads_per_ngram": 8,
        "ple_layer_multipliers": [11400714819323198485, 14029467366897019727, 1609587929392839161],
        "ple_head_offsets": [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500],
        "ple_head_vocab_sizes": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
        "eos_token_id": 248044,
        "image_token_id": 248056,
        "tie_word_embeddings": false
    }"#;

    pub(crate) fn fixture_config() -> Config {
        serde_json::from_str(OFFICIAL_TEXT_CONFIG).unwrap()
    }

    #[test]
    fn official_dimensions_validate_and_select_sigmoid_gdn() {
        let config = fixture_config();
        config.validate().unwrap();
        assert_eq!(config.hc_count, 4);
        assert_eq!(config.num_experts, 512);
        assert_eq!(config.output_gate(), GdnOutputGate::Sigmoid);
        assert_eq!(config.ple_embed_dim / 16, 160);
        assert_eq!(config.ple_layer_multipliers[0], 11_400_714_819_323_198_485);
        assert!(config.ple_layer_multipliers[0] > (1u64 << 53));
    }

    #[test]
    fn invalid_hyper_connection_count_is_rejected() {
        let mut config = fixture_config();
        config.hc_count = 1;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("hc_count"));
    }

    #[test]
    fn invalid_qsa_and_ple_configuration_is_rejected() {
        let mut config = fixture_config();
        config.indexer_kv_heads = 2;
        assert!(config.validate().unwrap_err().to_string().contains("QSA"));

        let mut config = fixture_config();
        config.ple_layer_ids = vec![3];
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("PLE layer 3"));

        let mut config = fixture_config();
        config.ple_head_offsets.pop();
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("PLE hash arrays"));

        let mut config = fixture_config();
        config.ple_embed_dim = 2559;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("must be divisible by 16 PLE heads"));
    }
}
