use mistralrs_quant::QuantizedConfig;

use crate::gdn::{GdnStateDType, GdnVHeadLayout};
use crate::layers::{Activation, YarnRopeConfig};
use crate::serde_default_fn;
use crate::vision_models::qwen3_5::config::RopeParameters;

// Re-export vision config from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(Vec<usize>, default_mlp_only_layers, Vec::new());
serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(bool, default_norm_topk_prob, true);

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
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
    // MoE fields
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    #[serde(default = "default_norm_topk_prob")]
    pub norm_topk_prob: bool,
    // Hybrid attention fields
    #[serde(default = "default_full_attn_interval")]
    pub full_attention_interval: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<LayerType>>,
    #[serde(default = "default_conv_kernel")]
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
    // Other
    #[serde(default = "default_mlp_only_layers")]
    pub mlp_only_layers: Vec<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_gdn_v_head_layout")]
    pub(crate) gdn_v_head_layout: GdnVHeadLayout,
}

impl TextConfig {
    pub fn validate(&self) -> candle_core::Result<()> {
        if self.max_position_embeddings == 0 {
            candle_core::bail!("Qwen3.5 maximum position embeddings must be positive");
        }
        if !self.rope_theta().is_finite() || self.rope_theta() <= 0.0 {
            candle_core::bail!("Qwen3.5 rope theta must be finite and positive");
        }
        if !self.partial_rotary_factor().is_finite()
            || self.partial_rotary_factor() <= 0.0
            || self.partial_rotary_factor() > 1.0
        {
            candle_core::bail!("Qwen3.5 partial rotary factor must be in (0, 1]");
        }
        if self.num_hidden_layers == 0 {
            candle_core::bail!("Qwen3.5 requires at least one hidden layer");
        }
        if self.full_attention_interval == 0
            || self.full_attention_interval > self.num_hidden_layers
        {
            candle_core::bail!(
                "Qwen3.5 full_attention_interval {} is invalid for {} layers",
                self.full_attention_interval,
                self.num_hidden_layers
            );
        }
        if let Some(layer_types) = &self.layer_types {
            if layer_types.len() != self.num_hidden_layers
                || !layer_types.contains(&LayerType::FullAttention)
            {
                candle_core::bail!(
                    "Qwen3.5 layer_types must list {} layers with at least one full_attention entry",
                    self.num_hidden_layers
                );
            }
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            candle_core::bail!(
                "Qwen3.5 has incompatible attention head counts: {} query and {} KV",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || !self
                .linear_num_value_heads
                .is_multiple_of(self.linear_num_key_heads)
        {
            candle_core::bail!(
                "Qwen3.5 has incompatible GDN head counts: {} key and {} value",
                self.linear_num_key_heads,
                self.linear_num_value_heads
            );
        }
        if self.linear_key_head_dim == 0
            || self.linear_value_head_dim == 0
            || self.linear_conv_kernel_dim == 0
        {
            candle_core::bail!("Qwen3.5 GDN dimensions must be non-zero");
        }
        let rot_dim = self.rot_dim();
        if rot_dim == 0 || rot_dim > self.head_dim || !rot_dim.is_multiple_of(2) {
            candle_core::bail!(
                "Qwen3.5 rotary dimension {rot_dim} must be positive, even, and no larger than head_dim {}",
                self.head_dim
            );
        }
        if self.mrope_section().len() != 3 || self.mrope_section().contains(&0) {
            candle_core::bail!("Qwen3.5 MRoPE requires three non-zero sections");
        }
        if !self.rope_parameters.mrope_interleaved {
            candle_core::bail!("Qwen3.5 requires interleaved MRoPE");
        }
        let section_width = self
            .mrope_section()
            .iter()
            .try_fold(0usize, |sum, section| sum.checked_add(*section))
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 MRoPE section width overflow"))?;
        if section_width.checked_mul(2) != Some(rot_dim) {
            candle_core::bail!(
                "Qwen3.5 MRoPE sections span {} dimensions, expected {rot_dim}",
                section_width.saturating_mul(2)
            );
        }
        if self.num_experts == 0
            || self.num_experts_per_tok == 0
            || self.num_experts_per_tok > self.num_experts
        {
            candle_core::bail!(
                "Qwen3.5 has invalid MoE routing: {} of {} experts",
                self.num_experts_per_tok,
                self.num_experts
            );
        }
        self.rope_parameters
            .validate_scaling(self.max_position_embeddings)
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_parameters.rope_theta
    }

    pub fn partial_rotary_factor(&self) -> f64 {
        self.rope_parameters.partial_rotary_factor
    }

    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_parameters.mrope_section
    }

    pub fn yarn_rope_config(&self) -> candle_core::Result<Option<YarnRopeConfig>> {
        self.rope_parameters
            .yarn_rope_config(self.max_position_embeddings, self.rot_dim())
    }

    pub fn layer_types(&self) -> Vec<LayerType> {
        if let Some(layer_types) = &self.layer_types {
            return layer_types.clone();
        }
        (0..self.num_hidden_layers)
            .map(|i| {
                if (i + 1) % self.full_attention_interval == 0 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect()
    }

    pub fn linear_key_dim(&self) -> usize {
        self.linear_num_key_heads * self.linear_key_head_dim
    }

    pub fn linear_value_dim(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn linear_conv_dim(&self) -> usize {
        2 * self.linear_key_dim() + self.linear_value_dim()
    }

    pub fn rot_dim(&self) -> usize {
        (self.head_dim as f64 * self.partial_rotary_factor()) as usize
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub tie_word_embeddings: bool,
    /// Top-level quantization_config takes precedence
    pub quantization_config: Option<QuantizedConfig>,
}
