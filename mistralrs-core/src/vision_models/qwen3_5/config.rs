use mistralrs_quant::QuantizedConfig;

use crate::gdn::GdnVHeadLayout;
use crate::layers::Activation;
use crate::serde_default_fn;

// Re-export vision config from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(f64, default_rope_theta, 10_000_000.0);

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

/// Nested rope_parameters from the config JSON.
/// Contains rope_theta, mrope_section, partial_rotary_factor, etc.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub head_dim: usize,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_parameters: RopeParameters,
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
    // Other
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_gdn_v_head_layout")]
    pub(crate) gdn_v_head_layout: GdnVHeadLayout,
    // Multi-token prediction head shipped in the checkpoint (`mtp.*` weights)
    #[serde(default)]
    pub mtp_num_hidden_layers: usize,
    #[serde(default)]
    pub mtp_use_dedicated_embeddings: bool,
}

impl TextConfig {
    pub fn validate(&self) -> candle_core::Result<()> {
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
        Ok(())
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

    /// Absolute layer index of the first MTP block; its paged KV cache lives past the main stack.
    pub fn mtp_layer_idx(&self) -> usize {
        self.num_hidden_layers
    }

    pub fn mtp_layers(&self, mtp: bool) -> usize {
        if mtp {
            self.mtp_num_hidden_layers
        } else {
            0
        }
    }

    /// Paged-KV mask over the main stack plus any MTP blocks appended after it.
    pub fn paged_kv_layers(&self, mtp: bool) -> Vec<bool> {
        let mut layers = self
            .layer_types()
            .into_iter()
            .map(|ty| ty == LayerType::FullAttention)
            .collect::<Vec<_>>();
        layers.extend(std::iter::repeat_n(true, self.mtp_layers(mtp)));
        layers
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
    /// Injected by the loader when the built-in MTP head should be loaded (see `MTP_CONFIG_KEY`).
    #[serde(default, rename = "_mistralrs_mtp")]
    pub mtp: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_layers_extend_the_paged_kv_mask() {
        let cfg: TextConfig = serde_json::from_str(
            r#"{
                "head_dim": 64, "vocab_size": 32, "hidden_size": 128, "intermediate_size": 256,
                "num_hidden_layers": 8, "num_attention_heads": 4, "num_key_value_heads": 2,
                "hidden_act": "silu", "max_position_embeddings": 1024, "rms_norm_eps": 1e-6,
                "rope_parameters": { "mrope_section": [8, 4, 4] },
                "linear_key_head_dim": 16, "linear_value_head_dim": 16,
                "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                "mtp_num_hidden_layers": 1
            }"#,
        )
        .unwrap();
        let base = cfg.paged_kv_layers(false);
        assert_eq!(base.len(), 8);
        assert_eq!(base.iter().filter(|x| **x).count(), 2);
        let with_mtp = cfg.paged_kv_layers(true);
        assert_eq!(with_mtp.len(), 9);
        assert!(with_mtp[8]);
        assert_eq!(cfg.mtp_layer_idx(), 8);
        assert_eq!(cfg.mtp_layers(true), 1);
        assert_eq!(cfg.mtp_layers(false), 0);
    }
}
