use mistralrs_quant::QuantizedConfig;

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(usize, default_vocab_size, 202_048);
serde_default_fn!(usize, default_text_hidden_size, 6656);
serde_default_fn!(usize, default_text_intermediate_size, 19_968);
serde_default_fn!(usize, default_text_layers, 52);
serde_default_fn!(usize, default_text_heads, 32);
serde_default_fn!(usize, default_text_kv_heads, 2);
serde_default_fn!(usize, default_head_dim, 128);
serde_default_fn!(Activation, default_text_activation, Activation::Silu);
serde_default_fn!(usize, default_max_position_embeddings, 131_072);
serde_default_fn!(f64, default_rms_norm_eps, 1e-5);
serde_default_fn!(f64, default_post_norm_eps, 1e-8);
serde_default_fn!(usize, default_sliding_window, 2048);
serde_default_fn!(f64, default_text_rope_theta, 500_000.0);
serde_default_fn!(f64, default_qk_scale_factor, 3.87);
serde_default_fn!(f64, default_output_multiplier, 0.196_116_135_138_184_04);
serde_default_fn!(f64, default_final_logit_softcapping, 20.0);

serde_default_fn!(usize, default_vision_hidden_size, 1536);
serde_default_fn!(usize, default_vision_intermediate_size, 8960);
serde_default_fn!(usize, default_vision_heads, 16);
serde_default_fn!(usize, default_vision_layers, 50);
serde_default_fn!(Activation, default_vision_activation, Activation::Gelu);
serde_default_fn!(usize, default_patch_size, 14);
serde_default_fn!(usize, default_patch_temporal, 2);
serde_default_fn!(usize, default_merge_size, 2);
serde_default_fn!(usize, default_pos_emb_side, 32);
serde_default_fn!(usize, default_vision_max_positions, 1024);
serde_default_fn!(f64, default_layer_norm_eps, 1e-5);
serde_default_fn!(f64, default_vision_rope_theta, 10_000.0);

serde_default_fn!(usize, default_out_hidden_size, 6144);
serde_default_fn!(usize, default_projector_hidden_size, 4096);
serde_default_fn!(Activation, default_projector_activation, Activation::Gelu);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TextAttentionType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VisionAttentionType {
    FullAttention,
    WindowAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default = "default_text_rope_theta")]
    pub rope_theta: f64,
}

impl Default for RopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_text_rope_theta(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionRopeParameters {
    #[serde(default = "default_vision_rope_theta")]
    pub rope_theta: f64,
}

impl Default for VisionRopeParameters {
    fn default() -> Self {
        Self {
            rope_theta: default_vision_rope_theta(),
        }
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_text_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_text_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_text_kv_heads")]
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_text_activation", alias = "hidden_act")]
    pub hidden_activation: Activation,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_post_norm_eps")]
    pub post_norm_eps: f64,
    #[serde(default = "default_sliding_window")]
    pub sliding_window: usize,
    #[serde(default)]
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub layer_types: Option<Vec<TextAttentionType>>,
    #[serde(default)]
    pub layer_rope_theta: Option<Vec<f64>>,
    #[serde(default = "default_qk_scale_factor")]
    pub qk_scale_factor: f64,
    #[serde(default = "default_output_multiplier")]
    pub output_multiplier: f64,
    #[serde(default = "default_final_logit_softcapping")]
    pub final_logit_softcapping: f64,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

impl TextConfig {
    pub fn layer_types(&self) -> candle_core::Result<Vec<TextAttentionType>> {
        let layer_types = self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|layer_idx| {
                    if (self.num_hidden_layers - 1 - layer_idx).is_multiple_of(4) {
                        TextAttentionType::FullAttention
                    } else {
                        TextAttentionType::SlidingAttention
                    }
                })
                .collect()
        });
        if layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Muse-Glimmer text layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        Ok(layer_types)
    }

    pub fn layer_rope_theta(&self) -> candle_core::Result<Vec<f64>> {
        let theta = self.layer_rope_theta.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|layer_idx| {
                    if (self.num_hidden_layers - 1 - layer_idx).is_multiple_of(4) {
                        0.0
                    } else {
                        self.rope_parameters.rope_theta
                    }
                })
                .collect()
        });
        if theta.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Muse-Glimmer layer_rope_theta has {} entries for {} layers",
                theta.len(),
                self.num_hidden_layers
            );
        }
        if theta.iter().any(|theta| !theta.is_finite() || *theta < 0.0) {
            candle_core::bail!(
                "Muse-Glimmer layer RoPE theta values must be finite and nonnegative"
            );
        }
        Ok(theta)
    }

    pub fn validate(&self) -> candle_core::Result<()> {
        if self.num_hidden_layers == 0 || self.hidden_size == 0 || self.head_dim == 0 {
            candle_core::bail!("Muse-Glimmer text dimensions must be nonzero");
        }
        if self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self
                .num_attention_heads
                .is_multiple_of(self.num_key_value_heads)
        {
            candle_core::bail!(
                "Muse-Glimmer has incompatible attention head counts: {} query and {} KV",
                self.num_attention_heads,
                self.num_key_value_heads
            );
        }
        if self.sliding_window == 0 || self.max_position_embeddings == 0 {
            candle_core::bail!("Muse-Glimmer context and sliding-window sizes must be nonzero");
        }
        for (name, value) in [
            ("rms_norm_eps", self.rms_norm_eps),
            ("post_norm_eps", self.post_norm_eps),
            ("qk_scale_factor", self.qk_scale_factor),
            ("output_multiplier", self.output_multiplier),
            ("final_logit_softcapping", self.final_logit_softcapping),
        ] {
            if !value.is_finite() || value <= 0.0 {
                candle_core::bail!("Muse-Glimmer {name} must be finite and positive");
            }
        }
        self.layer_types()?;
        self.layer_rope_theta()?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_vision_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_vision_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_vision_activation")]
    pub hidden_act: Activation,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_patch_temporal")]
    pub patch_temporal: usize,
    #[serde(default = "default_merge_size")]
    pub merge_size: usize,
    #[serde(default = "default_pos_emb_side")]
    pub pos_emb_height: usize,
    #[serde(default = "default_pos_emb_side")]
    pub pos_emb_width: usize,
    #[serde(default = "default_vision_max_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
    #[serde(default)]
    pub rope_parameters: VisionRopeParameters,
    #[serde(default)]
    pub layer_types: Option<Vec<VisionAttentionType>>,
}

impl VisionConfig {
    pub fn layer_types(&self) -> candle_core::Result<Vec<VisionAttentionType>> {
        let layer_types = self.layer_types.clone().unwrap_or_else(|| {
            (0..self.num_hidden_layers)
                .map(|layer_idx| {
                    if (layer_idx + 1) % 4 == 0 || layer_idx + 1 == self.num_hidden_layers {
                        VisionAttentionType::FullAttention
                    } else {
                        VisionAttentionType::WindowAttention
                    }
                })
                .collect()
        });
        if layer_types.len() != self.num_hidden_layers {
            candle_core::bail!(
                "Muse-Glimmer vision layer_types has {} entries for {} layers",
                layer_types.len(),
                self.num_hidden_layers
            );
        }
        Ok(layer_types)
    }

    pub fn validate(&self) -> candle_core::Result<()> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
        {
            candle_core::bail!("Muse-Glimmer vision attention dimensions are incompatible");
        }
        let head_dim = self.hidden_size / self.num_attention_heads;
        if !head_dim.is_multiple_of(4) {
            candle_core::bail!("Muse-Glimmer vision head dimension must be divisible by four");
        }
        if self.num_hidden_layers == 0
            || self.patch_size == 0
            || self.patch_temporal == 0
            || self.merge_size == 0
            || self.pos_emb_height == 0
            || self.pos_emb_width == 0
            || self.max_position_embeddings == 0
        {
            candle_core::bail!("Muse-Glimmer vision dimensions must be nonzero");
        }
        if self.pos_emb_height != self.pos_emb_width {
            candle_core::bail!("Muse-Glimmer requires a square learned vision position grid");
        }
        if !self.layer_norm_eps.is_finite()
            || self.layer_norm_eps <= 0.0
            || !self.rope_parameters.rope_theta.is_finite()
            || self.rope_parameters.rope_theta <= 0.0
        {
            candle_core::bail!("Muse-Glimmer vision norm epsilon and RoPE theta must be positive");
        }
        self.layer_types()?;
        Ok(())
    }
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_projector_hidden_size")]
    pub projector_hidden_size: usize,
    #[serde(default = "default_projector_activation")]
    pub projector_hidden_act: Activation,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
    #[serde(default, rename = "_mistralrs_muse_glimmer_gguf_collapsed_temporal")]
    pub(crate) gguf_collapsed_temporal: bool,
}

impl Config {
    pub fn validate(&self) -> candle_core::Result<()> {
        self.text_config.validate()?;
        self.vision_config.validate()?;
        let expected = self
            .vision_config
            .hidden_size
            .checked_mul(self.vision_config.merge_size.pow(2))
            .ok_or_else(|| candle_core::Error::msg("Muse-Glimmer vision output size overflow"))?;
        if self.out_hidden_size != expected {
            candle_core::bail!(
                "Muse-Glimmer out_hidden_size {} does not match merged vision size {expected}",
                self.out_hidden_size
            );
        }
        if self.image_token_id == self.video_token_id
            || self.image_token_id as usize >= self.text_config.vocab_size
            || self.video_token_id as usize >= self.text_config.vocab_size
        {
            candle_core::bail!(
                "Muse-Glimmer image and video token ids must be distinct and within the vocabulary"
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn released_layer_patterns_match_transformers() -> candle_core::Result<()> {
        let config: Config = serde_json::from_str(
            r#"{
                "text_config": {},
                "vision_config": {},
                "image_token_id": 200092,
                "video_token_id": 200091
            }"#,
        )
        .unwrap();
        config.validate()?;
        let text = config.text_config.layer_types()?;
        assert_eq!(text.len(), 52);
        assert!(text.iter().enumerate().all(|(index, kind)| {
            (*kind == TextAttentionType::FullAttention) == (index % 4 == 3)
        }));
        let rope = config.text_config.layer_rope_theta()?;
        assert!(rope
            .iter()
            .enumerate()
            .all(|(index, theta)| (*theta == 0.0) == (index % 4 == 3)));
        let vision = config.vision_config.layer_types()?;
        assert_eq!(vision.len(), 50);
        assert!(vision.iter().enumerate().all(|(index, kind)| {
            (*kind == VisionAttentionType::FullAttention) == ((index + 1) % 4 == 0 || index == 49)
        }));
        Ok(())
    }
}
