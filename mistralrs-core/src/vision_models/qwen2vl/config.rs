// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py

use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

use crate::serde_default_fn;

serde_default_fn!(Activation, default_vision_hidden_act, Activation::QuickGelu);
serde_default_fn!(usize, default_in_channels, 3);
serde_default_fn!(usize, default_max_window_layers, 28);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    #[serde(default = "default_in_channels", alias = "in_chans")]
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct MRopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub tie_word_embeddings: bool,
    pub rope_theta: f64,
    #[serde(default)]
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    #[serde(default = "default_max_window_layers")]
    pub max_window_layers: usize,
    #[serde(default)]
    pub layer_types: Option<Vec<AttentionType>>,
    pub vision_config: VisionConfig,
    pub rope_scaling: MRopeScaling,
    pub quantization_config: Option<QuantizedConfig>,
    pub image_token_id: u32,
    pub video_token_id: u32,
}

fn resolve_layer_sliding_windows(
    num_hidden_layers: usize,
    use_sliding_window: bool,
    sliding_window: Option<usize>,
    max_window_layers: usize,
    layer_types: Option<Vec<AttentionType>>,
) -> candle_core::Result<Vec<Option<usize>>> {
    let sliding_window = use_sliding_window.then_some(sliding_window).flatten();
    let layer_types = layer_types.unwrap_or_else(|| {
        (0..num_hidden_layers)
            .map(|layer_idx| {
                if sliding_window.is_some() && layer_idx >= max_window_layers {
                    AttentionType::SlidingAttention
                } else {
                    AttentionType::FullAttention
                }
            })
            .collect()
    });
    if layer_types.len() != num_hidden_layers {
        candle_core::bail!(
            "Qwen2-VL layer_types has {} entries for {} layers",
            layer_types.len(),
            num_hidden_layers
        );
    }
    layer_types
        .into_iter()
        .map(|layer_type| match layer_type {
            AttentionType::FullAttention => Ok(None),
            AttentionType::SlidingAttention => sliding_window.map(Some).ok_or_else(|| {
                candle_core::Error::msg(
                    "Qwen2-VL sliding_attention requires use_sliding_window and sliding_window",
                )
            }),
        })
        .collect()
}

impl Config {
    pub(super) fn layer_sliding_windows(&self) -> candle_core::Result<Vec<Option<usize>>> {
        resolve_layer_sliding_windows(
            self.num_hidden_layers,
            self.use_sliding_window,
            self.sliding_window,
            self.max_window_layers,
            self.layer_types.clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_layer_windows_match_transformers_semantics() -> candle_core::Result<()> {
        assert_eq!(
            resolve_layer_sliding_windows(4, true, Some(128), 2, None)?,
            vec![None, None, Some(128), Some(128)]
        );
        assert_eq!(
            resolve_layer_sliding_windows(4, false, Some(128), 2, None)?,
            vec![None; 4]
        );
        Ok(())
    }

    #[test]
    fn explicit_layer_types_are_validated() {
        assert!(resolve_layer_sliding_windows(
            2,
            true,
            Some(128),
            2,
            Some(vec![AttentionType::SlidingAttention]),
        )
        .is_err());
        assert!(resolve_layer_sliding_windows(
            1,
            false,
            Some(128),
            0,
            Some(vec![AttentionType::SlidingAttention]),
        )
        .is_err());
    }

    #[test]
    fn vision_config_accepts_transformers_in_chans() {
        let config: VisionConfig = serde_json::from_str(
            r#"{
                "depth": 2,
                "embed_dim": 8,
                "hidden_size": 16,
                "mlp_ratio": 4.0,
                "num_heads": 2,
                "in_chans": 5,
                "patch_size": 14,
                "spatial_merge_size": 2,
                "temporal_patch_size": 2
            }"#,
        )
        .unwrap();
        assert_eq!(config.in_channels, 5);
    }
}
