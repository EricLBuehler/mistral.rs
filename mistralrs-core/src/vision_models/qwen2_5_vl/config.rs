// https://github.com/huggingface/transformers/blob/f2c388e3f946862f657acc1e21b272ec946fc66c/src/transformers/models/qwen2_vl/configuration_qwen2_vl.py

use mistralrs_quant::QuantizedConfig;

use crate::layers::Activation;

use crate::serde_default_fn;

serde_default_fn!(Activation, default_vision_hidden_act, Activation::QuickGelu);
serde_default_fn!(usize, default_in_channels, 3);
serde_default_fn!(usize, default_depth, 32);
serde_default_fn!(usize, default_hidden_size, 3584);
serde_default_fn!(usize, default_out_hidden_size, 3584);
serde_default_fn!(usize, default_intermediate_size, 3420);
serde_default_fn!(usize, default_num_heads, 16);
serde_default_fn!(usize, default_patch_size, 14);
serde_default_fn!(usize, default_spatial_merge_size, 2);
serde_default_fn!(usize, default_temporal_patch_size, 2);
serde_default_fn!(usize, default_window_size, 112);
serde_default_fn!(usize, default_max_window_layers, 28);
serde_default_fn!(
    Vec<usize>,
    default_fullatt_block_indexes,
    vec![7, 15, 23, 31]
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AttentionType {
    FullAttention,
    SlidingAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    #[serde(default = "default_depth")]
    pub depth: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_out_hidden_size")]
    pub out_hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_chans: usize,
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_spatial_merge_size")]
    pub spatial_merge_size: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
    #[serde(default = "default_window_size")]
    pub window_size: usize,
    #[serde(default = "default_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,
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
            "Qwen2.5-VL layer_types has {} entries for {} layers",
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
                    "Qwen2.5-VL sliding_attention requires use_sliding_window and sliding_window",
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
            resolve_layer_sliding_windows(5, true, Some(256), 3, None)?,
            vec![None, None, None, Some(256), Some(256)]
        );
        assert_eq!(
            resolve_layer_sliding_windows(5, false, Some(256), 3, None)?,
            vec![None; 5]
        );
        Ok(())
    }

    #[test]
    fn explicit_layer_types_are_validated() {
        assert!(resolve_layer_sliding_windows(
            2,
            true,
            Some(256),
            2,
            Some(vec![AttentionType::FullAttention]),
        )
        .is_err());
        assert!(resolve_layer_sliding_windows(
            1,
            true,
            None,
            0,
            Some(vec![AttentionType::SlidingAttention]),
        )
        .is_err());
    }
}
