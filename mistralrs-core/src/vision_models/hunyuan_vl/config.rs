use mistralrs_quant::QuantizedConfig;

use crate::{layers::Activation, serde_default_fn};

serde_default_fn!(bool, default_attention_bias, false);
serde_default_fn!(bool, default_tie_word_embeddings, true);
serde_default_fn!(usize, default_head_dim, 128);
serde_default_fn!(f64, default_rope_theta, 10_000.0);
serde_default_fn!(usize, default_image_start_token_id, 120_118);
serde_default_fn!(usize, default_image_end_token_id, 120_119);
serde_default_fn!(usize, default_image_token_id, 120_120);
serde_default_fn!(usize, default_image_newline_token_id, 120_121);
serde_default_fn!(usize, default_max_image_size, 2048);
serde_default_fn!(usize, default_min_image_size, 512);
serde_default_fn!(usize, default_img_max_token_num, 16_384);
serde_default_fn!(usize, default_max_vit_seq_len, 65_536);
serde_default_fn!(usize, default_temporal_patch_size, 1);

// HF normalizes `xdrope` to `dynamic` and aliases `xdrope_section` to `mrope_section`.
#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(default)]
    pub rope_theta: Option<f64>,
    #[serde(default, alias = "mrope_section")]
    pub xdrope_section: Vec<usize>,
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default, rename = "type")]
    pub ty: Option<String>,
}

impl RopeParameters {
    /// Official configs normalize the on-disk `xdrope` alias to `dynamic` before use.
    pub fn resolved_rope_type(&self) -> &str {
        match self
            .rope_type
            .as_deref()
            .or(self.ty.as_deref())
            .unwrap_or("default")
        {
            "xdrope" => "dynamic",
            rope_type => rope_type,
        }
    }

    pub fn theta(&self) -> f64 {
        self.rope_theta.unwrap_or_else(default_rope_theta)
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    pub rms_norm_eps: f64,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_attention_bias")]
    pub attention_bias: bool,
    pub rope_parameters: RopeParameters,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    const HUNYUAN_OCR_1_5: &str = r#"{
        "architectures": ["HunYuanVLForConditionalGeneration"],
        "model_type": "hunyuan_vl",
        "tie_word_embeddings": true,
        "image_token_id": 120120,
        "image_start_token_id": 120118,
        "image_end_token_id": 120119,
        "image_newline_token_id": 120121,
        "text_config": {
            "model_type": "hunyuan_vl_text",
            "attention_bias": false,
            "head_dim": 128,
            "hidden_act": "silu",
            "hidden_size": 1024,
            "intermediate_size": 3584,
            "max_position_embeddings": 131072,
            "num_attention_heads": 16,
            "num_hidden_layers": 24,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": true,
            "vocab_size": 120818,
            "rope_parameters": {
                "alpha": 1000.0,
                "rope_theta": 10000.0,
                "rope_type": "xdrope",
                "type": "xdrope",
                "xdrope_section": [16, 16, 16, 16]
            }
        },
        "vision_config": {
            "model_type": "hunyuan_vl_vision",
            "hidden_act": "gelu",
            "hidden_size": 1152,
            "intermediate_size": 4304,
            "max_image_size": 2048,
            "min_image_size": 512,
            "img_max_token_num": 16384,
            "max_vit_seq_len": 65536,
            "temporal_patch_size": 1,
            "num_attention_heads": 16,
            "num_key_value_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 27,
            "out_hidden_size": 1024,
            "patch_size": 16,
            "rms_norm_eps": 1e-05,
            "spatial_merge_size": 2,
            "text_hidden_size": 1024
        }
    }"#;

    #[test]
    fn parses_checkpoint_nested_text_and_rope_parameters() {
        let cfg: Config = serde_json::from_str(HUNYUAN_OCR_1_5).unwrap();
        assert_eq!(cfg.text_config.vocab_size, 120_818);
        assert_eq!(cfg.text_config.hidden_size, 1024);
        assert_eq!(cfg.text_config.max_position_embeddings, 131_072);
        assert_eq!(cfg.text_config.head_dim, 128);
        assert_eq!(cfg.text_config.num_key_value_heads, 8);
        assert_eq!(
            cfg.text_config.rope_parameters.xdrope_section,
            [16, 16, 16, 16]
        );
        assert_eq!(
            cfg.text_config.rope_parameters.resolved_rope_type(),
            "dynamic"
        );
        assert_eq!(cfg.text_config.rope_parameters.theta(), 10_000.0);
        assert_eq!(cfg.text_config.rope_parameters.alpha, Some(1000.0));
        assert!(cfg.tie_word_embeddings);
        assert_eq!(cfg.vision_config.img_max_token_num, 16_384);
        assert_eq!(cfg.vision_config.max_vit_seq_len, 65_536);
        assert_eq!(cfg.vision_config.max_image_size, 2048);
        assert_eq!(cfg.vision_config.min_image_size, 512);
        assert_eq!(cfg.vision_config.temporal_patch_size, 1);
        assert_eq!(cfg.vision_config.num_key_value_heads, Some(16));
    }

    #[test]
    fn rope_parameters_accept_the_mrope_section_alias() {
        let cfg: Config = serde_json::from_str(HUNYUAN_OCR_1_5).unwrap();
        let rope_parameters: RopeParameters = serde_json::from_str(
            r#"{"rope_theta": 500000.0, "type": "xdrope", "mrope_section": [8, 8, 8, 8]}"#,
        )
        .unwrap();
        assert_eq!(rope_parameters.xdrope_section, [8, 8, 8, 8]);
        assert_eq!(rope_parameters.resolved_rope_type(), "dynamic");
        assert_eq!(rope_parameters.theta(), 500_000.0);
        assert_eq!(cfg.text_config.rope_parameters.xdrope_section.len(), 4);
    }

    #[test]
    fn flat_pre_1_5_configuration_is_rejected() {
        let err = serde_json::from_str::<Config>(
            r#"{"model_type": "hunyuan_vl", "vocab_size": 120818, "hidden_size": 1024}"#,
        )
        .unwrap_err();
        assert!(err.to_string().contains("text_config"), "{err}");
    }
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct VisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: Activation,
    #[serde(default)]
    pub attention_dropout: f64,
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,
    #[serde(default)]
    pub interpolate_mode: Option<String>,
    #[serde(default = "default_max_image_size")]
    pub max_image_size: usize,
    #[serde(default = "default_min_image_size")]
    pub min_image_size: usize,
    #[serde(default = "default_img_max_token_num")]
    pub img_max_token_num: usize,
    #[serde(default = "default_max_vit_seq_len")]
    pub max_vit_seq_len: usize,
    #[serde(default = "default_temporal_patch_size")]
    pub temporal_patch_size: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Config {
    pub text_config: TextConfig,
    pub vision_config: VisionConfig,
    #[serde(default = "default_tie_word_embeddings")]
    pub tie_word_embeddings: bool,
    #[serde(default = "default_image_start_token_id")]
    pub image_start_token_id: usize,
    #[serde(default = "default_image_end_token_id")]
    pub image_end_token_id: usize,
    #[serde(default = "default_image_token_id")]
    pub image_token_id: usize,
    #[serde(default = "default_image_newline_token_id")]
    pub image_newline_token_id: usize,
    #[serde(default)]
    pub quantization_config: Option<QuantizedConfig>,
}
