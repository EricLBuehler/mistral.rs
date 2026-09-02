use mistralrs_quant::QuantizedConfig;

use crate::gdn::{GdnStateDType, GdnVHeadLayout};
use crate::layers::{Activation, YarnRopeConfig};
use crate::serde_default_fn;

// Re-export vision config from qwen3_vl
pub use crate::vision_models::qwen3_vl::config::VisionConfig;

serde_default_fn!(usize, default_full_attn_interval, 4);
serde_default_fn!(usize, default_conv_kernel, 4);
serde_default_fn!(f64, default_partial_rotary_factor, 0.25);
serde_default_fn!(f64, default_rope_theta, 10_000_000.0);
serde_default_fn!(f64, default_yarn_beta_fast, 32.0);
serde_default_fn!(f64, default_yarn_beta_slow, 1.0);
serde_default_fn!(bool, default_true, true);

#[derive(Debug, Default, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RopeType {
    #[default]
    #[serde(alias = "mrope")]
    Default,
    Yarn,
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    FullAttention,
    LinearAttention,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct RopeParameters {
    #[serde(default, alias = "type")]
    pub rope_type: RopeType,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_partial_rotary_factor")]
    pub partial_rotary_factor: f64,
    #[serde(default = "default_true")]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<usize>,
    #[serde(default = "default_yarn_beta_fast")]
    pub beta_fast: f64,
    #[serde(default = "default_yarn_beta_slow")]
    pub beta_slow: f64,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
    #[serde(default)]
    pub attention_factor: Option<f64>,
    #[serde(default = "default_true")]
    pub truncate: bool,
}

impl RopeParameters {
    pub(crate) fn supported_max_position_embeddings(
        &self,
        declared_max_position_embeddings: usize,
    ) -> candle_core::Result<usize> {
        if self.rope_type != RopeType::Yarn {
            return Ok(declared_max_position_embeddings);
        }
        self.validate_scaling(declared_max_position_embeddings)?;
        let supported = self
            .original_max_position_embeddings
            .expect("validated YaRN original context length") as f64
            * self.factor.expect("validated YaRN factor");
        if supported > usize::MAX as f64 {
            candle_core::bail!("Qwen3.5 YaRN context length overflows usize");
        }
        Ok(supported.floor() as usize)
    }

    pub(crate) fn validate_scaling(
        &self,
        max_position_embeddings: usize,
    ) -> candle_core::Result<()> {
        if self.rope_type != RopeType::Yarn {
            return Ok(());
        }
        if !self.truncate {
            candle_core::bail!("Qwen3.5 YaRN truncate=false is not supported");
        }
        let factor = self
            .factor
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 YaRN requires a factor"))?;
        if !factor.is_finite() || factor < 1.0 {
            candle_core::bail!("Qwen3.5 YaRN factor must be finite and at least 1");
        }
        let original = self.original_max_position_embeddings.ok_or_else(|| {
            candle_core::Error::msg("Qwen3.5 YaRN requires original_max_position_embeddings")
        })?;
        if original == 0 {
            candle_core::bail!("Qwen3.5 YaRN original context length must be positive");
        }
        let supported = original as f64 * factor;
        if max_position_embeddings as f64 > supported {
            candle_core::bail!(
                "Qwen3.5 maximum position embeddings {max_position_embeddings} exceed the YaRN limit {supported:.0}"
            );
        }
        if !self.beta_fast.is_finite()
            || self.beta_fast <= 0.0
            || !self.beta_slow.is_finite()
            || self.beta_slow <= 0.0
        {
            candle_core::bail!("Qwen3.5 YaRN beta values must be finite and positive");
        }
        if self.beta_fast < self.beta_slow {
            candle_core::bail!("Qwen3.5 YaRN beta_fast must be at least beta_slow");
        }
        if self
            .mscale
            .is_some_and(|mscale| !mscale.is_finite() || mscale < 0.0)
            || self
                .mscale_all_dim
                .is_some_and(|mscale| !mscale.is_finite() || mscale < 0.0)
        {
            candle_core::bail!("Qwen3.5 YaRN mscale values must be finite and non-negative");
        }
        if self
            .attention_factor
            .is_some_and(|factor| !factor.is_finite() || factor <= 0.0)
        {
            candle_core::bail!("Qwen3.5 YaRN attention factor must be finite and positive");
        }
        Ok(())
    }

    pub(crate) fn yarn_rope_config(
        &self,
        max_position_embeddings: usize,
        head_dim: usize,
    ) -> candle_core::Result<Option<YarnRopeConfig>> {
        self.validate_scaling(max_position_embeddings)?;
        if self.rope_type != RopeType::Yarn {
            return Ok(None);
        }
        Ok(Some(YarnRopeConfig {
            base: self.rope_theta as f32,
            head_dim,
            max_position_embeddings,
            original_max_position_embeddings: self
                .original_max_position_embeddings
                .expect("validated YaRN original context length"),
            factor: self.factor.expect("validated YaRN factor") as f32,
            beta_fast: self.beta_fast as f32,
            beta_slow: self.beta_slow as f32,
            mscale: match (self.mscale, self.mscale_all_dim) {
                (Some(mscale), Some(mscale_all_dim)) if mscale != 0.0 && mscale_all_dim != 0.0 => {
                    mscale as f32
                }
                _ => 1.0,
            },
            mscale_all_dim: match (self.mscale, self.mscale_all_dim) {
                (Some(mscale), Some(mscale_all_dim)) if mscale != 0.0 && mscale_all_dim != 0.0 => {
                    mscale_all_dim as f32
                }
                _ => 0.0,
            },
            attention_factor: self.attention_factor.map(|x| x as f32),
        }))
    }
}

pub(crate) fn apply_max_model_len(
    config: &str,
    max_model_len: usize,
) -> candle_core::Result<String> {
    if max_model_len == 0 {
        candle_core::bail!("Qwen3.5 max_model_len must be positive");
    }
    let mut config: serde_json::Value =
        serde_json::from_str(config).map_err(|err| candle_core::Error::msg(err.to_string()))?;
    let root = config
        .as_object_mut()
        .ok_or_else(|| candle_core::Error::msg("Qwen3.5 config must be a JSON object"))?;
    let text_config = if root.contains_key("text_config") {
        root.get_mut("text_config")
            .expect("checked text_config presence")
            .as_object_mut()
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 text_config must be a JSON object"))?
    } else {
        root
    };
    let declared_max = text_config
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| {
            candle_core::Error::msg("Qwen3.5 max_position_embeddings must be a positive integer")
        })?;
    let rope_parameters: RopeParameters = serde_json::from_value(
        text_config
            .get("rope_parameters")
            .ok_or_else(|| candle_core::Error::msg("Qwen3.5 rope_parameters are required"))?
            .clone(),
    )
    .map_err(|err| candle_core::Error::msg(err.to_string()))?;
    let supported = rope_parameters.supported_max_position_embeddings(declared_max)?;
    if max_model_len > supported {
        candle_core::bail!(
            "Qwen3.5 max_model_len {max_model_len} exceeds the model-supported context length {supported}"
        );
    }
    text_config.insert(
        "max_position_embeddings".to_string(),
        serde_json::Value::from(max_model_len),
    );
    serde_json::to_string(&config).map_err(|err| candle_core::Error::msg(err.to_string()))
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
    #[serde(default)]
    pub mamba_ssm_dtype: GdnStateDType,
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
        self.validate_rope_scaling()?;
        Ok(())
    }

    fn validate_rope_scaling(&self) -> candle_core::Result<()> {
        self.rope_parameters
            .validate_scaling(self.max_position_embeddings)
    }

    pub fn yarn_rope_config(&self) -> candle_core::Result<Option<YarnRopeConfig>> {
        self.rope_parameters
            .yarn_rope_config(self.max_position_embeddings, self.rot_dim())
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

    fn text_config(
        max_position_embeddings: usize,
        rope_parameters: serde_json::Value,
    ) -> TextConfig {
        serde_json::from_value(serde_json::json!({
            "head_dim": 64,
            "vocab_size": 32,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 8,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "hidden_act": "silu",
            "max_position_embeddings": max_position_embeddings,
            "rms_norm_eps": 1e-6,
            "rope_parameters": rope_parameters,
            "linear_key_head_dim": 16,
            "linear_value_head_dim": 16,
            "linear_num_key_heads": 2,
            "linear_num_value_heads": 4
        }))
        .unwrap()
    }

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

    #[test]
    fn native_rope_remains_unscaled() {
        let cfg = text_config(
            262_144,
            serde_json::json!({
                "rope_type": "default",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [4, 2, 2]
            }),
        );
        cfg.validate().unwrap();
        assert!(cfg.yarn_rope_config().unwrap().is_none());
        assert_eq!(
            cfg.rope_parameters
                .supported_max_position_embeddings(cfg.max_position_embeddings)
                .unwrap(),
            262_144
        );
    }

    #[test]
    fn legacy_mrope_type_is_canonicalized_to_default() {
        let cfg = text_config(
            262_144,
            serde_json::json!({
                "rope_type": "mrope",
                "mrope_section": [4, 2, 2]
            }),
        );
        cfg.validate().unwrap();
        assert_eq!(cfg.rope_parameters.rope_type, RopeType::Default);
        assert!(cfg.yarn_rope_config().unwrap().is_none());
    }

    #[test]
    fn yarn_factor_four_accepts_qwen_ultra_long_limit() {
        let cfg = text_config(
            1_010_000,
            serde_json::json!({
                "rope_type": "yarn",
                "rope_theta": 10_000_000,
                "partial_rotary_factor": 0.25,
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }),
        );
        cfg.validate().unwrap();
        let yarn = cfg.yarn_rope_config().unwrap().unwrap();
        assert_eq!(yarn.original_max_position_embeddings, 262_144);
        assert_eq!(yarn.factor, 4.0);
        assert_eq!(yarn.max_position_embeddings, 1_010_000);
        assert!((yarn.mscale - 1.0).abs() < f32::EPSILON);
        assert!((yarn.mscale_all_dim - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn yarn_rejects_length_above_derived_limit() {
        let cfg = text_config(
            1_048_577,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }),
        );
        let error = cfg.validate().unwrap_err().to_string();
        assert!(error.contains("exceed the YaRN limit"));
    }

    #[test]
    fn yarn_rejects_unimplemented_non_truncating_correction_range() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "truncate": false
            }),
        );
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("truncate=false"));
    }

    #[test]
    fn yarn_rejects_reversed_beta_range() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "beta_fast": 1.0,
                "beta_slow": 32.0
            }),
        );
        assert!(cfg
            .validate()
            .unwrap_err()
            .to_string()
            .contains("beta_fast"));
    }

    #[test]
    fn yarn_zero_mscale_uses_hf_attention_default() {
        let cfg = text_config(
            524_288,
            serde_json::json!({
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 2.0,
                "original_max_position_embeddings": 262144,
                "mscale": 0.0,
                "mscale_all_dim": 1.0
            }),
        );
        let yarn = cfg.yarn_rope_config().unwrap().unwrap();
        assert_eq!(yarn.mscale, 1.0);
        assert_eq!(yarn.mscale_all_dim, 0.0);
    }

    #[test]
    fn runtime_limit_handles_nested_and_text_configs() {
        let text = serde_json::json!({
            "max_position_embeddings": 262144,
            "rope_parameters": {
                "rope_type": "yarn",
                "mrope_section": [4, 2, 2],
                "factor": 4.0,
                "original_max_position_embeddings": 262144
            }
        });
        let nested = apply_max_model_len(
            &serde_json::json!({ "text_config": text.clone(), "vision_config": {} }).to_string(),
            1_010_000,
        )
        .unwrap();
        let nested: serde_json::Value = serde_json::from_str(&nested).unwrap();
        assert_eq!(nested["text_config"]["max_position_embeddings"], 1_010_000);
        assert_eq!(nested["vision_config"], serde_json::json!({}));

        let standalone = apply_max_model_len(&text.to_string(), 524_288).unwrap();
        let standalone: serde_json::Value = serde_json::from_str(&standalone).unwrap();
        assert_eq!(standalone["max_position_embeddings"], 524_288);
        assert!(apply_max_model_len(&text.to_string(), 1_048_577).is_err());
    }
}
