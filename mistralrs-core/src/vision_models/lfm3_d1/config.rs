use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use super::calibration::DecisionCalibration;
use crate::{
    paged_attention::ModelConfigLike,
    vision_models::{lfm3_omni::audio::AudioEncoderConfig, lfm3_vl},
};

pub(crate) const CALIBRATION_FILENAME: &str = "calibration.json";

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub vision: lfm3_vl::Config,
    pub audio_encoder_config: Option<AudioEncoderConfig>,
    pub d1_vision_max_pixels: usize,
    #[serde(flatten)]
    pub(super) decision: DecisionConfig,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub(super) struct DecisionConfig {
    pub d1_attention: String,
    pub d1_codes: usize,
    pub d1_numbers: bool,
    pub d1_set_size: usize,
    #[serde(default = "default_prompt")]
    pub d1_prompt: usize,
    #[serde(default = "default_edges")]
    pub d1_box_edges: String,
    pub d1_calibration: DecisionCalibration,
    #[serde(default = "default_calibration_complete")]
    pub d1_calibration_complete: bool,
}

fn default_prompt() -> usize {
    1
}

fn default_edges() -> String {
    "median".to_owned()
}

fn default_calibration_complete() -> bool {
    true
}

pub(crate) fn inject_calibration(config: &mut serde_json::Value, config_path: &Path) -> Result<()> {
    if config.get("d1_calibration").is_none() {
        let path = config_path.with_file_name(CALIBRATION_FILENAME);
        config["d1_calibration"] = serde_json::from_str(
            &std::fs::read_to_string(&path)
                .with_context(|| format!("Cannot load d1 calibration {}", path.display()))?,
        )?;
    }
    let calibration: DecisionCalibration =
        serde_json::from_value(config["d1_calibration"].clone())?;
    calibration.validate()
}

pub(crate) struct ModelConfig<T>(pub T);

impl<T: std::ops::Deref> ModelConfigLike for ModelConfig<T>
where
    T::Target: ModelConfigLike,
{
    fn max_seq_len(&self) -> usize {
        self.0.max_seq_len()
    }
    fn num_layers(&self) -> usize {
        self.0.num_layers()
    }
    fn hidden_size(&self) -> usize {
        self.0.hidden_size()
    }
    fn num_kv_heads(&self) -> usize {
        self.0.num_kv_heads()
    }
    fn num_attn_heads(&self) -> usize {
        self.0.num_attn_heads()
    }
    fn k_head_dim(&self) -> usize {
        self.0.k_head_dim()
    }
    fn v_head_dim(&self) -> usize {
        self.0.v_head_dim()
    }
    fn layer_has_paged_kv_cache(&self, _layer: usize) -> bool {
        false
    }
}
