use std::collections::HashMap;

use anyhow::{bail, ensure, Context, Result};
use candle_core::quantized::gguf_file::Value;
use mistralrs_quant::GgufArchive;
use serde_json::json;

use super::{
    lfm3_audio_config::audio_encoder_config,
    lfm3_vl_config::{metadata_f64, prepare_lfm3_vl_config},
    multimodal_binding_utils::{metadata_string, TensorInventory},
};
use crate::vision_models::lfm3_d1::config::Config;

const DECISION_TYPE: &str = "lfm3.decision.type";
const CAUSAL_ATTENTION: &str = "lfm3.attention.causal";
const DECISION_CODES: usize = 2;
const DEFAULT_SET_SIZE: usize = 16;
const DEFAULT_VISION_MAX_PIXELS: usize = 1_638_400;

pub(crate) fn decision_model_type(
    metadata: &HashMap<String, Value>,
) -> Result<Option<&'static str>> {
    if !matches!(metadata.get("general.architecture"), Some(Value::String(architecture)) if architecture.eq_ignore_ascii_case("lfm3"))
    {
        return Ok(None);
    }
    let model_type = match metadata.get(DECISION_TYPE) {
        None => return Ok(None),
        Some(Value::String(kind)) => match kind.as_str() {
            "lfm3-d1" => "lfm3_vl",
            "lfm3-d1-omni" => "lfm3_omni",
            _ => bail!("Unsupported LFM3 GGUF decision type `{kind}`"),
        },
        Some(_) => bail!("GGUF metadata `{DECISION_TYPE}` must be a string"),
    };
    ensure!(
        matches!(metadata.get(CAUSAL_ATTENTION), Some(Value::Bool(false))),
        "LFM3 d1 GGUF requires `{CAUSAL_ATTENTION}=false`"
    );
    Ok(Some(model_type))
}

pub(crate) fn prepare_lfm3_d1_config(
    external_config: Option<&str>,
    archive: &GgufArchive,
) -> Result<String> {
    let model_type = decision_model_type(archive.metadata())?
        .context("LFM3 d1 GGUF requires decision metadata")?;
    let audio = model_type == "lfm3_omni";
    if let Some(external) = external_config {
        let value: serde_json::Value = serde_json::from_str(external)?;
        ensure!(
            value["architectures"]
                .as_array()
                .is_some_and(|architectures| architectures.iter().any(|name| name == "D1Model"))
                && value["model_type"].as_str() == Some(model_type),
            "LFM3 d1 GGUF requires the matching original D1Model configuration"
        );
    }
    let mut config = prepare_lfm3_vl_config(external_config, archive)?;
    if external_config.is_none() {
        let noul = metadata_f64(archive, "lfm3.decision.temperature.noul")?;
        let choice = metadata_f64(archive, "lfm3.decision.temperature.choice")?;
        let score = metadata_f64(archive, "lfm3.decision.temperature.score")?;
        ensure!(
            [noul, choice, score]
                .iter()
                .all(|temperature| temperature.is_finite() && *temperature > 0.0),
            "LFM3 d1 GGUF decision temperatures must be finite and positive"
        );
        let encoder = audio
            .then(|| audio_encoder_config(archive, &TensorInventory::from_archive(archive)))
            .transpose()?;
        let mut value: serde_json::Value = serde_json::from_str(&config)?;
        value.as_object_mut().unwrap().extend(
            json!({
                "architectures": ["D1Model"],
                "model_type": model_type,
                "audio_encoder_config": encoder,
                "d1_attention": "set",
                "d1_codes": DECISION_CODES,
                "d1_numbers": true,
                "d1_set_size": DEFAULT_SET_SIZE,
                "d1_vision_max_pixels": DEFAULT_VISION_MAX_PIXELS,
                "d1_calibration_complete": false,
                "d1_calibration": {
                    "default": score,
                    "temperatures": {
                        "noul": noul,
                        "choice:small": choice,
                        "choice:mid": choice,
                        "choice:large": choice,
                        "choice:inventory": choice,
                    },
                },
            })
            .as_object()
            .unwrap()
            .clone(),
        );
        config = serde_json::to_string(&value)?;
    }
    let parsed: Config = serde_json::from_str(&config)
        .context("LFM3 d1 GGUF configuration is incompatible with the native loader")?;
    ensure!(
        parsed.audio_encoder_config.is_some() == audio
            && metadata_string(archive, "clip.audio.projector_type")? == audio.then_some("lfm3a"),
        "LFM3 d1 GGUF audio components disagree with its decision type"
    );
    Ok(config)
}
