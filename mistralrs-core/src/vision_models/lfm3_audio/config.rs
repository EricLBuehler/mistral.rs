use anyhow::Result;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::{models::lfm3, vision_models::lfm3_omni::config::SpeechConfig};

const DEFAULT_CODEBOOKS: u64 = 12;
const DEFAULT_CODEBOOK_SIZE: u64 = 2017;
const DEFAULT_MASKGIT_STEPS: u64 = 3;
const DEFAULT_INTERLEAVED_TEXT: u64 = 6;
const DEFAULT_INTERLEAVED_AUDIO: u64 = 12;
const AUDIO_TOKEN_ID: u32 = 125031;
const AUDIO_OUT_TOKEN_ID: u32 = 125032;
const AUDIO_START_TOKEN_ID: u32 = 125012;
const TEXT_END_TOKEN_ID: u32 = 125014;
const MIXED_START_TOKEN_ID: u32 = 125015;
const LEGACY_SUBCONFIG_KEYS: [&str; 6] = [
    "architectures",
    "transformers_version",
    "is_causal",
    "auto_map",
    "_name_or_path",
    "dtype",
];

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct AudioConfig {
    pub text_config: lfm3::Config,
    #[serde(flatten)]
    pub speech: SpeechConfig,
    pub tie_word_embeddings: bool,
}

pub(crate) fn normalize_config(value: &mut Value) -> Result<()> {
    let object = value.as_object_mut().expect("Model config is an object");
    for (legacy, canonical) in [
        ("lfm", "text_config"),
        ("encoder", "audio_encoder_config"),
        ("depthformer", "depth_config"),
        ("codebooks", "num_codebooks"),
    ] {
        if let Some(legacy) = object.remove(legacy) {
            object.insert(canonical.to_string(), legacy);
        }
    }
    if let Some(size) = object.remove("audio_vocab_size") {
        let count = size.as_u64().unwrap_or_default();
        anyhow::ensure!(count > 0, "LFM3-Audio audio_vocab_size must be positive");
        object.insert("audio_eos_code".to_string(), (count - 1).into());
        object.insert("codebook_size".to_string(), size);
    }
    for key in ["format_version", "preprocessor", "tokenizer"] {
        object.remove(key);
    }
    for (key, default) in [
        ("num_codebooks", json!(DEFAULT_CODEBOOKS)),
        ("codebook_size", json!(DEFAULT_CODEBOOK_SIZE)),
        ("audio_eos_code", json!(DEFAULT_CODEBOOK_SIZE - 1)),
        ("interleaved_n_text", json!(DEFAULT_INTERLEAVED_TEXT)),
        ("interleaved_n_audio", json!(DEFAULT_INTERLEAVED_AUDIO)),
        ("audio_token_id", json!(AUDIO_TOKEN_ID)),
        ("audio_out_token_id", json!(AUDIO_OUT_TOKEN_ID)),
        ("audio_start_token_id", json!(AUDIO_START_TOKEN_ID)),
        ("text_end_token_id", json!(TEXT_END_TOKEN_ID)),
        ("mixed_start_token_id", json!(MIXED_START_TOKEN_ID)),
        ("tie_word_embeddings", json!(true)),
    ] {
        object.entry(key).or_insert(default);
    }
    let steps = object
        .remove("maskgit_steps")
        .and_then(|value| value.as_u64())
        .unwrap_or(DEFAULT_MASKGIT_STEPS);
    if object.get("maskgit_schedule").is_none_or(Value::is_null) {
        let codebooks = object["num_codebooks"].as_u64().unwrap_or_default();
        anyhow::ensure!(
            steps > 0 && steps <= codebooks,
            "LFM3-Audio maskgit_steps must be between 1 and num_codebooks"
        );
        let mut schedule = vec![codebooks / steps; usize::try_from(steps)?];
        *schedule.last_mut().unwrap() += codebooks % steps;
        object.insert("maskgit_schedule".to_string(), json!(schedule));
    }
    if let Some(encoder) = value["audio_encoder_config"].as_object_mut() {
        for (legacy, canonical) in [
            ("dim_in", "num_mel_bins"),
            ("dim_conv", "conv_channels"),
            ("dim_out", "hidden_size"),
            ("lfm", "text_config"),
        ] {
            if let Some(legacy) = encoder.remove(legacy) {
                encoder.insert(canonical.to_string(), legacy);
            }
        }
    }
    for path in [
        "/text_config",
        "/audio_encoder_config/text_config",
        "/depth_config",
    ] {
        if let Some(config) = value.pointer_mut(path).and_then(Value::as_object_mut) {
            for key in LEGACY_SUBCONFIG_KEYS {
                config.remove(key);
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalization_preserves_overrides_and_validates_legacy_speech_config() -> Result<()> {
        let mut value = json!({
            "lfm": {"hidden_size": 2048, "dtype": "float32"},
            "encoder": {"dim_in": 128, "dim_conv": 256, "dim_out": 768,
                "lfm": {"hidden_size": 768, "dtype": "float32"}},
            "depthformer": {"hidden_size": 1024, "is_causal": false},
            "codebooks": 12, "audio_vocab_size": 2017, "maskgit_steps": 5,
            "format_version": 1, "preprocessor": {}, "tokenizer": {}
        });
        normalize_config(&mut value)?;
        assert_eq!(value["text_config"]["hidden_size"], 2048);
        assert_eq!(value["audio_encoder_config"]["num_mel_bins"], 128);
        assert_eq!(
            value["audio_encoder_config"]["text_config"]["hidden_size"],
            768
        );
        assert_eq!(value["depth_config"]["hidden_size"], 1024);
        assert_eq!(value["maskgit_schedule"], json!([2, 2, 2, 2, 4]));
        assert_eq!(value["audio_eos_code"], 2016);
        assert_eq!(value["audio_token_id"], 125031);
        assert!(value["text_config"].get("dtype").is_none());
        assert!(value.get("lfm").is_none());
        assert!(value.get("format_version").is_none());
        let once = value.clone();
        normalize_config(&mut value)?;
        assert_eq!(value, once);
        let mut value = json!({"num_codebooks": 4, "maskgit_schedule": [1, 3],
            "audio_token_id": 42, "audio_eos_code": 15});
        normalize_config(&mut value)?;
        assert_eq!(value["maskgit_schedule"], json!([1, 3]));
        assert_eq!(value["audio_token_id"], 42);
        assert_eq!(value["audio_eos_code"], 15);
        Ok(())
    }
}
