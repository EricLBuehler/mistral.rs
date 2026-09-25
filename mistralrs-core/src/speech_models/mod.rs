mod bs1770;
mod dia;
mod pockettts;
pub mod utils;

use std::{str::FromStr, sync::Arc};

pub use dia::{DiaConfig, DiaPipeline};
pub use pockettts::{PocketTtsConfig, PocketTtsPipeline};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, strum::EnumIter)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "pockettts")]
    PocketTts,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "pockettts" => Ok(Self::PocketTts),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `pockettts`."
            )),
        }
    }
}

/// Marker file that identifies a pocket-tts repo (which ships no `config.json`).
pub const POCKETTTS_WEIGHTS_FILE: &str = "tts_b6369a24.safetensors";

impl SpeechLoaderType {
    /// Auto-detect speech loader type from a config.json string.
    /// Extend this when adding new speech pipelines.
    pub fn auto_detect_from_config(config: &str) -> Option<Self> {
        if serde_json::from_str::<DiaConfig>(config).is_ok() {
            return Some(Self::Dia);
        }
        None
    }

    /// Auto-detect speech loader type from the repo file list, for models that ship no
    /// `config.json` (e.g. pocket-tts). Extend this when adding new such pipelines.
    pub fn auto_detect_from_files(files: &[String]) -> Option<Self> {
        if files
            .iter()
            .any(|f| f.rsplit('/').next() == Some(POCKETTTS_WEIGHTS_FILE))
        {
            return Some(Self::PocketTts);
        }
        None
    }
}

#[derive(Clone, Copy, Debug)]
pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        top_k: Option<usize>,
    },
    PocketTts {
        temperature: f32,
        lsd_decode_steps: usize,
        eos_threshold: f32,
    },
}

impl SpeechGenerationConfig {
    pub fn default(ty: SpeechLoaderType) -> Self {
        match ty {
            SpeechLoaderType::Dia => Self::Dia {
                max_tokens: None,
                cfg_scale: 3.,
                temperature: 1.3,
                top_p: 0.95,
                top_k: Some(35),
            },
            SpeechLoaderType::PocketTts => Self::PocketTts {
                temperature: 0.7,
                lsd_decode_steps: 1,
                eos_threshold: -4.0,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SpeechGenerationOutput {
    pub pcm: Arc<Vec<f32>>,
    pub rate: usize,
    pub channels: usize,
}
