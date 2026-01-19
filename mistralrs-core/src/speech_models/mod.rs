mod bs1770;
mod dia;
pub mod utils;
pub mod vibevoice;

use std::{str::FromStr, sync::Arc};

pub use dia::{DiaConfig, DiaPipeline};
use serde::Deserialize;
pub use vibevoice::{VibeVoiceConfig, VibeVoiceGenerationConfig, VibeVoicePipeline};

#[derive(Clone, Copy, Debug, Deserialize, PartialEq)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
    #[serde(rename = "vibevoice")]
    VibeVoice,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            "vibevoice" => Ok(Self::VibeVoice),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`, `vibevoice`."
            )),
        }
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
    VibeVoice {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
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
            SpeechLoaderType::VibeVoice => Self::VibeVoice {
                max_tokens: None,
                cfg_scale: 3.0,
                temperature: 1.0,
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
