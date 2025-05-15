mod bs1770;
mod dia;
pub mod utils;

use std::str::FromStr;

pub use dia::{DiaConfig, DiaPipeline};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub enum SpeechLoaderType {
    #[serde(rename = "dia")]
    Dia,
}

impl FromStr for SpeechLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "dia" => Ok(Self::Dia),
            a => Err(format!(
                "Unknown architecture `{a}`. Possible architectures: `dia`."
            )),
        }
    }
}

pub enum SpeechGenerationConfig {
    Dia {
        max_tokens: Option<usize>,
        cfg_scale: f32,
        temperature: f32,
        top_p: f32,
        too_k: Option<usize>,
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
                too_k: Some(35),
            },
        }
    }
}
