// Model + inference code ported near-verbatim from the pocket-tts Rust crate
// (https://github.com/babybirdprd/pocket-tts), compiled against candle 0.11 with `crate::` paths
// rewritten. The raw-audio voice-cloning path (audio encoder / pause helpers) is retained but
// unused here, so allow the dead code.
#![allow(dead_code)]

mod conditioners;
mod config;
mod models;
mod modules;
mod pause;
mod tts_model;
mod voice_state;

use std::path::Path;
use std::sync::Arc;

use candle_core::Device;
use candle_nn::VarBuilder;

pub use config::PocketTtsConfig;
use tts_model::TTSModel;
use voice_state::ModelState;

use super::{SpeechGenerationConfig, SpeechGenerationOutput};

pub struct PocketTtsPipeline {
    model: TTSModel,
    voice_state: ModelState,
    device: Device,
}

impl PocketTtsPipeline {
    pub fn new(
        cfg: &PocketTtsConfig,
        vb: VarBuilder,
        tokenizer_path: &Path,
        voice_prompt_path: &Path,
    ) -> candle_core::Result<Self> {
        let device = vb.device().clone();
        let model = TTSModel::new(cfg, vb, tokenizer_path).map_err(candle_core::Error::wrap)?;
        let voice_state = model
            .voice_state_from_prompt_file(voice_prompt_path)
            .map_err(candle_core::Error::wrap)?;
        Ok(Self {
            model,
            voice_state,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn generate(
        &self,
        prompt: &str,
        cfg: &SpeechGenerationConfig,
    ) -> candle_core::Result<SpeechGenerationOutput> {
        let mut model = self.model.clone();
        if let SpeechGenerationConfig::PocketTts {
            temperature,
            lsd_decode_steps,
            eos_threshold,
        } = cfg
        {
            model.temp = *temperature;
            model.lsd_decode_steps = *lsd_decode_steps;
            model.eos_threshold = *eos_threshold;
        }

        let audio = model
            .generate(prompt, &self.voice_state)
            .map_err(candle_core::Error::wrap)?;
        let pcm = audio.flatten_all()?.to_vec1::<f32>()?;

        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: model.sample_rate,
            channels: self.model.mimi.channels,
        })
    }
}
