//! VibeVoice streaming text-to-speech implementation.
//!
//! VibeVoice is a real-time TTS model that uses:
//! - Dual Qwen2 language models (base LM + TTS LM)
//! - Diffusion head for generating acoustic latents
//! - σ-VAE acoustic decoder for converting latents to audio
//!
//! Architecture:
//! ```text
//! Text → Tokenize → Base LM (layers 0-3) → TTS LM (layers 4-23)
//!                                              ↓
//!                                    Hidden states + Type embeddings
//!                                              ↓
//!                               Diffusion Head (AdaLN + DPM-Solver)
//!                                              ↓
//!                                    Acoustic latents (64-dim @ 7.5Hz)
//!                                              ↓
//!                                    σ-VAE Decoder (3200x upsampling)
//!                                              ↓
//!                                    Audio waveform (24kHz)
//! ```

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

mod acoustic_decoder;
pub mod config;
mod diffusion_head;
mod language_model;
mod scheduler;

use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use mistralrs_quant::ShardedVarBuilder;
use rand::{Rng, SeedableRng};
use rand_isaac::Isaac64Rng;
use tokenizers::Tokenizer;
use tracing::info;

use acoustic_decoder::AcousticDecoder;
pub use config::VibeVoiceConfig;
use diffusion_head::DiffusionHead;
use language_model::{
    AcousticConnector, BaseLM, EosClassifier, LayerKvCache, TtsLM, TypeEmbeddings,
};
use scheduler::{ClassifierFreeGuidance, DpmSolverConfig, DpmSolverScheduler};

use super::{utils::normalize_loudness, SpeechGenerationOutput};

/// Sample rate for VibeVoice output audio (24kHz)
pub const SAMPLE_RATE: usize = 24000;

/// Number of audio channels (mono)
pub const CHANNELS: usize = 1;

/// Tokens per second of audio (7.5 Hz acoustic frame rate)
pub const _TOKENS_PER_SECOND: f32 = 7.5;

/// Speech tokens generated per text window
pub const SPEECH_TOKENS_PER_WINDOW: usize = 6;

/// Text window size for streaming
pub const _TEXT_WINDOW_SIZE: usize = 5;

/// VibeVoice generation configuration
#[derive(Clone, Copy, Debug)]
pub struct VibeVoiceGenerationConfig {
    /// Maximum number of speech tokens to generate
    pub max_tokens: Option<usize>,
    /// Classifier-free guidance scale (default: 3.0)
    pub cfg_scale: f32,
    /// Temperature for sampling (not used in diffusion, but for any LM sampling)
    #[allow(dead_code)]
    pub temperature: f32,
}

impl Default for VibeVoiceGenerationConfig {
    fn default() -> Self {
        Self {
            max_tokens: None,
            cfg_scale: 3.0,
            temperature: 1.0,
        }
    }
}

/// VibeVoice pipeline for text-to-speech generation
pub struct VibeVoicePipeline {
    // Model components
    base_lm: BaseLM,
    tts_lm: TtsLM,
    diffusion_head: DiffusionHead,
    acoustic_decoder: AcousticDecoder,
    #[allow(dead_code)]
    acoustic_connector: AcousticConnector,
    type_embeddings: TypeEmbeddings,
    #[allow(dead_code)]
    eos_classifier: EosClassifier,

    // Scaling factors for acoustic latents
    speech_scaling_factor: Tensor,
    speech_bias_factor: Tensor,

    // Tokenizer
    tokenizer: Arc<Tokenizer>,

    // Config
    cfg: VibeVoiceConfig,
    device: Device,
    dtype: DType,
}

impl VibeVoicePipeline {
    /// Create a new VibeVoice pipeline
    pub fn new(
        cfg: &VibeVoiceConfig,
        tokenizer: Arc<Tokenizer>,
        vb: ShardedVarBuilder,
    ) -> Result<Self> {
        let device = vb.device().clone();
        let dtype = vb.dtype();

        // Load model components
        let base_lm = BaseLM::new(cfg, vb.pp("model").pp("language_model"))?;
        let tts_lm = TtsLM::new(cfg, vb.pp("model").pp("tts_language_model"))?;

        let diffusion_head = DiffusionHead::new(
            &cfg.diffusion_head_config,
            vb.pp("model").pp("prediction_head"),
        )?;

        let acoustic_decoder = AcousticDecoder::new(
            &cfg.acoustic_tokenizer_config,
            vb.pp("model").pp("acoustic_tokenizer").pp("decoder"),
        )?;

        let acoustic_connector = AcousticConnector::new(
            cfg.acoustic_vae_dim,
            cfg.decoder_config.hidden_size,
            cfg.decoder_config.rms_norm_eps,
            vb.pp("model").pp("acoustic_connector"),
        )?;

        let type_embeddings = TypeEmbeddings::new(
            cfg.decoder_config.hidden_size,
            vb.pp("model").pp("tts_input_types"),
        )?;

        let eos_classifier =
            EosClassifier::new(cfg.decoder_config.hidden_size, vb.pp("tts_eos_classifier"))?;

        // Load scaling factors
        let speech_scaling_factor = vb.get((), "model.speech_scaling_factor")?;
        let speech_bias_factor = vb.get((), "model.speech_bias_factor")?;

        Ok(Self {
            base_lm,
            tts_lm,
            diffusion_head,
            acoustic_decoder,
            acoustic_connector,
            type_embeddings,
            eos_classifier,
            speech_scaling_factor,
            speech_bias_factor,
            tokenizer,
            cfg: cfg.clone(),
            device,
            dtype,
        })
    }

    /// Tokenize input text
    fn tokenize(&self, text: &str) -> Result<Tensor> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| candle_core::Error::Msg(format!("Tokenization error: {}", e)))?;

        let ids: Vec<u32> = encoding.get_ids().to_vec();
        Tensor::new(ids, &self.device)?.unsqueeze(0)
    }

    /// Sample speech tokens using diffusion
    fn sample_speech_tokens(
        &self,
        condition: &Tensor,
        uncond_condition: &Tensor,
        scheduler: &mut DpmSolverScheduler,
        cfg_guidance: &ClassifierFreeGuidance,
        rng: &mut Isaac64Rng,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = condition.dims3()?;

        // Initialize with random noise
        let shape = (batch_size, seq_len, self.cfg.acoustic_vae_dim);
        let noise: Vec<f32> = (0..shape.0 * shape.1 * shape.2)
            .map(|_| rng.random::<f32>() * 2.0 - 1.0) // Simple uniform noise, could use normal
            .collect();
        let mut sample = Tensor::from_vec(noise, shape, &self.device)?.to_dtype(self.dtype)?;

        // Reset scheduler state
        scheduler.reset();

        // Diffusion sampling loop
        let timesteps = scheduler.timesteps().to_vec();
        for (step_idx, &timestep) in timesteps.iter().enumerate() {
            let t =
                Tensor::full(timestep as f32, batch_size, &self.device)?.to_dtype(self.dtype)?;

            // Conditional prediction
            let cond_pred = self.diffusion_head.forward(&sample, &t, condition)?;

            // Unconditional prediction
            let uncond_pred = self.diffusion_head.forward(&sample, &t, uncond_condition)?;

            // Apply classifier-free guidance
            let model_output = cfg_guidance.apply(&cond_pred, &uncond_pred)?;

            // Scheduler step
            sample = scheduler.step(&model_output, &sample, timestep, step_idx)?;
        }

        // Apply scaling and bias
        let sample = sample
            .broadcast_mul(&self.speech_scaling_factor)?
            .broadcast_add(&self.speech_bias_factor)?;

        Ok(sample)
    }

    /// Generate speech from text
    pub fn generate(
        &self,
        text: &str,
        gen_cfg: &VibeVoiceGenerationConfig,
    ) -> Result<SpeechGenerationOutput> {
        let mut rng = Isaac64Rng::seed_from_u64(42);

        // Tokenize input
        let input_ids = self.tokenize(text)?;
        let seq_len = input_ids.dim(1)?;

        info!("Generating speech for {} tokens", seq_len);

        // Initialize scheduler
        let scheduler_cfg = DpmSolverConfig {
            num_train_timesteps: self.cfg.diffusion_head_config.ddpm_num_steps,
            num_inference_steps: self.cfg.diffusion_head_config.ddpm_num_inference_steps,
            beta_schedule: self.cfg.diffusion_head_config.ddpm_beta_schedule.clone(),
            prediction_type: self.cfg.diffusion_head_config.prediction_type.clone(),
        };
        let mut scheduler = DpmSolverScheduler::new(&scheduler_cfg);
        let cfg_guidance = ClassifierFreeGuidance::new(gen_cfg.cfg_scale);

        // Initialize KV caches
        let mut base_lm_cache = Some(LayerKvCache::new(self.base_lm.num_layers()));
        let mut tts_lm_cond_cache = Some(LayerKvCache::new(self.tts_lm.num_layers()));
        let mut tts_lm_uncond_cache = Some(LayerKvCache::new(self.tts_lm.num_layers()));

        // Process text through base LM first (get all hidden states)
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let base_hidden = self
            .base_lm
            .forward(&input_ids, &position_ids, &mut base_lm_cache)?;

        // Create type embeddings (0 = text)
        let text_type_ids = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;
        let text_type_embeds = self.type_embeddings.forward(&text_type_ids)?;

        // Process through TTS LM (conditional path)
        let tts_hidden_cond = self.tts_lm.forward(
            &base_hidden,
            Some(&text_type_embeds),
            &position_ids,
            &mut tts_lm_cond_cache,
        )?;

        // Unconditional path (zeros)
        let uncond_embeds = base_hidden.zeros_like()?;
        let tts_hidden_uncond = self.tts_lm.forward(
            &uncond_embeds,
            None,
            &position_ids,
            &mut tts_lm_uncond_cache,
        )?;

        // Generate speech latents
        let mut all_latents = Vec::new();
        let _max_speech_tokens = gen_cfg
            .max_tokens
            .unwrap_or(seq_len * SPEECH_TOKENS_PER_WINDOW);

        // For simplicity, generate all speech tokens at once based on the text hidden states
        // In streaming mode, this would be done window by window
        let speech_latents = self.sample_speech_tokens(
            &tts_hidden_cond,
            &tts_hidden_uncond,
            &mut scheduler,
            &cfg_guidance,
            &mut rng,
        )?;

        all_latents.push(speech_latents);

        // Concatenate all latents
        let latents = if all_latents.len() == 1 {
            all_latents.pop().unwrap()
        } else {
            Tensor::cat(&all_latents, 1)?
        };

        info!("Generated {} speech latent frames", latents.dim(1)?);

        // Decode to audio
        let audio = self.acoustic_decoder.decode(&latents)?;

        // Get audio samples
        let audio = audio.squeeze(0)?; // Remove batch dimension
        let audio = audio.to_dtype(DType::F32)?;

        // Normalize loudness
        let audio = normalize_loudness(&audio, SAMPLE_RATE as u32, true)?;

        let pcm = audio.to_vec1::<f32>()?;

        info!(
            "Generated {:.2}s of audio ({} samples)",
            pcm.len() as f32 / SAMPLE_RATE as f32,
            pcm.len()
        );

        Ok(SpeechGenerationOutput {
            pcm: Arc::new(pcm),
            rate: SAMPLE_RATE,
            channels: CHANNELS,
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
}
