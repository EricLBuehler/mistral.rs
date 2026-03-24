#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    dead_code
)]

use serde::Deserialize;

use crate::serde_default_fn;

serde_default_fn!(bool, default_false, false);
serde_default_fn!(bool, default_true, true);
serde_default_fn!(f64, default_rope_theta, 1_000_000.0);
serde_default_fn!(f64, default_norm_eps, 1e-5);
serde_default_fn!(usize, default_ada_dim, 32);

/// Audio encoding parameters for mel spectrogram computation.
#[derive(Debug, Clone, Deserialize)]
pub struct AudioEncodingArgs {
    pub sampling_rate: u32,
    pub frame_rate: f64,
    pub num_mel_bins: usize,
    pub hop_length: usize,
    pub window_size: usize,
    pub global_log_mel_max: f64,
}

/// Downsampling configuration for the temporal adapter.
#[derive(Debug, Clone, Deserialize)]
pub struct DownsampleArgs {
    pub downsample_factor: usize,
}

/// Whisper-based causal encoder configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperEncoderArgs {
    pub audio_encoding_args: AudioEncodingArgs,
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    #[serde(default = "default_false")]
    pub use_biases: bool,
    #[serde(default = "default_true")]
    pub causal: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    pub sliding_window: Option<usize>,
    pub n_kv_heads: usize,
}

/// Whisper model arguments containing encoder and downsampling config.
#[derive(Debug, Clone, Deserialize)]
pub struct WhisperModelArgs {
    pub encoder_args: WhisperEncoderArgs,
    pub downsample_args: DownsampleArgs,
}

/// Multimodal configuration section.
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralMultimodalConfig {
    pub whisper_model_args: WhisperModelArgs,
}

/// Top-level Voxtral configuration, deserialized from params.json.
#[derive(Debug, Clone, Deserialize)]
pub struct VoxtralConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    #[serde(default = "default_false")]
    pub use_biases: bool,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_norm_eps")]
    pub norm_eps: f64,
    pub vocab_size: usize,
    #[serde(default = "default_true")]
    pub tied_embeddings: bool,
    pub sliding_window: Option<usize>,
    pub model_max_length: usize,
    pub multimodal: VoxtralMultimodalConfig,
    #[serde(default = "default_false")]
    pub ada_rms_norm_t_cond: bool,
    #[serde(default = "default_ada_dim")]
    pub ada_rms_norm_t_cond_dim: usize,
}
