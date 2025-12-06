//! VibeVoice configuration structures

#![allow(dead_code)]

use serde::Deserialize;

/// Acoustic tokenizer (σ-VAE) configuration for converting latents to audio waveforms
#[derive(Debug, Clone, Deserialize)]
pub struct AcousticTokenizerConfig {
    /// Whether to use causal convolutions for streaming
    #[serde(default = "default_causal")]
    pub causal: bool,

    /// Number of audio channels (mono = 1)
    #[serde(default = "default_channels")]
    pub channels: usize,

    /// Whether to use bias in convolutions
    #[serde(default = "default_conv_bias")]
    pub conv_bias: bool,

    /// Number of filters in the decoder stem
    #[serde(default = "default_decoder_n_filters")]
    pub decoder_n_filters: usize,

    /// Upsampling ratios for each decoder stage: [8, 5, 5, 4, 2, 2] = 3200x total
    #[serde(default = "default_decoder_ratios")]
    pub decoder_ratios: Vec<usize>,

    /// Encoder depths as a dash-separated string (e.g., "3-3-3-3-3-3-8")
    /// Decoder depths mirror encoder depths if not specified
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: String,

    /// Fixed standard deviation for σ-VAE
    #[serde(default = "default_fix_std")]
    pub fix_std: f32,

    /// Layer scale initialization value
    #[serde(default = "default_layer_scale_init_value")]
    pub layer_scale_init_value: f64,

    /// Type of normalization layer ("RMSNorm" or "LayerNorm")
    #[serde(default = "default_layernorm")]
    pub layernorm: String,

    /// Layer normalization epsilon
    #[serde(default = "default_layernorm_eps")]
    pub layernorm_eps: f64,

    /// Mixer layer type ("depthwise_conv" or "conv")
    #[serde(default = "default_mixer_layer")]
    pub mixer_layer: String,

    /// Latent dimension (VAE latent space dimension)
    #[serde(default = "default_vae_dim")]
    pub vae_dim: usize,
}

impl Default for AcousticTokenizerConfig {
    fn default() -> Self {
        Self {
            causal: default_causal(),
            channels: default_channels(),
            conv_bias: default_conv_bias(),
            decoder_n_filters: default_decoder_n_filters(),
            decoder_ratios: default_decoder_ratios(),
            encoder_depths: default_encoder_depths(),
            fix_std: default_fix_std(),
            layer_scale_init_value: default_layer_scale_init_value(),
            layernorm: default_layernorm(),
            layernorm_eps: default_layernorm_eps(),
            mixer_layer: default_mixer_layer(),
            vae_dim: default_vae_dim(),
        }
    }
}

impl AcousticTokenizerConfig {
    /// Parse encoder_depths string to get depths for each stage
    pub fn get_depths(&self) -> Vec<usize> {
        self.encoder_depths
            .split('-')
            .map(|s| s.parse().unwrap_or(3))
            .collect()
    }
}

fn default_causal() -> bool {
    true
}
fn default_channels() -> usize {
    1
}
fn default_conv_bias() -> bool {
    true
}
fn default_decoder_n_filters() -> usize {
    32
}
fn default_decoder_ratios() -> Vec<usize> {
    vec![8, 5, 5, 4, 2, 2]
}
fn default_encoder_depths() -> String {
    "3-3-3-3-3-3-8".to_string()
}
fn default_fix_std() -> f32 {
    0.5
}
fn default_layer_scale_init_value() -> f64 {
    1e-6
}
fn default_layernorm() -> String {
    "RMSNorm".to_string()
}
fn default_layernorm_eps() -> f64 {
    1e-5
}
fn default_mixer_layer() -> String {
    "depthwise_conv".to_string()
}
fn default_vae_dim() -> usize {
    64
}

/// Diffusion head configuration for generating acoustic latents
#[derive(Debug, Clone, Deserialize)]
pub struct DiffusionHeadConfig {
    /// Batch multiplier for classifier-free guidance (4 = [cond, cond, cond, uncond])
    #[serde(default = "default_ddpm_batch_mul")]
    pub ddpm_batch_mul: usize,

    /// Beta schedule type ("cosine" or "linear")
    #[serde(default = "default_ddpm_beta_schedule")]
    pub ddpm_beta_schedule: String,

    /// Number of inference steps (typically 20)
    #[serde(default = "default_ddpm_num_inference_steps")]
    pub ddpm_num_inference_steps: usize,

    /// Number of training timesteps (typically 1000)
    #[serde(default = "default_ddpm_num_steps")]
    pub ddpm_num_steps: usize,

    /// Diffusion type ("ddpm")
    #[serde(default = "default_diffusion_type")]
    pub diffusion_type: String,

    /// FFN ratio for diffusion head layers
    #[serde(default = "default_head_ffn_ratio")]
    pub head_ffn_ratio: f32,

    /// Number of diffusion head layers
    #[serde(default = "default_head_layers")]
    pub head_layers: usize,

    /// Hidden size (must match LM hidden size)
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Latent size (output dimension, must match acoustic VAE dim)
    #[serde(default = "default_latent_size")]
    pub latent_size: usize,

    /// Prediction type ("v_prediction" or "epsilon")
    #[serde(default = "default_prediction_type")]
    pub prediction_type: String,

    /// RMS normalization epsilon
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
}

impl Default for DiffusionHeadConfig {
    fn default() -> Self {
        Self {
            ddpm_batch_mul: default_ddpm_batch_mul(),
            ddpm_beta_schedule: default_ddpm_beta_schedule(),
            ddpm_num_inference_steps: default_ddpm_num_inference_steps(),
            ddpm_num_steps: default_ddpm_num_steps(),
            diffusion_type: default_diffusion_type(),
            head_ffn_ratio: default_head_ffn_ratio(),
            head_layers: default_head_layers(),
            hidden_size: default_hidden_size(),
            latent_size: default_latent_size(),
            prediction_type: default_prediction_type(),
            rms_norm_eps: default_rms_norm_eps(),
        }
    }
}

fn default_ddpm_batch_mul() -> usize {
    4
}
fn default_ddpm_beta_schedule() -> String {
    "cosine".to_string()
}
fn default_ddpm_num_inference_steps() -> usize {
    20
}
fn default_ddpm_num_steps() -> usize {
    1000
}
fn default_diffusion_type() -> String {
    "ddpm".to_string()
}
fn default_head_ffn_ratio() -> f32 {
    3.0
}
fn default_head_layers() -> usize {
    4
}
fn default_hidden_size() -> usize {
    896
}
fn default_latent_size() -> usize {
    64
}
fn default_prediction_type() -> String {
    "v_prediction".to_string()
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}

/// Decoder (Qwen2) configuration for language model backbone
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub rope_theta: f64,
    #[serde(default = "default_decoder_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub hidden_act: Option<String>,
}

fn default_decoder_rms_norm_eps() -> f64 {
    1e-6
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 8192,
            rope_theta: 1000000.0,
            rms_norm_eps: 1e-6,
            hidden_act: Some("silu".to_string()),
        }
    }
}

/// Main VibeVoice streaming configuration
#[derive(Debug, Clone, Deserialize)]
pub struct VibeVoiceConfig {
    /// Acoustic VAE latent dimension (typically 64)
    #[serde(default = "default_acoustic_vae_dim")]
    pub acoustic_vae_dim: usize,

    /// Acoustic tokenizer configuration
    #[serde(default)]
    pub acoustic_tokenizer_config: AcousticTokenizerConfig,

    /// Decoder (Qwen2 LM) configuration
    #[serde(default)]
    pub decoder_config: DecoderConfig,

    /// Diffusion head configuration
    #[serde(default)]
    pub diffusion_head_config: DiffusionHeadConfig,

    /// Number of TTS backbone layers (upper layers for speech)
    #[serde(default = "default_tts_backbone_num_hidden_layers")]
    pub tts_backbone_num_hidden_layers: usize,
}

fn default_acoustic_vae_dim() -> usize {
    64
}
fn default_tts_backbone_num_hidden_layers() -> usize {
    20
}

impl Default for VibeVoiceConfig {
    fn default() -> Self {
        Self {
            acoustic_vae_dim: default_acoustic_vae_dim(),
            acoustic_tokenizer_config: AcousticTokenizerConfig::default(),
            decoder_config: DecoderConfig::default(),
            diffusion_head_config: DiffusionHeadConfig::default(),
            tts_backbone_num_hidden_layers: default_tts_backbone_num_hidden_layers(),
        }
    }
}

impl VibeVoiceConfig {
    /// Get the number of base LM layers (lower layers for text encoding)
    pub fn base_lm_layers(&self) -> usize {
        self.decoder_config
            .num_hidden_layers
            .saturating_sub(self.tts_backbone_num_hidden_layers)
    }

    /// Get head dimension
    pub fn head_dim(&self) -> usize {
        self.decoder_config.hidden_size / self.decoder_config.num_attention_heads
    }
}
