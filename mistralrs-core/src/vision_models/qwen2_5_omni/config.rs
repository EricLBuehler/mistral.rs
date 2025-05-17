use mistralrs_quant::QuantizedConfig;
use serde::{Deserialize, Serialize};

use crate::{layers::Activation, serde_default_fn};

// Vision Encoder Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniVisionEncoderConfig {
    #[serde(default = "default_vision_depth")]
    pub depth: usize,

    #[serde(default = "default_vision_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_vision_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_vision_num_heads")]
    pub num_heads: usize,

    #[serde(default = "default_vision_in_channels")]
    pub in_channels: usize,

    #[serde(default = "default_vision_patch_size")]
    pub patch_size: usize,

    #[serde(default = "default_vision_spatial_merge_size")]
    pub spatial_merge_size: usize,

    #[serde(default = "default_vision_temporal_patch_size")]
    pub temporal_patch_size: usize,

    #[serde(default = "default_vision_window_size")]
    pub window_size: usize,

    #[serde(default = "default_vision_out_hidden_size")]
    pub out_hidden_size: usize,

    #[serde(default = "default_vision_fullatt_block_indexes")]
    pub fullatt_block_indexes: Vec<usize>,

    #[serde(default = "default_vision_initializer_range")]
    pub initializer_range: f32,
}

// Default value functions for vision config
serde_default_fn!(usize, default_vision_depth, 32);
serde_default_fn!(usize, default_vision_hidden_size, 3584);
serde_default_fn!(Activation, default_vision_hidden_act, Activation::Silu);
serde_default_fn!(usize, default_vision_intermediate_size, 3420);
serde_default_fn!(usize, default_vision_num_heads, 16);
serde_default_fn!(usize, default_vision_in_channels, 3);
serde_default_fn!(usize, default_vision_patch_size, 14);
serde_default_fn!(usize, default_vision_spatial_merge_size, 2);
serde_default_fn!(usize, default_vision_temporal_patch_size, 2);
serde_default_fn!(usize, default_vision_window_size, 112);
serde_default_fn!(usize, default_vision_out_hidden_size, 3584);
serde_default_fn!(
    Vec<usize>,
    default_vision_fullatt_block_indexes,
    vec![7, 15, 23, 31]
);
serde_default_fn!(f32, default_vision_initializer_range, 0.02);

// Audio Encoder Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniAudioEncoderConfig {
    #[serde(default = "default_audio_num_mel_bins")]
    pub num_mel_bins: usize,

    #[serde(default = "default_audio_encoder_layers")]
    pub encoder_layers: usize,

    #[serde(default = "default_audio_encoder_attention_heads")]
    pub encoder_attention_heads: usize,

    #[serde(default = "default_audio_encoder_ffn_dim")]
    pub encoder_ffn_dim: usize,

    #[serde(default = "default_audio_d_model")]
    pub d_model: usize,

    #[serde(default = "default_audio_dropout")]
    pub dropout: f32,

    #[serde(default = "default_audio_attention_dropout")]
    pub attention_dropout: f32,

    #[serde(default = "default_audio_activation_function")]
    pub activation_function: String,

    #[serde(default = "default_audio_activation_dropout")]
    pub activation_dropout: f32,

    #[serde(default = "default_audio_scale_embedding")]
    pub scale_embedding: bool,

    #[serde(default = "default_audio_initializer_range")]
    pub initializer_range: f32,

    #[serde(default = "default_audio_max_source_positions")]
    pub max_source_positions: usize,

    #[serde(default = "default_audio_n_window")]
    pub n_window: usize,

    #[serde(default = "default_audio_output_dim")]
    pub output_dim: usize,
}

// Default value functions for audio config
serde_default_fn!(usize, default_audio_num_mel_bins, 128);
serde_default_fn!(usize, default_audio_encoder_layers, 32);
serde_default_fn!(usize, default_audio_encoder_attention_heads, 20);
serde_default_fn!(usize, default_audio_encoder_ffn_dim, 5120);
serde_default_fn!(usize, default_audio_d_model, 1280);
serde_default_fn!(f32, default_audio_dropout, 0.0);
serde_default_fn!(f32, default_audio_attention_dropout, 0.0);
serde_default_fn!(
    String,
    default_audio_activation_function,
    "gelu".to_string()
);
serde_default_fn!(f32, default_audio_activation_dropout, 0.0);
serde_default_fn!(bool, default_audio_scale_embedding, false);
serde_default_fn!(f32, default_audio_initializer_range, 0.02);
serde_default_fn!(usize, default_audio_max_source_positions, 1500);
serde_default_fn!(usize, default_audio_n_window, 100);
serde_default_fn!(usize, default_audio_output_dim, 3584);

// Rope scaling configuration
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RopeScalingConfig {
    pub mrope_section: Vec<usize>,
}

// Text Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniTextConfig {
    #[serde(default = "default_text_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_text_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_text_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_text_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_text_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_text_initializer_range")]
    pub initializer_range: f32,

    #[serde(default = "default_text_rms_norm_eps")]
    pub rms_norm_eps: f64,

    #[serde(default = "default_text_use_cache")]
    pub use_cache: bool,

    #[serde(default = "default_text_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_text_rope_theta")]
    pub rope_theta: f64,

    pub rope_scaling: RopeScalingConfig,

    #[serde(default = "default_text_use_sliding_window")]
    pub use_sliding_window: bool,

    #[serde(default = "default_text_sliding_window")]
    pub sliding_window: usize,

    #[serde(default = "default_text_max_window_layers")]
    pub max_window_layers: usize,

    #[serde(default = "default_text_attention_dropout")]
    pub attention_dropout: f32,

    pub quantization_config: Option<QuantizedConfig>,
}

// Default value functions for text config
serde_default_fn!(usize, default_text_vocab_size, 152064);
serde_default_fn!(usize, default_text_hidden_size, 3584);
serde_default_fn!(usize, default_text_intermediate_size, 18944);
serde_default_fn!(usize, default_text_num_hidden_layers, 28);
serde_default_fn!(usize, default_text_num_attention_heads, 28);
serde_default_fn!(usize, default_text_num_key_value_heads, 4);
serde_default_fn!(Activation, default_text_hidden_act, Activation::Silu);
serde_default_fn!(usize, default_text_max_position_embeddings, 32768);
serde_default_fn!(f32, default_text_initializer_range, 0.02);
serde_default_fn!(f64, default_text_rms_norm_eps, 1e-6);
serde_default_fn!(bool, default_text_use_cache, true);
serde_default_fn!(bool, default_text_tie_word_embeddings, false);
serde_default_fn!(f64, default_text_rope_theta, 1000000.0);
serde_default_fn!(bool, default_text_use_sliding_window, false);
serde_default_fn!(usize, default_text_sliding_window, 32768);
serde_default_fn!(usize, default_text_max_window_layers, 28);
serde_default_fn!(f32, default_text_attention_dropout, 0.0);

// Thinker Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniThinkerConfig {
    pub audio_config: Qwen25OmniAudioEncoderConfig,

    pub vision_config: Qwen25OmniVisionEncoderConfig,

    pub text_config: Qwen25OmniTextConfig,

    #[serde(default = "default_thinker_audio_token_index")]
    pub audio_token_index: usize,

    #[serde(default = "default_thinker_image_token_index")]
    pub image_token_index: usize,

    #[serde(default = "default_thinker_video_token_index")]
    pub video_token_index: usize,

    #[serde(default = "default_thinker_position_id_per_seconds")]
    pub position_id_per_seconds: usize,

    #[serde(default = "default_thinker_seconds_per_chunk")]
    pub seconds_per_chunk: usize,

    #[serde(default = "default_thinker_audio_start_token_id")]
    pub audio_start_token_id: usize,

    #[serde(default = "default_thinker_audio_end_token_id")]
    pub audio_end_token_id: usize,

    #[serde(default = "default_thinker_user_token_id")]
    pub user_token_id: usize,

    #[serde(default = "default_thinker_initializer_range")]
    pub initializer_range: f32,
}

// Default value functions for thinker config
serde_default_fn!(usize, default_thinker_audio_token_index, 151646);
serde_default_fn!(usize, default_thinker_image_token_index, 151655);
serde_default_fn!(usize, default_thinker_video_token_index, 151656);
serde_default_fn!(usize, default_thinker_position_id_per_seconds, 25);
serde_default_fn!(usize, default_thinker_seconds_per_chunk, 2);
serde_default_fn!(usize, default_thinker_audio_start_token_id, 151647);
serde_default_fn!(usize, default_thinker_audio_end_token_id, 151648);
serde_default_fn!(usize, default_thinker_user_token_id, 872);
serde_default_fn!(f32, default_thinker_initializer_range, 0.02);

// Talker Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniTalkerConfig {
    #[serde(default = "default_talker_audio_token_index")]
    pub audio_token_index: usize,

    #[serde(default = "default_talker_image_token_index")]
    pub image_token_index: usize,

    #[serde(default = "default_talker_video_token_index")]
    pub video_token_index: usize,

    #[serde(default = "default_talker_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_talker_tts_text_start_token_id")]
    pub tts_text_start_token_id: usize,

    #[serde(default = "default_talker_tts_text_end_token_id")]
    pub tts_text_end_token_id: usize,

    #[serde(default = "default_talker_tts_text_pad_token_id")]
    pub tts_text_pad_token_id: usize,

    #[serde(default = "default_talker_tts_codec_start_token_id")]
    pub tts_codec_start_token_id: usize,

    #[serde(default = "default_talker_tts_codec_end_token_id")]
    pub tts_codec_end_token_id: usize,

    #[serde(default = "default_talker_tts_codec_pad_token_id")]
    pub tts_codec_pad_token_id: usize,

    #[serde(default = "default_talker_tts_codec_mask_token_id")]
    pub tts_codec_mask_token_id: usize,

    #[serde(default = "default_talker_vision_start_token_id")]
    pub vision_start_token_id: usize,

    #[serde(default = "default_talker_vision_end_token_id")]
    pub vision_end_token_id: usize,

    #[serde(default = "default_talker_embedding_size")]
    pub embedding_size: usize,

    #[serde(default = "default_talker_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_talker_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_talker_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_talker_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_talker_num_key_value_heads")]
    pub num_key_value_heads: usize,

    #[serde(default = "default_talker_hidden_act")]
    pub hidden_act: Activation,

    #[serde(default = "default_talker_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_talker_rms_norm_eps")]
    pub rms_norm_eps: f32,

    #[serde(default = "default_talker_head_dim")]
    pub head_dim: usize,

    #[serde(default = "default_talker_use_cache")]
    pub use_cache: bool,

    #[serde(default = "default_talker_tie_word_embeddings")]
    pub tie_word_embeddings: bool,

    #[serde(default = "default_talker_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_talker_use_sliding_window")]
    pub use_sliding_window: bool,

    #[serde(default = "default_talker_sliding_window")]
    pub sliding_window: usize,

    #[serde(default = "default_talker_max_window_layers")]
    pub max_window_layers: usize,

    #[serde(default = "default_talker_attention_dropout")]
    pub attention_dropout: f32,

    pub rope_scaling: Option<RopeScalingConfig>,

    #[serde(default = "default_talker_position_id_per_seconds")]
    pub position_id_per_seconds: usize,

    #[serde(default = "default_talker_seconds_per_chunk")]
    pub seconds_per_chunk: usize,

    #[serde(default = "default_talker_audio_start_token_id")]
    pub audio_start_token_id: usize,

    #[serde(default = "default_talker_audio_end_token_id")]
    pub audio_end_token_id: usize,

    #[serde(default = "default_talker_initializer_range")]
    pub initializer_range: f32,

    #[serde(default = "default_talker_spatial_merge_size")]
    pub spatial_merge_size: usize,
}

// Default value functions for talker config
serde_default_fn!(usize, default_talker_audio_token_index, 151646);
serde_default_fn!(usize, default_talker_image_token_index, 151655);
serde_default_fn!(usize, default_talker_video_token_index, 151656);
serde_default_fn!(usize, default_talker_vocab_size, 8448);
serde_default_fn!(usize, default_talker_tts_text_start_token_id, 151860);
serde_default_fn!(usize, default_talker_tts_text_end_token_id, 151861);
serde_default_fn!(usize, default_talker_tts_text_pad_token_id, 151859);
serde_default_fn!(usize, default_talker_tts_codec_start_token_id, 8293);
serde_default_fn!(usize, default_talker_tts_codec_end_token_id, 8294);
serde_default_fn!(usize, default_talker_tts_codec_pad_token_id, 8292);
serde_default_fn!(usize, default_talker_tts_codec_mask_token_id, 8296);
serde_default_fn!(usize, default_talker_vision_start_token_id, 151652);
serde_default_fn!(usize, default_talker_vision_end_token_id, 151653);
serde_default_fn!(usize, default_talker_embedding_size, 3584);
serde_default_fn!(usize, default_talker_hidden_size, 3584);
serde_default_fn!(usize, default_talker_intermediate_size, 18944);
serde_default_fn!(usize, default_talker_num_hidden_layers, 28);
serde_default_fn!(usize, default_talker_num_attention_heads, 28);
serde_default_fn!(usize, default_talker_num_key_value_heads, 4);
serde_default_fn!(Activation, default_talker_hidden_act, Activation::Silu);
serde_default_fn!(usize, default_talker_max_position_embeddings, 32768);
serde_default_fn!(f32, default_talker_rms_norm_eps, 1e-6);
serde_default_fn!(usize, default_talker_head_dim, 128);
serde_default_fn!(bool, default_talker_use_cache, true);
serde_default_fn!(bool, default_talker_tie_word_embeddings, false);
serde_default_fn!(f32, default_talker_rope_theta, 1000000.0);
serde_default_fn!(bool, default_talker_use_sliding_window, false);
serde_default_fn!(usize, default_talker_sliding_window, 32768);
serde_default_fn!(usize, default_talker_max_window_layers, 28);
serde_default_fn!(f32, default_talker_attention_dropout, 0.0);
serde_default_fn!(usize, default_talker_position_id_per_seconds, 25);
serde_default_fn!(usize, default_talker_seconds_per_chunk, 2);
serde_default_fn!(usize, default_talker_audio_start_token_id, 151647);
serde_default_fn!(usize, default_talker_audio_end_token_id, 151648);
serde_default_fn!(f32, default_talker_initializer_range, 0.02);
serde_default_fn!(usize, default_talker_spatial_merge_size, 2);

// DiT Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniDiTConfig {
    #[serde(default = "default_dit_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_dit_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_dit_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_dit_ff_mult")]
    pub ff_mult: usize,

    #[serde(default = "default_dit_emb_dim")]
    pub emb_dim: usize,

    #[serde(default = "default_dit_head_dim")]
    pub head_dim: usize,

    #[serde(default = "default_dit_rope_theta")]
    pub rope_theta: f32,

    #[serde(default = "default_dit_max_position_embeddings")]
    pub max_position_embeddings: usize,

    #[serde(default = "default_dit_block_size")]
    pub block_size: usize,

    #[serde(default = "default_dit_look_ahead_layers")]
    pub look_ahead_layers: Vec<usize>,

    #[serde(default = "default_dit_look_backward_layers")]
    pub look_backward_layers: Vec<usize>,

    #[serde(default = "default_dit_repeats")]
    pub repeats: usize,

    #[serde(default = "default_dit_num_embeds")]
    pub num_embeds: usize,

    #[serde(default = "default_dit_mel_dim")]
    pub mel_dim: usize,

    #[serde(default = "default_dit_dropout")]
    pub dropout: f32,

    #[serde(default = "default_dit_enc_emb_dim")]
    pub enc_emb_dim: usize,

    #[serde(default = "default_dit_enc_dim")]
    pub enc_dim: usize,

    #[serde(default = "default_dit_enc_channels")]
    pub enc_channels: Vec<usize>,

    #[serde(default = "default_dit_enc_kernel_sizes")]
    pub enc_kernel_sizes: Vec<usize>,

    #[serde(default = "default_dit_enc_dilations")]
    pub enc_dilations: Vec<usize>,

    #[serde(default = "default_dit_enc_attention_channels")]
    pub enc_attention_channels: usize,

    #[serde(default = "default_dit_enc_res2net_scale")]
    pub enc_res2net_scale: usize,

    #[serde(default = "default_dit_enc_se_channels")]
    pub enc_se_channels: usize,
}

// Default value functions for DiT config
serde_default_fn!(usize, default_dit_hidden_size, 1024);
serde_default_fn!(usize, default_dit_num_hidden_layers, 22);
serde_default_fn!(usize, default_dit_num_attention_heads, 16);
serde_default_fn!(usize, default_dit_ff_mult, 2);
serde_default_fn!(usize, default_dit_emb_dim, 512);
serde_default_fn!(usize, default_dit_head_dim, 64);
serde_default_fn!(f32, default_dit_rope_theta, 10000.0);
serde_default_fn!(usize, default_dit_max_position_embeddings, 32768);
serde_default_fn!(usize, default_dit_block_size, 24);
serde_default_fn!(Vec<usize>, default_dit_look_ahead_layers, vec![10]);
serde_default_fn!(Vec<usize>, default_dit_look_backward_layers, vec![0, 20]);
serde_default_fn!(usize, default_dit_repeats, 2);
serde_default_fn!(usize, default_dit_num_embeds, 8193);
serde_default_fn!(usize, default_dit_mel_dim, 80);
serde_default_fn!(f32, default_dit_dropout, 0.1);
serde_default_fn!(usize, default_dit_enc_emb_dim, 192);
serde_default_fn!(usize, default_dit_enc_dim, 128);
serde_default_fn!(
    Vec<usize>,
    default_dit_enc_channels,
    vec![256, 256, 256, 256, 768]
);
serde_default_fn!(
    Vec<usize>,
    default_dit_enc_kernel_sizes,
    vec![5, 3, 3, 3, 1]
);
serde_default_fn!(Vec<usize>, default_dit_enc_dilations, vec![1, 2, 3, 4, 1]);
serde_default_fn!(usize, default_dit_enc_attention_channels, 64);
serde_default_fn!(usize, default_dit_enc_res2net_scale, 2);
serde_default_fn!(usize, default_dit_enc_se_channels, 64);

// BigVGAN Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniBigVGANConfig {
    #[serde(default = "default_bigvgan_mel_dim")]
    pub mel_dim: usize,

    #[serde(default = "default_bigvgan_upsample_initial_channel")]
    pub upsample_initial_channel: usize,

    #[serde(default = "default_bigvgan_resblock_kernel_sizes")]
    pub resblock_kernel_sizes: Vec<usize>,

    #[serde(default = "default_bigvgan_resblock_dilation_sizes")]
    pub resblock_dilation_sizes: Vec<Vec<usize>>,

    #[serde(default = "default_bigvgan_upsample_rates")]
    pub upsample_rates: Vec<usize>,

    #[serde(default = "default_bigvgan_upsample_kernel_sizes")]
    pub upsample_kernel_sizes: Vec<usize>,
}

// Default value functions for BigVGAN config
serde_default_fn!(usize, default_bigvgan_mel_dim, 80);
serde_default_fn!(usize, default_bigvgan_upsample_initial_channel, 1536);
serde_default_fn!(
    Vec<usize>,
    default_bigvgan_resblock_kernel_sizes,
    vec![3, 7, 11]
);
serde_default_fn!(
    Vec<Vec<usize>>,
    default_bigvgan_resblock_dilation_sizes,
    vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]]
);
serde_default_fn!(
    Vec<usize>,
    default_bigvgan_upsample_rates,
    vec![5, 3, 2, 2, 2, 2]
);
serde_default_fn!(
    Vec<usize>,
    default_bigvgan_upsample_kernel_sizes,
    vec![11, 7, 4, 4, 4, 4]
);

// Token2Wav Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniToken2WavConfig {
    pub dit_config: Qwen25OmniDiTConfig,

    pub bigvgan_config: Qwen25OmniBigVGANConfig,
}

// Main Omni Config
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Qwen25OmniConfig {
    pub thinker_config: Qwen25OmniThinkerConfig,

    pub talker_config: Qwen25OmniTalkerConfig,

    pub token2wav_config: Qwen25OmniToken2WavConfig,

    #[serde(default = "default_omni_enable_audio_output")]
    pub enable_audio_output: bool,
}

serde_default_fn!(bool, default_omni_enable_audio_output, true);
