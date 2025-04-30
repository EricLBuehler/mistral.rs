use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataConfig {
    pub text_length: usize,
    pub audio_length: usize,
    pub channels: usize,
    pub text_pad_value: i32,
    pub audio_eos_value: i32,
    pub audio_pad_value: i32,
    pub audio_bos_value: i32,
    pub delay_pattern: Vec<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncoderConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_hidden: usize,
    pub n_head: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecoderConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_hidden: usize,
    pub gqa_query_heads: usize,
    pub kv_heads: usize,
    pub gqa_head_dim: usize,
    pub cross_query_heads: usize,
    pub cross_head_dim: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub encoder: EncoderConfig,
    pub decoder: DecoderConfig,
    pub src_vocab_size: usize,
    pub tgt_vocab_size: usize,
    pub normalization_layer_epsilon: f64,
    pub weight_dtype: String,
    pub rope_min_timescale: f32,
    pub rope_max_timescale: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiaConfig {
    pub version: String,
    pub model: ModelConfig,
    pub data: DataConfig,
}
