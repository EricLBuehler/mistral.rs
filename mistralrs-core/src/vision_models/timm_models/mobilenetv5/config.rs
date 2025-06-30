#[derive(Debug, Clone, serde::Deserialize)]
pub struct MobileNetV5Config {
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub num_channels: usize,
    pub rms_norm_eps: f64,
    pub vocab_size: usize,
    pub vocab_offset: usize,
}
