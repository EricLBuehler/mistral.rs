use serde::{Deserialize, Serialize};

use crate::{
    paged_attention::ModelConfigLike,
    vision_models::{lfm3_omni::audio::AudioEncoderConfig, lfm3_vl},
};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub(crate) struct Config {
    #[serde(flatten)]
    pub vision: lfm3_vl::Config,
    pub audio_encoder_config: Option<AudioEncoderConfig>,
    pub audio_token_id: Option<u32>,
    pub d1_codes: usize,
    pub d1_attention: String,
    pub d1_numbers: bool,
    pub d1_set_size: usize,
    pub d1_vision_max_pixels: usize,
}

pub(crate) struct ModelConfig<T>(pub T);

impl<T: std::ops::Deref> ModelConfigLike for ModelConfig<T>
where
    T::Target: ModelConfigLike,
{
    fn max_seq_len(&self) -> usize {
        self.0.max_seq_len()
    }
    fn num_layers(&self) -> usize {
        self.0.num_layers()
    }
    fn hidden_size(&self) -> usize {
        self.0.hidden_size()
    }
    fn num_kv_heads(&self) -> usize {
        self.0.num_kv_heads()
    }
    fn num_attn_heads(&self) -> usize {
        self.0.num_attn_heads()
    }
    fn k_head_dim(&self) -> usize {
        self.0.k_head_dim()
    }
    fn v_head_dim(&self) -> usize {
        self.0.v_head_dim()
    }
    fn layer_has_paged_kv_cache(&self, _layer: usize) -> bool {
        false
    }
}
