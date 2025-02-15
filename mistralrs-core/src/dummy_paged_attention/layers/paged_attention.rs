use candle_core::{Device, Result, Tensor};

use crate::{
    attention::SdpaParams,
    pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata},
};

#[allow(dead_code)]
pub struct PagedAttention;

impl PagedAttention {
    pub fn new(
        _head_dim: usize,
        _device: &Device,
        _alibi_slopes: Option<Vec<f32>>,
    ) -> Result<Self> {
        unreachable!();
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(unused_variables)]
    /// query: shape = [batch_size, seq_len, num_heads * head_size]
    /// key: shape = [batch_size, seq_len, num_kv_heads * head_size]
    /// value: shape = [batch_size, num_kv_heads * head_size]
    /// key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
    ///     block_size, x]
    /// value_cache: shape = [num_blocks, num_kv_heads, head_size,
    ///     block_size]
    /// input_metadata: metadata for paged attention.
    pub fn forward(
        &self,
        _query: &Tensor,
        _key: &Tensor,
        _value: &Tensor,
        _attention_mask: Option<&Tensor>,
        _key_cache: Option<Tensor>,
        _value_cache: Option<Tensor>,
        _input_metadata: &PagedAttentionInputMetadata,
        _sdpa_params: &SdpaParams,
        _flash_params: Option<&FlashParams>,
    ) -> Result<Tensor> {
        unreachable!();
    }
}
