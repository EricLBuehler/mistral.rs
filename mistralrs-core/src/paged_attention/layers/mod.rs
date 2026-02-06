#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub mod paged_attention;
#[cfg(any(all(feature = "cuda", target_family = "unix"), feature = "metal"))]
pub use paged_attention::PagedAttention;

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub mod paged_attention {
    use candle_core::{Device, Result, Tensor};

    use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
    use crate::{attention::SdpaParams, pipeline::text_models_inputs_processor::FlashParams};

    pub struct PagedAttention;

    impl PagedAttention {
        pub fn new(
            _head_dim: usize,
            _device: &Device,
            _alibi_slopes: Option<Vec<f32>>,
        ) -> Result<Self> {
            candle_core::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }

        #[allow(clippy::too_many_arguments)]
        #[allow(unused_variables)]
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
            _sinks: Option<&Tensor>,
        ) -> Result<Tensor> {
            candle_core::bail!("Paged attention requires the CUDA or Metal feature flags.");
        }
    }
}

#[cfg(not(any(all(feature = "cuda", target_family = "unix"), feature = "metal")))]
pub use paged_attention::PagedAttention;
