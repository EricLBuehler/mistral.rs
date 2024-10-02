use std::any::Any;

use candle_core::Tensor;

pub(crate) mod clip;
pub(crate) mod idefics2;
pub(crate) mod idefics2_input_processor;
pub(crate) mod image_processor;
pub(crate) mod mllama;

pub(crate) mod llava;
pub(crate) mod phi3;
pub(crate) mod phi3_inputs_processor;
pub(crate) mod preprocessor_config;
pub(crate) mod processor_config;
pub(crate) use llava::llava15;
pub(crate) use llava::llava_inputs_processor;
pub(crate) use llava::llava_next;
pub(crate) use llava::llava_next_inputs_processor;

use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};

pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub seqlen_offsets_kernel: Tensor,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
}
