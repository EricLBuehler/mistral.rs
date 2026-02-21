use std::any::Any;

use candle_core::Tensor;

pub(crate) mod clip;
pub(crate) mod conformer;
pub(crate) mod idefics2;
pub(crate) use idefics2::idefics2_input_processor;
pub(crate) mod image_processor;
pub(crate) mod llava;
pub(crate) mod mllama;
pub(crate) mod phi3;
pub(crate) use phi3::phi3_inputs_processor;
pub(crate) mod preprocessor_config;
pub(crate) mod processor_config;
pub(crate) mod qwen2_5_vl;
pub(crate) mod qwen2vl;
pub(crate) use llava::llava15;
pub(crate) use llava::llava_inputs_processor;
pub(crate) use llava::llava_next;
pub(crate) use llava::llava_next_inputs_processor;
pub(crate) mod idefics3;
pub(crate) mod minicpmo;
pub(crate) mod phi4;
pub(crate) use phi4::inputs_processor;
pub(crate) mod gemma3;
pub(crate) mod gemma3n;
pub(crate) mod llama4;
pub(crate) mod mistral3;
pub(crate) mod qwen3_vl;
pub(crate) mod qwen3_vl_moe;
pub(crate) mod siglip;
pub(crate) mod voxtral;

use crate::pipeline::text_models_inputs_processor::{FlashParams, PagedAttentionInputMetadata};

pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
    pub paged_attn_meta: Option<PagedAttentionInputMetadata>,
    pub flash_meta: FlashParams,
}
