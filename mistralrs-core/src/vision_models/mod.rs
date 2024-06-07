use std::any::Any;

use candle_core::Tensor;

pub(crate) mod clip;
pub(crate) mod image_processor;
pub(crate) mod phi3;
pub(crate) mod phi3_inputs_processor;
pub(crate) mod preprocessor_config;
pub(crate) mod processor_config;

pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub seqlen_offsets_kernel: Tensor,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub model_specific_args: Box<dyn Any>,
}
