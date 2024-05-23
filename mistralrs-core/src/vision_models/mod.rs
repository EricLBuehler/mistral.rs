use candle_core::Tensor;

pub(crate) mod idefics2;
pub(crate) mod idefics2_image_processor;
pub(crate) mod image_processor;

#[derive(Clone)]
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub seqlen_offsets: Vec<usize>,
    pub seqlen_offsets_kernel: Tensor,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Option<Tensor>,
    pub pixel_attention_mask: Option<Tensor>,
}
