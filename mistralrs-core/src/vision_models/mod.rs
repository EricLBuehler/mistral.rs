use candle_core::{Device, Tensor};

use crate::{
    pipeline::{InputsProcessor, InputsProcessorType},
    sequence::Sequence,
};

pub(crate) mod idefics2;
pub(crate) mod idefics2_image_processor;
pub(crate) mod image_processor;

pub struct VisionInputsProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub input_ids: Tensor,
    pub input_ids_full: Option<Tensor>,
    pub seqlen_offsets: Vec<usize>,
    pub seqlen_offsets_full: Option<Vec<usize>>,
    pub seqlen_offsets_kernel: Tensor,
    pub seqlen_offsets_kernel_full: Option<Tensor>,
    pub context_lens: Vec<(usize, usize)>,
    pub position_ids: Vec<usize>,
    pub pixel_values: Tensor,
    pub pixel_attention_mask: Option<Tensor>,
}

impl InputsProcessor for VisionInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        input_seqs: &[&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
    ) -> anyhow::Result<Box<dyn std::any::Any>> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        todo!()
    }
}
