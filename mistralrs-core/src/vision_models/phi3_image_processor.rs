use std::{any::Any, sync::Arc};

use candle_core::{Device, Result};
use indexmap::IndexMap;

use crate::{
    pipeline::{
        text_models_inputs_processor::{self, get_completion_input, get_prompt_input},
        InputsProcessor, InputsProcessorType, Processor,
    },
    sequence::Sequence,
    Content, Pipeline,
};

use super::{
    image_processor::{ImagePreProcessor, PreprocessedImages},
    phi3::Phi3VisionSpecificArgs,
    preprocessor_config::PreProcessorConfig,
    processor_config::ProcessorConfig,
    ModelInputs,
};

// Input processor
pub struct Phi3ImageProcessor;
// Processor
pub struct Phi3Processor;

impl Phi3Processor {
    pub fn new(config: ProcessorConfig, preprocessor_config: PreProcessorConfig) -> Self {
        todo!()
    }
}

impl Processor for Phi3Processor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, Content>>,
        add_generation_prompt: bool,
    ) -> anyhow::Result<Vec<u32>> {
        todo!()
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Phi3ImageProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
}

impl InputsProcessor for Phi3ImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
    ) -> anyhow::Result<Box<dyn Any>> {
        if is_xlora {
            anyhow::bail!("Cannot make inputs for X-LoRA vision model.");
        }
        if no_kv_cache {
            anyhow::bail!("Vision model must have kv cache.");
        }
        let text_models_inputs_processor::InputMetadata {
            input,
            positions,
            positions_kernel,
            context_lens,
            position_ids,
        } = if is_prompt {
            get_prompt_input(input_seqs, device, last_n_context_len)?
        } else {
            get_completion_input(input_seqs, device, no_kv_cache, last_n_context_len)?
        };
        let config = other_config.expect("Need a PreProcessorConfig config.");
        let config: &PreProcessorConfig = config.downcast_ref().expect("Downcast failed.");
        Ok(Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            seqlen_offsets_kernel: positions_kernel,
            context_lens,
            position_ids,
            pixel_values: todo!(),
            model_specific_args: Box::new(Phi3VisionSpecificArgs {
                image_sizes: todo!(),
            }),
        }))
    }
}

impl ImagePreProcessor for Phi3ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    fn preprocess(
        &self,
        images: Vec<image::DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
    ) -> Result<PreprocessedImages> {
        todo!()
    }
}
