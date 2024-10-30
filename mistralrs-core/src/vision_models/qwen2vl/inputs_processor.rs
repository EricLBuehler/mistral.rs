use std::{any::Any, num::NonZeroUsize, sync::Arc};

use anyhow::Result;
use candle_core::Device;
use image::DynamicImage;
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
    },
};

// Input processor
struct Qwen2VLImageProcessor {}
// Processor
pub struct Qwen2VLProcessor;

impl Qwen2VLProcessor {
    pub fn new() -> Self {
        Self
    }
}

impl Processor for Qwen2VLProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Qwen2VLImageProcessor {})
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Qwen2VLImageProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Vision
    }
    fn process_inputs(
        &self,
        tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        is_prompt: bool,
        is_xlora: bool,
        device: &Device,
        no_kv_cache: bool,
        last_n_context_len: Option<(usize, usize)>,
        other_config: Option<Arc<dyn Any>>,
        paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_batchsize: Option<NonZeroUsize>,
    ) -> Box<dyn Iterator<Item = Result<InputProcessorOutput>>> {
        todo!()
    }
}

impl ImagePreProcessor for Qwen2VLImageProcessor {
    const DEFAULT_MEAN: [f64; 3] = todo!();
    const DEFAULT_STD: [f64; 3] = todo!();

    fn preprocess(
        &self,
        images: Vec<DynamicImage>,
        config: &PreProcessorConfig,
        device: &Device,
        batch_info: (usize, usize),
    ) -> candle_core::Result<PreprocessedImages> {
        todo!()
    }
}
