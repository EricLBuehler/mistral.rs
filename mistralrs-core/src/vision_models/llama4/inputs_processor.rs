#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::{any::Any, num::NonZeroUsize, sync::Arc};

use candle_core::{Device, Result};
use image::DynamicImage;
use tokenizers::Tokenizer;
use tracing::warn;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::{
            self, get_completion_input, get_prompt_input, PagedAttentionMeta,
        },
        InputProcessorOutput, InputsProcessor, InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    vision_models::{
        image_processor::{ImagePreProcessor, PreprocessedImages},
        preprocessor_config::PreProcessorConfig,
        processor_config::ProcessorConfig,
        ModelInputs,
    },
};

use super::Llama4ModelSpecificArgs;

struct Llama4ImageProcessor;

pub struct Llama4Processor;

impl Llama4Processor {
    pub fn new(processor_config: ProcessorConfig) -> Self {
        Self
    }
}

impl Processor for Llama4Processor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(Llama4ImageProcessor)
    }

    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }

    fn template_action(&self) -> MessagesAction {
        MessagesAction::Keep
    }
}

impl InputsProcessor for Llama4ImageProcessor {
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
        return_raw_logits: bool,
        other_config: Option<Arc<dyn Any>>,
        mut paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_chunksize: Option<NonZeroUsize>,
        mapper: Option<&dyn DeviceMapper>,
    ) -> Box<dyn Iterator<Item = anyhow::Result<InputProcessorOutput>>> {
        if is_xlora {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Cannot make inputs for X-LoRA vision model.",
            ))));
        }
        if no_kv_cache {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Vision model must have kv cache.",
            ))));
        }
        // TODO(EricLBuehler): support this? Would require some handling of image tokens.
        if prompt_chunksize.is_some() {
            warn!("`prompt_chunksize` is set. Llama4 does not support prompt batching.");
        }

        let text_models_inputs_processor::InnerInputProcessorOutput {
            inputs:
                text_models_inputs_processor::InputMetadata {
                    input,
                    positions,
                    context_lens,
                    position_ids,
                    paged_attn_meta,
                    flash_meta,
                },
            seq_indices,
        } = if is_prompt {
            get_prompt_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
            .nth(0)
            .unwrap()
            .unwrap()
        } else {
            get_completion_input(
                input_seqs
                    .iter()
                    .map(|seq| seq.get_toks().to_vec())
                    .collect::<Vec<_>>(),
                input_seqs,
                device,
                no_kv_cache,
                last_n_context_len,
                return_raw_logits,
                paged_attn_metadata.as_mut(),
                None, // TODO: evaluate if it is possible to batch this
                mapper,
            )
            .nth(0)
            .unwrap()
            .unwrap()
        };

        let inputs: Box<dyn Any> = Box::new(ModelInputs {
            input_ids: input,
            seqlen_offsets: positions,
            context_lens,
            position_ids,
            pixel_values: None,
            model_specific_args: Box::new(Llama4ModelSpecificArgs),
            paged_attn_meta,
            flash_meta,
        });
        Box::new(std::iter::once(Ok(InputProcessorOutput {
            inputs,
            seq_indices,
        })))
    }
}

impl ImagePreProcessor for Llama4ImageProcessor {
    #[allow(clippy::excessive_precision)]
    const DEFAULT_MEAN: [f64; 3] = [0.48145466, 0.4578275, 0.40821073];
    #[allow(clippy::excessive_precision)]
    const DEFAULT_STD: [f64; 3] = [0.26862954, 0.26130258, 0.27577711];

    // https://github.com/huggingface/transformers/blob/main/src/transformers/models/pixtral/image_processing_pixtral.py
    fn preprocess(
        &self,
        mut images: Vec<DynamicImage>,
        videos: Vec<Vec<DynamicImage>>,
        config: &PreProcessorConfig,
        device: &Device,
        (_bs, _max_num_images): (usize, usize),
    ) -> Result<PreprocessedImages> {
        todo!()
    }
}
