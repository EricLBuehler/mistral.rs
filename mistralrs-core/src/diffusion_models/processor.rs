use std::{any::Any, num::NonZeroUsize, sync::Arc};

use anyhow::Result;
use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use crate::{
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType,
    },
    sequence::Sequence,
};

pub struct DiffusionProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub(crate) input_ids: Tensor,
}

impl InputsProcessor for DiffusionProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    fn process_inputs(
        &self,
        _tokenizer: Arc<Tokenizer>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta<'_>>,
        prompt_batchsize: Option<NonZeroUsize>,
    ) -> Box<dyn Iterator<Item = Result<InputProcessorOutput>>> {
        let mut make_value = if prompt_batchsize.is_some() {
            return Box::new(std::iter::once(Err(anyhow::Error::msg(
                "Prompt batching is unsupported for diffusion models",
            ))));
        } else {
            || {
                let mut tokenized = Vec::new();
                for seq in &mut *input_seqs {
                    tokenized.push(Tensor::new(seq.get_toks(), device)?)
                }

                let inputs = ModelInputs {
                    input_ids: Tensor::stack(&tokenized, 0)?,
                };
                Ok(InputProcessorOutput {
                    inputs: Box::new(inputs),
                    seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
                })
            }
        };
        Box::new(std::iter::once(make_value()))
    }
}
