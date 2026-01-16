use std::{any::Any, sync::Arc};

use anyhow::{Context, Result};
use candle_core::Device;
use indexmap::IndexMap;
use tokenizers::Tokenizer;

use crate::{
    device_map::DeviceMapper,
    pipeline::{
        text_models_inputs_processor::PagedAttentionMeta, InputProcessorOutput, InputsProcessor,
        InputsProcessorType, MessagesAction, Processor,
    },
    sequence::Sequence,
    MessageContent, Pipeline,
};

use super::DiffusionGenerationParams;

pub struct DiffusionProcessor;

impl Processor for DiffusionProcessor {
    fn process(
        &self,
        _pipeline: &dyn Pipeline,
        _messages: Vec<IndexMap<String, MessageContent>>,
        _add_generation_prompt: bool,
        _add_special_tokens: bool,
        _enable_thinking: Option<bool>,
        _reasoning_effort: Option<crate::request::ReasoningEffort>,
        _tools: Vec<crate::Tool>,
    ) -> Result<(Vec<u32>, String)> {
        anyhow::bail!(
            "DiffusionProcessor::process should not be used. It does not expect chat messages."
        )
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(DiffusionInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        // Just a default
        MessagesAction::FlattenOnlyText
    }
}

pub struct DiffusionInputsProcessor;

#[derive(Clone)]
pub struct ModelInputs {
    pub(crate) prompts: Vec<String>,
    pub(crate) params: DiffusionGenerationParams,
}

impl InputsProcessor for DiffusionInputsProcessor {
    fn get_type(&self) -> InputsProcessorType {
        InputsProcessorType::Text
    }

    fn process_inputs(
        &self,
        _tokenizer: Option<Arc<Tokenizer>>,
        input_seqs: &mut [&mut Sequence],
        _is_prompt: bool,
        _is_xlora: bool,
        _device: &Device,
        _no_kv_cache: bool,
        _last_n_context_len: Option<(usize, usize)>,
        _return_raw_logits: bool,
        _other_config: Option<Arc<dyn Any>>,
        _paged_attn_metadata: Option<PagedAttentionMeta>,
        _mapper: Option<&dyn DeviceMapper>,
    ) -> Result<InputProcessorOutput> {
        let inputs = ModelInputs {
            prompts: input_seqs
                .iter_mut()
                .map(|seq| seq.get_initial_prompt().to_string())
                .collect::<Vec<_>>(),
            params: input_seqs[0]
                .get_diffusion_diffusion_params()
                .context("Diffusion model params must be present")?,
        };
        Ok(InputProcessorOutput {
            inputs: Box::new(inputs),
            seq_indices: (0..input_seqs.len()).collect::<Vec<_>>(),
        })
    }
}
