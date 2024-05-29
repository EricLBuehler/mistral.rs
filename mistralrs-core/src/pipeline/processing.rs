use std::sync::Arc;

use anyhow::Result;
use either::Either;
use indexmap::IndexMap;

use crate::{Content, Pipeline};

use super::{chat_template::apply_chat_template_to, text_models_inputs_processor, InputsProcessor};

/// Processor for messages.
/// Also includes method to retrieve the input processor for processing inputs for the
/// model.
pub trait Processor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, Content>>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>>;
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor>;
    fn get_special_tokens(&self) -> &[&'static str];
}

pub(crate) fn apply_chat_template(
    pipeline: &dyn Pipeline,
    messages: Vec<IndexMap<String, Content>>,
    add_generation_prompt: bool,
) -> Result<String> {
    let chat_template = pipeline.get_chat_template();
    let template = chat_template.chat_template.as_ref().unwrap();
    let bos_tok = if let Some(ref bos) = pipeline.get_chat_template().bos_token {
        match bos.0 {
            Either::Left(ref lit) => Some(lit.to_string()),
            Either::Right(ref added) => Some(added.content.to_string()),
        }
    } else {
        None
    };
    let eos_tok = if let Some(ref eos) = pipeline.get_chat_template().eos_token {
        match eos.0 {
            Either::Left(ref lit) => Some(lit.to_string()),
            Either::Right(ref added) => Some(added.content.to_string()),
        }
    } else {
        None
    };
    let unk_tok = if let Some(ref unk) = pipeline.get_chat_template().unk_token {
        match unk.0 {
            Either::Left(ref lit) => Some(lit.to_string()),
            Either::Right(ref added) => Some(added.content.to_string()),
        }
    } else {
        None
    };
    apply_chat_template_to(
        messages,
        add_generation_prompt,
        template,
        bos_tok,
        eos_tok,
        unk_tok,
    )
}

pub struct BasicProcessor;

impl Processor for BasicProcessor {
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, Content>>,
        add_generation_prompt: bool,
    ) -> Result<Vec<u32>> {
        let prompt = apply_chat_template(pipeline, messages, add_generation_prompt)?;
        let encoding = pipeline
            .tokenizer()
            .encode(prompt, false)
            .map_err(|e| anyhow::Error::msg(e.to_string()))?;
        Ok(encoding.get_ids().to_vec())
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(text_models_inputs_processor::TextInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
}
