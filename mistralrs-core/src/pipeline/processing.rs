use std::sync::Arc;

use anyhow::{Context, Result};
use either::Either;
use indexmap::IndexMap;

use crate::{
    vision_models::{preprocessor_config::PreProcessorConfig, processor_config::ProcessorConfig},
    MessageContent, Pipeline, Tool,
};

use super::{chat_template::apply_chat_template_to, text_models_inputs_processor, InputsProcessor};

/// Trait to create processors.
pub trait ProcessorCreator {
    fn new_processor(
        _: Option<ProcessorConfig>,
        _: PreProcessorConfig,
    ) -> Arc<dyn Processor + Send + Sync>;
}

pub enum MessagesAction {
    // For idefics2, others which use the "new" openai format
    Keep,
    // For everything else
    FlattenOnlyText,
}

/// Processor for messages.
/// Also includes method to retrieve the input processor for processing inputs for the
/// model.
pub trait Processor {
    /// Get the tokens and the untokenized prompt
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        tools: Vec<Tool>,
    ) -> Result<(Vec<u32>, String)> {
        let prompt = apply_chat_template(
            pipeline,
            messages,
            add_generation_prompt,
            self.template_action(),
            tools,
        )?;
        let encoding = pipeline
            .tokenizer()
            .with_context(|| {
                "Default `Processor::process` requires the model to have a tokenizer."
            })?
            .encode(prompt.clone(), true)
            .map_err(anyhow::Error::msg)?;
        Ok((encoding.get_ids().to_vec(), prompt))
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor>;
    fn get_special_tokens(&self) -> &[&'static str];
    fn template_action(&self) -> MessagesAction;
}

pub(crate) fn apply_chat_template(
    pipeline: &dyn Pipeline,
    messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
    action: MessagesAction,
    tools: Vec<Tool>,
) -> Result<String> {
    let messages = match action {
        MessagesAction::Keep => messages,
        MessagesAction::FlattenOnlyText => {
            // This is really only for image models. If they need to flatten it s.t. they only see
            // the text, do that.
            let mut new_messages = Vec::new();
            for message in messages {
                let mut new_message = IndexMap::new();
                for (k, v) in message {
                    if k == "content" {
                        match v {
                            Either::Left(lv) => {
                                new_message.insert(k, Either::Left(lv));
                            }
                            Either::Right(rv) => {
                                'outer: for content_row in rv {
                                    for (content_k, content_v) in content_row {
                                        if content_k == "text" {
                                            new_message.insert(k, Either::Left(content_v));
                                            break 'outer;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        new_message.insert(k, Either::Left(v.left().unwrap()));
                    }
                }
                new_messages.push(new_message)
            }
            new_messages
        }
    };
    let chat_template = pipeline
        .get_chat_template()
        .with_context(|| "`apply_chat_template` expects the pipeline to have a chat template.")?;
    let template = chat_template.chat_template.as_ref().unwrap();
    let bos_tok = if let Some(ref bos) = chat_template.bos_token {
        match bos.0 {
            Either::Left(ref lit) => Some(lit.to_string()),
            Either::Right(ref added) => Some(added.content.to_string()),
        }
    } else {
        None
    };
    let eos_tok = if let Some(ref eos) = chat_template.eos_token {
        match eos.0 {
            Either::Left(ref lit) => Some(lit.to_string()),
            Either::Right(ref added) => Some(added.content.to_string()),
        }
    } else {
        None
    };
    let unk_tok = if let Some(ref unk) = chat_template.unk_token {
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
        tools,
    )
}

pub struct BasicProcessor;

impl Processor for BasicProcessor {
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor> {
        Arc::new(text_models_inputs_processor::TextInputsProcessor)
    }
    fn get_special_tokens(&self) -> &[&'static str] {
        &[]
    }
    fn template_action(&self) -> MessagesAction {
        MessagesAction::FlattenOnlyText
    }
}
