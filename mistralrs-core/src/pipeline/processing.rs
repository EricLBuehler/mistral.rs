use std::sync::Arc;

use anyhow::{Context, Result};
use either::Either;
use indexmap::IndexMap;

use crate::{
    request::ReasoningEffort,
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
    // Gemma 4 is trained with audio after the instruction.
    KeepWithAudioAfterText,
    // For everything else
    FlattenOnlyText,
}

/// Processor for messages.
/// Also includes method to retrieve the input processor for processing inputs for the
/// model.
pub trait Processor {
    /// Get the tokens and the untokenized prompt. `add_special_tokens` should usually be true.
    #[allow(clippy::too_many_arguments)]
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
        reasoning_effort: Option<ReasoningEffort>,
        tools: Vec<Tool>,
    ) -> Result<(Vec<u32>, String)> {
        // for message in messages.iter_mut() {
        //     if message["role"].as_ref().left().is_some_and(|x| x == "tool") {
        //         message["role"] = Either::Left("ipython".to_string());
        //         message["content"] = Either::Left(format!(
        //             "{{\"output\": \"{}\"}}",
        //             message["content"].as_ref().unwrap_left()
        //         ));
        //     }
        // }

        let prompt = apply_chat_template(
            pipeline,
            messages,
            add_generation_prompt,
            enable_thinking,
            reasoning_effort,
            self.template_action(),
            tools,
        )?;
        // Templates own their special tokens (HF apply_chat_template convention): when
        // the rendered prompt already starts with bos, letting the tokenizer add it
        // again doubles it (gemma-family tokenizers add bos in their post-processor).
        let add_special_tokens = add_special_tokens
            && !pipeline
                .get_chat_template()
                .and_then(|t| t.bos_tok())
                .is_some_and(|bos| prompt.starts_with(&bos));
        let encoding = pipeline
            .tokenizer()
            .with_context(|| {
                "Default `Processor::process` requires the model to have a tokenizer."
            })?
            .encode_fast(prompt.clone(), add_special_tokens)
            .map_err(anyhow::Error::msg)?;
        Ok((encoding.get_ids().to_vec(), prompt))
    }
    fn inputs_processor(&self) -> Arc<dyn InputsProcessor>;
    fn retain_prefix_cached_images(&self) -> bool {
        false
    }
    fn get_special_tokens(&self) -> &[&'static str];
    fn template_action(&self) -> MessagesAction;
}

/// Helper function to extract token string from BeginEndUnkPadTok
fn extract_token_string(token: &super::chat_template::BeginEndUnkPadTok) -> String {
    match &token.0 {
        Either::Left(lit) => lit.clone(),
        Either::Right(added) => added.content.clone(),
    }
}

/// Flatten a content field to extract only text from structured content
fn flatten_content(content: MessageContent) -> MessageContent {
    match content {
        Either::Left(_) => content,
        Either::Right(content_rows) => {
            // Find the first "text" field in the content rows
            content_rows
                .into_iter()
                .find_map(|content_row| {
                    content_row
                        .get("text")
                        .and_then(|v| v.as_str())
                        .map(|s| Either::Left(s.to_string()))
                })
                .unwrap_or(Either::Right(Vec::new()))
        }
    }
}

fn move_audio_after_text(content: MessageContent) -> MessageContent {
    let Either::Right(content_rows) = content else {
        return content;
    };
    let (mut non_audio, audio): (Vec<_>, Vec<_>) = content_rows
        .into_iter()
        .partition(|row| row.get("type").and_then(|value| value.as_str()) != Some("audio"));
    non_audio.extend(audio);
    Either::Right(non_audio)
}

pub(crate) fn apply_chat_template(
    pipeline: &dyn Pipeline,
    messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    action: MessagesAction,
    tools: Vec<Tool>,
) -> Result<String> {
    let messages = match action {
        MessagesAction::Keep => messages,
        MessagesAction::KeepWithAudioAfterText => messages
            .into_iter()
            .map(|message| {
                message
                    .into_iter()
                    .map(|(key, value)| {
                        let value = if key == "content" {
                            move_audio_after_text(value)
                        } else {
                            value
                        };
                        (key, value)
                    })
                    .collect()
            })
            .collect(),
        MessagesAction::FlattenOnlyText => {
            // This is really only for image models. If they need to flatten it s.t. they only see
            // the text, do that.
            messages
                .into_iter()
                .map(|message| {
                    message
                        .into_iter()
                        .map(|(key, value)| {
                            let new_value = if key == "content" {
                                flatten_content(value)
                            } else {
                                value
                            };
                            (key, new_value)
                        })
                        .collect()
                })
                .collect()
        }
    };

    let chat_template = pipeline
        .get_chat_template()
        .with_context(|| "`apply_chat_template` expects the pipeline to have a chat template.")?;
    let template = chat_template
        .chat_template
        .as_ref()
        .with_context(|| "The selected model does not define a chat template.")?;

    let bos_tok = chat_template.bos_token.as_ref().map(extract_token_string);
    let eos_tok = chat_template.eos_token.as_ref().map(extract_token_string);
    let unk_tok = chat_template.unk_token.as_ref().map(extract_token_string);

    apply_chat_template_to(
        messages,
        add_generation_prompt,
        enable_thinking,
        reasoning_effort,
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
        MessagesAction::Keep
    }
}

#[cfg(test)]
mod tests {
    use either::Either;
    use indexmap::IndexMap;
    use serde_json::Value;

    use super::move_audio_after_text;

    #[test]
    fn moves_audio_after_text_without_reordering_other_parts() {
        let content = Either::Right(vec![
            IndexMap::from([("type".to_string(), Value::String("image".to_string()))]),
            IndexMap::from([
                ("type".to_string(), Value::String("audio".to_string())),
                ("id".to_string(), Value::String("first".to_string())),
            ]),
            IndexMap::from([("type".to_string(), Value::String("video".to_string()))]),
            IndexMap::from([
                ("type".to_string(), Value::String("text".to_string())),
                ("text".to_string(), Value::String("transcribe".to_string())),
            ]),
            IndexMap::from([
                ("type".to_string(), Value::String("audio".to_string())),
                ("id".to_string(), Value::String("second".to_string())),
            ]),
        ]);

        let Either::Right(parts) = move_audio_after_text(content) else {
            panic!("expected multimodal content");
        };
        let types = parts
            .iter()
            .map(|part| part.get("type").and_then(Value::as_str).unwrap())
            .collect::<Vec<_>>();
        let audio_ids = parts
            .iter()
            .filter_map(|part| part.get("id").and_then(Value::as_str))
            .collect::<Vec<_>>();

        assert_eq!(types, ["image", "video", "text", "audio", "audio"]);
        assert_eq!(audio_ids, ["first", "second"]);
    }

    #[test]
    fn leaves_string_content_unchanged() {
        let content = Either::Left("hello".to_string());
        assert_eq!(move_audio_after_text(content.clone()), content);
    }
}
