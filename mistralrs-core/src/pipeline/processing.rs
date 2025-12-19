use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use either::Either;
use indexmap::IndexMap;
use serde_json::{json, Value};

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
    /// Get the tokens and the untokenized prompt. `add_special_tokens` should usually be true.
    fn process(
        &self,
        pipeline: &dyn Pipeline,
        messages: Vec<IndexMap<String, MessageContent>>,
        add_generation_prompt: bool,
        add_special_tokens: bool,
        enable_thinking: Option<bool>,
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
            self.template_action(),
            tools,
        )?;
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

pub(crate) fn apply_chat_template(
    pipeline: &dyn Pipeline,
    messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    action: MessagesAction,
    tools: Vec<Tool>,
) -> Result<String> {
    let messages = match action {
        MessagesAction::Keep => messages,
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

    // Best-effort debug dump of the rendered prompt + inputs to the template renderer.
    // This is intentionally in `mistralrs-core` so it captures the exact final prompt that is
    // being tokenized/executed (including chat template behavior).
    static DUMP_COUNTER: AtomicUsize = AtomicUsize::new(0);
    let dump_enabled = std::env::var("MISTRALRS_DEBUG_DUMP_PROMPT")
        .ok()
        .as_deref()
        .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
    let dump_messages = dump_enabled.then(|| messages.clone());
    let dump_tools = dump_enabled.then(|| tools.clone());

    let chat_template = pipeline
        .get_chat_template()
        .with_context(|| "`apply_chat_template` expects the pipeline to have a chat template.")?;
    let template = chat_template.chat_template.as_ref().unwrap();

    let bos_tok = chat_template.bos_token.as_ref().map(extract_token_string);
    let eos_tok = chat_template.eos_token.as_ref().map(extract_token_string);
    let unk_tok = chat_template.unk_token.as_ref().map(extract_token_string);

    let render_result = apply_chat_template_to(
        messages,
        add_generation_prompt,
        enable_thinking,
        template,
        bos_tok,
        eos_tok,
        unk_tok,
        tools,
    );

    if dump_enabled {
        let dir = std::env::var("MISTRALRS_DEBUG_DUMP_PROMPT_DIR")
            .ok()
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "request_dumps".to_string());

        let ts_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let n = DUMP_COUNTER.fetch_add(1, Ordering::Relaxed);
        let path = std::path::Path::new(&dir).join(format!("{ts_ms}_prompt_{n}.json"));

        let _ = std::fs::create_dir_all(&dir);
        let dump: Value = match &render_result {
            Ok(prompt) => json!({
                "ts_ms": ts_ms,
                "counter": n,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "messages": dump_messages,
                "tools": dump_tools,
                "prompt": prompt,
            }),
            Err(e) => json!({
                "ts_ms": ts_ms,
                "counter": n,
                "add_generation_prompt": add_generation_prompt,
                "enable_thinking": enable_thinking,
                "messages": dump_messages,
                "tools": dump_tools,
                "error": e.to_string(),
            }),
        };
        if let Ok(pretty) = serde_json::to_string_pretty(&dump) {
            let _ = std::fs::write(&path, pretty);
        }
    }

    render_result
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
