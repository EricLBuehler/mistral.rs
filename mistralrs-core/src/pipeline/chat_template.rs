use std::collections::HashMap;

use anyhow::Result;
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use minijinja::{context, value::Kwargs, Environment, Error, ErrorKind, Value};
use regex::Regex;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::{trace, warn};

use crate::{tools::ToolCallFormat, MessageContent, ModelGenerationDefaults, Tool};

const SUPPORTED_ALTERNATE_EOS: &[&str] = &[
    "<|im_end|>",      // Handle ChatML case
    "<end_of_turn>",   // Handle Gemma2 chat case
    "<|end_of_text|>", // Hermes
    "<|end|>",         // Phi-3, Phi-3.5, Harmony
    "<|eot_id|>",      // Llama 3
];

const HARMONY_ALTERNATE_EOS: &[&str] = &[
    "<|message|>", // Harmony
    "<|start|>",   // Harmony
    "<|channel|>", // Harmony
];

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub(crate) struct ChatTemplateRequestError(String);

#[doc(hidden)]
pub fn is_chat_template_request_error(error: &anyhow::Error) -> bool {
    error
        .chain()
        .any(|source| source.is::<ChatTemplateRequestError>())
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(
        minijinja::Error::new(ErrorKind::InvalidOperation, msg.clone())
            .with_source(ChatTemplateRequestError(msg)),
    )
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BeginEndUnkPadTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[derive(Debug, Deserialize, Serialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Serialize, Default)]
/// Template for chat models including bos/eos/unk as well as the chat template.
pub struct ChatTemplate {
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    pub bos_token: Option<BeginEndUnkPadTok>,

    /// Jinja format [chat templating] for chat completion.
    ///
    /// [chat templating]: https://huggingface.co/docs/transformers/chat_templating
    pub chat_template: Option<ChatTemplateValue>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    pub eos_token: Option<BeginEndUnkPadTok>,
    legacy: Option<bool>,
    model_max_length: Option<f64>,
    pub pad_token: Option<BeginEndUnkPadTok>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: Option<String>,
    truncation_size: Option<String>,
    pub unk_token: Option<BeginEndUnkPadTok>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub(crate) fn get_template_contents(&self) -> Vec<String> {
        match self.chat_template.as_ref() {
            Some(t) => match &t.0 {
                Either::Left(s) => vec![s.clone()],
                Either::Right(vec) => vec.iter().flat_map(|m| m.values().cloned()).collect(),
            },
            None => vec![],
        }
    }

    /// Check if this chat template uses OpenAI Harmony format.
    pub fn is_harmony_format(&self) -> bool {
        self.get_template_contents()
            .iter()
            .any(|t| crate::reasoning_parsers::harmony::is_harmony_template(t))
    }

    pub(crate) fn tool_call_format(&self) -> Option<ToolCallFormat> {
        self.get_template_contents()
            .iter()
            .find_map(|template| template_tool_call_format(template))
    }

    /// Check if this chat template uses `<think>...</think>` tags for reasoning.
    ///
    /// This is mutually exclusive with Harmony format - if the template uses
    /// Harmony format, this returns false even if think tags are present.
    pub fn uses_think_tags(&self) -> bool {
        // Don't enable if Harmony format is detected (mutual exclusivity)
        if self.is_harmony_format() {
            return false;
        }

        self.get_template_contents()
            .iter()
            .any(|t| crate::reasoning_parsers::tag_based::is_think_tag_template(t))
    }

    /// Check if the template uses Gemma 4 channel-based reasoning tags.
    pub fn uses_channel_tags(&self) -> bool {
        self.get_template_contents()
            .iter()
            .any(|t| crate::reasoning_parsers::tag_based::is_channel_tag_template(t))
    }

    pub fn uses_gemma_turns(&self) -> bool {
        self.get_template_contents()
            .iter()
            .any(|t| crate::reasoning_parsers::tag_based::is_gemma_turn_template(t))
    }

    pub fn eos_tok(&self) -> Option<String> {
        match self.eos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn bos_tok(&self) -> Option<String> {
        match self.bos_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

pub fn calculate_eos_tokens(
    chat_template: &ChatTemplate,
    gen_conf: Option<&GenerationConfig>,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut eos_tok_ids = chat_template.eos_tok().map(|x| vec![x]).unwrap_or_default();
    let mut bos_tok_ids = chat_template.bos_tok().map(|b| vec![b]).unwrap_or_default();

    let templates = chat_template.get_template_contents();

    for alternate in SUPPORTED_ALTERNATE_EOS {
        if tokenizer.get_vocab(true).contains_key(*alternate)
            && templates.iter().any(|t| t.contains(*alternate))
        {
            eos_tok_ids.push(alternate.to_string())
        }
    }
    if chat_template.is_harmony_format() {
        for alternate in HARMONY_ALTERNATE_EOS {
            if tokenizer.get_vocab(true).contains_key(*alternate)
                && templates
                    .iter()
                    .any(|template| template.contains(*alternate))
            {
                eos_tok_ids.push(alternate.to_string());
            }
        }
    }

    if let Some(gen_conf) = gen_conf {
        if let Some(eos_field) = gen_conf.eos_token_id.as_ref() {
            let ids = match eos_field {
                Either::Left(id) => vec![*id],
                Either::Right(ids) => ids.clone(),
            };
            for id in ids {
                let Ok(s) = tokenizer.decode(&[id], false) else {
                    warn!("Ignoring generation config EOS token id {id}: not in the tokenizer vocabulary");
                    continue;
                };
                if !eos_tok_ids.contains(&s) {
                    eos_tok_ids.push(s);
                }
            }
        }

        if let Some(bos_field) = gen_conf.bos_token_id.as_ref() {
            let ids = match bos_field {
                Either::Left(id) => vec![*id],
                Either::Right(ids) => ids.clone(),
            };
            for id in ids {
                let s = tokenizer
                    .decode(&[id], false)
                    .unwrap_or_else(|_| panic!("Unable to decode id {id})"));
                if !bos_tok_ids.contains(&s) {
                    bos_tok_ids.push(s);
                }
            }
        }
    }

    eos_tok_ids = eos_tok_ids.into_iter().dedup().collect::<Vec<_>>();
    bos_tok_ids = bos_tok_ids.into_iter().dedup().collect::<Vec<_>>();

    let bos_render = bos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");
    let eos_render = eos_tok_ids
        .iter()
        .map(|val| format!("{val:?}"))
        .collect::<Vec<String>>()
        .join(", ");

    trace!(
        "bos_toks = {bos_render}, eos_toks = {eos_render}, unk_tok = {}",
        chat_template.unk_tok().unwrap_or("`None`".to_string()),
    );

    let mut eos_toks = Vec::new();
    for eos_tok in eos_tok_ids {
        eos_toks.push(
            tokenizer
                .get_vocab(true)
                .get(&eos_tok)
                .copied()
                .unwrap_or_else(|| panic!("Unable to extract `{eos_tok}` EOS token.")),
        )
    }
    eos_toks
}

#[allow(dead_code)]
#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    bos_token_id: Option<Either<u32, Vec<u32>>>,
    #[serde(default)]
    #[serde(with = "either::serde_untagged_optional")]
    eos_token_id: Option<Either<u32, Vec<u32>>>,
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    temperature: Option<f64>,
    #[serde(default)]
    top_k: Option<usize>,
    #[serde(default)]
    top_p: Option<f64>,
    #[serde(default)]
    min_p: Option<f64>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    max_new_tokens: Option<usize>,
    #[serde(default)]
    max_length: Option<usize>,
    #[serde(default)]
    suppress_tokens: Option<Vec<u32>>,
}

impl GenerationConfig {
    /// HF `GenerationConfig.from_model_config`: without a generation_config.json the model config's own
    /// generation fields apply, with a nested `text_config` filling anything the top level leaves unset.
    pub fn from_model_config(config_json: &str) -> Option<Self> {
        let raw: serde_json::Value = serde_json::from_str(config_json).ok()?;
        let mut conf: GenerationConfig = serde_json::from_value(raw.clone()).ok()?;
        if let Some(nested) = raw.get("text_config").cloned() {
            if let Ok(nested) = serde_json::from_value::<GenerationConfig>(nested) {
                conf.bos_token_id = conf.bos_token_id.or(nested.bos_token_id);
                conf.eos_token_id = conf.eos_token_id.or(nested.eos_token_id);
                conf.do_sample = conf.do_sample.or(nested.do_sample);
                conf.temperature = conf.temperature.or(nested.temperature);
                conf.top_k = conf.top_k.or(nested.top_k);
                conf.top_p = conf.top_p.or(nested.top_p);
                conf.min_p = conf.min_p.or(nested.min_p);
                conf.repetition_penalty = conf.repetition_penalty.or(nested.repetition_penalty);
            }
        }
        conf.max_new_tokens = None;
        conf.max_length = None;
        Some(conf)
    }

    pub(crate) fn validate_token_ids(&self, vocab_size: usize) -> Result<()> {
        for (field, value) in [
            ("bos_token_id", self.bos_token_id.as_ref()),
            ("eos_token_id", self.eos_token_id.as_ref()),
        ] {
            let Some(value) = value else {
                continue;
            };
            let ids = match value {
                Either::Left(id) => std::slice::from_ref(id),
                Either::Right(ids) => ids.as_slice(),
            };
            for id in ids {
                anyhow::ensure!(
                    usize::try_from(*id).is_ok_and(|id| id < vocab_size),
                    "generation config `{field}` contains token ID {id}, but the tokenizer vocabulary has {vocab_size} entries"
                );
            }
        }
        if let Some(ids) = self.suppress_tokens.as_ref() {
            for id in ids {
                anyhow::ensure!(
                    usize::try_from(*id).is_ok_and(|id| id < vocab_size),
                    "generation config `suppress_tokens` contains token ID {id}, but the tokenizer vocabulary has {vocab_size} entries"
                );
            }
        }
        Ok(())
    }

    pub fn generation_defaults(&self) -> Option<ModelGenerationDefaults> {
        let defaults = ModelGenerationDefaults {
            do_sample: self.do_sample,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            min_p: self.min_p,
            repetition_penalty: self.repetition_penalty,
            max_new_tokens: self.max_new_tokens,
            max_length: self.max_length,
            suppress_tokens: self.suppress_tokens.clone(),
        };

        if defaults.is_empty() {
            None
        } else {
            Some(defaults)
        }
    }
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    if let Ok(indent) = kwargs.get::<usize>("indent") {
        // Cap the indent: it feeds `b" ".repeat(indent)`, so an attacker-controlled template could request a huge allocation or capacity-overflow panic.
        const MAX_INDENT: usize = 256;
        if indent > MAX_INDENT {
            return Err(Error::new(
                ErrorKind::InvalidOperation,
                format!("tojson `indent` of {indent} exceeds the maximum of {MAX_INDENT}"),
            ));
        }
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut ser).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })?;
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        // Python's json.dumps default separators, which is what HF templates were rendered with
        let mut buf = Vec::new();
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonCompactFormatter);
        value.serialize(&mut ser).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })?;
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    // HF's tojson does not HTML-escape, so neither can we without changing the prompt
    .map(Value::from_safe_string)
}

#[derive(Default)]
struct PythonCompactFormatter;

impl serde_json::ser::Formatter for PythonCompactFormatter {
    fn begin_array_value<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_key<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
        first: bool,
    ) -> std::io::Result<()> {
        if !first {
            writer.write_all(b", ")?;
        }
        Ok(())
    }

    fn begin_object_value<W: std::io::Write + ?Sized>(
        &mut self,
        writer: &mut W,
    ) -> std::io::Result<()> {
        writer.write_all(b": ")
    }
}

fn strftime_now(fmt: String) -> Result<String, minijinja::Error> {
    let date = chrono::Local::now();
    let date_string = date.format(&fmt).to_string();
    Ok(date_string)
}

use crate::request::{resolve_reasoning_controls, ReasoningEffort};

/// Check if a chat template uses Gemma 4 tool call tokens.
fn is_gemma4_tool_template(template: &str) -> bool {
    template.contains("<|tool_call>") && template.contains("<tool_call|>")
}

fn is_liquid_tool_template(template: &str) -> bool {
    template.contains("<|tool_call_start|>") && template.contains("<|tool_call_end|>")
}

fn is_atem_tool_template(template: &str) -> bool {
    template.contains("<atem:function_calls>") && template.contains("<atem:invoke")
}

/// Whether the template walks tool call arguments as key/value pairs.
///
/// OpenAI sends `arguments` as a JSON string while templates are authored against the map that
/// transformers passes, so those templates fail on the string unless it is parsed first.
fn iterates_tool_call_arguments(template: &str) -> bool {
    let compact = template
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect::<String>();
    compact.contains("arguments|items") || compact.contains("arguments.items()")
}

fn normalize_minijinja_compatibility(template: &str) -> String {
    template.replace(
        "namespace(name=tcid if tcid else '')",
        "namespace(name=(tcid if tcid else ''))",
    )
}

fn template_tool_call_format(template: &str) -> Option<ToolCallFormat> {
    if is_atem_tool_template(template) {
        Some(ToolCallFormat::Atem)
    } else if crate::reasoning_parsers::harmony::is_harmony_template(template) {
        Some(ToolCallFormat::Harmony)
    } else if is_gemma4_tool_template(template) {
        Some(ToolCallFormat::Gemma4)
    } else if is_liquid_tool_template(template) {
        Some(ToolCallFormat::Liquid)
    } else if template.contains("<|python_tag|>") {
        Some(ToolCallFormat::Llama)
    } else if template.contains("[TOOL_CALLS]") {
        Some(ToolCallFormat::MistralNemo)
    } else if template.contains("<tool_calls>") && template.contains("</tool_calls>") {
        Some(ToolCallFormat::Hunyuan)
    } else if template.contains("<｜tool▁call▁begin｜>") {
        Some(ToolCallFormat::DeepSeek)
    } else if template.contains("<tool_call>") && template.contains("</tool_call>") {
        Some(ToolCallFormat::Qwen)
    } else {
        None
    }
}

fn parse_tool_call_arguments(messages: &mut [IndexMap<String, MessageContent>]) {
    for message in messages.iter_mut() {
        let is_assistant = message
            .get("role")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .is_some_and(|r| r == "assistant");
        if !is_assistant {
            continue;
        }

        let Some(Either::Right(tool_calls)) = message.get_mut("tool_calls") else {
            continue;
        };
        for tc in tool_calls.iter_mut() {
            // tool_calls[i].function.arguments
            let Some(serde_json::Value::Object(func)) = tc.get_mut("function") else {
                continue;
            };
            if let Some(serde_json::Value::String(json_str)) = func.get("arguments") {
                if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(json_str) {
                    if parsed.is_object() {
                        func.insert("arguments".to_string(), parsed);
                    }
                }
            }
        }
    }
}

// Schema keys most templates were trained on come first; the rest is alphabetical so renders stay deterministic
const TOOL_PARAMETER_KEY_ORDER: &[&str] =
    &["type", "properties", "required", "additionalProperties"];

fn tool_template_value(tool: &Tool) -> serde_json::Value {
    let mut function = serde_json::Map::new();
    function.insert(
        "name".to_string(),
        serde_json::Value::String(tool.function.name.clone()),
    );
    if let Some(description) = &tool.function.description {
        function.insert(
            "description".to_string(),
            serde_json::Value::String(description.clone()),
        );
    }
    if let Some(parameters) = &tool.function.parameters {
        let mut keys = parameters.keys().collect::<Vec<_>>();
        keys.sort_by_key(|key| {
            (
                TOOL_PARAMETER_KEY_ORDER
                    .iter()
                    .position(|known| known == key)
                    .unwrap_or(TOOL_PARAMETER_KEY_ORDER.len()),
                key.as_str(),
            )
        });
        let ordered = keys
            .into_iter()
            .map(|key| (key.clone(), parameters[key].clone()))
            .collect::<serde_json::Map<_, _>>();
        function.insert("parameters".to_string(), serde_json::Value::Object(ordered));
    }
    if let Some(strict) = tool.function.strict {
        function.insert("strict".to_string(), serde_json::Value::Bool(strict));
    }
    serde_json::json!({ "type": tool.tp, "function": function })
}

fn clear_assistant_tool_call_content(messages: &mut [IndexMap<String, MessageContent>]) {
    for message in messages.iter_mut() {
        let is_assistant = message
            .get("role")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .is_some_and(|r| r == "assistant");
        if is_assistant && message.contains_key("tool_calls") {
            message.insert("content".to_string(), Either::Left(String::new()));
        }
    }
}

/// Pre-process messages for Gemma 4 tool templates.
///
/// The Gemma 4 chat template expects `tool_responses` as a field on a
/// **user** message, but the OpenAI API sends `role: "tool"` as separate
/// messages. This function replaces consecutive `role: "tool"` messages
/// with a single `role: "user"` message carrying the `tool_responses`
/// field, matching the format used by the reference implementations
/// (llama.cpp `convert_tool_responses_gemma4`, HF transformers).
///
/// Additionally, when the preceding assistant message has structured
/// `tool_calls`, its raw-JSON `content` is cleared so the template only
/// renders the `<|tool_call>` tags.
fn preprocess_gemma4_tool_messages(messages: &mut Vec<IndexMap<String, MessageContent>>) {
    let mut result: Vec<IndexMap<String, MessageContent>> = Vec::with_capacity(messages.len());
    let mut i = 0;

    while i < messages.len() {
        let is_tool = messages[i]
            .get("role")
            .and_then(|v| match v {
                Either::Left(s) => Some(s.as_str()),
                _ => None,
            })
            .is_some_and(|r| r == "tool");

        if !is_tool {
            let mut msg = std::mem::take(&mut messages[i]);

            // When an assistant message has structured tool_calls, clear the
            // raw-JSON content so the template only renders <|tool_call> tags.
            let is_assistant = msg
                .get("role")
                .and_then(|v| match v {
                    Either::Left(s) => Some(s.as_str()),
                    _ => None,
                })
                .is_some_and(|r| r == "assistant");
            if is_assistant && (msg.contains_key("tool_calls") || !msg.contains_key("content")) {
                msg.insert("content".to_string(), Either::Left(String::new()));
            }

            result.push(msg);
            i += 1;
            continue;
        }

        // Collect consecutive tool messages into a single tool_responses list.
        let mut tool_responses: Vec<IndexMap<String, serde_json::Value>> = Vec::new();
        let mut media_parts: Vec<IndexMap<String, serde_json::Value>> = Vec::new();
        while i < messages.len() {
            let is_tool = messages[i]
                .get("role")
                .and_then(|v| match v {
                    Either::Left(s) => Some(s.as_str()),
                    _ => None,
                })
                .is_some_and(|r| r == "tool");
            if !is_tool {
                break;
            }

            let tool_msg = &messages[i];

            let name = tool_msg
                .get("name")
                .and_then(|v| match v {
                    Either::Left(s) => Some(s.clone()),
                    _ => None,
                })
                .unwrap_or_else(|| "unknown".to_string());

            let content = match tool_msg.get("content") {
                Some(Either::Left(s)) => s.clone(),
                Some(Either::Right(parts)) => {
                    let mut text = String::new();
                    for part in parts {
                        match part.get("type").and_then(|v| v.as_str()) {
                            Some("text") => {
                                if let Some(t) = part.get("text").and_then(|v| v.as_str()) {
                                    text.push_str(t);
                                }
                            }
                            Some("image") | Some("audio") | Some("video") => {
                                media_parts.push(part.clone());
                            }
                            _ => {}
                        }
                    }
                    text
                }
                _ => String::new(),
            };

            let response_value: serde_json::Value =
                serde_json::from_str(&content).unwrap_or(serde_json::Value::String(content));

            let mut entry = IndexMap::new();
            entry.insert("name".to_string(), serde_json::Value::String(name));
            entry.insert("response".to_string(), response_value);
            tool_responses.push(entry);

            i += 1;
        }

        // Create a user message with the collected tool_responses.
        let mut user_msg: IndexMap<String, MessageContent> = IndexMap::new();
        user_msg.insert("role".to_string(), Either::Left("user".to_string()));
        user_msg.insert("tool_responses".to_string(), Either::Right(tool_responses));
        if !media_parts.is_empty() {
            user_msg.insert("content".to_string(), Either::Right(media_parts));
        }
        result.push(user_msg);
    }

    *messages = result;
}

#[allow(clippy::too_many_arguments)]
pub fn apply_chat_template_to(
    mut messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<ReasoningEffort>,
    template: &ChatTemplateValue,
    bos_tok: Option<String>,
    eos_tok: Option<String>,
    unk_tok: Option<String>,
    tools: Vec<Tool>,
) -> Result<String> {
    let mut env = Environment::new();

    // enable python methods such as .strip()
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);

    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    #[derive(Serialize, Deserialize)]
    struct UntaggedContent(#[serde(with = "either::serde_untagged")] MessageContent);

    // Resolve template string early so we can check for Gemma 4 format
    let resolved_template = match &template.0 {
        Either::Left(x) => x.clone(),
        Either::Right(map) => {
            let has_tool_use = map.iter().any(|t| {
                t.get("name").is_some_and(|name| name == "tool_use") || t.contains_key("tool_use")
            });
            let must_use_tool_template = !tools.is_empty();

            if must_use_tool_template && !has_tool_use {
                return Err(ChatTemplateRequestError(
                    "Tools were provided but this chat template does not handle tool usage"
                        .to_string(),
                )
                .into());
            }

            let mut found_template = None;
            for t in map {
                let name = t.get("name");
                if let Some(name) = name {
                    found_template = Some(t["template"].clone());
                    #[allow(clippy::if_same_then_else)]
                    if name == "tool_use" && !tools.is_empty() {
                        break;
                    } else if name == "default" && !must_use_tool_template {
                        break;
                    }
                } else if t.contains_key("tool_use") && !tools.is_empty() {
                    found_template = Some(t["tool_use"].clone());
                    break;
                } else if t.contains_key("default") && !must_use_tool_template {
                    found_template = Some(t["default"].clone());
                    break;
                }
            }

            found_template.ok_or_else(|| anyhow::anyhow!("Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools."))?
        }
    };

    let is_gemma4_template = is_gemma4_tool_template(&resolved_template);
    let is_liquid_template = is_liquid_tool_template(&resolved_template);

    // HF templates expect tool_calls[].function.arguments as a mapping, not the OpenAI wire string
    parse_tool_call_arguments(&mut messages);
    if is_gemma4_template {
        preprocess_gemma4_tool_messages(&mut messages);
    } else if is_liquid_template {
        clear_assistant_tool_call_content(&mut messages);
    } else if iterates_tool_call_arguments(&resolved_template) {
        parse_tool_call_arguments(&mut messages);
    }

    let mut new_messages = Vec::new();
    for message in messages {
        let mut new_message = IndexMap::new();
        for (k, v) in message {
            new_message.insert(k, UntaggedContent(v));
        }
        new_messages.push(new_message);
    }

    // Use the already-resolved template string
    let mut template = normalize_minijinja_compatibility(&resolved_template);
    template = template.replace("[::-1]", "|reverse");
    // Convert Python‑style descending ranges `range(..., -1, -1)` to a forward
    // range followed by Jinja’s `|reverse` filter so it works even when
    // negative‑step ranges aren’t supported.
    let re = Regex::new(r"range\((?P<expr>[^,]+),\s*-1,\s*-1\)").unwrap();
    template = re
        .replace_all(&template, |caps: &regex::Captures| {
            format!("range({})|reverse", &caps["expr"])
        })
        .into_owned();

    if template.contains("{{ meta }}") {
        // Fix for GLM4 models
        template = template.replace("{%- set meta = message.get(\"metadata\", \"\") %}", "");
        template = template.replace("{{ meta }}", "");
    }
    let generation_re = Regex::new(r"\{%-?\s*(?:end)?generation\s*-?%\}").unwrap();
    template = generation_re.replace_all(&template, "").into_owned();

    env.add_template("chat_template", &template)?;
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", tojson);
    env.add_function("strftime_now", strftime_now);
    let tmpl = env.get_template("chat_template")?;

    let date = chrono::Local::now();
    let date_string = date.format("%d, %B, %Y").to_string();

    let reasoning_controls = resolve_reasoning_controls(enable_thinking, reasoning_effort)?;
    let reasoning_effort_value = reasoning_controls
        .reasoning_effort
        .map(|effort| Value::from(effort.as_str()))
        .unwrap_or(Value::UNDEFINED);

    // Detect builtin tools from the tools list
    // Known builtin tools for GPT-OSS/Harmony format: "browser", "python"
    // Known builtin tools for Llama 3.x: "wolfram_alpha", "web_search", "brave_search", "python", "code_interpreter"
    let builtin_tool_names = [
        "browser",
        "python",
        "code_interpreter",
        "web_search",
        "brave_search",
        "wolfram_alpha",
    ];
    let builtin_tools: Vec<&str> = tools
        .iter()
        .filter_map(|t| {
            let name = t.function.name.as_str();
            if builtin_tool_names.contains(&name) {
                Some(name)
            } else {
                None
            }
        })
        .collect();

    let is_gemma4 = is_gemma4_tool_template(&resolved_template);

    let tools = tools.iter().map(tool_template_value).collect::<Vec<_>>();
    let mut rendered = if tools.is_empty() {
        tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            date_string => date_string,
            enable_thinking => reasoning_controls.enable_thinking,
            reasoning_effort => &reasoning_effort_value,
            reasoning_strength => &reasoning_effort_value,
        })?
    } else {
        tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            xml_tools => tools.clone(), // SmolLM3
            tools => tools,
            builtin_tools => builtin_tools,
            date_string => date_string,
            enable_thinking => reasoning_controls.enable_thinking,
            reasoning_effort => &reasoning_effort_value,
            reasoning_strength => &reasoning_effort_value,
        })?
    };

    // Gemma 4 fix: when tool_responses are in a user turn (the correct
    // format), the template's generation-prompt logic skips `<|turn>model\n`
    // because it checks `prev_message_type != 'tool_response'`.  But the
    // training data ALWAYS has `<|turn>model\n` before the model generates.
    // Append it when the template left it out.
    if is_gemma4 && add_generation_prompt && rendered.ends_with("<tool_response|>") {
        rendered.push_str("<|turn>model\n");
    }

    Ok(rendered)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use either::Either;
    use indexmap::IndexMap;
    use serde_json::Value;

    use super::{
        apply_chat_template_to, calculate_eos_tokens, preprocess_gemma4_tool_messages,
        template_tool_call_format, ChatTemplate, ChatTemplateValue, GenerationConfig,
        ReasoningEffort,
    };
    use crate::{
        tools::ToolCallFormat, Function, MessageContent, Tool, ToolType, DEFAULT_ENABLE_THINKING,
    };
    use tokenizers::Tokenizer;

    fn user_text_message(text: &str) -> IndexMap<String, MessageContent> {
        IndexMap::from([
            ("role".to_string(), Either::Left("user".to_string())),
            ("content".to_string(), Either::Left(text.to_string())),
        ])
    }

    #[test]
    fn generation_config_token_ids_must_fit_the_tokenizer_vocabulary() {
        let valid: GenerationConfig = serde_json::from_value(serde_json::json!({
            "bos_token_id": 0,
            "eos_token_id": [1, 2],
            "suppress_tokens": [3]
        }))
        .unwrap();
        assert!(valid.validate_token_ids(4).is_ok());

        let invalid: GenerationConfig = serde_json::from_value(serde_json::json!({
            "eos_token_id": [1, 4]
        }))
        .unwrap();
        assert!(invalid.validate_token_ids(4).is_err());
    }

    #[test]
    fn intentional_template_rejections_keep_validation_identity() {
        let template = ChatTemplateValue(Either::Left(
            "{{ raise_exception('messages must alternate') }}".to_string(),
        ));
        let error = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            None,
            &template,
            None,
            None,
            None,
            Vec::new(),
        )
        .unwrap_err();

        assert!(super::is_chat_template_request_error(&error));
    }

    #[test]
    fn tools_require_a_compatible_template() {
        let template = ChatTemplateValue(Either::Right(vec![HashMap::from([
            ("name".to_string(), "default".to_string()),
            ("template".to_string(), "{{ messages }}".to_string()),
        ])]));
        let tool = Tool {
            tp: ToolType::Function,
            function: Function {
                name: "lookup".to_string(),
                description: None,
                parameters: None,
                strict: None,
            },
        };
        let error = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            None,
            &template,
            None,
            None,
            None,
            vec![tool],
        )
        .unwrap_err();

        assert!(super::is_chat_template_request_error(&error));
    }

    #[test]
    fn detects_tool_call_format_from_template() {
        let cases = [
            (
                "<|tool_call>call:name{}<tool_call|>",
                ToolCallFormat::Gemma4,
            ),
            (
                "<|tool_call_start|>[name()]<|tool_call_end|>",
                ToolCallFormat::Liquid,
            ),
            ("<|python_tag|>{{ tool }}", ToolCallFormat::Llama),
            ("[TOOL_CALLS]{{ tool_calls }}", ToolCallFormat::MistralNemo),
            (
                "<tool_calls>{{ tool_calls }}</tool_calls>",
                ToolCallFormat::Hunyuan,
            ),
            ("<｜tool▁call▁begin｜>function", ToolCallFormat::DeepSeek),
            ("<tool_call>{{ tool }}</tool_call>", ToolCallFormat::Qwen),
            (
                "<|start|>assistant<|channel|>commentary<|message|>",
                ToolCallFormat::Harmony,
            ),
            (
                "<|start|>assistant to=tool<|message|><atem:function_calls><atem:invoke",
                ToolCallFormat::Atem,
            ),
        ];

        for (template, expected) in cases {
            assert_eq!(template_tool_call_format(template), Some(expected));
        }
    }

    #[test]
    fn muse_channel_tokens_are_not_alternate_eos() {
        use ahash::AHashMap;
        use tokenizers::models::wordlevel::WordLevel;

        let vocab = [
            ("<unk>".to_string(), 0),
            ("<|eot|>".to_string(), 1),
            ("<|start|>".to_string(), 2),
            ("<|message|>".to_string(), 3),
        ]
        .into_iter()
        .collect::<AHashMap<_, _>>();
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".to_string())
                .build()
                .unwrap(),
        );
        let template: ChatTemplate = serde_json::from_value(serde_json::json!({
            "eos_token": "<|eot|>",
            "chat_template": "<|start|>assistant to=user<|message|>"
        }))
        .unwrap();

        assert_eq!(calculate_eos_tokens(&template, None, &tokenizer), vec![1]);
    }

    #[test]
    fn atem_template_receives_xhigh_reasoning_strength_and_mapping_arguments() {
        let template = ChatTemplateValue(Either::Left(
            "{{ reasoning_strength }}:{{ messages[0]['tool_calls'][0]['function']['arguments']['city'] }}<atem:function_calls><atem:invoke"
                .to_string(),
        ));
        let messages = vec![assistant_message_with_tool_calls()];

        let rendered = apply_chat_template_to(
            messages,
            false,
            None,
            Some(ReasoningEffort::XHigh),
            &template,
            None,
            None,
            None,
            Vec::new(),
        )
        .unwrap();

        assert!(rendered.starts_with("xhigh:Boston"));
    }

    #[test]
    fn atem_template_uses_its_default_reasoning_strength() {
        let template = ChatTemplateValue(Either::Left(
            "{% set value = reasoning_strength if reasoning_strength is defined and reasoning_strength else 'high' %}{{ value }}<atem:function_calls><atem:invoke"
                .to_string(),
        ));

        let rendered = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            None,
            &template,
            None,
            None,
            None,
            Vec::new(),
        )
        .unwrap();

        assert!(rendered.starts_with("high"));
    }

    #[test]
    fn atem_template_accepts_inline_conditionals_in_namespace_arguments() {
        let template = ChatTemplateValue(Either::Left(
            "{% set tcid = '' %}{% set rns = namespace(name=tcid if tcid else '') %}{{ rns.name }}<atem:function_calls><atem:invoke"
                .to_string(),
        ));

        let rendered = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            None,
            &template,
            None,
            None,
            None,
            Vec::new(),
        )
        .unwrap();

        assert_eq!(rendered, "<atem:function_calls><atem:invoke");
    }

    #[test]
    fn unspecified_thinking_enables_template_thinking() {
        let template = ChatTemplateValue(Either::Left(
            "{% if enable_thinking is defined and enable_thinking %}<|think|>{% endif %}{{ bos_token }}{{ messages[0]['content'] }}".to_string(),
        ));
        let messages = vec![user_text_message("hello")];

        let rendered = apply_chat_template_to(
            messages,
            false,
            None,
            None,
            &template,
            Some("<bos>".to_string()),
            None,
            None,
            vec![],
        )
        .unwrap();
        let enabled = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            Some(true),
            None,
            &template,
            Some("<bos>".to_string()),
            None,
            None,
            vec![],
        )
        .unwrap();

        const { assert!(DEFAULT_ENABLE_THINKING) };
        assert_eq!(rendered, "<|think|><bos>hello");
        assert_eq!(rendered, enabled);
    }

    #[test]
    fn unspecified_effort_is_undefined_in_templates() {
        let template = ChatTemplateValue(Either::Left(
            "{% if reasoning_effort is defined %}effort{% else %}no-effort{% endif %}:{% if reasoning_strength is defined %}strength{% else %}no-strength{% endif %}:{% if enable_thinking %}enabled{% else %}disabled{% endif %}"
                .to_string(),
        ));

        let rendered = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            None,
            &template,
            None,
            None,
            None,
            vec![],
        )
        .unwrap();

        assert_eq!(rendered, "no-effort:no-strength:enabled");
    }

    #[test]
    fn explicit_effort_sets_both_template_names_and_toggle() {
        let template = ChatTemplateValue(Either::Left(
            "{{ reasoning_effort }}:{{ reasoning_strength }}:{% if enable_thinking %}enabled{% else %}disabled{% endif %}"
                .to_string(),
        ));

        let xhigh = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            Some(ReasoningEffort::XHigh),
            &template,
            None,
            None,
            None,
            vec![],
        )
        .unwrap();
        let off = apply_chat_template_to(
            vec![user_text_message("hello")],
            false,
            None,
            Some(ReasoningEffort::Off),
            &template,
            None,
            None,
            None,
            vec![],
        )
        .unwrap();

        assert_eq!(xhigh, "xhigh:xhigh:enabled");
        assert_eq!(off, "off:off:disabled");
    }

    #[test]
    fn generation_config_exposes_sampling_defaults() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "temperature": 1.0,
                "top_k": 32,
                "top_p": 0.9,
                "min_p": 0.05,
                "repetition_penalty": 1.1,
                "max_new_tokens": 512,
                "suppress_tokens": [258882, 258883]
            }"#,
        )
        .unwrap();

        let defaults = config.generation_defaults().unwrap();
        assert_eq!(defaults.do_sample, Some(true));
        assert_eq!(defaults.temperature, Some(1.0));
        assert_eq!(defaults.top_k, Some(32));
        assert_eq!(defaults.top_p, Some(0.9));
        assert_eq!(defaults.min_p, Some(0.05));
        assert_eq!(defaults.repetition_penalty, Some(1.1));
        assert_eq!(defaults.max_new_tokens, Some(512));
        assert_eq!(defaults.suppress_tokens, Some(vec![258882, 258883]));
    }

    fn assistant_message_with_tool_calls() -> IndexMap<String, MessageContent> {
        let mut tc_map = IndexMap::new();
        tc_map.insert("id".to_string(), Value::String("call-1".to_string()));
        tc_map.insert("type".to_string(), Value::String("function".to_string()));
        let mut func = serde_json::Map::new();
        func.insert("name".to_string(), Value::String("get_weather".to_string()));
        func.insert(
            "arguments".to_string(),
            Value::String(r#"{"city":"Boston"}"#.to_string()),
        );
        tc_map.insert("function".to_string(), Value::Object(func));

        IndexMap::from([
            ("role".to_string(), Either::Left("assistant".to_string())),
            (
                "content".to_string(),
                Either::Left(
                    r#"{"name":"get_weather","arguments":"{\"city\":\"Boston\"}"}"#.to_string(),
                ),
            ),
            ("tool_calls".to_string(), Either::Right(vec![tc_map])),
        ])
    }

    fn tool_result_message(name: &str, content: &str) -> IndexMap<String, MessageContent> {
        IndexMap::from([
            ("role".to_string(), Either::Left("tool".to_string())),
            ("name".to_string(), Either::Left(name.to_string())),
            ("content".to_string(), Either::Left(content.to_string())),
        ])
    }

    #[test]
    fn gemma4_preprocess_creates_user_msg_for_tool_responses() {
        let mut messages = vec![
            user_text_message("What's the weather?"),
            assistant_message_with_tool_calls(),
            tool_result_message("get_weather", r#"{"temp":72}"#),
        ];

        preprocess_gemma4_tool_messages(&mut messages);

        // Tool message replaced by a user message with tool_responses
        assert_eq!(messages.len(), 3);
        // Assistant message should NOT have tool_responses
        assert!(!messages[1].contains_key("tool_responses"));
        // Content should be cleared (had tool_calls)
        let content = messages[1].get("content").unwrap();
        assert_eq!(content, &Either::Left(String::new()));
        // New user message should have tool_responses
        let role = messages[2].get("role").unwrap();
        assert_eq!(role, &Either::Left("user".to_string()));
        assert!(messages[2].contains_key("tool_responses"));
    }

    #[test]
    fn gemma4_preprocess_tool_response_has_correct_structure() {
        let mut messages = vec![
            user_text_message("hi"),
            assistant_message_with_tool_calls(),
            tool_result_message("get_weather", r#"{"temp":72}"#),
        ];

        preprocess_gemma4_tool_messages(&mut messages);

        let tool_responses = match messages[2].get("tool_responses").unwrap() {
            Either::Right(v) => v,
            _ => panic!("Expected Either::Right"),
        };
        assert_eq!(tool_responses.len(), 1);
        assert_eq!(tool_responses[0]["name"], "get_weather");
        // Content was valid JSON → parsed into a Value, not a string
        assert_eq!(tool_responses[0]["response"]["temp"], 72);
    }

    #[test]
    fn detects_templates_that_walk_tool_call_arguments() {
        assert!(super::iterates_tool_call_arguments(
            "{%- for k, v in tool_call.arguments|items %}"
        ));
        assert!(super::iterates_tool_call_arguments(
            "{%- for k, v in tool_call.arguments | items %}"
        ));
        assert!(super::iterates_tool_call_arguments(
            "{% for k, v in tool_call.arguments.items() %}"
        ));
        assert!(!super::iterates_tool_call_arguments(
            "{{ tool_call.arguments | tojson }}"
        ));
    }

    #[test]
    fn parse_tool_call_arguments_converts_json_string_to_object() {
        let mut messages = vec![
            user_text_message("call something"),
            assistant_message_with_tool_calls(),
        ];
        // Before: arguments is a JSON string
        if let Some(Either::Right(ref tcs)) = messages[1].get("tool_calls") {
            let func = tcs[0].get("function").unwrap();
            assert!(func.get("arguments").unwrap().is_string());
        }

        super::parse_tool_call_arguments(&mut messages);

        // After: arguments should be a parsed object
        if let Some(Either::Right(ref tcs)) = messages[1].get("tool_calls") {
            let func = tcs[0].get("function").unwrap();
            let args = func.get("arguments").unwrap();
            assert!(args.is_object(), "arguments should be parsed to object");
            assert_eq!(args.get("city").unwrap(), "Boston");
        } else {
            panic!("expected tool_calls");
        }
    }

    #[test]
    fn gemma4_preprocess_multiple_tool_messages() {
        let mut messages = vec![
            user_text_message("hi"),
            assistant_message_with_tool_calls(),
            tool_result_message("get_weather", r#"{"temp":72}"#),
            tool_result_message("get_forecast", "sunny"),
        ];

        preprocess_gemma4_tool_messages(&mut messages);

        // assistant + one user msg replaces the two tool msgs
        assert_eq!(messages.len(), 3);
        let tool_responses = match messages[2].get("tool_responses").unwrap() {
            Either::Right(v) => v,
            _ => panic!("Expected Either::Right"),
        };
        assert_eq!(tool_responses.len(), 2);
        assert_eq!(tool_responses[0]["name"], "get_weather");
        assert_eq!(tool_responses[1]["name"], "get_forecast");
        // Non-JSON content falls back to string
        assert_eq!(tool_responses[1]["response"], "sunny");
    }

    #[test]
    fn gemma4_preprocess_no_tool_messages_is_noop() {
        let mut messages = vec![
            user_text_message("hello"),
            IndexMap::from([
                ("role".to_string(), Either::Left("assistant".to_string())),
                ("content".to_string(), Either::Left("hi there".to_string())),
            ]),
        ];
        let original_len = messages.len();

        preprocess_gemma4_tool_messages(&mut messages);

        assert_eq!(messages.len(), original_len);
    }

    #[test]
    fn gemma4_preprocess_tool_without_name_defaults_to_unknown() {
        let mut messages = vec![
            user_text_message("hi"),
            assistant_message_with_tool_calls(),
            // Tool message without "name" field
            IndexMap::from([
                ("role".to_string(), Either::Left("tool".to_string())),
                ("content".to_string(), Either::Left("result".to_string())),
            ]),
        ];

        preprocess_gemma4_tool_messages(&mut messages);

        let tool_responses = match messages[2].get("tool_responses").unwrap() {
            Either::Right(v) => v,
            _ => panic!("Expected Either::Right"),
        };
        assert_eq!(tool_responses[0]["name"], "unknown");
    }

    #[test]
    fn generation_config_keeps_omitted_sampling_fields_unset() {
        let config: GenerationConfig = serde_json::from_str(
            r#"{
                "do_sample": true,
                "temperature": 1.0
            }"#,
        )
        .unwrap();

        let defaults = config.generation_defaults().unwrap();
        assert_eq!(defaults.do_sample, Some(true));
        assert_eq!(defaults.temperature, Some(1.0));
        assert_eq!(defaults.top_k, None);
        assert_eq!(defaults.top_p, None);
        assert_eq!(defaults.repetition_penalty, None);
        assert_eq!(defaults.max_new_tokens, None);
        assert_eq!(defaults.max_length, None);
        assert_eq!(defaults.suppress_tokens, None);
    }
}
