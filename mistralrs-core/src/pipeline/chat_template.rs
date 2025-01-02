use std::collections::HashMap;

use anyhow::Result;
use either::Either;
use indexmap::IndexMap;
use itertools::Itertools;
use minijinja::{context, value::Kwargs, Environment, Error, ErrorKind, Value};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;
use tracing::info;

use crate::{MessageContent, Tool};

const SUPPORTED_ALTERNATE_EOS: &[&str] = &[
    "<|im_end|>",    // Handle ChatML case
    "<end_of_turn>", // Handle Gemma2 chat case
];

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct AddedTokensDecoder {
    __type: Option<String>,
    pub content: String,
    lstrip: bool,
    normalized: bool,
    rstrip: bool,
    single_word: bool,
    special: Option<bool>,
}

fn raise_exception(msg: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::InvalidOperation, msg))
}

#[derive(Debug, Deserialize)]
pub struct BeginEndUnkPadTok(
    #[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>,
);

#[derive(Debug, Deserialize)]
pub struct ChatTemplateValue(
    #[serde(with = "either::serde_untagged")] pub Either<String, Vec<HashMap<String, String>>>,
);

#[allow(dead_code)]
#[derive(Debug, Deserialize, Default)]
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
    gen_conf: Option<GenerationConfig>,
    tokenizer: &Tokenizer,
) -> Vec<u32> {
    let mut eos_tok_ids = chat_template.eos_tok().map(|x| vec![x]).unwrap_or_default();
    let mut bos_tok_ids = chat_template.bos_tok().map(|b| vec![b]).unwrap_or_default();

    for alternate in SUPPORTED_ALTERNATE_EOS {
        if tokenizer.get_vocab(true).contains_key(*alternate) {
            eos_tok_ids.push(alternate.to_string())
        }
    }

    if let Some(gen_conf) = gen_conf {
        let ids = match gen_conf.eos_token_id {
            Either::Left(id) => vec![id],
            Either::Right(ids) => ids,
        };
        for id in ids {
            let s = tokenizer
                .decode(&[id], false)
                .unwrap_or_else(|_| panic!("Unable to decode id {id})"));
            if !eos_tok_ids.contains(&s) {
                eos_tok_ids.push(s);
            }
        }

        let ids = match gen_conf.bos_token_id {
            Either::Left(id) => vec![id],
            Either::Right(ids) => ids,
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

    eos_tok_ids = eos_tok_ids.into_iter().dedup().collect::<Vec<_>>();
    bos_tok_ids = bos_tok_ids.into_iter().dedup().collect::<Vec<_>>();

    let bos_render = bos_tok_ids
        .iter()
        .map(|val| format!("{:?}", val))
        .collect::<Vec<String>>()
        .join(", ");
    let eos_render = eos_tok_ids
        .iter()
        .map(|val| format!("{:?}", val))
        .collect::<Vec<String>>()
        .join(", ");

    info!(
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
#[derive(Debug, Deserialize)]
pub struct GenerationConfig {
    #[serde(with = "either::serde_untagged")]
    bos_token_id: Either<u32, Vec<u32>>,
    #[serde(with = "either::serde_untagged")]
    eos_token_id: Either<u32, Vec<u32>>,
}

fn tojson(value: Value, kwargs: Kwargs) -> Result<Value, Error> {
    if let Ok(indent) = kwargs.get("indent") {
        let mut buf = Vec::new();
        let repeat = b" ".repeat(indent);
        let formatter = serde_json::ser::PrettyFormatter::with_indent(&repeat);
        let mut ser = serde_json::Serializer::with_formatter(&mut buf, formatter);
        value.serialize(&mut ser).unwrap();
        String::from_utf8(buf).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    } else {
        serde_json::to_string(&value).map_err(|err| {
            Error::new(ErrorKind::BadSerialization, "cannot serialize to JSON").with_source(err)
        })
    }
    .map_err(|err| {
        Error::new(ErrorKind::InvalidOperation, "cannot serialize to JSON").with_source(err)
    })
    .map(|s| {
        // When this filter is used the return value is safe for both HTML and JSON
        let mut rv = String::with_capacity(s.len());
        for c in s.chars() {
            match c {
                '<' => rv.push_str("\\u003c"),
                '>' => rv.push_str("\\u003e"),
                '&' => rv.push_str("\\u0026"),
                '\'' => rv.push_str("\\u0027"),
                _ => rv.push(c),
            }
        }
        Value::from_safe_string(rv)
    })
}

pub fn apply_chat_template_to(
    messages: Vec<IndexMap<String, MessageContent>>,
    add_generation_prompt: bool,
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
    let mut new_messages = Vec::new();
    for message in messages {
        let mut new_message = IndexMap::new();
        for (k, v) in message {
            new_message.insert(k, UntaggedContent(v));
        }
        new_messages.push(new_message);
    }

    let template = match &template.0 {
        Either::Left(x) => x.clone(),
        Either::Right(map) => {
            let mut template = "".to_string();
            for t in map {
                if t.contains_key("tool_use") && !tools.is_empty() {
                    template = t["tool_use"].clone();
                    break;
                } else if t.contains_key("default") {
                    template = t["default"].clone();
                    break;
                }
            }
            if template.is_empty() {
                anyhow::bail!("Chat template does not contain a `tool_use` or `default` key. Please ensure it contains at least a `default` key, although `tool_use` should be specified for using tools.");
            }
            template
        }
    };

    env.add_template("chat_template", &template)?;
    env.add_function("raise_exception", raise_exception);
    env.add_filter("tojson", tojson);
    let tmpl = env.get_template("chat_template").unwrap();

    let date = chrono::Utc::now();
    let date_string = date.format("%d, %B, %Y").to_string();

    if tools.is_empty() {
        Ok(tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            date_string => date_string,
        })?)
    } else {
        Ok(tmpl.render(context! {
            messages => new_messages,
            add_generation_prompt => add_generation_prompt,
            bos_token => bos_tok,
            eos_token => eos_tok,
            unk_token => unk_tok,
            tools => tools,
            date_string => date_string,
        })?)
    }
}
