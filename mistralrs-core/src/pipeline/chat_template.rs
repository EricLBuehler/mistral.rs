use std::collections::HashMap;

use anyhow::Result;
use either::Either;
use indexmap::IndexMap;
use minijinja::{context, Environment, ErrorKind};
use serde::Deserialize;
use tokenizers::Tokenizer;
use tracing::info;

const SUPPORTED_ALTERNATE_EOS: [&str; 2] = [
    "<|eot_id|>", // Handle Llama3 chat case
    "<|im_end|>", // Handle ChatML case
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
pub struct Unk(#[serde(with = "either::serde_untagged")] pub Either<String, AddedTokensDecoder>);

#[allow(dead_code)]
#[derive(Debug, Deserialize)]
pub struct ChatTemplate {
    add_bos_token: Option<bool>,
    add_eos_token: Option<bool>,
    added_tokens_decoder: Option<HashMap<String, AddedTokensDecoder>>,
    additional_special_tokens: Option<Vec<String>>,
    #[serde(with = "either::serde_untagged")]
    pub bos_token: Either<String, AddedTokensDecoder>,
    pub chat_template: Option<String>,
    clean_up_tokenization_spaces: Option<bool>,
    device_map: Option<String>,
    #[serde(with = "either::serde_untagged")]
    pub eos_token: Either<String, AddedTokensDecoder>,
    legacy: Option<bool>,
    model_max_length: f64,
    pad_token: Option<String>,
    sp_model_kwargs: Option<HashMap<String, String>>,
    spaces_between_special_tokens: Option<bool>,
    tokenizer_class: String,
    truncation_size: Option<String>,
    pub unk_token: Option<Unk>,
    use_default_system_prompt: Option<bool>,
}

impl ChatTemplate {
    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    pub fn eos_tok(&self) -> String {
        match self.eos_token {
            Either::Left(ref lit) => lit.clone(),
            Either::Right(ref added) => added.content.clone(),
        }
    }

    pub fn bos_tok(&self) -> String {
        match self.bos_token {
            Either::Left(ref lit) => lit.clone(),
            Either::Right(ref added) => added.content.clone(),
        }
    }

    pub fn unk_tok(&self) -> Option<String> {
        match self.unk_token.as_ref()?.0 {
            Either::Left(ref lit) => Some(lit.clone()),
            Either::Right(ref added) => Some(added.content.clone()),
        }
    }
}

pub fn calculate_eos_tokens(chat_template: &ChatTemplate, tokenizer: &Tokenizer) -> Vec<u32> {
    let mut eos_tok_ids = vec![chat_template.eos_tok()];

    for alternate in SUPPORTED_ALTERNATE_EOS {
        if tokenizer.get_vocab(true).get(alternate).is_some() {
            eos_tok_ids.push(alternate.to_string())
        }
    }

    info!(
        "bos_tok = {}, eos_tok = {:?}, unk_tok = {}",
        chat_template.bos_tok(),
        eos_tok_ids,
        chat_template.eos_tok()
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

pub fn apply_chat_template_to(
    messages: Vec<IndexMap<String, String>>,
    add_generation_prompt: bool,
    template: &str,
    bos_tok: &str,
    eos_tok: &str,
    unk_tok: Option<String>,
) -> Result<String> {
    let mut env = Environment::new();
    // https://github.com/huggingface/transformers/blob/76a33a10923ccc1074917f6b6a1e719e626b7dc9/src/transformers/tokenization_utils_base.py#L1842
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);

    let template = template.replace(".strip()", "|trim");
    env.add_template("chat_template", template.as_str())?;
    env.add_function("raise_exception", raise_exception);
    let tmpl = env.get_template("chat_template").unwrap();
    Ok(tmpl.render(context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_tok,
        eos_token => eos_tok,
        unk_token => unk_tok,
    })?)
}
