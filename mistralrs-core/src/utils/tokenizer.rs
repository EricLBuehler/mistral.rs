use std::{collections::HashMap, path::Path};

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::{tokenizer, Tokenizer};

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
}

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub(crate) fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    let mut tokenizer = {
        let raw = std::fs::read(p.clone()).map_err(anyhow::Error::msg)?;
        let mut tokenizer: Value = serde_json::from_slice(&raw).unwrap();
        let added_tokens: Vec<AddedToken> =
            serde_json::from_value(tokenizer["added_tokens"].clone()).unwrap();
        let vocab: HashMap<String, usize> =
            serde_json::from_value(tokenizer["model"]["vocab"].clone()).unwrap();
        for token in added_tokens {
            if !vocab.contains_key(&token.content) {
                tokenizer["model"]["vocab"]
                    .as_object_mut()
                    .unwrap()
                    .insert(token.content, token.id.into())
                    .ok_or(())
                    .unwrap_err();
            }
        }
        let raw_fixed = serde_json::to_vec_pretty(&tokenizer).unwrap();
        Tokenizer::from_bytes(&raw_fixed).map_err(anyhow::Error::msg)?
    };
    if let Some(added_tokens) = processor_added_tokens {
        tokenizer.add_special_tokens(
            &added_tokens
                .iter()
                .map(|x| tokenizer::AddedToken::from(x.to_string(), true))
                .collect::<Vec<_>>(),
        );
    }
    Ok(tokenizer)
}
