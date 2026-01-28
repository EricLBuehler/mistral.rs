use std::{collections::HashMap, path::Path};

use anyhow::Result;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::{tokenizer, Tokenizer};
use tracing::warn;

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
}

#[derive(Deserialize)]
struct TokenizerConfigAddedToken {
    content: String,
    #[serde(default)]
    special: bool,
}

#[derive(Deserialize)]
struct TokenizerConfig {
    added_tokens_decoder: Option<HashMap<String, TokenizerConfigAddedToken>>,
}

/// Load special tokens from a `tokenizer_config.json` file and add them to the tokenizer.
fn add_special_tokens_from_tokenizer_config(
    tokenizer: &mut Tokenizer,
    config_path: &Path,
) -> Result<()> {
    let raw = std::fs::read_to_string(config_path)?;
    let config: TokenizerConfig = serde_json::from_str(&raw)?;

    if let Some(added_tokens) = config.added_tokens_decoder {
        let special_tokens: Vec<tokenizer::AddedToken> = added_tokens
            .into_iter()
            .filter(|(_, v)| v.special)
            .map(|(_, v)| tokenizer::AddedToken::from(v.content, true))
            .collect();

        if !special_tokens.is_empty() {
            tokenizer.add_special_tokens(&special_tokens);
        }
    }

    Ok(())
}

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub(crate) fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    let path = p.as_ref();
    let is_tiktoken = path
        .extension()
        .map_or(false, |ext| ext == "model");

    let mut tokenizer = if is_tiktoken {
        let mut tok = crate::utils::tiktoken::convert_tiktoken_to_tokenizers(path)?;

        // Try to load special tokens from tokenizer_config.json in same directory
        if let Some(dir) = path.parent() {
            let config_path = dir.join("tokenizer_config.json");
            if config_path.exists() {
                if let Err(e) = add_special_tokens_from_tokenizer_config(&mut tok, &config_path) {
                    warn!(
                        "Failed to load special tokens from tokenizer_config.json: {e}"
                    );
                }
            }
        }

        tok
    } else {
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
