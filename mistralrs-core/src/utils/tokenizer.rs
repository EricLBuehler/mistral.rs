use std::{collections::HashMap, path::Path};

use anyhow::Result;
use base64::Engine;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::{tokenizer, Tokenizer};

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
}

#[derive(Deserialize)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
    token_str: Option<String>,
}

#[derive(Deserialize)]
struct TekkenTokenizer {
    vocab: Vec<TekkenVocabEntry>,
    pattern: String,
    special_tokens: HashMap<String, usize>,
}

fn load_tekken_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = std::fs::read(path)?;
    let tekken: TekkenTokenizer = serde_json::from_slice(&raw)?;
    
    // Build vocabulary mapping from bytes to token IDs
    let mut vocab_map = HashMap::new();
    for entry in &tekken.vocab {
        // Decode base64 bytes
        let token_bytes = base64::engine::general_purpose::STANDARD.decode(&entry.token_bytes)?;
        let token_str = if let Some(ref s) = entry.token_str {
            s.clone()
        } else {
            // Convert bytes to string representation for HF tokenizer
            String::from_utf8_lossy(&token_bytes).to_string()
        };
        vocab_map.insert(token_str, entry.rank);
    }
    
    // Create HuggingFace tokenizer compatible JSON structure
    let tokenizer_json = serde_json::json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": tekken.special_tokens.iter().map(|(k, v)| {
            serde_json::json!({
                "id": v,
                "content": k,
                "single_word": false,
                "lstrip": false,
                "rstrip": false,
                "normalized": false,
                "special": true
            })
        }).collect::<Vec<_>>(),
        "normalizer": null,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {
                "Regex": tekken.pattern
            },
            "behavior": "Isolated",
            "invert": false
        },
        "post_processor": null,
        "decoder": {
            "type": "ByteLevel",
            "add_prefix_space": false,
            "trim_offsets": true,
            "use_regex": true
        },
        "model": {
            "type": "BPE",
            "dropout": null,
            "unk_token": null,
            "continuing_subword_prefix": null,
            "end_of_word_suffix": null,
            "fuse_unk": false,
            "vocab": vocab_map,
            "merges": []
        }
    });
    
    let tokenizer_bytes = serde_json::to_vec(&tokenizer_json)?;
    Tokenizer::from_bytes(&tokenizer_bytes).map_err(anyhow::Error::msg)
}

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    // Check if this is a tekken.json file
    if p.as_ref().file_name()
        .and_then(|name| name.to_str())
        .map(|name| name == "tekken.json")
        .unwrap_or(false)
    {
        let mut tokenizer = load_tekken_tokenizer(p)?;
        if let Some(added_tokens) = processor_added_tokens {
            tokenizer.add_special_tokens(
                &added_tokens
                    .iter()
                    .map(|x| tokenizer::AddedToken::from(x.to_string(), true))
                    .collect::<Vec<_>>(),
            );
        }
        return Ok(tokenizer);
    }

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
