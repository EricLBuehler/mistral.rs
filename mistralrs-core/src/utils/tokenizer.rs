use std::{collections::HashMap, path::Path};

use ahash::AHashMap;
use anyhow::{anyhow, Result};
use base64::Engine;
use serde::Deserialize;
use serde_json::Value;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder,
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    tokenizer::{self, normalizer::SplitDelimiterBehavior, Tokenizer},
};

use super::tiktoken::token_bytes_to_string;

#[derive(Deserialize)]
struct AddedToken {
    id: usize,
    content: String,
}

#[derive(Deserialize)]
struct TekkenVocabEntry {
    rank: usize,
    token_bytes: String,
}

#[derive(Deserialize)]
struct TekkenSpecialToken {
    rank: usize,
    token_str: String,
}

#[derive(Deserialize)]
struct TekkenConfig {
    pattern: String,
    default_vocab_size: usize,
    default_num_special_tokens: usize,
}

#[derive(Deserialize)]
struct TekkenTokenizer {
    config: TekkenConfig,
    vocab: Vec<TekkenVocabEntry>,
    #[serde(default)]
    special_tokens: Vec<TekkenSpecialToken>,
}

fn load_tekken_tokenizer<P: AsRef<Path>>(path: P) -> Result<Tokenizer> {
    let raw = std::fs::read(path)?;
    let tekken: TekkenTokenizer = serde_json::from_slice(&raw)?;

    let num_special = tekken.config.default_num_special_tokens;
    let inner_vocab_size = tekken.config.default_vocab_size - num_special;

    // Build bpe_ranks: bytes -> rank (only ranks < inner_vocab_size)
    let mut bpe_ranks: HashMap<Vec<u8>, u32> = HashMap::new();
    for entry in &tekken.vocab {
        if entry.rank >= inner_vocab_size {
            continue;
        }
        let token_bytes = base64::engine::general_purpose::STANDARD.decode(&entry.token_bytes)?;
        #[allow(clippy::cast_possible_truncation)]
        bpe_ranks.insert(token_bytes, entry.rank as u32);
    }

    // Build vocab with IDs offset by num_special (special tokens occupy 0..num_special)
    let mut vocab = AHashMap::new();
    for (token_bytes, rank) in &bpe_ranks {
        let token_str = token_bytes_to_string(token_bytes);
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(token_str, *rank + num_special as u32);
    }

    // Extract BPE merges: for each multi-byte token, find the best split
    // into two pieces that are both in bpe_ranks (same algorithm as tiktoken.rs)
    let mut merges = Vec::new();
    for (token, rank) in &bpe_ranks {
        if token.len() == 1 {
            continue;
        }
        let mut local = Vec::new();
        for index in 1..token.len() {
            let piece_l = &token[..index];
            let piece_r = &token[index..];
            if let (Some(&rank_l), Some(&rank_r)) = (bpe_ranks.get(piece_l), bpe_ranks.get(piece_r))
            {
                local.push((piece_l.to_vec(), piece_r.to_vec(), *rank, rank_l, rank_r));
            }
        }
        local.sort_by_key(|(_, _, _, rank_l, rank_r)| (*rank_l, *rank_r));
        for (piece_l, piece_r, rank, _, _) in local {
            merges.push((piece_l, piece_r, rank));
        }
    }
    merges.sort_by_key(|(_, _, rank)| *rank);
    let merges: Vec<(String, String)> = merges
        .into_iter()
        .map(|(l, r, _)| (token_bytes_to_string(&l), token_bytes_to_string(&r)))
        .collect();

    // Build special token name lookup from the parsed special_tokens field
    let mut special_token_map: HashMap<usize, String> = HashMap::new();
    for st in &tekken.special_tokens {
        special_token_map.insert(st.rank, st.token_str.clone());
    }

    // Add special tokens to the BPE vocab so IDs 0..num_special are defined
    #[allow(clippy::cast_possible_truncation)]
    let special_token_names: Vec<String> = (0..num_special)
        .map(|id| {
            special_token_map
                .get(&id)
                .cloned()
                .unwrap_or_else(|| format!("<SPECIAL_{id}>"))
        })
        .collect();
    for (id, name) in special_token_names.iter().enumerate() {
        #[allow(clippy::cast_possible_truncation)]
        vocab.insert(name.clone(), id as u32);
    }

    // Build BPE model with proper vocab and merges
    let bpe_model = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("Failed to build BPE model: {}", e))?;

    let mut tokenizer = Tokenizer::new(bpe_model);

    // Pre-tokenizer: Split on tekken pattern, then ByteLevel
    let split = Split::new(
        SplitPattern::Regex(tekken.config.pattern),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("Failed to create split pre-tokenizer: {}", e))?;
    let byte_level = ByteLevel::new(false, true, true);
    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ]);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    // Decoder
    let decoder = ByteLevelDecoder::new(true, false, false);
    tokenizer.with_decoder(Some(decoder));

    // Register special tokens so the tokenizer treats them as special
    let special_tokens: Vec<tokenizer::AddedToken> = special_token_names
        .into_iter()
        .map(|name| tokenizer::AddedToken::from(name, true))
        .collect();
    tokenizer.add_special_tokens(&special_tokens);

    Ok(tokenizer)
}

/// May fix the tokenizer according to: https://gist.github.com/jneuff/682d47b786329f19291d166957b3274a
pub(crate) fn get_tokenizer<P: AsRef<Path> + Clone>(
    p: P,
    processor_added_tokens: Option<&[&str]>,
) -> Result<Tokenizer> {
    // Check if this is a tekken.json file
    if p.as_ref()
        .file_name()
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
