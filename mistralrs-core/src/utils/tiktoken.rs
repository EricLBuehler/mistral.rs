use ahash::AHashMap;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{BpeBuilder, Merges, Vocab},
    pre_tokenizers::byte_level::ByteLevel,
    tokenizer::Tokenizer,
};

pub fn convert_tiktoken_to_tokenizers<P: AsRef<Path>>(
    tokenizer_model_path: P,
) -> Result<Tokenizer> {
    let model_bytes = fs::read(&tokenizer_model_path)?;

    let (vocab, merges) = extract_vocab_merges_from_model(&model_bytes)?;

    let bpe_model = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .build()
        .map_err(|e| anyhow!("Failed to build BPE model: {}", e))?;

    let mut tokenizer = Tokenizer::new(bpe_model);

    // Set up pre-tokenizer with ByteLevel
    let pre_tokenizer = ByteLevel::new(false, false, false);
    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    // Set up decoder
    let decoder = ByteLevelDecoder::new(false, false, false);
    tokenizer.with_decoder(Some(decoder));

    Ok(tokenizer)
}

fn extract_vocab_merges_from_model(model_bytes: &[u8]) -> Result<(Vocab, Merges)> {
    let mut vocab = AHashMap::new();
    let mut token_to_rank = HashMap::new();

    let lines = String::from_utf8_lossy(model_bytes);

    for line in lines.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            continue;
        }

        let token_base64 = parts[0];
        let rank_str = parts[1];

        let token_bytes = STANDARD.decode(token_base64)?;
        let rank: u32 = rank_str.parse()?;

        let token = bytes_to_unicode(&token_bytes);

        vocab.insert(token.clone(), rank);
        token_to_rank.insert(token_bytes, (token, rank));
    }

    let mut sorted_tokens: Vec<_> = token_to_rank.into_iter().collect();
    sorted_tokens.sort_by_key(|(_, (_, rank))| *rank);

    let mut merges = Vec::new();

    for i in 256..sorted_tokens.len() {
        let (token_bytes, (_, _)) = &sorted_tokens[i];

        if token_bytes.len() >= 2 {
            let mut best_pair = None;
            let mut best_rank_sum = u32::MAX;

            // Try all possible splits to find the best pair based on BPE merge rules
            for split_pos in 1..token_bytes.len() {
                let left_bytes = &token_bytes[..split_pos];
                let right_bytes = &token_bytes[split_pos..];

                let left_str = bytes_to_unicode(left_bytes);
                let right_str = bytes_to_unicode(right_bytes);

                // Check if both parts exist in vocab (they should have lower ranks)
                if let (Some(&left_rank), Some(&right_rank)) =
                    (vocab.get(&left_str), vocab.get(&right_str))
                {
                    // In BPE, pairs that were merged earlier have lower combined rank
                    let rank_sum = left_rank + right_rank;
                    if rank_sum < best_rank_sum && left_rank < i as u32 && right_rank < i as u32 {
                        best_rank_sum = rank_sum;
                        best_pair = Some((left_str, right_str));
                    }
                }
            }

            if let Some((left, right)) = best_pair {
                merges.push((left, right));
            }
        }
    }

    Ok((vocab, merges))
}

fn bytes_to_unicode(bytes: &[u8]) -> String {
    let mut bs: Vec<u32> = vec![];
    for i in 0..=255u8 {
        bs.push(i as u32);
    }

    let mut cs: Vec<u32> = bs.clone();
    let mut n = 0;

    for b in 0..=255u8 {
        if !(b.is_ascii_alphanumeric() || b"!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ ".contains(&b)) {
            bs.push(256 + n);
            cs.push(b as u32);
            n += 1;
        }
    }

    let cs_chars: Vec<char> = cs.iter().map(|&c| char::from_u32(c).unwrap()).collect();

    let mut result = String::new();
    for &byte in bytes {
        let idx = bs.iter().position(|&b| b == byte as u32).unwrap();
        result.push(cs_chars[idx]);
    }

    result
}

#[cfg(test)]
mod tests {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let test_bytes = b"hello";
        let result = bytes_to_unicode(test_bytes);
        assert_eq!(result, "hello");

        let test_bytes = vec![0, 1, 2, 255];
        let result = bytes_to_unicode(&test_bytes);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_tiktoken_conversion() -> anyhow::Result<()> {
        let api = ApiBuilder::new().with_progress(true).build().unwrap();
        let api = api.repo(Repo::with_revision(
            "EricB/mistralrs_tests".to_string(),
            RepoType::Model,
            "main".to_string(),
        ));

        let converted_tokenizer = {
            let tokenizer_filename = api.get("tokenizer_llama3.model").unwrap();
            convert_tiktoken_to_tokenizers(tokenizer_filename).unwrap()
        };

        let truth_tokenizer = {
            let tokenizer_filename = api.get("tokenizer_llama3.json").unwrap();
            Tokenizer::from_file(tokenizer_filename).unwrap()
        };

        // Test encoding
        let converted_encoding = converted_tokenizer
            .encode("hello world", false)
            .map_err(|e| anyhow!("Failed to encode with converted_tokenizer: {}", e))?;
        let converted_ids = converted_encoding.get_ids();

        let truth_encoding = truth_tokenizer
            .encode("hello world", false)
            .map_err(|e| anyhow!("Failed to encode with truth_tokenizer: {}", e))?;
        let truth_ids = truth_encoding.get_ids();

        assert_eq!(truth_ids, converted_ids);

        Ok(())
    }
}
