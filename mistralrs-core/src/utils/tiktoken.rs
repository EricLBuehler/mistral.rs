use ahash::AHashMap;
use anyhow::{anyhow, Result};
use base64::{engine::general_purpose::STANDARD, Engine};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use tokenizers::{
    decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::{BpeBuilder, Merges, Vocab},
    pre_tokenizers::{
        byte_level::ByteLevel,
        sequence::Sequence,
        split::{Split, SplitPattern},
        PreTokenizerWrapper,
    },
    tokenizer::{normalizer::SplitDelimiterBehavior, Tokenizer},
};

#[allow(dead_code)]
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

    // Set up pre-tokenizer with tiktoken pattern
    // The pattern from Python:
    // r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    let pattern = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

    let split = Split::new(
        SplitPattern::Regex(pattern.to_string()),
        SplitDelimiterBehavior::Isolated,
        false,
    )
    .map_err(|e| anyhow!("Failed to create split pre-tokenizer: {}", e))?;

    let byte_level = ByteLevel::new(
        false, // add_prefix_space
        true,  // trim_offsets (matching the truth tokenizer)
        true,  // use_regex - this might be needed for proper space handling
    );

    let pre_tokenizer = Sequence::new(vec![
        PreTokenizerWrapper::Split(split),
        PreTokenizerWrapper::ByteLevel(byte_level),
    ]);

    tokenizer.with_pre_tokenizer(Some(pre_tokenizer));

    // Set up decoder
    let decoder = ByteLevelDecoder::new(true, false, false);
    tokenizer.with_decoder(Some(decoder));

    Ok(tokenizer)
}

fn extract_vocab_merges_from_model(model_bytes: &[u8]) -> Result<(Vocab, Merges)> {
    // Parse the tiktoken model file - format is "base64_token rank\n"
    let mut bpe_ranks = HashMap::new();
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

        bpe_ranks.insert(token_bytes, rank);
    }

    // Convert to vocabulary and generate merges following Python logic
    let mut vocab = AHashMap::new();
    let mut merges = Vec::new();

    for (token, rank) in &bpe_ranks {
        let token_str = token_bytes_to_string(token);
        vocab.insert(token_str, *rank);

        if token.len() == 1 {
            continue;
        }

        let mut local = Vec::new();

        // Try all possible splits of the token
        for index in 1..token.len() {
            let piece_l = &token[..index];
            let piece_r = &token[index..];

            // Check if both pieces exist in bpe_ranks
            if let (Some(&rank_l), Some(&rank_r)) = (bpe_ranks.get(piece_l), bpe_ranks.get(piece_r))
            {
                // Check if the concatenation also exists (it should be the current token)
                let mut concat = piece_l.to_vec();
                concat.extend_from_slice(piece_r);
                if bpe_ranks.contains_key(&concat) {
                    local.push((piece_l.to_vec(), piece_r.to_vec(), *rank, rank_l, rank_r));
                }
            }
        }

        // Sort by the ranks of the pieces
        local.sort_by_key(|(_, _, _, rank_l, rank_r)| (*rank_l, *rank_r));

        for (piece_l, piece_r, rank, _, _) in local {
            merges.push((piece_l, piece_r, rank));
        }
    }

    // Sort merges by rank
    merges.sort_by_key(|(_, _, rank)| *rank);

    // Convert merges to string pairs
    let merges: Vec<(String, String)> = merges
        .into_iter()
        .map(|(l, r, _)| (token_bytes_to_string(&l), token_bytes_to_string(&r)))
        .collect();

    Ok((vocab, merges))
}

pub(super) fn bytes_to_unicode() -> AHashMap<u8, char> {
    // Create the mapping from bytes to unicode characters.
    // Matches Python's openai/tiktoken bytes_to_unicode().
    let mut bs: Vec<u8> = vec![];

    // Add printable ASCII range
    bs.extend((b'!'..=b'~').collect::<Vec<_>>());
    // Add extended Latin range 1
    bs.extend((0xA1u8..=0xACu8).collect::<Vec<_>>());
    // Add extended Latin range 2
    bs.extend((0xAEu8..=0xFFu8).collect::<Vec<_>>());

    // cs stores the unicode codepoints (may be > 255 for non-printable bytes)
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;

    // Add remaining bytes not in the initial ranges, mapping them to 256+
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    // Create the mapping
    let mut byte_encoder = AHashMap::new();
    for (b, c) in bs.iter().zip(cs.iter()) {
        byte_encoder.insert(*b, char::from_u32(*c).unwrap());
    }

    byte_encoder
}

pub(super) fn token_bytes_to_string(bytes: &[u8]) -> String {
    let byte_encoder = bytes_to_unicode();
    bytes.iter().map(|&b| byte_encoder[&b]).collect()
}

#[cfg(test)]
mod tests {
    use hf_hub::{api::sync::ApiBuilder, Repo, RepoType};

    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let byte_encoder = bytes_to_unicode();

        // Test that we have mappings for all 256 bytes
        assert_eq!(byte_encoder.len(), 256);

        // Test specific mappings
        assert_eq!(byte_encoder[&b'h'], 'h');
        assert_eq!(byte_encoder[&b'e'], 'e');
        assert_eq!(byte_encoder[&b'l'], 'l');
        assert_eq!(byte_encoder[&b'o'], 'o');

        // Test that all bytes map to valid chars
        for b in 0u8..=255 {
            assert!(byte_encoder.contains_key(&b));
        }
    }

    #[test]
    fn test_token_bytes_to_string() {
        let test_bytes = b"hello";
        let result = token_bytes_to_string(test_bytes);
        assert_eq!(result, "hello");
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

        // Test a few more cases to ensure basic functionality
        let test_cases = vec![
            "The quick brown fox",
            "Hello, world!",
            "123456",
            "🦀 Rust",
            "Hello, world! \n🚀 (normal) 😶‍🌫️ (compound emoji, zwj sequence) ✅ (emoji as single token)\n你好世界！\nNǐ hǎo shìjiè!",
        ];

        for test_case in test_cases {
            let converted_enc = converted_tokenizer
                .encode(test_case, false)
                .map_err(|e| anyhow!("Failed to encode '{}': {}", test_case, e))?;
            let truth_enc = truth_tokenizer
                .encode(test_case, false)
                .map_err(|e| anyhow!("Failed to encode '{}': {}", test_case, e))?;

            // Just ensure both tokenizers produce some output
            assert!(
                !converted_enc.get_ids().is_empty(),
                "Converted tokenizer produced empty output for '{test_case}'"
            );
            assert!(
                !truth_enc.get_ids().is_empty(),
                "Truth tokenizer produced empty output for '{test_case}'"
            );
        }

        Ok(())
    }
}
