// Originally from https://github.com/microsoft/aici/blob/64f0b551dee49e320e9b3b92289f3d6f2e888276/aicirt/src/bintokens.rs
// Licensed under the MIT license

use crate::aici::{bytes::TokRxInfo, toktree::TokTrie};
use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use tokenizers::{normalizers::Sequence, NormalizerWrapper, Tokenizer};
use tracing::{error, warn};

#[derive(Serialize, Deserialize)]
pub struct ByteTokenizer {
    pub hf_model: String,
    pub hf_tokenizer: Tokenizer,
    pub eos_token: u32,
    pub vocab_size: u32,
    token_bytes: Vec<Vec<u8>>,
    pub special: BTreeMap<String, u32>,
}
fn is_self_mapped(c: char) -> bool {
    matches!(c, '!'..='~' | '\u{00A1}'..='\u{00AC}' | '\u{00AE}'..='\u{00FF}')
}
fn build_char_map() -> HashMap<char, u8> {
    let mut res = HashMap::default();
    let mut k = 0x100u32;
    for byte in 0..=255u8 {
        let c = byte as char;
        if is_self_mapped(c) {
            res.insert(c, byte);
        } else {
            res.insert(char::from_u32(k).unwrap(), byte);
            k += 1;
        }
    }
    res
}

impl ByteTokenizer {
    pub fn from_tokenizer(mut hft: Tokenizer) -> Result<ByteTokenizer> {
        let mut is_byte_level = false;
        let mut is_byte_fallback = false;
        let mut space_ch = ' ';

        // remove the "Prepend space"
        if let Some(n) = hft.get_normalizer() {
            let n = match n {
                NormalizerWrapper::Sequence(x) => NormalizerWrapper::Sequence(Sequence::new(
                    x.get_normalizers()
                        .iter()
                        .filter_map(|n| match n {
                            NormalizerWrapper::Prepend(_) => None,
                            _ => Some(n.clone()),
                        })
                        .collect(),
                )),
                _ => n.clone(),
            };
            hft.with_normalizer(n);
        }

        if let Some(d) = hft.get_decoder() {
            // DecoderWrapper::Sequence() doesn't let one access the decoders
            // so we resort to json munching
            let v = serde_json::to_value(d).unwrap();
            if v["type"].as_str() == Some("ByteLevel") {
                is_byte_level = true;
            } else if v["type"].as_str() == Some("Sequence") {
                if let Some(decoders) = v["decoders"].as_array() {
                    for decoder in decoders {
                        if decoder["type"].as_str() == Some("ByteFallback") {
                            is_byte_fallback = true;
                        } else if decoder["type"].as_str() == Some("Replace")
                            && decoder["content"].as_str() == Some(" ")
                        {
                            if let Some(s) = decoder["pattern"]["String"].as_str() {
                                let s: Vec<char> = s.chars().collect();
                                if s.len() == 1 {
                                    space_ch = s[0];
                                }
                            }
                        }
                    }
                }
            }
        }

        if !is_byte_fallback && !is_byte_level {
            bail!("can't determine decoder type: {:?}", hft.get_decoder());
        }

        #[allow(clippy::cast_possible_truncation)]
        let vocab_size = hft.get_vocab_size(true) as u32;
        let added = hft.get_added_tokens_decoder();

        let mut res = ByteTokenizer {
            hf_model: "foobar".to_string(),
            eos_token: 0,
            vocab_size,
            special: BTreeMap::new(),
            token_bytes: (0..vocab_size).map(|_| Vec::new()).collect(),
            hf_tokenizer: hft,
        };

        for (id, info) in added.iter() {
            if info.special {
                match info.content.as_str() {
                    "</s>" | "<|endoftext|>" => res.eos_token = *id,
                    _ => {}
                }
                res.special.insert(info.content.clone(), *id);
            } else {
                res.token_bytes[*id as usize] = info.content.clone().into_bytes();
            }
        }

        let char_map = build_char_map();

        for tok_id in 0..vocab_size {
            if added.contains_key(&tok_id) {
                continue;
            }
            if let Some(tok_name) = res.hf_tokenizer.id_to_token(tok_id) {
                if is_byte_fallback {
                    if tok_name.len() == 6 && tok_name.starts_with("<0x") && tok_name.ends_with('>')
                    {
                        // parse hex number from tok_name
                        let hex_str = &tok_name[3..5];
                        let byte = u8::from_str_radix(hex_str, 16).unwrap();
                        res.token_bytes[tok_id as usize] = vec![byte];
                    } else {
                        assert!(!tok_name.starts_with("<0x"));
                        let tok_name = tok_name.replace(space_ch, " ");
                        res.token_bytes[tok_id as usize] = tok_name.as_bytes().to_vec();
                    }
                } else if is_byte_level {
                    let bytes: Result<Vec<u8>> = tok_name
                        .chars()
                        .map(|c| {
                            char_map
                                .get(&c)
                                .copied()
                                .ok_or_else(|| anyhow!("missing char: {}", c))
                        })
                        .collect();
                    let bytes = match bytes {
                        Ok(b) => b,
                        Err(e) => {
                            error!("error: {} for {:?}", e, tok_name);
                            continue;
                        }
                    };

                    res.token_bytes[tok_id as usize] = bytes;
                } else {
                    panic!();
                }
            } else {
                warn!("missing token: {}", tok_id);
            }
        }

        Ok(res)
    }
}

impl ByteTokenizer {
    pub fn tokrx_info(&self) -> TokRxInfo {
        TokRxInfo {
            vocab_size: self.vocab_size,
            tok_eos: self.eos_token,
        }
    }
    pub fn token_bytes(&self) -> Vec<Vec<u8>> {
        self.token_bytes.clone()
    }
}

pub(crate) fn build_tok_trie(tokenizer: Tokenizer) -> TokTrie {
    let bt = ByteTokenizer::from_tokenizer(tokenizer).unwrap();
    TokTrie::from(&bt.tokrx_info(), &bt.token_bytes())
}
