use anyhow::Result;
use std::collections::HashMap;
use tiktoken_rs::CoreBPE;
use tokenizers::{
    tokenizer::{EncodeInput, InputSequence},
    AddedToken, Tokenizer,
};

/// A simple Encoding type that holds token IDs
#[derive(Clone, Debug)]
pub struct Encoding {
    ids: Vec<u32>,
    attention_mask: Vec<u32>,
}

impl Encoding {
    /// Create a new Encoding from token IDs
    pub fn new(ids: Vec<u32>, attention_mask: Vec<u32>) -> Self {
        Self {
            ids,
            attention_mask,
        }
    }

    /// Get the token IDs
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get the attention mask
    pub fn get_attention_mask(&self) -> &[u32] {
        &self.attention_mask
    }

    /// Get the length of the encoding
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if the encoding is empty
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }
}

/// Wrapper for tokenizer implementations
#[derive(Clone)]
pub enum TokenizerImpl {
    /// Tokenizer instance with optional special tokens
    Tokenizer {
        tokenizer: Box<Tokenizer>,
        bos: Option<String>,
        eos: Option<String>,
        unk: Option<String>,
    },
    /// tiktoken-rs CoreBPE tokenizer
    TikToken {
        tokenizer: Box<CoreBPE>,
        bos: Option<String>,
        eos: Option<String>,
        unk: Option<String>,
    },
}

impl TokenizerImpl {
    /// Encode text into tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        match self {
            Self::Tokenizer { tokenizer, .. } => {
                let encoding = tokenizer
                    .encode(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(
                    encoding.get_ids().to_vec(),
                    encoding.get_attention_mask().to_vec(),
                ))
            }
            Self::TikToken { tokenizer, .. } => {
                let ids = if add_special_tokens {
                    tokenizer.encode_with_special_tokens(text)
                } else {
                    tokenizer.encode_ordinary(text)
                };
                let attention_mask = vec![1u32; ids.len()];
                Ok(Encoding::new(ids, attention_mask))
            }
        }
    }

    /// Fast encode text into tokens
    pub fn encode_fast<'a>(
        &self,
        text: impl Into<tokenizers::EncodeInput<'a>>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        match self {
            Self::Tokenizer { tokenizer, .. } => {
                let encoding = tokenizer
                    .encode_fast(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(
                    encoding.get_ids().to_vec(),
                    encoding.get_attention_mask().to_vec(),
                ))
            }
            Self::TikToken { tokenizer, .. } => {
                // tiktoken expects raw string input
                let enc_input: EncodeInput = text.into();
                let s = if let EncodeInput::Single(InputSequence::Raw(cow)) = enc_input {
                    cow.into_owned()
                } else {
                    return Err(anyhow::anyhow!("Unsupported input for tiktoken"));
                };
                let ids = if add_special_tokens {
                    tokenizer.encode_with_special_tokens(&s)
                } else {
                    tokenizer.encode_ordinary(&s)
                };
                let attention_mask = vec![1u32; ids.len()];
                Ok(Encoding::new(ids, attention_mask))
            }
        }
    }

    /// Decode tokens into text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::TikToken { tokenizer, .. } => {
                let ranks: Vec<tiktoken_rs::Rank> = ids.to_vec();
                tokenizer.decode(ranks).map_err(anyhow::Error::msg)
            }
        }
    }

    /// Get the vocabulary
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.get_vocab(with_added_tokens),
            Self::TikToken { .. } => HashMap::new(),
        }
    }

    /// Get vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.get_vocab_size(with_added_tokens),
            Self::TikToken { .. } => 0,
        }
    }

    /// Add special tokens
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.add_special_tokens(tokens),
            Self::TikToken { .. } => 0,
        }
    }

    /// Get the underlying tokenizer
    pub fn get_tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer,
            Self::TikToken { .. } => panic!("Token access not supported for tiktoken"),
        }
    }

    /// Get BOS token if available
    pub fn bos_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { bos, .. } => bos.as_deref(),
            Self::TikToken { bos, .. } => bos.as_deref(),
        }
    }

    /// Get EOS token if available
    pub fn eos_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { eos, .. } => eos.as_deref(),
            Self::TikToken { eos, .. } => eos.as_deref(),
        }
    }

    /// Get UNK token if available
    pub fn unk_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { unk, .. } => unk.as_deref(),
            Self::TikToken { unk, .. } => unk.as_deref(),
        }
    }

    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.token_to_id(token),
            Self::TikToken { tokenizer, .. } => {
                let ids = tokenizer.encode_with_special_tokens(token);
                if ids.len() == 1 {
                    Some(ids[0])
                } else {
                    None
                }
            }
        }
    }

    /// Set padding parameters
    pub fn with_padding(&mut self, padding: Option<tokenizers::PaddingParams>) -> &mut Self {
        match self {
            Self::Tokenizer { tokenizer, .. } => {
                tokenizer.with_padding(padding);
            }
            Self::TikToken { .. } => {}
        }
        self
    }

    /// Encode batch of texts
    pub fn encode_batch<E>(&self, inputs: E, add_special_tokens: bool) -> Result<Vec<Encoding>>
    where
        E: IntoIterator,
        E::Item: AsRef<str>,
    {
        // Collect into strings first to simplify the trait bounds
        let input_strings: Vec<String> =
            inputs.into_iter().map(|s| s.as_ref().to_string()).collect();

        match self {
            Self::Tokenizer { tokenizer, .. } => {
                let encodings = tokenizer
                    .encode_batch(input_strings, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(encodings
                    .into_iter()
                    .map(|e| Encoding::new(e.get_ids().to_vec(), e.get_attention_mask().to_vec()))
                    .collect())
            }
            Self::TikToken { tokenizer, .. } => {
                let mut result = Vec::new();
                for text in input_strings {
                    let ids = if add_special_tokens {
                        tokenizer.encode_with_special_tokens(&text)
                    } else {
                        tokenizer.encode_ordinary(&text)
                    };
                    let attention_mask = vec![1u32; ids.len()];
                    result.push(Encoding::new(ids, attention_mask));
                }
                Ok(result)
            }
        }
    }

    /// Decode batch of token sequences
    pub fn decode_batch(
        &self,
        token_sequences: &[&[u32]],
        skip_special_tokens: bool,
    ) -> Result<Vec<String>> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer
                .decode_batch(token_sequences, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::TikToken { tokenizer, .. } => {
                let mut res = Vec::new();
                for seq in token_sequences {
                    let ranks: Vec<_> = seq.to_vec();
                    res.push(tokenizer.decode(ranks).map_err(anyhow::Error::msg)?);
                }
                Ok(res)
            }
        }
    }
}
