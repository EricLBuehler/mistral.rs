use anyhow::Result;
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::{AddedToken, Tokenizer};
use tekken::{Tekkenizer, SpecialTokenPolicy};

/// A simple Encoding type that holds token IDs
#[derive(Clone, Debug)]
pub struct Encoding {
    ids: Vec<u32>,
    // We could add more fields in the future like:
    // attention_mask: Vec<u32>,
    // offsets: Vec<(usize, usize)>,
    // etc.
}

impl Encoding {
    /// Create a new Encoding from token IDs
    pub fn new(ids: Vec<u32>) -> Self {
        Self { ids }
    }

    /// Get the token IDs
    pub fn get_ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get the attention mask (all 1s for now)
    pub fn get_attention_mask(&self) -> Vec<u32> {
        vec![1; self.ids.len()]
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

/// Enum representing different tokenizer implementations
#[derive(Clone)]
pub enum TokenizerImpl {
    /// HuggingFace tokenizers (from tokenizer.json)
    HuggingFace(Tokenizer),
    /// GGUF embedded tokenizers
    Gguf {
        tokenizer: Tokenizer,
        bos: Option<String>,
        eos: Option<String>,
        unk: Option<String>,
    },
    /// Tekken tokenizer (using tekken-rs)
    Tekken {
        tokenizer: Arc<Tekkenizer>,
        vocab_size: usize,
    },
}

impl TokenizerImpl {
    /// Create a new HuggingFace tokenizer variant
    pub fn new_huggingface(tokenizer: Tokenizer) -> Self {
        Self::HuggingFace(tokenizer)
    }

    /// Create a new GGUF tokenizer variant
    pub fn new_gguf(
        tokenizer: Tokenizer,
        bos: Option<String>,
        eos: Option<String>,
        unk: Option<String>,
    ) -> Self {
        Self::Gguf {
            tokenizer,
            bos,
            eos,
            unk,
        }
    }

    /// Create a new Tekken tokenizer variant
    pub fn new_tekken(tokenizer: Tekkenizer) -> Result<Self> {
        let vocab_size = tokenizer.vocab_size();
        Ok(Self::Tekken {
            tokenizer: Arc::new(tokenizer),
            vocab_size,
        })
    }

    /// Encode text into tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        match self {
            Self::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(encoding.get_ids().to_vec()))
            }
            Self::Gguf { tokenizer, .. } => {
                let encoding = tokenizer
                    .encode(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(encoding.get_ids().to_vec()))
            }
            Self::Tekken { tokenizer, .. } => {
                let tokens = tokenizer.encode(text, add_special_tokens, add_special_tokens)
                    .map_err(|e| anyhow::anyhow!("Tekken encoding error: {}", e))?;
                Ok(Encoding::new(tokens))
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
            Self::HuggingFace(tokenizer) => {
                let encoding = tokenizer
                    .encode_fast(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(encoding.get_ids().to_vec()))
            }
            Self::Gguf { tokenizer, .. } => {
                let encoding = tokenizer
                    .encode_fast(text, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(Encoding::new(encoding.get_ids().to_vec()))
            }
            Self::Tekken { .. } => {
                // For Tekken, we'll fall back to regular encode
                // since it doesn't have a separate fast encode method
                let text_str = match text.into() {
                    tokenizers::EncodeInput::Single(s) => match s {
                        tokenizers::InputSequence::Raw(text) => text.into(),
                        tokenizers::InputSequence::PreTokenized(tokens) => tokens.join(" "),
                        tokenizers::InputSequence::PreTokenizedOwned(tokens) => {
                            tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
                        }
                        tokenizers::InputSequence::PreTokenizedCow(tokens) => {
                            tokens.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(" ")
                        }
                    },
                    tokenizers::EncodeInput::Dual(s1, s2) => {
                        let s1_str = match s1 {
                            tokenizers::InputSequence::Raw(text) => text.into(),
                            tokenizers::InputSequence::PreTokenized(tokens) => tokens.join(" "),
                            tokenizers::InputSequence::PreTokenizedOwned(tokens) => {
                                tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
                            }
                            tokenizers::InputSequence::PreTokenizedCow(tokens) => {
                                tokens.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(" ")
                            }
                        };
                        let s2_str = match s2 {
                            tokenizers::InputSequence::Raw(text) => text.into(),
                            tokenizers::InputSequence::PreTokenized(tokens) => tokens.join(" "),
                            tokenizers::InputSequence::PreTokenizedOwned(tokens) => {
                                tokens.iter().map(|s| s.as_str()).collect::<Vec<_>>().join(" ")
                            }
                            tokenizers::InputSequence::PreTokenizedCow(tokens) => {
                                tokens.iter().map(|s| s.as_ref()).collect::<Vec<_>>().join(" ")
                            }
                        };
                        format!("{} {}", s1_str, s2_str)
                    }
                };
                self.encode(&text_str, add_special_tokens)
            }
        }
    }

    /// Decode tokens into text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Gguf { tokenizer, .. } => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Tekken { tokenizer, .. } => {
                let policy = if skip_special_tokens {
                    SpecialTokenPolicy::Ignore
                } else {
                    SpecialTokenPolicy::Keep
                };
                tokenizer.decode(ids, policy)
                    .map_err(|e| anyhow::anyhow!("Tekken decoding error: {}", e))
            }
        }
    }

    /// Get the vocabulary
    pub fn get_vocab(&self, with_added_tokens: bool) -> Option<HashMap<String, u32>> {
        match self {
            Self::HuggingFace(tokenizer) => Some(tokenizer.get_vocab(with_added_tokens)),
            Self::Gguf { tokenizer, .. } => Some(tokenizer.get_vocab(with_added_tokens)),
            Self::Tekken { .. } => {
                // Tekken doesn't expose vocabulary directly
                None
            }
        }
    }

    /// Get vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.get_vocab_size(with_added_tokens),
            Self::Gguf { tokenizer, .. } => tokenizer.get_vocab_size(with_added_tokens),
            Self::Tekken { vocab_size, .. } => *vocab_size,
        }
    }

    /// Add special tokens
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.add_special_tokens(tokens),
            Self::Gguf { tokenizer, .. } => tokenizer.add_special_tokens(tokens),
            Self::Tekken { .. } => {
                // Tekken handles special tokens internally
                0
            }
        }
    }

    /// Get the underlying HuggingFace tokenizer if available
    pub fn get_hf_tokenizer(&self) -> Option<&Tokenizer> {
        match self {
            Self::HuggingFace(tokenizer) => Some(tokenizer),
            Self::Gguf { tokenizer, .. } => Some(tokenizer),
            Self::Tekken { .. } => None,
        }
    }

    /// Get BOS token if available
    pub fn bos_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { bos, .. } => bos.as_deref(),
            Self::Tekken { .. } => None, // Tekken handles special tokens internally
        }
    }

    /// Get EOS token if available
    pub fn eos_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { eos, .. } => eos.as_deref(),
            Self::Tekken { .. } => None, // Tekken handles special tokens internally
        }
    }

    /// Get UNK token if available
    pub fn unk_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { unk, .. } => unk.as_deref(),
            Self::Tekken { .. } => None, // Tekken handles special tokens internally
        }
    }

    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.token_to_id(token),
            Self::Gguf { tokenizer, .. } => tokenizer.token_to_id(token),
            Self::Tekken { .. } => {
                // Tekken doesn't expose token to ID mapping
                None
            }
        }
    }

    /// Set padding parameters
    pub fn with_padding(&mut self, padding: Option<tokenizers::PaddingParams>) -> &mut Self {
        match self {
            Self::HuggingFace(tokenizer) => {
                tokenizer.with_padding(padding);
            }
            Self::Gguf { tokenizer, .. } => {
                tokenizer.with_padding(padding);
            }
            Self::Tekken { .. } => {
                // Tekken doesn't support padding configuration
            }
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
            Self::HuggingFace(tokenizer) => {
                let encodings = tokenizer
                    .encode_batch(input_strings, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(encodings.into_iter()
                    .map(|e| Encoding::new(e.get_ids().to_vec()))
                    .collect())
            }
            Self::Gguf { tokenizer, .. } => {
                let encodings = tokenizer
                    .encode_batch(input_strings, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(encodings.into_iter()
                    .map(|e| Encoding::new(e.get_ids().to_vec()))
                    .collect())
            }
            Self::Tekken { tokenizer, .. } => {
                // Tekken doesn't support batch encoding directly
                // Encode each string individually
                let mut results = Vec::new();
                for text in input_strings {
                    let tokens = tokenizer.encode(&text, add_special_tokens, add_special_tokens)
                        .map_err(|e| anyhow::anyhow!("Tekken encoding error: {}", e))?;
                    results.push(Encoding::new(tokens));
                }
                Ok(results)
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
            Self::HuggingFace(tokenizer) => tokenizer
                .decode_batch(token_sequences, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Gguf { tokenizer, .. } => tokenizer
                .decode_batch(token_sequences, skip_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Tekken { tokenizer, .. } => {
                // Decode each sequence individually
                let policy = if skip_special_tokens {
                    SpecialTokenPolicy::Ignore
                } else {
                    SpecialTokenPolicy::Keep
                };
                let mut results = Vec::new();
                for seq in token_sequences {
                    results.push(tokenizer.decode(seq, policy)?);
                }
                Ok(results)
            }
        }
    }
}
