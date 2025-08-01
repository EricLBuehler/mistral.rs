use anyhow::Result;
use std::collections::HashMap;
use tokenizers::{AddedToken, Tokenizer};

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
        tokenizer: Tokenizer,
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
        }
    }

    /// Decode tokens into text
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> Result<String> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer
                .decode(ids, skip_special_tokens)
                .map_err(anyhow::Error::msg),
        }
    }

    /// Get the vocabulary
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.get_vocab(with_added_tokens),
        }
    }

    /// Get vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.get_vocab_size(with_added_tokens),
        }
    }

    /// Add special tokens
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.add_special_tokens(tokens),
        }
    }

    /// Get the underlying tokenizer
    pub fn get_tokenizer(&self) -> &Tokenizer {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer,
        }
    }

    /// Get BOS token if available
    pub fn bos_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { bos, .. } => bos.as_deref(),
        }
    }

    /// Get EOS token if available
    pub fn eos_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { eos, .. } => eos.as_deref(),
        }
    }

    /// Get UNK token if available
    pub fn unk_token(&self) -> Option<&str> {
        match self {
            Self::Tokenizer { unk, .. } => unk.as_deref(),
        }
    }

    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::Tokenizer { tokenizer, .. } => tokenizer.token_to_id(token),
        }
    }

    /// Set padding parameters
    pub fn with_padding(&mut self, padding: Option<tokenizers::PaddingParams>) -> &mut Self {
        match self {
            Self::Tokenizer { tokenizer, .. } => {
                tokenizer.with_padding(padding);
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
            Self::Tokenizer { tokenizer, .. } => {
                let encodings = tokenizer
                    .encode_batch(input_strings, add_special_tokens)
                    .map_err(anyhow::Error::msg)?;
                Ok(encodings
                    .into_iter()
                    .map(|e| Encoding::new(e.get_ids().to_vec(), e.get_attention_mask().to_vec()))
                    .collect())
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
        }
    }
}
