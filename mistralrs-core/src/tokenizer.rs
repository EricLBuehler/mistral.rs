use anyhow::Result;
use std::collections::HashMap;
use tokenizers::{AddedToken, Encoding, Tokenizer};

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

    /// Encode text into tokens
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> Result<Encoding> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .encode(text, add_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Gguf { tokenizer, .. } => tokenizer
                .encode(text, add_special_tokens)
                .map_err(anyhow::Error::msg),
        }
    }

    /// Fast encode text into tokens
    pub fn encode_fast<'a>(
        &self,
        text: impl Into<tokenizers::EncodeInput<'a>>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer
                .encode_fast(text, add_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Gguf { tokenizer, .. } => tokenizer
                .encode_fast(text, add_special_tokens)
                .map_err(anyhow::Error::msg),
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
        }
    }

    /// Get the vocabulary
    pub fn get_vocab(&self, with_added_tokens: bool) -> HashMap<String, u32> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.get_vocab(with_added_tokens),
            Self::Gguf { tokenizer, .. } => tokenizer.get_vocab(with_added_tokens),
        }
    }

    /// Get vocabulary size
    pub fn get_vocab_size(&self, with_added_tokens: bool) -> usize {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.get_vocab_size(with_added_tokens),
            Self::Gguf { tokenizer, .. } => tokenizer.get_vocab_size(with_added_tokens),
        }
    }

    /// Add special tokens
    pub fn add_special_tokens(&mut self, tokens: &[AddedToken]) -> usize {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.add_special_tokens(tokens),
            Self::Gguf { tokenizer, .. } => tokenizer.add_special_tokens(tokens),
        }
    }

    /// Get the underlying tokenizer (for compatibility during transition)
    pub fn get_base_tokenizer(&self) -> &Tokenizer {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer,
            Self::Gguf { tokenizer, .. } => tokenizer,
        }
    }

    /// Get the underlying tokenizer mutably (for compatibility during transition)
    pub fn get_base_tokenizer_mut(&mut self) -> &mut Tokenizer {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer,
            Self::Gguf { tokenizer, .. } => tokenizer,
        }
    }

    /// Get BOS token if available
    pub fn bos_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { bos, .. } => bos.as_deref(),
        }
    }

    /// Get EOS token if available
    pub fn eos_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { eos, .. } => eos.as_deref(),
        }
    }

    /// Get UNK token if available
    pub fn unk_token(&self) -> Option<&str> {
        match self {
            Self::HuggingFace(_) => None, // HF tokenizers handle this internally
            Self::Gguf { unk, .. } => unk.as_deref(),
        }
    }

    /// Get token ID from token string
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(tokenizer) => tokenizer.token_to_id(token),
            Self::Gguf { tokenizer, .. } => tokenizer.token_to_id(token),
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
            Self::HuggingFace(tokenizer) => tokenizer
                .encode_batch(input_strings, add_special_tokens)
                .map_err(anyhow::Error::msg),
            Self::Gguf { tokenizer, .. } => tokenizer
                .encode_batch(input_strings, add_special_tokens)
                .map_err(anyhow::Error::msg),
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
        }
    }
}
