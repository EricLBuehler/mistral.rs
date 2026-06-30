mod chat_template;
mod content;
mod gguf_tokenizer;
use strum::EnumString;

use anyhow::{Context, Result};
pub(crate) use chat_template::get_gguf_chat_template;
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{convert_gguf_to_hf_tokenizer, GgufTokenizerConversion};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";

const MAX_SHOWN_ARCH_LEN: usize = 64;

#[derive(Debug, EnumString, Clone, Copy, strum::Display, strum::VariantNames)]
#[strum(serialize_all = "lowercase")]
pub enum GGUFArchitecture {
    Llama,
    Mpt,
    Gptneox,
    Gptj,
    Gpt2,
    Bloom,
    Falcon,
    Mamba,
    Rwkv,
    Phi2,
    Phi3,
    Starcoder2,
    Qwen2,
    Qwen3,
    Qwen3MoE,
    Mistral3,
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl GGUFArchitecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        let value = value.as_ref();
        Self::from_str(&value.to_ascii_lowercase())
            .with_context(|| {
                // Untrusted file metadata headed for terminal/logs: cap and escape it.
                let mut chars = value.chars();
                let shown: String = chars
                    .by_ref()
                    .take(MAX_SHOWN_ARCH_LEN)
                    .flat_map(char::escape_default)
                    .collect();
                let ellipsis = if chars.next().is_some() { "..." } else { "" };
                format!(
                    "Unknown GGUF architecture `{shown}{ellipsis}`: the file's metadata declares \
                    an architecture this version of mistral.rs does not support (the model may \
                    be newer than this build). Supported architectures: {}",
                    <Self as strum::VariantNames>::VARIANTS.join(", ")
                )
            })
            .map_err(anyhow::Error::msg)
    }
}
