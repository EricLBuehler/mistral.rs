pub(crate) mod base_model;
mod chat_template;
mod content;
pub(crate) mod gemma3_bindings;
pub(crate) mod gemma3_config;
pub(crate) mod gemma3n_bindings;
mod gguf_tokenizer;
pub(crate) mod idefics3_bindings;
mod lfm2_vl_bindings;
pub(crate) mod llama4_bindings;
mod mistral3_bindings;
mod multimodal_binding_utils;
pub(crate) mod multimodal_bindings;
pub(crate) mod multimodal_vision_registry;
pub(crate) mod normal_bindings;
pub(crate) mod normal_config;
pub(crate) mod normal_registry;
pub(crate) mod qwen_multimodal_bindings;
use strum::EnumString;

use anyhow::{Context, Result};
pub(crate) use chat_template::{get_gguf_chat_template, get_gguf_chat_template_from_metadata};
pub(crate) use content::Content;
pub(crate) use gguf_tokenizer::{
    convert_gguf_metadata_to_hf_tokenizer, validate_external_gguf_tokenizer,
    GgufTokenizerConversion,
};
use std::str::FromStr;

pub const GGUF_MULTI_FILE_DELIMITER: &str = ";";

#[derive(Debug, EnumString, Clone, Copy, strum::Display)]
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
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown GGUF architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }
}
