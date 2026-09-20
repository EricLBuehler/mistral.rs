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
pub(crate) mod muse_glimmer_bindings;
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

/// Direct-dispatch architectures in `pipeline::gguf::load_native_multimodal` that don't go
/// through `multimodal_vision_registry::family_from_names` (they don't need the projector's
/// own `clip.vision.projector_type` / `clip.audio.projector_type` to disambiguate).
/// Keep in sync with the `match architecture.as_str()` arm there.
const DIRECT_MULTIMODAL_ARCHITECTURES: &[&str] = &[
    "gemma4",
    "qwen2vl",
    "qwen3vl",
    "qwen3vlmoe",
    "qwen35",
    "qwen35moe",
];

/// Whether a `general.architecture` / projector-type pairing from GGUF metadata would actually
/// be accepted by the native multimodal GGUF loader. Used by callers (e.g. the CLI's directory
/// scan for a companion `mmproj*.gguf`) that want to predict, without loading the model, whether
/// attaching a given projector to a given base model is meaningful, rather than just present in
/// the same directory.
pub fn gguf_architecture_accepts_projector(
    architecture: &str,
    vision_projector_type: Option<&str>,
    audio_projector_type: Option<&str>,
) -> bool {
    let architecture = architecture.to_ascii_lowercase();
    if DIRECT_MULTIMODAL_ARCHITECTURES.contains(&architecture.as_str()) {
        return true;
    }
    let family_matches = |projector: Option<&str>| {
        multimodal_vision_registry::family_from_names(&architecture, projector)
            .ok()
            .flatten()
            .is_some()
    };
    family_matches(vision_projector_type) || family_matches(audio_projector_type)
}

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

#[cfg(test)]
mod projector_compat_tests {
    use super::gguf_architecture_accepts_projector;

    #[test]
    fn direct_dispatch_architectures_accept_any_projector_claim() {
        assert!(gguf_architecture_accepts_projector("gemma4", None, None));
        assert!(gguf_architecture_accepts_projector("qwen2vl", None, None));
        assert!(gguf_architecture_accepts_projector("QWEN3VL", None, None));
    }

    #[test]
    fn matched_family_pairing_is_accepted() {
        assert!(gguf_architecture_accepts_projector(
            "llama",
            Some("idefics3"),
            None
        ));
        assert!(gguf_architecture_accepts_projector(
            "mistral3",
            Some("pixtral"),
            None
        ));
        assert!(gguf_architecture_accepts_projector(
            "gemma3",
            Some("gemma3"),
            None
        ));
    }

    #[test]
    fn mismatched_family_pairing_is_rejected() {
        // The exact repro from mistral.rs issue #2421: a text-only qwen2 GGUF sitting next
        // to an unrelated idefics3 (llama-arch) projector in the same directory.
        assert!(!gguf_architecture_accepts_projector(
            "qwen2",
            Some("idefics3"),
            None
        ));
        assert!(!gguf_architecture_accepts_projector("qwen2", None, None));
    }

    #[test]
    fn unknown_projector_type_is_rejected() {
        assert!(!gguf_architecture_accepts_projector(
            "llama",
            Some("some-future-projector"),
            None
        ));
    }
}
