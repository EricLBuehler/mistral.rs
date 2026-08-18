use anyhow::{bail, Result};
use mistralrs_quant::{GgufArchive, GgufBindingMap};

use crate::MultimodalLoaderType;

use super::{
    gemma3_bindings::build_gemma3_bindings,
    gemma3n_bindings::build_gemma3n_bindings,
    idefics3_bindings::build_idefics3_bindings,
    lfm2_vl_bindings::build_lfm2_vl_bindings,
    llama4_bindings::build_llama4_bindings,
    mistral3_bindings::build_mistral3_bindings,
    multimodal_binding_utils::{metadata_string, projector_type},
    muse_glimmer_bindings::build_muse_glimmer_bindings,
    normal_registry::RopePairing,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum NativeMultimodalGgufFamily {
    Gemma3,
    Gemma3n,
    Idefics3,
    Mistral3,
    Llama4,
    Lfm2Vl,
    MuseGlimmer,
}

pub(crate) struct NativeMultimodalGguf {
    pub loader_type: MultimodalLoaderType,
    pub bindings: GgufBindingMap,
    pub rope_pairing: RopePairing,
}

pub(crate) fn resolve_native_multimodal_gguf(
    archive: &GgufArchive,
) -> Result<Option<NativeMultimodalGguf>> {
    let architecture = metadata_string(archive, "general.architecture")?
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata `general.architecture` is required"))?;
    let Some(family) = family_from_names(architecture, projector_type(archive)?)? else {
        return Ok(None);
    };
    Ok(Some(NativeMultimodalGguf {
        loader_type: family.loader_type(),
        bindings: family.build_bindings(archive)?,
        rope_pairing: family.rope_pairing(),
    }))
}

impl NativeMultimodalGgufFamily {
    fn loader_type(self) -> MultimodalLoaderType {
        match self {
            Self::Gemma3 => MultimodalLoaderType::Gemma3,
            Self::Gemma3n => MultimodalLoaderType::Gemma3n,
            Self::Idefics3 => MultimodalLoaderType::Idefics3,
            Self::Mistral3 => MultimodalLoaderType::Mistral3,
            Self::Llama4 => MultimodalLoaderType::Llama4,
            Self::Lfm2Vl => MultimodalLoaderType::Lfm2Vl,
            Self::MuseGlimmer => MultimodalLoaderType::MuseGlimmer,
        }
    }

    fn rope_pairing(self) -> RopePairing {
        match self {
            Self::Gemma3 | Self::Gemma3n | Self::Lfm2Vl => RopePairing::HalfSplit,
            Self::Idefics3 | Self::Mistral3 | Self::Llama4 | Self::MuseGlimmer => {
                RopePairing::Adjacent
            }
        }
    }

    fn build_bindings(self, archive: &GgufArchive) -> Result<GgufBindingMap> {
        match self {
            Self::Gemma3 => build_gemma3_bindings(archive),
            Self::Gemma3n => build_gemma3n_bindings(archive),
            Self::Idefics3 => build_idefics3_bindings(archive),
            Self::Mistral3 => build_mistral3_bindings(archive),
            Self::Llama4 => build_llama4_bindings(archive),
            Self::Lfm2Vl => build_lfm2_vl_bindings(archive),
            Self::MuseGlimmer => build_muse_glimmer_bindings(archive),
        }
    }
}

fn family_from_names(
    architecture: &str,
    projector: Option<&str>,
) -> Result<Option<NativeMultimodalGgufFamily>> {
    let family = match projector {
        Some("gemma3") => {
            require_architecture(architecture, "gemma3", "gemma3")?;
            NativeMultimodalGgufFamily::Gemma3
        }
        Some("gemma3nv") => {
            require_architecture(architecture, "gemma3n", "gemma3nv")?;
            NativeMultimodalGgufFamily::Gemma3n
        }
        Some("idefics3") => {
            require_architecture(architecture, "llama", "idefics3")?;
            NativeMultimodalGgufFamily::Idefics3
        }
        Some("pixtral") => {
            require_architecture(architecture, "mistral3", "pixtral")?;
            NativeMultimodalGgufFamily::Mistral3
        }
        Some("llama4") => {
            require_architecture(architecture, "llama4", "llama4")?;
            NativeMultimodalGgufFamily::Llama4
        }
        Some("lfm2") => {
            require_architecture(architecture, "lfm2", "lfm2")?;
            NativeMultimodalGgufFamily::Lfm2Vl
        }
        Some("muse-glimmer") => {
            require_architecture(architecture, "muse-glimmer", "muse-glimmer")?;
            NativeMultimodalGgufFamily::MuseGlimmer
        }
        Some(_) | None => return Ok(None),
    };
    Ok(Some(family))
}

fn require_architecture(architecture: &str, expected: &str, projector: &str) -> Result<()> {
    if architecture != expected {
        bail!(
            "vision projector `{projector}` requires `{expected}` GGUF architecture, \
             found `{architecture}`"
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_supported_converter_pairs() {
        for (architecture, projector, family, pairing) in [
            (
                "gemma3",
                "gemma3",
                NativeMultimodalGgufFamily::Gemma3,
                RopePairing::HalfSplit,
            ),
            (
                "gemma3n",
                "gemma3nv",
                NativeMultimodalGgufFamily::Gemma3n,
                RopePairing::HalfSplit,
            ),
            (
                "llama",
                "idefics3",
                NativeMultimodalGgufFamily::Idefics3,
                RopePairing::Adjacent,
            ),
            (
                "mistral3",
                "pixtral",
                NativeMultimodalGgufFamily::Mistral3,
                RopePairing::Adjacent,
            ),
            (
                "llama4",
                "llama4",
                NativeMultimodalGgufFamily::Llama4,
                RopePairing::Adjacent,
            ),
            (
                "lfm2",
                "lfm2",
                NativeMultimodalGgufFamily::Lfm2Vl,
                RopePairing::HalfSplit,
            ),
            (
                "muse-glimmer",
                "muse-glimmer",
                NativeMultimodalGgufFamily::MuseGlimmer,
                RopePairing::Adjacent,
            ),
        ] {
            let detected = family_from_names(architecture, Some(projector))
                .unwrap()
                .unwrap();
            assert_eq!(detected, family);
            assert_eq!(detected.rope_pairing(), pairing);
        }
    }

    #[test]
    fn rejects_cross_family_projector_pair() {
        let error = family_from_names("llama", Some("pixtral")).unwrap_err();
        assert!(error.to_string().contains("requires `mistral3`"));
    }

    #[test]
    fn leaves_other_projectors_for_other_registries() {
        assert_eq!(
            family_from_names("qwen3vl", Some("qwen3vl_merger")).unwrap(),
            None
        );
        assert_eq!(family_from_names("gemma4", Some("gemma4v")).unwrap(), None);
    }

    #[test]
    fn excludes_non_intersecting_converter_formats() {
        for (architecture, projector) in [
            ("qwen2", "resampler"),
            ("qwen3", "minicpmv4_6"),
            ("phi3", "phi3v"),
            ("llama", "idefics2"),
        ] {
            assert_eq!(
                family_from_names(architecture, Some(projector)).unwrap(),
                None
            );
        }
    }
}
