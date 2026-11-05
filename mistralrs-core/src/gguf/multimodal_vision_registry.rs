use anyhow::{bail, Result};
use candle_core::DType;
use mistralrs_quant::{GgufArchive, GgufBindingMap};

use crate::MultimodalLoaderType;

use super::{
    gemma3_bindings::build_gemma3_bindings,
    gemma3n_bindings::build_gemma3n_bindings,
    idefics3_bindings::build_idefics3_bindings,
    lfm2_vl_bindings::build_lfm2_vl_bindings,
    lfm3_audio_bindings::{
        build_lfm3_audio_bindings, build_lfm3_audio_input_bindings, build_lfm3_omni_bindings,
    },
    lfm3_d1_config::decision_model_type,
    lfm3_vl_bindings::build_lfm3_vl_bindings,
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
    Lfm3Vl,
    Lfm3Audio,
    Lfm3Omni,
    Lfm3D1,
    MuseGlimmer,
}

pub(crate) struct NativeMultimodalGguf {
    pub loader_type: MultimodalLoaderType,
    pub bindings: GgufBindingMap,
    pub rope_pairing: RopePairing,
}

pub(crate) fn resolve_native_multimodal_gguf(
    archive: &GgufArchive,
    dtype: DType,
) -> Result<Option<NativeMultimodalGguf>> {
    let architecture = metadata_string(archive, "general.architecture")?
        .ok_or_else(|| anyhow::anyhow!("GGUF metadata `general.architecture` is required"))?;
    let family = if decision_model_type(archive.metadata())?.is_some() {
        Some(NativeMultimodalGgufFamily::Lfm3D1)
    } else {
        family_from_names(
            architecture,
            projector_type(archive)?,
            metadata_string(archive, "clip.audio.projector_type")?,
        )?
    };
    let Some(family) = family else {
        return Ok(None);
    };
    Ok(Some(NativeMultimodalGguf {
        loader_type: family.loader_type(),
        bindings: family.build_bindings(archive, dtype)?,
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
            Self::Lfm3Vl => MultimodalLoaderType::Lfm3Vl,
            Self::Lfm3Audio => MultimodalLoaderType::Lfm3Audio,
            Self::Lfm3Omni => MultimodalLoaderType::Lfm3Omni,
            Self::Lfm3D1 => MultimodalLoaderType::Lfm3D1,
            Self::MuseGlimmer => MultimodalLoaderType::MuseGlimmer,
        }
    }

    fn rope_pairing(self) -> RopePairing {
        match self {
            Self::Gemma3
            | Self::Gemma3n
            | Self::Lfm2Vl
            | Self::Lfm3Vl
            | Self::Lfm3Audio
            | Self::Lfm3Omni
            | Self::Lfm3D1 => RopePairing::HalfSplit,
            Self::Idefics3 | Self::Mistral3 | Self::Llama4 | Self::MuseGlimmer => {
                RopePairing::Adjacent
            }
        }
    }

    fn build_bindings(self, archive: &GgufArchive, dtype: DType) -> Result<GgufBindingMap> {
        match self {
            Self::Gemma3 => build_gemma3_bindings(archive),
            Self::Gemma3n => build_gemma3n_bindings(archive),
            Self::Idefics3 => build_idefics3_bindings(archive),
            Self::Mistral3 => build_mistral3_bindings(archive),
            Self::Llama4 => build_llama4_bindings(archive),
            Self::Lfm2Vl => build_lfm2_vl_bindings(archive),
            Self::Lfm3Vl => build_lfm3_vl_bindings(archive, dtype),
            Self::Lfm3Audio => build_lfm3_audio_bindings(archive, dtype),
            Self::Lfm3Omni => build_lfm3_omni_bindings(archive, dtype),
            Self::Lfm3D1 => {
                let bindings = build_lfm3_vl_bindings(archive, dtype)?;
                if decision_model_type(archive.metadata())? == Some("lfm3_omni") {
                    build_lfm3_audio_input_bindings(archive, bindings, dtype)
                } else {
                    Ok(bindings)
                }
            }
            Self::MuseGlimmer => build_muse_glimmer_bindings(archive),
        }
    }
}

fn family_from_names(
    architecture: &str,
    projector: Option<&str>,
    audio_projector: Option<&str>,
) -> Result<Option<NativeMultimodalGgufFamily>> {
    if audio_projector == Some("lfm3a") {
        require_architecture(architecture, "lfm3", "lfm3a")?;
        return match projector {
            None => Ok(Some(NativeMultimodalGgufFamily::Lfm3Audio)),
            Some("lfm3vl") => Ok(Some(NativeMultimodalGgufFamily::Lfm3Omni)),
            Some(projector) => {
                bail!("unsupported LFM3 audio and vision projector combination: `{projector}`")
            }
        };
    }
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
        Some("lfm3vl") => {
            require_architecture(architecture, "lfm3", "lfm3vl")?;
            NativeMultimodalGgufFamily::Lfm3Vl
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
            "multimodal projector `{projector}` requires `{expected}` GGUF architecture, \
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
            (
                "lfm3",
                "lfm3vl",
                NativeMultimodalGgufFamily::Lfm3Vl,
                RopePairing::HalfSplit,
            ),
        ] {
            let detected = family_from_names(architecture, Some(projector), None)
                .unwrap()
                .unwrap();
            assert_eq!(detected, family);
            assert_eq!(detected.rope_pairing(), pairing);
        }
        for (vision, family) in [
            (None, NativeMultimodalGgufFamily::Lfm3Audio),
            (Some("lfm3vl"), NativeMultimodalGgufFamily::Lfm3Omni),
        ] {
            let detected = family_from_names("lfm3", vision, Some("lfm3a"))
                .unwrap()
                .unwrap();
            assert_eq!(detected, family);
            assert_eq!(detected.rope_pairing(), RopePairing::HalfSplit);
        }
    }

    #[test]
    fn rejects_cross_family_projector_pair() {
        for (architecture, projector, expected) in [
            ("llama", "pixtral", "mistral3"),
            ("lfm2", "lfm3vl", "lfm3"),
            ("lfm3", "lfm2", "lfm2"),
        ] {
            let error = family_from_names(architecture, Some(projector), None).unwrap_err();
            assert!(error
                .to_string()
                .contains(&format!("requires `{expected}`")));
        }
    }

    #[test]
    fn leaves_other_projectors_for_other_registries() {
        assert_eq!(
            family_from_names("qwen3vl", Some("qwen3vl_merger"), None).unwrap(),
            None
        );
        assert_eq!(
            family_from_names("gemma4", Some("gemma4v"), None).unwrap(),
            None
        );
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
                family_from_names(architecture, Some(projector), None).unwrap(),
                None
            );
        }
    }
}
