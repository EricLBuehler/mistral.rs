use std::path::Path;

use anyhow::{bail, Result};
use candle_core::DType;
use mistralrs_quant::{GgufArchive, GgufBindingMap};

use crate::MultimodalLoaderType;

use super::{
    gemma3_bindings::synthesize_gemma3_text,
    gemma3_config::ensure_gemma3_vision_config,
    lfm3_vl_config::synthesize_lfm3_multimodal,
    multimodal_bindings::build_gemma4_bindings,
    multimodal_vision_registry::{resolve_native_multimodal_gguf, NativeMultimodalGguf},
    muse_glimmer_bindings::normalize_muse_glimmer_config,
    normal_registry::RopePairing,
    qwen4exp::{self, synthesize_qwen4exp},
    qwen_multimodal_bindings::{
        build_qwen_multimodal_bindings, normalize_qwen_multimodal_config,
        qwen_multimodal_loader_type,
    },
};

/// Inputs for an architecture that builds its own config and bindings from the GGUF.
pub(crate) struct GgufSynthesisInputs<'a> {
    pub archive: &'a GgufArchive,
    pub dtype: DType,
    pub external_config: Option<&'a str>,
    pub processor_config: Option<&'a Path>,
    pub preprocessor_config: Option<&'a Path>,
}

/// The caller stamps `rope_pairing` into `config`.
pub(crate) struct SynthesizedGguf {
    pub config: String,
    pub bindings: GgufBindingMap,
    pub loader_type: MultimodalLoaderType,
    pub processor_config: Option<String>,
    pub preprocessor_config: Option<String>,
    pub rope_pairing: RopePairing,
}

pub(crate) type SynthesizeFn = fn(&GgufSynthesisInputs<'_>) -> Result<SynthesizedGguf>;
pub(crate) type ResolveMultimodalFn = fn(&GgufArchive, DType) -> Result<NativeMultimodalGguf>;

/// How a GGUF without a projector is loaded.
#[derive(Clone, Copy)]
pub(crate) enum GgufTextRoute {
    NormalRegistry,
    ProjectorRequired,
    Synthesized(SynthesizeFn),
}

/// How a GGUF with merged projector components is loaded.
#[derive(Clone, Copy)]
pub(crate) enum GgufMultimodalRoute {
    ProjectorRegistry,
    Resolve(ResolveMultimodalFn),
    Synthesized(SynthesizeFn),
}

pub(crate) struct GgufArchSpec {
    architectures: &'static [&'static str],
    pub text: GgufTextRoute,
    pub multimodal: GgufMultimodalRoute,
    pub infer_multimodal_assets: bool,
    pub validate_multimodal_config: Option<fn(&str) -> Result<()>>,
    pub normalize_multimodal_config: Option<fn(&MultimodalLoaderType, &str) -> Result<String>>,
}

impl GgufArchSpec {
    const fn new(architectures: &'static [&'static str]) -> Self {
        Self {
            architectures,
            text: GgufTextRoute::NormalRegistry,
            multimodal: GgufMultimodalRoute::ProjectorRegistry,
            infer_multimodal_assets: true,
            validate_multimodal_config: None,
            normalize_multimodal_config: None,
        }
    }

    const fn text(mut self, text: GgufTextRoute) -> Self {
        self.text = text;
        self
    }

    const fn multimodal(mut self, multimodal: GgufMultimodalRoute) -> Self {
        self.multimodal = multimodal;
        self
    }

    const fn validate_multimodal_config(mut self, validate: fn(&str) -> Result<()>) -> Self {
        self.validate_multimodal_config = Some(validate);
        self
    }

    const fn infer_multimodal_assets(mut self, infer: bool) -> Self {
        self.infer_multimodal_assets = infer;
        self
    }

    const fn normalize_multimodal_config(
        mut self,
        normalize: fn(&MultimodalLoaderType, &str) -> Result<String>,
    ) -> Self {
        self.normalize_multimodal_config = Some(normalize);
        self
    }

    /// Whether the multimodal path needs no external `config.json`.
    pub(crate) fn synthesizes_multimodal_config(&self) -> bool {
        matches!(self.multimodal, GgufMultimodalRoute::Synthesized(_))
    }

    pub(crate) fn resolve_multimodal(
        &self,
        archive: &GgufArchive,
        architecture: &str,
        dtype: DType,
    ) -> Result<NativeMultimodalGguf> {
        match self.multimodal {
            GgufMultimodalRoute::Resolve(resolve) => resolve(archive, dtype),
            GgufMultimodalRoute::ProjectorRegistry | GgufMultimodalRoute::Synthesized(_) => {
                let Some(resolved) = resolve_native_multimodal_gguf(archive, dtype)? else {
                    bail!(
                        "multimodal GGUF architecture `{architecture}` is not supported by the native multimodal loader"
                    );
                };
                Ok(resolved)
            }
        }
    }
}

const DEFAULT_SPEC: GgufArchSpec = GgufArchSpec::new(&[]);

static ARCH_SPECS: &[GgufArchSpec] = &[
    GgufArchSpec::new(&["lfm3"])
        .multimodal(GgufMultimodalRoute::Synthesized(synthesize_lfm3_multimodal))
        .infer_multimodal_assets(false),
    GgufArchSpec::new(&[qwen4exp::ARCHITECTURE])
        .text(GgufTextRoute::Synthesized(synthesize_qwen4exp))
        .multimodal(GgufMultimodalRoute::Synthesized(synthesize_qwen4exp)),
    GgufArchSpec::new(&["gemma3"])
        .text(GgufTextRoute::Synthesized(synthesize_gemma3_text))
        .validate_multimodal_config(ensure_gemma3_vision_config),
    GgufArchSpec::new(&["gemma4"])
        .text(GgufTextRoute::ProjectorRequired)
        .multimodal(GgufMultimodalRoute::Resolve(resolve_gemma4)),
    GgufArchSpec::new(&["qwen2vl", "qwen3vl", "qwen3vlmoe"])
        .text(GgufTextRoute::ProjectorRequired)
        .multimodal(GgufMultimodalRoute::Resolve(resolve_qwen_multimodal)),
    GgufArchSpec::new(&["qwen35", "qwen35moe"])
        .multimodal(GgufMultimodalRoute::Resolve(resolve_qwen_multimodal))
        .normalize_multimodal_config(normalize_qwen_multimodal_config),
    GgufArchSpec::new(&["muse-glimmer"])
        .text(GgufTextRoute::ProjectorRequired)
        .normalize_multimodal_config(normalize_muse_glimmer_config),
    GgufArchSpec::new(&["gemma3n", "llama4"]).text(GgufTextRoute::ProjectorRequired),
];

pub(crate) fn arch_spec(architecture: &str) -> &'static GgufArchSpec {
    ARCH_SPECS
        .iter()
        .find(|spec| {
            spec.architectures
                .iter()
                .any(|candidate| candidate.eq_ignore_ascii_case(architecture))
        })
        .unwrap_or(&DEFAULT_SPEC)
}

fn resolve_gemma4(archive: &GgufArchive, _dtype: DType) -> Result<NativeMultimodalGguf> {
    Ok(NativeMultimodalGguf {
        loader_type: MultimodalLoaderType::Gemma4,
        bindings: build_gemma4_bindings(archive)?,
        rope_pairing: RopePairing::HalfSplit,
    })
}

fn resolve_qwen_multimodal(archive: &GgufArchive, _dtype: DType) -> Result<NativeMultimodalGguf> {
    Ok(NativeMultimodalGguf {
        loader_type: qwen_multimodal_loader_type(archive)?,
        bindings: build_qwen_multimodal_bindings(archive)?,
        rope_pairing: RopePairing::HalfSplit,
    })
}
