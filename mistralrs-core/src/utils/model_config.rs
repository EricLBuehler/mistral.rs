use anyhow::Result;
use candle_core::quantized::{gguf_file, ggml_file};
use candle_nn::VarBuilder;
use std::{collections::HashMap, path::PathBuf};
use super::varbuilder_utils::{from_mmaped_safetensors, load_preload_adapters};

use crate::{
    DeviceMapMetadata,
    lora::{Ordering, LoraConfig},
    xlora_models::XLoraConfig,
    pipeline::ModelPaths,
};

pub struct FileGGML {
    pub ct: ggml_file::Content,
    pub gqa: usize,
}

pub struct FileGGUF<'a> {
    pub ct: gguf_file::Content,
    pub reader: &'a mut std::fs::File,
}

pub struct Device<'a> {
    pub device: &'a candle_core::Device,
    pub mapper: DeviceMapMetadata,
}

pub struct Adapter<'a> {
    pub xlora_config: Option<XLoraConfig>,
    pub lora_config: &'a [((String, String), LoraConfig)],
    pub vb: VarBuilder<'a>,
    pub ordering: &'a Ordering,
    pub preload_adapters: Option<HashMap<String, (VarBuilder<'a>, LoraConfig)>>,
}

impl<'a> Adapter<'a> {
    // NOTE: It is not possible to store references for values returned by: load_preload_adapters() + from_mmaped_safetensors(),
    // As referenced value would drop after this method, Adapter takes ownership of vb + preload_adapters
    // and then passes by reference to the `from_gguf()` / `from_ggml()` methods when proxying to params.
    // NOTE: Due to reference usage persisting in returned struct, additional lifetime annotations were required.
    pub fn try_new<'b: 'a>(
        paths: &'b Box<dyn ModelPaths>,
        device: &'b candle_core::Device,
        silent: bool,
        is_xlora: bool,
    ) -> Result<Self> {
        let lora_config = paths.get_adapter_configs().as_ref().unwrap();
        let ordering = paths.get_ordering().as_ref().unwrap();
        let preload_adapters = load_preload_adapters(
            paths.get_lora_preload_adapter_info(),
            candle_core::DType::F32,
            device,
            silent,
        )?;

        // X-LoRA support:
        let mut xlora_paths: Vec<PathBuf> = vec![];
        let mut xlora_config: Option<XLoraConfig> = None;
        if is_xlora {
            xlora_paths = vec![paths.get_classifier_path().as_ref().unwrap().to_path_buf()];
            xlora_config = Some(paths.get_classifier_config().as_ref().unwrap().clone());
        }

        // Create VarBuilder:
        let vb = from_mmaped_safetensors(
            xlora_paths,
            paths
                .get_adapter_filenames()
                .as_ref()
                .unwrap()
                .iter()
                .map(|(_, x)| (*x).to_owned())
                .collect::<Vec<_>>(),
            candle_core::DType::F32,
            device,
            silent,
        )?;

        Ok(Self {
            lora_config,
            xlora_config,
            vb,
            ordering,
            preload_adapters,
        })
    }
}

// New type wrappers that segment the distinct parameter sets used by `from_ggml()` + `from_gguf()` methods:
pub struct GGML(pub FileGGML);
pub struct GGUF<'a>(pub FileGGUF<'a>, pub Device<'a>);

// A `None` type vs the `Some` type (`Adapter<'a>`)
pub struct NoAdapter {}

// Marker traits to restrict type input:
// (required workaround to support impl on subtypes, otherwise would use an enum)
pub trait QuantParams {}
impl QuantParams for GGML {}
impl QuantParams for GGUF<'_> {}

// Emulates `Option<Adapter>` but is compatible as a type bound in `impl<T>` for Some vs None
pub trait MaybeAdapter {}
impl MaybeAdapter for Adapter<'_> {}
impl MaybeAdapter for NoAdapter {}

pub struct Config<Q: QuantParams, A: MaybeAdapter> {
    pub quant: Q,
    pub adapter: A,
}

// NOTE: Variantly used for `.expect_quantized()` / `.expect_adapted()` methods
// `where` clause required due to bug with inline bounds:
// https://github.com/luker-os/variantly/pull/16
#[derive(variantly::Variantly)]
pub enum ModelParams<'a, Q> where Q: QuantParams {
    Quantized(Config::<Q, NoAdapter>),
    Adapted(Config::<Q, Adapter<'a>>),
}

impl<'a, Q: QuantParams> ModelParams<'a, Q> {
    pub fn new(quant: Q) -> Self {
        Self::Quantized(Config::<Q, NoAdapter> {
            quant,
            adapter: NoAdapter {},
        })
    }

    pub fn with_adapter<'b: 'a>(self, adapter: Adapter<'b>) -> Self {
        let q = self.expect_quantized("Config should be Quantized (without an Adapter)");

        Self::Adapted( Config::<Q, Adapter<'b>> {
            adapter,
            quant: q.quant
        })
    }
}

// Traits for the existing methods used across various model types to impl `from_ggml()` / `from_gguf()`
// Basic:
pub trait FromGGML {
    fn from_ggml(ct: ggml_file::Content, gqa: usize) -> Result<Self, candle_core::Error> where Self: Sized;
}

pub trait FromGGUF {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &candle_core::Device,
        mapper: DeviceMapMetadata,
    ) -> Result<Self, candle_core::Error> where Self: Sized;
}

// Extended variants:
pub trait FromAdapterGGML {
    fn from_ggml(
        ct: ggml_file::Content,
        gqa: usize,
        lora_config: &[((String, String), LoraConfig)],
        vb: &VarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self, candle_core::Error> where Self: Sized;
}
pub trait FromAdapterGGUF {
    #[allow(clippy::too_many_arguments)]
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &candle_core::Device,
        lora_config: &[((String, String), LoraConfig)],
        vb: &VarBuilder,
        ordering: &Ordering,
        xlora_config: Option<XLoraConfig>,
        mapper: DeviceMapMetadata,
        preload_adapters: &Option<HashMap<String, (VarBuilder, LoraConfig)>>,
    ) -> Result<Self, candle_core::Error> where Self: Sized;
}

// NOTE: Below is a workaround to proxy params to the existing API methods `get_gguf()` / `get_gmml()` traits covered above.
impl Config<GGML, NoAdapter> {
    pub fn try_into_model<T: FromGGML>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let GGML(
            FileGGML { ct, gqa },
        ) = self.quant;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_ggml(
            ct,
            gqa,
        )
    }
}

impl Config<GGML, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGML>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let GGML(
            FileGGML { ct, gqa },
        ) = self.quant;

        let Adapter {
            xlora_config,
            lora_config,
            vb,
            ordering,
            preload_adapters,
        } = self.adapter;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_ggml(
            ct,
            gqa,
            lora_config,
            &vb,
            ordering,
            xlora_config,
            &preload_adapters,
        )
    }
}

impl Config<GGUF<'_>, NoAdapter> {
    pub fn try_into_model<T: FromGGUF>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let GGUF(
            FileGGUF { ct, reader },
            Device { device, mapper },
        ) = self.quant;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_gguf(
            ct,
            reader,
            device,
            mapper,
        )
    }
}

impl Config<GGUF<'_>, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGUF>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let GGUF(
            FileGGUF { ct, reader },
            Device { device, mapper },
        ) = self.quant;

        let Adapter {
            xlora_config,
            lora_config,
            vb,
            ordering,
            preload_adapters,
        } = self.adapter;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_gguf(
            ct,
            reader,
            device,
            lora_config,
            &vb,
            ordering,
            xlora_config,
            mapper,
            &preload_adapters,
        )
    }
}

use akin::akin;
use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};

impl TryFrom<ModelParams<'_, GGML>> for QLlama {
    type Error = candle_core::Error;

    fn try_from(params: ModelParams<'_, GGML>) -> Result<Self, Self::Error> {
        let config = params.expect_quantized("`Config` should be GGML Quantized");
        config.try_into_model::<Self>()
    }
}

impl TryFrom<ModelParams<'_, GGML>> for XLoraQLlama {
    type Error = candle_core::Error;

    fn try_from(params: ModelParams<'_, GGML>) -> Result<Self, Self::Error> {
        let config = params.expect_adapted("`Config` should be GGML Quantized with an Adapter");
        config.try_into_model::<Self>()
    }
}

akin! {
    let &models_gguf = [QLlama, QPhi, QPhi3];

    impl TryFrom<ModelParams<'_, GGUF<'_>>> for *models_gguf {
        type Error = candle_core::Error;

        fn try_from(params: ModelParams<'_, GGUF<'_>>) -> Result<Self, Self::Error> {
            let config = params.expect_quantized("`Config` should be GGUF Quantized");
            config.try_into_model::<Self>()
        }
    }
}

akin! {
    let &models_gguf_a = [XLoraQLlama, XLoraQPhi3];

    impl TryFrom<ModelParams<'_, GGUF<'_>>> for *models_gguf_a {
        type Error = candle_core::Error;

        fn try_from(params: ModelParams<'_, GGUF<'_>>) -> Result<Self, Self::Error> {
            let config = params.expect_adapted("`Config` should be GGUF Quantized with an Adapter");
            config.try_into_model::<Self>()
        }
    }
}
