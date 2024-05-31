use super::varbuilder_utils::{from_mmaped_safetensors, load_preload_adapters};
use anyhow::Result;
use candle_core::quantized::{ggml_file, gguf_file};
use candle_nn::VarBuilder;
use std::{collections::HashMap, path::PathBuf};

use crate::{
    lora::{LoraConfig, Ordering},
    pipeline::ModelPaths,
    xlora_models::XLoraConfig,
    DeviceMapMetadata,
};

#[derive(derive_more::From)]
pub struct FileGGML {
    pub ct: ggml_file::Content,
    pub gqa: usize,
}

#[derive(derive_more::From)]
pub struct FileGGUF<'a> {
    pub ct: gguf_file::Content,
    pub reader: &'a mut std::fs::File,
}

#[derive(derive_more::From)]
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
    #[allow(clippy::borrowed_box)]
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
        // TODO: `from_mmaped_safetensors` has `xlora_paths` as the 2nd param (_valid but params need to be named better_)
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
pub struct ParamsGGML(pub FileGGML);
pub struct ParamsGGUF<'a>(pub FileGGUF<'a>, pub Device<'a>);

// A `None` type vs the `Some` type (`Adapter<'a>`)
pub struct NoAdapter {}

// Marker traits to restrict type input:
// (required workaround to support impl on subtypes, otherwise would use an enum)
pub trait QuantParams {}
impl QuantParams for ParamsGGML {}
impl QuantParams for ParamsGGUF<'_> {}

// Emulates `Option<Adapter>` but is compatible as a type bound in `impl<T>` for Some vs None
pub trait MaybeAdapter {}
impl MaybeAdapter for Adapter<'_> {}
impl MaybeAdapter for NoAdapter {}

// `derive_more::From` provides a terser construction for enum variants of `ModelParams`.
#[derive(derive_more::From)]
pub struct Config<Q: QuantParams, A: MaybeAdapter> {
    pub quant: Q,
    pub adapter: A,
}

// NOTE: Variantly used for `.expect_quantized()` / `.expect_adapted()` methods
// `where` clause required due to bug with inline bounds:
// https://github.com/luker-os/variantly/pull/16
#[allow(clippy::large_enum_variant)]
#[derive(variantly::Variantly)]
pub enum ModelParams<'a, Q>
where
    Q: QuantParams,
{
    Quantized(Config<Q, NoAdapter>),
    Adapted(Config<Q, Adapter<'a>>),
}

// A `builder()` method is derived from the `new()` method and it's params (derived builder struct fields).
// NOTE: Intended to be built via fluent API in a single line, cannot conditionally append params.
// `.adapter(Adapter<' >)` or for conditional usage `.and_adapter(Option<Adapter<' >)` can be used.
// Otherwise omitting an `.adapter()` call prior to calling `build()` is ok, defaults to `None`.
#[buildstructor::buildstructor]
impl<'a, Q: QuantParams> ModelParams<'a, Q> {
    #[builder]
    pub fn new<'b: 'a>(quant: Q, adapter: Option<Adapter<'b>>) -> Self {
        match adapter {
            None => Self::Quantized((quant, NoAdapter {}).into()),
            Some(a) => Self::Adapted((quant, a).into()),
        }
    }
}

// Traits for the existing methods used across various model types to impl `from_ggml()` / `from_gguf()`
// Basic:
pub trait FromGGML {
    fn from_ggml(ct: ggml_file::Content, gqa: usize) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

pub trait FromGGUF {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &candle_core::Device,
        mapper: DeviceMapMetadata,
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
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
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
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
    ) -> Result<Self, candle_core::Error>
    where
        Self: Sized;
}

// NOTE: Below is a workaround to proxy params to the existing API methods `get_gguf()` / `get_gmml()` traits covered above.
impl Config<ParamsGGML, NoAdapter> {
    pub fn try_into_model<T: FromGGML>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let ParamsGGML(FileGGML { ct, gqa }) = self.quant;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_ggml(ct, gqa)
    }
}

impl Config<ParamsGGML, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGML>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let ParamsGGML(FileGGML { ct, gqa }) = self.quant;

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

impl Config<ParamsGGUF<'_>, NoAdapter> {
    pub fn try_into_model<T: FromGGUF>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let ParamsGGUF(FileGGUF { ct, reader }, Device { device, mapper }) = self.quant;

        // Forwards all structured fields above into the required flattened param sequence:
        T::from_gguf(ct, reader, device, mapper)
    }
}

impl Config<ParamsGGUF<'_>, Adapter<'_>> {
    pub fn try_into_model<T: FromAdapterGGUF>(self) -> Result<T, candle_core::Error> {
        // Destructure props:
        let ParamsGGUF(FileGGUF { ct, reader }, Device { device, mapper }) = self.quant;

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

use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};
use akin::akin;

impl TryFrom<ModelParams<'_, ParamsGGML>> for QLlama {
    type Error = candle_core::Error;

    fn try_from(params: ModelParams<'_, ParamsGGML>) -> Result<Self, Self::Error> {
        let config = params.expect_quantized("`Config` should be GGML Quantized");
        config.try_into_model()
    }
}

impl TryFrom<ModelParams<'_, ParamsGGML>> for XLoraQLlama {
    type Error = candle_core::Error;

    fn try_from(params: ModelParams<'_, ParamsGGML>) -> Result<Self, Self::Error> {
        let config = params.expect_adapted("`Config` should be GGML Quantized with an Adapter");
        config.try_into_model()
    }
}

akin! {
    let &models_gguf = [QLlama, QPhi, QPhi3];

    impl TryFrom<ModelParams<'_, ParamsGGUF<'_>>> for *models_gguf {
        type Error = candle_core::Error;

        fn try_from(params: ModelParams<'_, ParamsGGUF<'_>>) -> Result<Self, Self::Error> {
            let config = params.expect_quantized("`Config` should be GGUF Quantized");
            config.try_into_model()
        }
    }
}

akin! {
    let &models_gguf_a = [XLoraQLlama, XLoraQPhi3];

    impl TryFrom<ModelParams<'_, ParamsGGUF<'_>>> for *models_gguf_a {
        type Error = candle_core::Error;

        fn try_from(params: ModelParams<'_, ParamsGGUF<'_>>) -> Result<Self, Self::Error> {
            let config = params.expect_adapted("`Config` should be GGUF Quantized with an Adapter");
            config.try_into_model()
        }
    }
}
