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

pub struct FileGGUF<'a> {
    pub ct: gguf_file::Content,
    pub reader: &'a mut std::fs::File,
}

pub struct FileGGML {
    pub ct: ggml_file::Content,
    pub gqa: usize,
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
        xlora_paths: Vec<PathBuf>,
        xlora_config: Option<XLoraConfig>,
    ) -> Result<Self> {
        let lora_config = paths.get_adapter_configs().as_ref().unwrap();
        let ordering = paths.get_ordering().as_ref().unwrap();
        let preload_adapters = load_preload_adapters(
            paths.get_lora_preload_adapter_info(),
            candle_core::DType::F32,
            device,
            silent,
        )?;

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

pub struct AdapterGGUF<'a>(pub FileGGUF<'a>, pub Device<'a>, pub Adapter<'a>);
pub struct AdapterGGML<'a>(pub FileGGML, pub Adapter<'a>);

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

// TODO: This trait is a workaround to proxy params to the existing API methods `get_gguf()` / `get_gmml()` it intends to replace.
pub trait MapParamsToModel<T> {
    type Error;

    fn try_from(self) -> Result<T, Self::Error>;
}

// Without sharing a common trait among other wrapper types, this could be used instead:
// (`try_into` reads more naturally during usage due to different syntax order for calling)
// impl AdapterGGUF<'_> {
    // pub fn try_into<T: FromAdapterGGUF>(self) -> Result<T, candle_core::Error> {
impl<T: FromAdapterGGUF> MapParamsToModel<T> for AdapterGGUF<'_> {
    type Error = candle_core::Error;

    // Technically mirrors the signature of `TryInto`
    fn try_from(self) -> Result<T, Self::Error> {
        // Destructure props:
        let AdapterGGUF(
            FileGGUF { ct, reader },
            Device { device, mapper },
            Adapter {
                xlora_config,
                lora_config,
                vb,
                ordering,
                preload_adapters,
            },
        ) = self;

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

impl<T: FromAdapterGGML> MapParamsToModel<T> for AdapterGGML<'_> {
    type Error = candle_core::Error;

    fn try_from(self) -> Result<T, Self::Error> {
        // Destructure props:
        let AdapterGGML(
            FileGGML { ct, gqa },
            Adapter {
                xlora_config,
                lora_config,
                vb,
                ordering,
                preload_adapters,
            },
        ) = self;

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

/*

// `TryFrom` has a blanket implementation that prevents this type of generics solution:
// `impl<T: FromAdapterGGUF> TryFrom<AdapterGGUF<'_>> for T {`
//
// This approach would need to copy/paste this impl for `XLoraQPhi3`
// (or use a macro like `akin` to generate a copy for each variant)
// Caveat: Requires importing each model explicitly:
// `use crate::xlora_models::{XLoraQLlama, XLoraQPhi3};`
//
impl TryFrom<AdapterGGUF<'_>> for XLoraQLlama {
    type Error = candle_core::Error;

    fn try_from(value: AdapterGGUF<'_>) -> Result<Self, Self::Error> {
        // Destructure props:
        let AdapterGGUF(
            FileGGUF { ct, reader },
            Device { device, mapper },
            Adapter {
                xlora_config,
                lora_config,
                vb,
                ordering,
                preload_adapters,
            },
        ) = value;

        // Forwards all structured fields above into the required flattened param sequence:
        Self::from_gguf(
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

*/
