use anyhow::{Context, Result};
use std::{num::NonZeroUsize, str::FromStr};
use strum::EnumString;

use crate::{pipeline::QuantizationKind, Loader, ModelDType, ModelKind, Topology};

pub const EXL2_MULTI_FILE_DELIMITER: &str = " ";

#[derive(Debug, EnumString, Clone, Copy)]
#[strum(serialize_all = "kebab-case")]
pub enum EXL2Architecture {
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
}

// Wraps from_str() for some convenience:
// - Case-insensitive variant matching (TODO: is this desirable?)
// - Customized error until potential upstream support: https://github.com/Peternator7/strum/issues/332
impl EXL2Architecture {
    pub fn from_value<T: AsRef<str> + std::fmt::Display>(value: T) -> Result<Self> {
        Self::from_str(&value.as_ref().to_ascii_lowercase())
            .with_context(|| format!("Unknown EXL2 architecture `{value}`"))
            .map_err(anyhow::Error::msg)
    }
}

pub struct EXL2LoaderBuilder {
    model_id: Option<String>,
    quantized_model_id: String,
    quantized_filenames: Vec<String>,
    kind: ModelKind,
    config: EXL2SpecificConfig,
}

pub struct EXL2SpecificConfig {
    pub topology: Option<Topology>,
    pub gpu_split: Option<String>,
    pub length: Option<usize>,
    pub rope_scale: Option<f32>,
    pub rope_alpha: Option<f32>,
    pub no_flash_attn: bool,
    pub no_xformers: bool,
    pub no_sdpa: bool,
    pub low_mem: bool,
    pub experts_per_token: Option<usize>,
    pub load_q4: bool,
    pub fast_safetensors: bool,
    pub ignore_compatibility: bool,
    pub chunk_size: Option<usize>,
}

impl EXL2LoaderBuilder {
    pub fn new(
        chat_template: Option<String>,
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filenames: Vec<String>,
        config: EXL2SpecificConfig,
    ) -> Self {
        let kind = ModelKind::Quantized {
            quant: QuantizationKind::Exl2,
        };

        Self {
            model_id: tok_model_id,
            quantized_model_id,
            quantized_filenames,
            kind,
            config,
        }
    }

    pub fn build(self) -> Result<Box<dyn Loader>> {
        // Implement the loading logic for EXL2 models here
        todo!("Implement EXL2 model loading")
    }
}
