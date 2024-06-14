use std::{fmt::Debug, str::FromStr};

use anyhow::Result;
use candle_nn::VarBuilder;

#[cfg(feature = "pyo3_macros")]
use pyo3::pyclass;

use serde::Deserialize;

use super::{NormalLoadingMetadata, Seq2SeqModel};
use crate::models::t5::{Config as T5Config, T5ForConditionalGeneration as T5Model};

pub trait Seq2SeqModelLoader {
    fn load(
        &self,
        config: &str,
        use_flash_attn: bool,
        vb: VarBuilder,
        normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn Seq2SeqModel + Send + Sync>>;
    fn is_gptx(&self) -> bool;
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>>;
}

#[cfg_attr(feature = "pyo3_macros", pyclass)]
#[derive(Clone, Debug, Deserialize)]
/// The architecture to load the vision model as.
pub enum Seq2SeqLoaderType {
    #[serde(rename = "t5")]
    T5,
}

impl FromStr for Seq2SeqLoaderType {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "t5" => Ok(Self::T5),
            a => Err(format!("Unknown architecture `{a}`")),
        }
    }
}

// ======================== T5

pub struct T5Loader;

impl Seq2SeqModelLoader for T5Loader {
    fn load(
        &self,
        config: &str,
        _use_flash_attn: bool,
        vb: VarBuilder,
        _normal_loading_metadata: NormalLoadingMetadata,
    ) -> Result<Box<dyn Seq2SeqModel + Send + Sync>> {
        let config: T5Config = serde_json::from_str(config)?;
        Ok(Box::new(T5Model::load(vb, &config)?))
    }
    fn is_gptx(&self) -> bool {
        true
    }
    fn get_config_repr(&self, config: &str) -> Result<Box<dyn Debug>> {
        let config: T5Config = serde_json::from_str(config)?;
        Ok(Box::new(config))
    }
}
