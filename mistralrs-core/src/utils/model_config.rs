use candle_core::quantized::gguf_file;
use candle_nn::VarBuilder;
use std::collections::HashMap;

use crate::{
    DeviceMapMetadata,
    lora::{Ordering, LoraConfig},
    xlora_models::XLoraConfig,
};

pub struct File<'a> {
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
    pub vb: &'a VarBuilder<'a>,
    pub ordering: &'a Ordering,
    pub preload_adapters: &'a Option<HashMap<String, (VarBuilder<'a>, LoraConfig)>>,
}
