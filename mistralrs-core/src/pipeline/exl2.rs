use crate::{ChatTemplate, Loader};
use anyhow::Result;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::{
    models::quantized_llama::ModelWeights as QLlama,
    models::quantized_phi2::ModelWeights as QPhi,
    models::quantized_phi3::ModelWeights as QPhi3,
    models::quantized_starcoder2::ModelWeights as QStarcoder2,
    xlora_models::{XLoraQLlama, XLoraQPhi3},
};

use super::GeneralMetadata;

enum Model {
    Llama(QLlama),
    Phi2(QPhi),
    XLoraLlama(XLoraQLlama),
    XLoraPhi3(XLoraQPhi3),
    Phi3(QPhi3),
    Starcoder2(QStarcoder2),
}

pub struct EXL2Pipeline {
    model: Model,
    tokenizer: Arc<Tokenizer>,
    chat_template: Arc<ChatTemplate>,
    model_id: String,
    metadata: Arc<GeneralMetadata>,
}

pub struct EXL2Loader {
    tok_model_id: Option<String>,
    quantized_model_id: String,
    quantized_filename: String,
    config: EXL2SpecificConfig,
}

pub struct EXL2SpecificConfig {
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

pub struct EXL2LoaderBuilder {
    tok_model_id: Option<String>,
    quantized_model_id: String,
    quantized_filename: String,
    topology: Option<String>,
    config: EXL2SpecificConfig,
}

impl EXL2LoaderBuilder {
    pub fn new(
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        topology: Option<String>,
        config: EXL2SpecificConfig,
    ) -> Self {
        Self {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            topology,
            config: EXL2SpecificConfig { ..config },
        }
    }

    pub fn build(self) -> Result<Box<dyn Loader>> {
        // Implementation details for building the EXL2 loader would go here
        // This is a placeholder and would need to be filled in with the actual implementation
        todo!("Implement EXL2 loader building")
    }
}
