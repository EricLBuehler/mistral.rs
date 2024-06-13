use mistralrs_core::{NormalLoaderType, VisionLoaderType};
use pyo3::pyclass;

#[pyclass]
#[derive(Debug, Clone)]
pub enum Architecture {
    Mistral,
    Gemma,
    Mixtral,
    Llama,
    Phi2,
    Phi3,
    Qwen2,
}

impl From<Architecture> for NormalLoaderType {
    fn from(value: Architecture) -> Self {
        match value {
            Architecture::Gemma => Self::Gemma,
            Architecture::Llama => Self::Llama,
            Architecture::Mistral => Self::Mistral,
            Architecture::Mixtral => Self::Mixtral,
            Architecture::Phi2 => Self::Phi2,
            Architecture::Phi3 => Self::Phi3,
            Architecture::Qwen2 => Self::Qwen2,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum VisionArchitecture {
    Phi3V,
}

impl From<VisionArchitecture> for VisionLoaderType {
    fn from(value: VisionArchitecture) -> Self {
        match value {
            VisionArchitecture::Phi3V => VisionLoaderType::Phi3V,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub enum Which {
    Plain {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
        arch: Architecture,
    },

    XLora {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
        arch: Architecture,
    },

    Lora {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        adapters_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        arch: Architecture,
    },

    #[allow(clippy::upper_case_acronyms)]
    GGUF {
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
    },

    XLoraGGUF {
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    LoraGGUF {
        tok_model_id: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
    },

    #[allow(clippy::upper_case_acronyms)]
    GGML {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        gqa: Option<usize>,
    },

    XLoraGGML {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
        gqa: Option<usize>,
    },

    LoraGGML {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
        gqa: Option<usize>,
    },

    VisionPlain {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
        arch: VisionArchitecture,
    },
}
