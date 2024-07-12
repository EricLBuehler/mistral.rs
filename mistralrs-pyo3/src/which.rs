use mistralrs_core::{NormalLoaderType, VisionLoaderType};
use pyo3::pyclass;

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum Architecture {
    Mistral,
    Gemma,
    Mixtral,
    Llama,
    Phi2,
    Phi3,
    Qwen2,
    Gemma2,
    Starcoder2,
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
            Architecture::Gemma2 => Self::Gemma2,
            Architecture::Starcoder2 => Self::Starcoder2,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum VisionArchitecture {
    Phi3V,
    Idefics2,
    LLaVANext,
    LLaVA,
}

impl From<VisionArchitecture> for VisionLoaderType {
    fn from(value: VisionArchitecture) -> Self {
        match value {
            VisionArchitecture::Phi3V => VisionLoaderType::Phi3V,
            VisionArchitecture::Idefics2 => VisionLoaderType::Idefics2,
            VisionArchitecture::LLaVANext => VisionLoaderType::LLaVANext,
            VisionArchitecture::LLaVA => VisionLoaderType::LLaVA,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub enum Which {
    #[pyo3(constructor = (
        model_id,
        arch,
        tokenizer_json = None,
        repeat_last_n = 64
    ))]
    Plain {
        model_id: String,
        arch: Architecture,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
    },

    #[pyo3(constructor = (
        xlora_model_id,
        order,
        arch,
        model_id = None,
        tokenizer_json = None,
        repeat_last_n = 64,
        tgt_non_granular_index = None
    ))]
    XLora {
        xlora_model_id: String,
        order: String,
        arch: Architecture,
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
        tgt_non_granular_index: Option<usize>,
    },

    #[pyo3(constructor = (
        adapters_model_id,
        order,
        arch,
        model_id = None,
        tokenizer_json = None,
        repeat_last_n = 64
    ))]
    Lora {
        adapters_model_id: String,
        order: String,
        arch: Architecture,
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id = None,
        repeat_last_n = 64
    ))]
    #[allow(clippy::upper_case_acronyms)]
    GGUF {
        quantized_model_id: String,
        quantized_filename: String,
        tok_model_id: Option<String>,
        repeat_last_n: usize,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        xlora_model_id,
        order,
        tok_model_id = None,
        repeat_last_n = 64,
        tgt_non_granular_index = None,
    ))]
    XLoraGGUF {
        quantized_model_id: String,
        quantized_filename: String,
        xlora_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        repeat_last_n: usize,
        tgt_non_granular_index: Option<usize>,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        adapters_model_id,
        order,
        tok_model_id = None,
        repeat_last_n = 64
    ))]
    LoraGGUF {
        quantized_model_id: String,
        quantized_filename: String,
        adapters_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        repeat_last_n: usize,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id,
        tokenizer_json = None,
        repeat_last_n = 64,
        gqa = 1,
    ))]
    #[allow(clippy::upper_case_acronyms)]
    GGML {
        quantized_model_id: String,
        quantized_filename: String,
        tok_model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
        gqa: usize,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        xlora_model_id,
        order,
        tok_model_id = None,
        tokenizer_json = None,
        repeat_last_n = 64,
        tgt_non_granular_index = None,
        gqa = 1,
    ))]
    XLoraGGML {
        quantized_model_id: String,
        quantized_filename: String,
        xlora_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
        tgt_non_granular_index: Option<usize>,
        gqa: usize,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        adapters_model_id,
        order,
        tok_model_id = None,
        tokenizer_json = None,
        repeat_last_n = 64,
        gqa = 1,
    ))]
    LoraGGML {
        quantized_model_id: String,
        quantized_filename: String,
        adapters_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
        gqa: usize,
    },

    #[pyo3(constructor = (
        model_id,
        arch,
        tokenizer_json = None,
        repeat_last_n = 64
    ))]
    VisionPlain {
        model_id: String,
        arch: VisionArchitecture,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
    },
}
