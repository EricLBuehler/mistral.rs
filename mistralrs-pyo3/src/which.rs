use std::path::PathBuf;

use either::Either;
use mistralrs_core::{DiffusionLoaderType, ModelDType, NormalLoaderType, VisionLoaderType};
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
    Phi3_5MoE,
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
            Architecture::Phi3_5MoE => Self::Phi3_5MoE,
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
    VLlama,
}

impl From<VisionArchitecture> for VisionLoaderType {
    fn from(value: VisionArchitecture) -> Self {
        match value {
            VisionArchitecture::Phi3V => VisionLoaderType::Phi3V,
            VisionArchitecture::Idefics2 => VisionLoaderType::Idefics2,
            VisionArchitecture::LLaVANext => VisionLoaderType::LLaVANext,
            VisionArchitecture::LLaVA => VisionLoaderType::LLaVA,
            VisionArchitecture::VLlama => VisionLoaderType::VLlama,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum DiffusionArchitecture {
    Flux,
    FluxOffloaded,
}

impl From<DiffusionArchitecture> for DiffusionLoaderType {
    fn from(value: DiffusionArchitecture) -> Self {
        match value {
            DiffusionArchitecture::Flux => DiffusionLoaderType::Flux,
            DiffusionArchitecture::FluxOffloaded => DiffusionLoaderType::FluxOffloaded,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum IsqOrganization {
    Default,
    MoQE,
}

impl From<IsqOrganization> for mistralrs_core::IsqOrganization {
    fn from(value: IsqOrganization) -> Self {
        match value {
            IsqOrganization::Default => mistralrs_core::IsqOrganization::Default,
            IsqOrganization::MoQE => mistralrs_core::IsqOrganization::MoeExpertsOnly,
        }
    }
}

#[pyclass]
#[derive(Clone)]
pub enum Which {
    #[pyo3(constructor = (
        model_id,
        arch = None,
        tokenizer_json = None,
        topology = None,
        organization = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
    ))]
    Plain {
        model_id: String,
        arch: Option<Architecture>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        organization: Option<IsqOrganization>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<PathBuf>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        xlora_model_id,
        order,
        arch = None,
        model_id = None,
        tokenizer_json = None,
        tgt_non_granular_index = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
    ))]
    XLora {
        xlora_model_id: String,
        order: String,
        arch: Option<Architecture>,
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<PathBuf>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        adapters_model_id,
        order,
        arch = None,
        model_id = None,
        tokenizer_json = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
    ))]
    Lora {
        adapters_model_id: String,
        order: String,
        arch: Option<Architecture>,
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<PathBuf>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id = None,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    #[allow(clippy::upper_case_acronyms)]
    GGUF {
        quantized_model_id: String,
        quantized_filename: Either<String, Vec<String>>,
        tok_model_id: Option<String>,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        xlora_model_id,
        order,
        tok_model_id = None,
        tgt_non_granular_index = None,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    XLoraGGUF {
        quantized_model_id: String,
        quantized_filename: Either<String, Vec<String>>,
        xlora_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        tgt_non_granular_index: Option<usize>,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        adapters_model_id,
        order,
        tok_model_id = None,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    LoraGGUF {
        quantized_model_id: String,
        quantized_filename: Either<String, Vec<String>>,
        adapters_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id,
        tokenizer_json = None,
        gqa = 1,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    #[allow(clippy::upper_case_acronyms)]
    GGML {
        quantized_model_id: String,
        quantized_filename: String,
        tok_model_id: String,
        tokenizer_json: Option<String>,
        gqa: usize,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        xlora_model_id,
        order,
        tok_model_id = None,
        tokenizer_json = None,
        tgt_non_granular_index = None,
        gqa = 1,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    XLoraGGML {
        quantized_model_id: String,
        quantized_filename: String,
        xlora_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        tgt_non_granular_index: Option<usize>,
        gqa: usize,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        adapters_model_id,
        order,
        tok_model_id = None,
        tokenizer_json = None,
        gqa = 1,
        topology = None,
        dtype = ModelDType::Auto,
    ))]
    LoraGGML {
        quantized_model_id: String,
        quantized_filename: String,
        adapters_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        gqa: usize,
        topology: Option<String>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        model_id,
        arch,
        tokenizer_json = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
    ))]
    VisionPlain {
        model_id: String,
        arch: VisionArchitecture,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<PathBuf>,
        dtype: ModelDType,
    },

    #[pyo3(constructor = (
        model_id,
        arch,
        dtype = ModelDType::Auto,
    ))]
    DiffusionPlain {
        model_id: String,
        arch: DiffusionArchitecture,
        dtype: ModelDType,
    },
}
