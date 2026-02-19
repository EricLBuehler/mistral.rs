use std::path::PathBuf;

use either::Either;
use mistralrs_core::{
    AutoDeviceMapParams, DiffusionLoaderType, EmbeddingLoaderType, ModelDType, NormalLoaderType,
    VisionLoaderType,
};
use pyo3::{pyclass, pymethods};

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
    DeepseekV2,
    DeepseekV3,
    Qwen3,
    GLM4,
    GLM4MoeLite,
    GLM4Moe,
    Qwen3Moe,
    SmolLm3,
    GraniteMoeHybrid,
    GptOss,
    Qwen3Next,
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
            Architecture::DeepseekV2 => Self::DeepSeekV2,
            Architecture::DeepseekV3 => Self::DeepSeekV3,
            Architecture::Qwen3 => Self::Qwen3,
            Architecture::GLM4 => Self::GLM4,
            Architecture::GLM4MoeLite => Self::GLM4MoeLite,
            Architecture::GLM4Moe => Self::GLM4Moe,
            Architecture::Qwen3Moe => Self::Qwen3Moe,
            Architecture::SmolLm3 => Self::SmolLm3,
            Architecture::GraniteMoeHybrid => Self::GraniteMoeHybrid,
            Architecture::GptOss => Self::GptOss,
            Architecture::Qwen3Next => Self::Qwen3Next,
        }
    }
}

#[pyclass(eq, eq_int)]
#[derive(Debug, Clone, PartialEq)]
pub enum EmbeddingArchitecture {
    EmbeddingGemma,
    Qwen3Embedding,
}

impl From<EmbeddingArchitecture> for EmbeddingLoaderType {
    fn from(value: EmbeddingArchitecture) -> Self {
        match value {
            EmbeddingArchitecture::EmbeddingGemma => EmbeddingLoaderType::EmbeddingGemma,
            EmbeddingArchitecture::Qwen3Embedding => EmbeddingLoaderType::Qwen3Embedding,
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
    Qwen2VL,
    Idefics3,
    MiniCpmO,
    Phi4MM,
    Qwen2_5VL,
    Gemma3,
    Mistral3,
    Llama4,
    Gemma3n,
    Qwen3VL,
    Qwen3VLMoE,
    Voxtral,
}

impl From<VisionArchitecture> for VisionLoaderType {
    fn from(value: VisionArchitecture) -> Self {
        match value {
            VisionArchitecture::Phi3V => VisionLoaderType::Phi3V,
            VisionArchitecture::Idefics2 => VisionLoaderType::Idefics2,
            VisionArchitecture::LLaVANext => VisionLoaderType::LLaVANext,
            VisionArchitecture::LLaVA => VisionLoaderType::LLaVA,
            VisionArchitecture::VLlama => VisionLoaderType::VLlama,
            VisionArchitecture::Qwen2VL => VisionLoaderType::Qwen2VL,
            VisionArchitecture::Idefics3 => VisionLoaderType::Idefics3,
            VisionArchitecture::MiniCpmO => VisionLoaderType::MiniCpmO,
            VisionArchitecture::Phi4MM => VisionLoaderType::Phi4MM,
            VisionArchitecture::Qwen2_5VL => VisionLoaderType::Qwen2_5VL,
            VisionArchitecture::Gemma3 => VisionLoaderType::Gemma3,
            VisionArchitecture::Mistral3 => VisionLoaderType::Mistral3,
            VisionArchitecture::Llama4 => VisionLoaderType::Llama4,
            VisionArchitecture::Gemma3n => VisionLoaderType::Gemma3n,
            VisionArchitecture::Qwen3VL => VisionLoaderType::Qwen3VL,
            VisionArchitecture::Qwen3VLMoE => VisionLoaderType::Qwen3VLMoE,
            VisionArchitecture::Voxtral => VisionLoaderType::Voxtral,
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
pub enum SpeechLoaderType {
    Dia,
}

impl From<SpeechLoaderType> for mistralrs_core::SpeechLoaderType {
    fn from(value: SpeechLoaderType) -> Self {
        match value {
            SpeechLoaderType::Dia => mistralrs_core::SpeechLoaderType::Dia,
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
#[pyo3(get_all)]
#[derive(Debug, Clone, PartialEq)]
pub struct TextAutoMapParams {
    pub max_seq_len: usize,
    pub max_batch_size: usize,
}

#[pymethods]
impl TextAutoMapParams {
    #[new]
    #[pyo3(signature = (
        max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
    ))]
    pub fn new(max_seq_len: usize, max_batch_size: usize) -> Self {
        Self {
            max_seq_len,
            max_batch_size,
        }
    }
}

#[pyclass]
#[pyo3(get_all)]
#[derive(Debug, Clone, PartialEq)]
pub struct VisionAutoMapParams {
    pub max_seq_len: usize,
    pub max_batch_size: usize,
    pub max_num_images: usize,
    pub max_image_length: usize,
}

#[pymethods]
impl VisionAutoMapParams {
    #[new]
    #[pyo3(signature = (
        max_seq_len = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        max_num_images = AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES,
        max_image_length = AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH,
    ))]
    pub fn new(
        max_seq_len: usize,
        max_batch_size: usize,
        max_num_images: usize,
        max_image_length: usize,
    ) -> Self {
        Self {
            max_seq_len,
            max_batch_size,
            max_num_images,
            max_image_length,
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
        imatrix = None,
        calibration_file = None,
        auto_map_params = None,
        hf_cache_path = None,
        matformer_config_path = None,
        matformer_slice_name = None,
    ))]
    Plain {
        model_id: String,
        arch: Option<Architecture>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        organization: Option<IsqOrganization>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<Either<String, Vec<String>>>,
        dtype: ModelDType,
        imatrix: Option<PathBuf>,
        calibration_file: Option<PathBuf>,
        auto_map_params: Option<TextAutoMapParams>,
        hf_cache_path: Option<PathBuf>,
        matformer_config_path: Option<PathBuf>,
        matformer_slice_name: Option<String>,
    },

    #[pyo3(constructor = (
        model_id,
        arch = None,
        tokenizer_json = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
        hf_cache_path = None,
    ))]
    Embedding {
        model_id: String,
        arch: Option<EmbeddingArchitecture>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<Either<String, Vec<String>>>,
        dtype: ModelDType,
        hf_cache_path: Option<PathBuf>,
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
        auto_map_params = None,
        hf_cache_path = None,
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
        from_uqff: Option<Either<String, Vec<String>>>,
        dtype: ModelDType,
        auto_map_params: Option<TextAutoMapParams>,
        hf_cache_path: Option<PathBuf>,
    },

    #[pyo3(constructor = (
        adapter_model_ids,
        arch = None,
        model_id = None,
        tokenizer_json = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
        auto_map_params = None,
        hf_cache_path = None,
    ))]
    Lora {
        adapter_model_ids: Vec<String>,
        arch: Option<Architecture>,
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<Either<String, Vec<String>>>,
        dtype: ModelDType,
        auto_map_params: Option<TextAutoMapParams>,
        hf_cache_path: Option<PathBuf>,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id = None,
        topology = None,
        dtype = ModelDType::Auto,
        auto_map_params = None,
    ))]
    #[allow(clippy::upper_case_acronyms)]
    GGUF {
        quantized_model_id: String,
        quantized_filename: Either<String, Vec<String>>,
        tok_model_id: Option<String>,
        topology: Option<String>,
        dtype: ModelDType,
        auto_map_params: Option<TextAutoMapParams>,
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
        auto_map_params = None,
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
        auto_map_params: Option<TextAutoMapParams>,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        adapters_model_id,
        order,
        tok_model_id = None,
        topology = None,
        dtype = ModelDType::Auto,
        auto_map_params = None,
    ))]
    LoraGGUF {
        quantized_model_id: String,
        quantized_filename: Either<String, Vec<String>>,
        adapters_model_id: String,
        order: String,
        tok_model_id: Option<String>,
        topology: Option<String>,
        dtype: ModelDType,
        auto_map_params: Option<TextAutoMapParams>,
    },

    #[pyo3(constructor = (
        quantized_model_id,
        quantized_filename,
        tok_model_id,
        tokenizer_json = None,
        gqa = 1,
        topology = None,
        dtype = ModelDType::Auto,
        auto_map_params = None,
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
        auto_map_params: Option<TextAutoMapParams>,
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
        auto_map_params = None,
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
        auto_map_params: Option<TextAutoMapParams>,
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
        auto_map_params = None,
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
        auto_map_params: Option<TextAutoMapParams>,
    },

    #[pyo3(constructor = (
        model_id,
        arch = None,
        tokenizer_json = None,
        topology = None,
        write_uqff = None,
        from_uqff = None,
        dtype = ModelDType::Auto,
        max_edge = None,
        calibration_file = None,
        imatrix = None,
        auto_map_params = None,
        hf_cache_path = None,
        matformer_config_path = None,
        matformer_slice_name = None,
        organization = None,
    ))]
    VisionPlain {
        model_id: String,
        arch: Option<VisionArchitecture>,
        tokenizer_json: Option<String>,
        topology: Option<String>,
        write_uqff: Option<PathBuf>,
        from_uqff: Option<Either<String, Vec<String>>>,
        dtype: ModelDType,
        max_edge: Option<u32>,
        calibration_file: Option<PathBuf>,
        imatrix: Option<PathBuf>,
        auto_map_params: Option<VisionAutoMapParams>,
        hf_cache_path: Option<PathBuf>,
        matformer_config_path: Option<PathBuf>,
        matformer_slice_name: Option<String>,
        organization: Option<IsqOrganization>,
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

    #[pyo3(constructor = (
        model_id,
        arch,
        dac_model_id = None,
        dtype = ModelDType::Auto,
    ))]
    Speech {
        model_id: String,
        arch: SpeechLoaderType,
        dac_model_id: Option<String>,
        dtype: ModelDType,
    },
}
