use mistralrs_core::NormalLoaderType;
use pyo3::pyclass;

#[pyclass]
#[derive(Clone)]
pub enum Which {
    Plain {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: usize,
        arch: NormalLoaderType,
    },

    XLora {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: usize,
        order: String,
        tgt_non_granular_index: Option<usize>,
        arch: NormalLoaderType,
    },

    Lora {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        adapters_model_id: String,
        repeat_last_n: usize,
        order: String,
        arch: NormalLoaderType,
    },

    Gguf {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
    },

    XLoraGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    LoraGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    Ggml {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        gqa: Option<usize>,
    },

    XLoraGGML {
        tok_model_id: String,
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
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
        gqa: Option<usize>,
    },
}
