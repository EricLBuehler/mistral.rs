use pyo3::pyclass;

#[pyclass]
#[derive(Clone)]
pub enum Which {
    Mistral {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    XLoraMistral {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    Gemma {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    XLoraGemma {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    Llama {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    XLoraLlama {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    Mixtral {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    XLoraMixtral {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    Phi2 {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    XLoraPhi2 {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        xlora_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    LoraMistral {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        adapters_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
    },

    LoraMixtral {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        adapters_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
    },

    LoraLlama {
        model_id: Option<String>,
        tokenizer_json: Option<String>,
        adapters_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
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
