use pyo3::pyclass;

#[pyclass]
#[derive(Clone)]
pub enum Which {
    Mistral {
        model_id: String,
        tokenizer_json: Option<String>,
        repeat_last_n: Option<usize>,
    },

    MistralGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
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

    LlamaGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
    },

    LlamaGGML {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        gqa: Option<usize>,
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

    MixtralGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
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

    XLoraMistralGGUF {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    XLoraLlamaGGUF {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        tgt_non_granular_index: Option<usize>,
    },

    XLoraLlamaGGML {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
        order: String,
        gqa: Option<usize>,
        tgt_non_granular_index: Option<usize>,
    },

    XLoraMixtralGGUF {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        xlora_model_id: String,
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

    LoraMistralGGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        adapters_model_id: String,
        repeat_last_n: Option<usize>,
        order: String,
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

    LoraLlamaGGUF {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
    },

    LoraLlamaGGML {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
        gqa: Option<usize>,
    },

    LoraMixtralGGUF {
        tok_model_id: Option<String>,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
        adapters_model_id: String,
        order: String,
    },

    Phi2GGUF {
        tok_model_id: String,
        tokenizer_json: Option<String>,
        quantized_model_id: String,
        quantized_filename: String,
        repeat_last_n: Option<usize>,
    },
}
