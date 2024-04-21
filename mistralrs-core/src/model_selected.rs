use clap::Subcommand;

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the mistral model.
    Mistral {
        /// Model ID to load from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized mistral model with gguf.
    MistralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mistral-7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(
            short = 'm',
            long,
            default_value = "TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
        )]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the mistral model, with X-LoRA.
    XLoraMistral {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the gemma model.
    Gemma {
        /// Model ID to load from
        #[arg(short, long, default_value = "google/gemma-7b-it")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the gemma model, with X-LoRA.
    XLoraGemma {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long, default_value = "lamm-mit/x-lora-gemma-7b")]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the llama model.
    Llama {
        /// Model ID to load from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized llama model with gguf.
    LlamaGGML {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "meta-llama/Llama-2-13b-chat-hf")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGML file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select the llama model, with X-LoRA.
    XLoraLlama {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the mixtral model.
    Mixtral {
        /// Model ID to load from
        #[arg(short, long, default_value = "mistralai/Mixtral-8x7B-Instruct-v0.1")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the quantized mixtral model with gguf.
    MixtralGGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "mistralai/Mixtral-8x7B-Instruct-v0.1")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Mixtral-8x7B-v0.1-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "mixtral-8x7b-v0.1.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the mixtral model, with X-LoRA.
    XLoraMixtral {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraMistralGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/zephyr-7B-beta-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "zephyr-7b-beta.Q8_0.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraLlamaGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraLlamaGGML {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGML file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the quantized mistral model with gguf and X-LoRA.
    XLoraMixtralGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Mixtral-8x7B-v0.1-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "mixtral-8x7b-v0.1.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the phi2 model.
    Phi2 {
        /// Model ID to load from
        #[arg(short, long, default_value = "microsoft/phi-2")]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select the phi2 model, with X-LoRA.
    XLoraPhi2 {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,
    },

    /// Select the mistral model, with LoRA and gguf.
    LoraMistralGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/zephyr-7B-beta-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "zephyr-7b-beta.Q8_0.gguf")]
        quantized_filename: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        adapters_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the mistral model, with LoRA.
    LoraMistral {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long, default_value = "lamm-mit/x-lora")]
        adapters_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the mixtral model, with LoRA.
    LoraMixtral {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the llama model, with LoRA.
    LoraLlama {
        /// Force a base model ID to load from instead of using the ordering file.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized mistral model with gguf and LoRA.
    LoraLlamaGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "llama-2-13b-chat.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized mistral model with gguf and LoRA.
    LoraLlamaGGML {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGML file.
        #[arg(short = 'm', long, default_value = "TheBloke/Llama-2-13B-chat-GGML")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(
            short = 'f',
            long,
            default_value = "llama-2-13b-chat.ggmlv3.q4_K_M.bin"
        )]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// GQA
        #[arg(long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select the quantized mistral model with gguf and LoRA.
    LoraMixtralGGUF {
        /// Force a base model ID to load the tokenizer from instead of using the ordering file.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/Mixtral-8x7B-v0.1-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "mixtral-8x7b-v0.1.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select the quantized phi2 model with gguf.
    Phi2GGUF {
        /// Model ID to load the tokenizer from
        #[arg(short, long, default_value = "microsoft/phi-2")]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// If it is set to an empty string then the quantized filename will be used as a path to the GGUF file.
        #[arg(short = 'm', long, default_value = "TheBloke/phi-2-GGUF")]
        quantized_model_id: Option<String>,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long, default_value = "phi-2.Q4_K_M.gguf")]
        quantized_filename: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },
}
