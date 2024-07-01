use clap::Subcommand;

use crate::{
    pipeline::{NormalLoaderType, VisionLoaderType},
    ModelDType,
};

fn parse_arch(x: &str) -> Result<NormalLoaderType, String> {
    x.parse()
}

fn parse_vision_arch(x: &str) -> Result<VisionLoaderType, String> {
    x.parse()
}

fn parse_model_dtype(x: &str) -> Result<ModelDType, String> {
    x.parse()
}

#[derive(Debug, Subcommand)]
pub enum ModelSelected {
    /// Select the model from a toml file
    Toml {
        /// .toml file containing the selector configuration.
        #[arg(short, long)]
        file: String,
    },

    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_arch)]
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },

    /// Select an X-LoRA architecture
    XLora {
        /// Force a base model ID to load from instead of using the ordering file. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
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

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_arch)]
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },

    /// Select a LoRA architecture
    Lora {
        /// Force a base model ID to load from instead of using the ordering file. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// The architecture of the model.
        #[arg(long, value_parser = parse_arch)]
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },

    /// Select a GGUF model.
    GGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,
    },

    /// Select a GGUF model with X-LoRA.
    XLoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
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

    /// Select a GGUF model with LoRA.
    LoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,
    },

    /// Select a GGML model.
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select a GGML model with X-LoRA.
    XLoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        xlora_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select a GGML model with LoRA.
    LoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,
    },

    /// Select a vision plain model, without quantization or adapters
    VisionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Control the application of repeat penalty for the last n tokens
        #[arg(long, default_value_t = 64)]
        repeat_last_n: usize,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_vision_arch)]
        arch: VisionLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },
}
