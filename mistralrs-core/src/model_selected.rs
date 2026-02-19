use std::path::PathBuf;

use clap::Subcommand;

use crate::{
    pipeline::{
        AutoDeviceMapParams, EmbeddingLoaderType, IsqOrganization, NormalLoaderType,
        VisionLoaderType,
    },
    DiffusionLoaderType, ModelDType, SpeechLoaderType,
};

// Default value functions for serde deserialization
fn default_model_dtype() -> ModelDType {
    ModelDType::Auto
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

fn parse_arch(x: &str) -> Result<NormalLoaderType, String> {
    x.parse()
}

fn parse_vision_arch(x: &str) -> Result<VisionLoaderType, String> {
    x.parse()
}

fn parse_embedding_arch(x: &str) -> Result<EmbeddingLoaderType, String> {
    x.parse()
}

fn parse_diffusion_arch(x: &str) -> Result<DiffusionLoaderType, String> {
    x.parse()
}

fn parse_speech_arch(x: &str) -> Result<SpeechLoaderType, String> {
    x.parse()
}

fn parse_model_dtype(x: &str) -> Result<ModelDType, String> {
    x.parse()
}

#[derive(Debug, Clone, Subcommand, serde::Deserialize, serde::Serialize)]
pub enum ModelSelected {
    /// Select the model from a toml file
    Toml {
        /// .toml file containing the selector configuration.
        #[arg(short, long)]
        file: String,
    },

    /// Select a model for running via auto loader
    Run {
        /// Model ID to load from. May be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If specified, it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe`.
        #[arg(short, long)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[arg(short, long)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        #[arg(short, long)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        #[arg(short, long)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[arg(short, long)]
        calibration_file: Option<PathBuf>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// Only supported on specific vision models.
        #[arg(short = 'e', long)]
        max_edge: Option<u32>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific vision models.
        #[arg(long)]
        max_num_images: Option<usize>,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        /// Only supported on specific vision models.
        #[arg(long)]
        max_image_length: Option<usize>,

        /// Cache path for Hugging Face models downloaded locally.
        #[arg(long)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[arg(long)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[arg(long)]
        matformer_slice_name: Option<String>,
    },

    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_arch)]
        #[serde(default)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        #[serde(default = "default_model_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        #[serde(default)]
        topology: Option<String>,

        #[allow(rustdoc::bare_urls)]
        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: https://arxiv.org/abs/2310.02410).
        #[arg(short, long)]
        #[serde(default)]
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        #[arg(short, long)]
        #[serde(default)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[arg(short, long)]
        #[serde(default)]
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--calibration-file/-c`
        #[arg(short, long)]
        #[serde(default)]
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        #[arg(short, long)]
        #[serde(default)]
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[arg(long)]
        #[serde(default)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[arg(long)]
        #[serde(default)]
        matformer_slice_name: Option<String>,
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

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        #[arg(long)]
        tgt_non_granular_index: Option<usize>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_arch)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[arg(short, long)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        #[arg(short, long)]
        from_uqff: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        hf_cache_path: Option<PathBuf>,
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
        adapter_model_id: String,

        /// The architecture of the model.
        #[arg(long, value_parser = parse_arch)]
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[arg(short, long)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        #[arg(short, long)]
        from_uqff: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        hf_cache_path: Option<PathBuf>,
    },

    /// Select a GGUF model.
    GGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a GGUF model with X-LoRA.
    XLoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        #[arg(short = 'f', long)]
        quantized_filename: String,

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

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a GGUF model with LoRA.
    LoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a GGML model.
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a GGML model with X-LoRA.
    XLoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename.
        #[arg(short = 'f', long)]
        quantized_filename: String,

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

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a GGML model with LoRA.
    LoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        tok_model_id: Option<String>,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(long)]
        tokenizer_json: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        #[arg(short = 'm', long)]
        quantized_model_id: String,

        /// Quantized filename.
        #[arg(short = 'f', long)]
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        adapters_model_id: String,

        /// Ordering JSON file
        #[arg(short, long)]
        order: String,

        /// GQA value
        #[arg(short, long, default_value_t = 1)]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,
    },

    /// Select a vision plain model, without quantization or adapters
    VisionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_vision_arch)]
        arch: Option<VisionLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[arg(short, long)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;).
        #[arg(short, long)]
        from_uqff: Option<String>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// This is only supported on the Qwen2-VL and Idefics models. Others handle this internally.
        #[arg(short = 'e', long)]
        max_edge: Option<u32>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        #[arg(short, long)]
        calibration_file: Option<PathBuf>,

        /// .cimatrix file to enhance GGUF quantizations with. This must be a .cimatrix file.
        #[arg(short, long)]
        imatrix: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN)]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE)]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES)]
        max_num_images: usize,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        #[arg(long, default_value_t = AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH)]
        max_image_length: usize,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        hf_cache_path: Option<PathBuf>,

        /// Path to local Matryoshka Transformer configuration CSV file
        #[arg(long)]
        matformer_config_path: Option<PathBuf>,

        /// Name of the Matryoshka Transformer slice to use
        #[arg(long)]
        matformer_slice_name: Option<String>,

        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: https://arxiv.org/abs/2310.02410).
        #[arg(long)]
        organization: Option<IsqOrganization>,
    },

    /// Select a diffusion model, without quantization or adapters
    #[command(name = "diffusion")]
    DiffusionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_diffusion_arch)]
        arch: DiffusionLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },

    Speech {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// DAC Model ID to load from. If not provided, this is automatically downloaded from the default path for the model.
        /// This may be a HF hub repo or a local path.
        #[arg(short, long)]
        dac_model_id: Option<String>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_speech_arch)]
        arch: SpeechLoaderType,

        /// Model data type. Defaults to `auto`.
        #[arg(long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        dtype: ModelDType,
    },

    /// Select multi-model mode with configuration file
    #[command(name = "multi-model")]
    MultiModel {
        /// Multi-model configuration file path (JSON format)
        #[arg(short, long)]
        config: String,

        /// Default model ID to use when no model is specified in requests
        #[arg(short, long)]
        default_model_id: Option<String>,
    },

    /// Select an embedding model, without quantization or adapters
    Embedding {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        #[arg(short, long)]
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[arg(short, long)]
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[arg(short, long, value_parser = parse_embedding_arch)]
        #[serde(default)]
        arch: Option<EmbeddingLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[arg(short, long, default_value_t = ModelDType::Auto, value_parser = parse_model_dtype)]
        #[serde(default = "default_model_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[arg(long)]
        #[serde(default)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[arg(short, long)]
        #[serde(default)]
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[arg(short, long)]
        #[serde(default)]
        from_uqff: Option<String>,

        /// Cache path for Hugging Face models downloaded locally
        #[arg(long)]
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,
    },
}
