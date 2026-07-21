use std::{fs::File, path::PathBuf, str::FromStr};

use serde::{de, Deserialize, Deserializer};

use crate::{
    amoe::AnyMoeConfig,
    pipeline::{EmbeddingLoaderType, IsqOrganization},
    AnyMoeLoader, AutoDeviceMapParams, EmbeddingLoaderBuilder, EmbeddingSpecificConfig,
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig, Loader,
    LoraAdapterSpec, LoraRuntimeConfig, ModelDType, MultimodalLoaderBuilder, MultimodalLoaderType,
    MultimodalSpecificConfig, NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig,
    Topology, UqffWriteConfig, GGUF_MULTI_FILE_DELIMITER, UQFF_MULTI_FILE_DELIMITER,
};

fn default_one() -> usize {
    1
}

fn default_dtype() -> ModelDType {
    ModelDType::Auto
}

fn default_empty_vec_usize() -> Vec<usize> {
    Vec::new()
}

fn default_max_seq_len() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN
}

fn default_max_batch_size() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE
}

fn default_max_num_images() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES
}

fn default_max_image_length() -> usize {
    AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind")]
pub enum TomlModelSelected {
    /// Select a plain model, without quantization or adapters
    #[serde(rename = "plain")]
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// ISQ organization: `default` or `moqe` (Mixture of Quantized Experts: https://arxiv.org/abs/2310.02410).
        organization: Option<IsqOrganization>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<String>,

        /// .imatrix file to enhance GGUF quantizations with.
        /// Incompatible with `--imatrix/-i`
        imatrix: Option<PathBuf>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        /// Incompatible with `--imatrix/-i`
        calibration_file: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,
    },

    /// Select an X-LoRA architecture
    #[serde(rename = "xlora")]
    XLora {
        /// Force a base model ID to load from instead of using the ordering file. This may be a HF hub repo or a local path.
        model_id: Option<String>,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,
    },

    /// Select a LoRA architecture
    #[serde(rename = "lora")]
    Lora {
        /// Base model ID. This may be a Hugging Face repository or a local path.
        model_id: String,

        /// LoRA adapters to preload.
        #[serde(default)]
        adapters: Vec<LoraAdapterSpec>,

        /// Dynamic LoRA runtime capacity and rank limits.
        #[serde(default)]
        runtime_config: LoraRuntimeConfig,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,
    },

    /// Select a GGUF model.
    #[allow(clippy::upper_case_acronyms)]
    #[serde(rename = "gguf")]
    GGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        quantized_filename: String,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a GGUF model with X-LoRA.
    #[serde(rename = "xlora_gguf")]
    XLoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        quantized_filename: String,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a GGUF model with legacy LoRA.
    #[serde(rename = "legacy_lora_gguf")]
    LoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename(s).
        /// May be a single filename, or use a delimiter of " " (a single space) for multiple files.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a GGML model.
    #[allow(clippy::upper_case_acronyms)]
    #[serde(rename = "ggml")]
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a GGML model with X-LoRA.
    #[serde(rename = "xlora_ggml")]
    XLoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a GGML model with legacy LoRA.
    #[serde(rename = "legacy_lora_ggml")]
    LoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,
    },

    /// Select a multimodal plain model, without quantization or adapters
    #[serde(rename = "multimodal")]
    MultimodalPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: Option<MultimodalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<String>,

        /// Automatically resize and pad images to this maximum edge length. Aspect ratio is preserved.
        /// This is only supported on the Qwen2-VL and Idefics 2 models. Others handle this internally.
        max_edge: Option<u32>,

        /// Generate and utilize an imatrix to enhance GGUF quantizations.
        calibration_file: Option<PathBuf>,

        /// .cimatrix file to enhance GGUF quantizations with. This must be a .cimatrix file.
        imatrix: Option<PathBuf>,

        /// Maximum prompt sequence length to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_seq_len")]
        max_seq_len: usize,

        /// Maximum prompt batch size to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_batch_size")]
        max_batch_size: usize,

        /// Maximum prompt number of images to expect for this model. This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_num_images")]
        max_num_images: usize,

        /// Maximum expected image size will have this edge length on both edges.
        /// This affects automatic device mapping but is not a hard limit.
        #[serde(default = "default_max_image_length")]
        max_image_length: usize,

        /// Cache path for Hugging Face models downloaded locally
        hf_cache_path: Option<PathBuf>,

        /// ISQ organization: `default` or `moqe`.
        organization: Option<IsqOrganization>,
    },

    /// Select an embedding model, without quantization or adapters
    #[serde(rename = "embedding")]
    Embedding {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
        #[serde(default)]
        tokenizer_json: Option<String>,

        /// The architecture of the model.
        #[serde(default)]
        arch: Option<EmbeddingLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        #[serde(default)]
        topology: Option<String>,

        /// UQFF path to write to.
        #[serde(default)]
        write_uqff: Option<UqffWriteConfig>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ. Specify multiple files using a semicolon delimiter (;)
        #[serde(default)]
        from_uqff: Option<String>,

        /// Cache path for Hugging Face models downloaded locally
        #[serde(default)]
        hf_cache_path: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Eq, PartialEq)]
enum LegacyModelFamily {
    XLora,
    Lora,
    LegacyLora,
    Multimodal,
    Embedding,
}

fn deserialize_model_selected<'de, D>(deserializer: D) -> Result<TomlModelSelected, D::Error>
where
    D: Deserializer<'de>,
{
    let mut value = toml::Value::deserialize(deserializer)?;
    let table = value
        .as_table_mut()
        .ok_or_else(|| de::Error::custom("model selector must be a TOML table"))?;

    if table.contains_key("adapter_model_ids") {
        return Err(de::Error::custom(
            "legacy `adapter_model_ids` is no longer supported; use `kind = \"lora\"` with `adapters`",
        ));
    }

    if !table.contains_key("kind") {
        let kind = infer_legacy_model_kind(table).map_err(de::Error::custom)?;
        table.insert("kind".to_string(), toml::Value::String(kind.to_string()));
    }

    value.try_into().map_err(de::Error::custom)
}

fn infer_legacy_model_kind(table: &toml::Table) -> Result<&'static str, String> {
    let quantized = [
        "tok_model_id",
        "quantized_model_id",
        "quantized_filename",
        "gqa",
    ]
    .iter()
    .any(|key| table.contains_key(*key));
    let has_gqa = table.contains_key("gqa");

    let mut families = Vec::new();
    if table.contains_key("xlora_model_id") || table.contains_key("tgt_non_granular_index") {
        families.push(LegacyModelFamily::XLora);
    }
    if table.contains_key("adapters") || table.contains_key("runtime_config") {
        families.push(LegacyModelFamily::Lora);
    }
    if table.contains_key("adapters_model_id") {
        families.push(LegacyModelFamily::LegacyLora);
    }
    if ["max_edge", "max_num_images", "max_image_length"]
        .iter()
        .any(|key| table.contains_key(*key))
    {
        families.push(LegacyModelFamily::Multimodal);
    }
    if let Some(family) = infer_legacy_arch_family(table)? {
        if !families.contains(&family) {
            families.push(family);
        }
    }

    if families.len() > 1 {
        return Err(
            "model selector without `kind` has conflicting fields; set `kind` explicitly"
                .to_string(),
        );
    }

    let family = families.first().copied();
    match (family, quantized, has_gqa) {
        (Some(LegacyModelFamily::XLora), false, _) => Ok("xlora"),
        (Some(LegacyModelFamily::XLora), true, false) => Ok("xlora_gguf"),
        (Some(LegacyModelFamily::XLora), true, true) => Ok("xlora_ggml"),
        (Some(LegacyModelFamily::Lora), false, _) => Ok("lora"),
        (Some(LegacyModelFamily::LegacyLora), true, false) => Ok("legacy_lora_gguf"),
        (Some(LegacyModelFamily::LegacyLora), true, true) => Ok("legacy_lora_ggml"),
        (Some(LegacyModelFamily::Multimodal), false, _) => Ok("multimodal"),
        (Some(LegacyModelFamily::Embedding), false, _) => Ok("embedding"),
        (None, true, false) => Ok("gguf"),
        (None, true, true) => Ok("ggml"),
        (None, false, _) => Ok("plain"),
        _ => Err(
            "model selector without `kind` mixes incompatible model fields; set `kind` explicitly"
                .to_string(),
        ),
    }
}

fn infer_legacy_arch_family(table: &toml::Table) -> Result<Option<LegacyModelFamily>, String> {
    let Some(arch) = table.get("arch") else {
        return Ok(None);
    };
    if NormalLoaderType::deserialize(arch.clone()).is_ok() {
        return Ok(None);
    }

    let multimodal = MultimodalLoaderType::deserialize(arch.clone()).is_ok();
    let embedding = EmbeddingLoaderType::deserialize(arch.clone()).is_ok();
    match (multimodal, embedding) {
        (true, false) => Ok(Some(LegacyModelFamily::Multimodal)),
        (false, true) => Ok(Some(LegacyModelFamily::Embedding)),
        (true, true) => {
            Err("model selector architecture is ambiguous; set `kind` explicitly".to_string())
        }
        (false, false) => Ok(None),
    }
}

#[derive(Deserialize)]
pub struct AnyMoeTomlModelSelected {
    /// Config
    config: AnyMoeConfig,

    /// Base model
    dataset_json: String,

    /// Prefix of the mlp key (the part before the layer number: "a.b.c" in "a.b.c.0.mlp")
    prefix: String,

    /// Name of the mlp key (the part before the layer number: "mlp" in "a.b.c.0.mlp")
    mlp: String,

    /// Expert model ids
    model_ids: Vec<String>,

    /// Layer ids (zero indexed) of layers to apply AnyMoE to, if empty will use all
    #[serde(default = "default_empty_vec_usize")]
    layers: Vec<usize>,
}

#[derive(Deserialize)]
pub struct TomlSelector {
    /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
    tokenizer_json: Option<String>,

    /// Selected model
    #[serde(deserialize_with = "deserialize_model_selected")]
    model: TomlModelSelected,

    /// Legacy target/draft speculative decoding was removed. Keep this field
    /// only to reject old configs explicitly instead of silently ignoring them.
    #[serde(default)]
    speculative: Option<serde::de::IgnoredAny>,

    /// AnyMoE config
    anymoe: Option<AnyMoeTomlModelSelected>,
}

#[derive(Clone)]
struct TomlLoaderInnerParams {
    chat_template: Option<String>,
    no_kv_cache: bool,
    tokenizer_json: Option<String>,
    jinja_explicit: Option<String>,
}

pub struct TomlLoaderArgs {
    pub chat_template: Option<String>,
    pub no_kv_cache: bool,
    pub jinja_explicit: Option<String>,
}

pub fn get_toml_selected_model_dtype(model: &TomlSelector) -> ModelDType {
    match model.model {
        TomlModelSelected::Plain { dtype, .. }
        | TomlModelSelected::Lora { dtype, .. }
        | TomlModelSelected::XLora { dtype, .. }
        | TomlModelSelected::MultimodalPlain { dtype, .. }
        | TomlModelSelected::GGUF { dtype, .. }
        | TomlModelSelected::GGML { dtype, .. }
        | TomlModelSelected::XLoraGGUF { dtype, .. }
        | TomlModelSelected::XLoraGGML { dtype, .. }
        | TomlModelSelected::LoraGGUF { dtype, .. }
        | TomlModelSelected::LoraGGML { dtype, .. }
        | TomlModelSelected::Embedding { dtype, .. } => dtype,
    }
}

pub fn get_toml_selected_model_device_map_params(
    model: &TomlSelector,
) -> anyhow::Result<AutoDeviceMapParams> {
    match model.model {
        TomlModelSelected::Plain {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::Lora {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::XLora {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::GGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::GGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::XLoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::XLoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::LoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | TomlModelSelected::LoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len,
            max_batch_size,
        }),
        TomlModelSelected::Embedding { .. } => Ok(AutoDeviceMapParams::default_text()),
        TomlModelSelected::MultimodalPlain {
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => Ok(AutoDeviceMapParams::Multimodal {
            max_seq_len,
            max_batch_size,
            max_image_shape: (max_image_length, max_image_length),
            max_num_images,
        }),
    }
}

fn loader_from_selected(
    args: TomlLoaderInnerParams,
    model: TomlModelSelected,
) -> anyhow::Result<Box<dyn Loader>> {
    let loader: Box<dyn Loader> = match model {
        TomlModelSelected::Plain {
            model_id,
            arch,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix,
                calibration_file,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(arch)?,
        TomlModelSelected::XLora {
            model_id,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix: None,
                calibration_file: None,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            args.chat_template,
            args.tokenizer_json,
            model_id,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            args.no_kv_cache,
            tgt_non_granular_index,
        )
        .build(arch)?,
        TomlModelSelected::Lora {
            model_id,
            adapters,
            runtime_config,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            max_seq_len: _,
            max_batch_size: _,
            hf_cache_path,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix: None,
                calibration_file: None,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_lora(adapters, runtime_config)
        .build(arch)?,
        TomlModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                topology: Topology::from_option_path(topology)?,
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        TomlModelSelected::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                topology: Topology::from_option_path(topology)?,
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            args.no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        TomlModelSelected::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            topology,
            ..
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                topology: Topology::from_option_path(topology)?,
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(),
        TomlModelSelected::GGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        TomlModelSelected::XLoraGGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            gqa,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_xlora(
            xlora_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
            args.no_kv_cache,
            tgt_non_granular_index,
        )
        .build(),
        TomlModelSelected::LoraGGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            gqa,
            topology,
            dtype: _,
            max_seq_len: _,
            max_batch_size: _,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(),
        TomlModelSelected::MultimodalPlain {
            model_id,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            max_edge,
            calibration_file,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            imatrix,
            hf_cache_path,
            organization,
        } => MultimodalLoaderBuilder::new(
            MultimodalSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                max_edge,
                calibration_file,
                imatrix,
                hf_cache_path,
                matformer_config_path: None,
                matformer_slice_name: None,
                organization: organization.unwrap_or_default(),
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
            args.jinja_explicit,
        )
        .build(arch),
        TomlModelSelected::Embedding {
            model_id,
            tokenizer_json,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
            hf_cache_path,
        } => EmbeddingLoaderBuilder::new(
            EmbeddingSpecificConfig {
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff: from_uqff.map(|x| {
                    x.split(UQFF_MULTI_FILE_DELIMITER)
                        .map(PathBuf::from_str)
                        .map(|x| x.unwrap())
                        .collect::<Vec<_>>()
                }),
                imatrix: None,
                calibration_file: None,
                hf_cache_path,
            },
            tokenizer_json,
            Some(model_id),
        )
        .build(arch),
    };
    Ok(loader)
}

impl TryInto<Box<dyn Loader>> for (TomlSelector, TomlLoaderArgs) {
    type Error = anyhow::Error;
    fn try_into(self) -> Result<Box<dyn Loader>, Self::Error> {
        let (selector, args) = self;
        let args = TomlLoaderInnerParams {
            chat_template: args.chat_template,
            no_kv_cache: args.no_kv_cache,
            tokenizer_json: selector.tokenizer_json,
            jinja_explicit: args.jinja_explicit,
        };
        if selector.speculative.is_some() {
            anyhow::bail!(
                "legacy target/draft speculative decoding in TOML configs was removed; use MTP through --mtp-model or the MTP API instead"
            );
        }
        if matches!(&selector.model, TomlModelSelected::Lora { .. }) && selector.anymoe.is_some() {
            anyhow::bail!(
                "dynamic LoRA cannot be combined with AnyMoE in one TOML model configuration"
            );
        }
        let loader = loader_from_selected(args.clone(), selector.model)?;
        let loader = if let Some(AnyMoeTomlModelSelected {
            config,
            dataset_json,
            prefix,
            mlp,
            model_ids,
            layers,
        }) = selector.anymoe
        {
            Box::new(AnyMoeLoader {
                target: loader,
                config,
                path: dataset_json,
                prefix,
                mlp,
                model_ids,
                layers,
            })
        } else {
            loader
        };
        Ok(loader)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_lora_example_selects_lora() {
        let selector: TomlSelector =
            toml::from_str(include_str!("../../toml-selectors/lora.toml")).unwrap();

        let TomlModelSelected::Lora {
            model_id,
            adapters,
            runtime_config,
            ..
        } = selector.model
        else {
            panic!("explicit lora kind selected a different model variant");
        };
        assert_eq!(model_id, "mistralai/Mistral-7B-Instruct-v0.1");
        assert_eq!(
            adapters,
            vec![LoraAdapterSpec::new("zephyr", "typeof/zephyr-7b-beta-lora")]
        );
        assert_eq!(runtime_config.max_adapters, 4);
        assert_eq!(runtime_config.max_rank, 128);
        assert_eq!(runtime_config.max_bytes, 2_147_483_648);
    }

    #[test]
    fn legacy_plain_without_kind_is_supported() {
        let selector = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "mistralai/Mistral-7B-Instruct-v0.1"
            "#,
        )
        .unwrap();

        assert!(matches!(selector.model, TomlModelSelected::Plain { .. }));
    }

    #[test]
    fn legacy_quantized_models_without_kind_are_inferred() {
        let gguf = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.gguf"
            "#,
        )
        .unwrap();
        assert!(matches!(gguf.model, TomlModelSelected::GGUF { .. }));

        let ggml = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.bin"
                gqa = 8
            "#,
        )
        .unwrap();
        assert!(matches!(ggml.model, TomlModelSelected::GGML { .. }));
    }

    #[test]
    fn legacy_xlora_models_without_kind_are_inferred() {
        let normal = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "base"
                xlora_model_id = "adapter"
                order = "order.json"
            "#,
        )
        .unwrap();
        assert!(matches!(normal.model, TomlModelSelected::XLora { .. }));

        let gguf = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.gguf"
                xlora_model_id = "adapter"
                order = "order.json"
            "#,
        )
        .unwrap();
        assert!(matches!(gguf.model, TomlModelSelected::XLoraGGUF { .. }));

        let ggml = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.bin"
                xlora_model_id = "adapter"
                order = "order.json"
                gqa = 8
            "#,
        )
        .unwrap();
        assert!(matches!(ggml.model, TomlModelSelected::XLoraGGML { .. }));
    }

    #[test]
    fn legacy_quantized_lora_models_without_kind_are_inferred() {
        let gguf = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.gguf"
                adapters_model_id = "adapter"
                order = "order.json"
            "#,
        )
        .unwrap();
        assert!(matches!(gguf.model, TomlModelSelected::LoraGGUF { .. }));

        let ggml = toml::from_str::<TomlSelector>(
            r#"
                [model]
                tok_model_id = "tokenizer"
                quantized_model_id = "weights"
                quantized_filename = "model.bin"
                adapters_model_id = "adapter"
                order = "order.json"
                gqa = 8
            "#,
        )
        .unwrap();
        assert!(matches!(ggml.model, TomlModelSelected::LoraGGML { .. }));
    }

    #[test]
    fn legacy_specialized_models_without_kind_are_inferred() {
        let multimodal = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "vision-model"
                arch = "qwen2vl"
            "#,
        )
        .unwrap();
        assert!(matches!(
            multimodal.model,
            TomlModelSelected::MultimodalPlain { .. }
        ));

        let embedding = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "embedding-model"
                arch = "qwen3embedding"
                tokenizer_json = "tokenizer.json"
            "#,
        )
        .unwrap();
        assert!(matches!(
            embedding.model,
            TomlModelSelected::Embedding { .. }
        ));
    }

    #[test]
    fn legacy_nested_tokenizer_json_keeps_plain_default() {
        let selector = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "text-model"
                tokenizer_json = "tokenizer.json"
            "#,
        )
        .unwrap();

        assert!(matches!(selector.model, TomlModelSelected::Plain { .. }));
    }

    #[test]
    fn explicit_kind_is_authoritative() {
        let selector = toml::from_str::<TomlSelector>(
            r#"
                [model]
                kind = "multimodal"
                model_id = "vision-model"
            "#,
        )
        .unwrap();

        assert!(matches!(
            selector.model,
            TomlModelSelected::MultimodalPlain { .. }
        ));
    }

    #[test]
    fn legacy_selector_with_conflicting_discriminators_is_rejected() {
        let error = toml::from_str::<TomlSelector>(
            r#"
                [model]
                model_id = "ambiguous-model"
                max_edge = 1024
                arch = "qwen3embedding"
            "#,
        )
        .err()
        .expect("conflicting model fields should not parse");

        assert!(error.to_string().contains("conflicting fields"));
    }

    #[test]
    fn dynamic_lora_and_anymoe_are_rejected_together() {
        let (_, anymoe) = include_str!("../../toml-selectors/anymoe_lora.toml")
            .split_once("[anymoe]")
            .unwrap();
        let source = format!(
            "{}\n[anymoe]{}",
            include_str!("../../toml-selectors/lora.toml"),
            anymoe
        );
        let selector: TomlSelector = toml::from_str(&source).unwrap();
        let result: anyhow::Result<Box<dyn Loader>> = (
            selector,
            TomlLoaderArgs {
                chat_template: None,
                no_kv_cache: false,
                jinja_explicit: None,
            },
        )
            .try_into();

        assert!(result
            .err()
            .expect("dynamic LoRA and AnyMoE must be rejected")
            .to_string()
            .contains("dynamic LoRA cannot be combined with AnyMoE"));
    }
}
