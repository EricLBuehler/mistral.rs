use std::{fs::File, num::NonZeroUsize, path::PathBuf};

use serde::Deserialize;

use crate::{
    amoe::AnyMoeConfig, pipeline::IsqOrganization, AnyMoeLoader, GGMLLoaderBuilder,
    GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig, Loader, ModelDType,
    NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, SpeculativeConfig,
    SpeculativeLoader, Topology, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
    GGUF_MULTI_FILE_DELIMITER,
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

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TomlModelSelected {
    /// Select a plain model, without quantization or adapters
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
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<PathBuf>,
    },

    /// Select an X-LoRA architecture
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
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<PathBuf>,
    },

    /// Select a LoRA architecture
    Lora {
        /// Force a base model ID to load from instead of using the ordering file. This may be a HF hub repo or a local path.
        model_id: Option<String>,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// The architecture of the model.
        arch: Option<NormalLoaderType>,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<PathBuf>,
    },

    /// Select a GGUF model.
    #[allow(clippy::upper_case_acronyms)]
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a GGUF model with X-LoRA.
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a GGUF model with LoRA.
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a GGML model.
    #[allow(clippy::upper_case_acronyms)]
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a GGML model with X-LoRA.
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a GGML model with LoRA.
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

        /// Path to a topology YAML file.
        topology: Option<String>,
    },

    /// Select a vision plain model, without quantization or adapters
    VisionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: VisionLoaderType,

        /// Model data type. Defaults to `auto`.
        #[serde(default = "default_dtype")]
        dtype: ModelDType,

        /// Path to a topology YAML file.
        topology: Option<String>,

        /// UQFF path to write to.
        write_uqff: Option<PathBuf>,

        /// UQFF path to load from. If provided, this takes precedence over applying ISQ.
        from_uqff: Option<PathBuf>,
    },
}

#[derive(Deserialize)]
pub struct SpeculativeTomlModelSelected {
    /// Gamma value for the model
    gamma: usize,

    /// Base model
    draft_model: TomlModelSelected,
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
    model: TomlModelSelected,

    /// Speculative model selector
    speculative: Option<SpeculativeTomlModelSelected>,

    /// AnyMoE config
    anymoe: Option<AnyMoeTomlModelSelected>,
}

#[derive(Clone)]
struct TomlLoaderInnerParams {
    use_flash_attn: bool,
    chat_template: Option<String>,
    no_kv_cache: bool,
    tokenizer_json: Option<String>,
    prompt_batchsize: Option<NonZeroUsize>,
}

pub struct TomlLoaderArgs {
    pub use_flash_attn: bool,
    pub chat_template: Option<String>,
    pub no_kv_cache: bool,
    pub prompt_batchsize: Option<NonZeroUsize>,
}

pub fn get_toml_selected_model_dtype(model: &TomlSelector) -> ModelDType {
    match model.model {
        TomlModelSelected::Plain { dtype, .. }
        | TomlModelSelected::Lora { dtype, .. }
        | TomlModelSelected::XLora { dtype, .. }
        | TomlModelSelected::VisionPlain { dtype, .. } => dtype,
        TomlModelSelected::GGUF { .. }
        | TomlModelSelected::LoraGGUF { .. }
        | TomlModelSelected::GGML { .. }
        | TomlModelSelected::LoraGGML { .. }
        | TomlModelSelected::XLoraGGUF { .. }
        | TomlModelSelected::XLoraGGML { .. } => ModelDType::Auto,
    }
}

fn loader_from_selected(
    args: TomlLoaderInnerParams,
    model: TomlModelSelected,
) -> anyhow::Result<Box<dyn Loader>> {
    let use_flash_attn = args.use_flash_attn;
    let loader: Box<dyn Loader> = match model {
        TomlModelSelected::Plain {
            model_id,
            arch,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
                organization: organization.unwrap_or_default(),
                write_uqff,
                from_uqff,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
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
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff,
            },
            args.chat_template,
            args.tokenizer_json,
            model_id,
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
            adapters_model_id,
            order,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
                organization: Default::default(),
                write_uqff,
                from_uqff,
            },
            args.chat_template,
            args.tokenizer_json,
            model_id,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(arch)?,
        TomlModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            topology,
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
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
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
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
        } => GGUFLoaderBuilder::new(
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename
                .split(GGUF_MULTI_FILE_DELIMITER)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>(),
            GGUFSpecificConfig {
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            args.tokenizer_json,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .with_lora(
            adapters_model_id,
            serde_json::from_reader(
                File::open(order.clone())
                    .unwrap_or_else(|_| panic!("Could not load ordering file at {order}")),
            )?,
        )
        .build(),
        TomlModelSelected::VisionPlain {
            model_id,
            arch,
            dtype: _,
            topology,
            write_uqff,
            from_uqff,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                prompt_batchsize: args.prompt_batchsize,
                topology: Topology::from_option_path(topology)?,
                write_uqff,
                from_uqff,
            },
            args.chat_template,
            args.tokenizer_json,
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
            use_flash_attn: args.use_flash_attn,
            chat_template: args.chat_template,
            no_kv_cache: args.no_kv_cache,
            tokenizer_json: selector.tokenizer_json,
            prompt_batchsize: args.prompt_batchsize,
        };
        let loader = loader_from_selected(args.clone(), selector.model)?;
        let loader = if let Some(speculative) = selector.speculative {
            let draft_loader = loader_from_selected(args, speculative.draft_model)?;
            Box::new(SpeculativeLoader {
                target: loader,
                draft: draft_loader,
                config: SpeculativeConfig {
                    gamma: speculative.gamma,
                },
            })
        } else {
            loader
        };
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
