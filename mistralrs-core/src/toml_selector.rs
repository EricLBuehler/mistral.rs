use std::fs::File;

use serde::Deserialize;

use crate::{
    GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig, Loader,
    ModelDType, NormalLoaderBuilder, NormalLoaderType, NormalSpecificConfig, SpeculativeConfig,
    SpeculativeLoader, VisionLoaderBuilder, VisionLoaderType, VisionSpecificConfig,
};

fn default_repeat_last_n() -> usize {
    64
}

fn default_one() -> usize {
    1
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum TomlModelSelected {
    /// Select a plain model, without quantization or adapters
    Plain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,
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
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,
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
        arch: NormalLoaderType,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,
    },

    /// Select a GGUF model.
    #[allow(clippy::upper_case_acronyms)]
    GGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        quantized_filename: String,
    },

    /// Select a GGUF model with X-LoRA.
    XLoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        quantized_filename: String,

        /// Model ID to load X-LoRA from. This may be a HF hub repo or a local path.
        xlora_model_id: String,

        /// Ordering JSON file
        order: String,

        /// Index of completion tokens to generate scalings up until. If this is 1, then there will be one completion token generated before it is cached.
        /// This makes the maximum running sequences 1.
        tgt_non_granular_index: Option<usize>,
    },

    /// Select a GGUF model with LoRA.
    LoraGGUF {
        /// `tok_model_id` is the local or remote model ID where you can find a `tokenizer_config.json` file.
        /// If the `chat_template` is specified, then it will be treated as a path and used over remote files,
        /// removing all remote accesses.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,
    },

    /// Select a GGML model.
    #[allow(clippy::upper_case_acronyms)]
    GGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: String,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        quantized_filename: String,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,
    },

    /// Select a GGML model with X-LoRA.
    XLoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
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
    },

    /// Select a GGML model with LoRA.
    LoraGGML {
        /// Model ID to load the tokenizer from. This may be a HF hub repo or a local path.
        tok_model_id: Option<String>,

        /// Quantized model ID to find the `quantized_filename`, only applicable if `quantized` is set.
        /// This may be a HF hub repo or a local path.
        quantized_model_id: String,

        /// Quantized filename, only applicable if `quantized` is set.
        quantized_filename: String,

        /// Model ID to load LoRA from. This may be a HF hub repo or a local path.
        adapters_model_id: String,

        /// Ordering JSON file
        order: String,

        /// GQA value
        #[serde(default = "default_one")]
        gqa: usize,
    },

    /// Select a vision plain model, without quantization or adapters
    VisionPlain {
        /// Model ID to load from. This may be a HF hub repo or a local path.
        model_id: String,

        /// The architecture of the model.
        arch: VisionLoaderType,

        /// Model data type. Defaults to `auto`.
        dtype: ModelDType,
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
pub struct TomlSelector {
    /// Path to local tokenizer.json file. If this is specified it is used over any remote file.
    tokenizer_json: Option<String>,

    /// Control the application of repeat penalty for the last n tokens
    #[serde(default = "default_repeat_last_n")]
    repeat_last_n: usize,

    /// Selected model
    model: TomlModelSelected,

    /// Speculative model selector
    speculative: Option<SpeculativeTomlModelSelected>,
}

#[derive(Clone)]
struct TomlLoaderInnerParams {
    use_flash_attn: bool,
    chat_template: Option<String>,
    no_kv_cache: bool,
    tokenizer_json: Option<String>,
    repeat_last_n: usize,
}

pub struct TomlLoaderArgs {
    pub use_flash_attn: bool,
    pub chat_template: Option<String>,
    pub no_kv_cache: bool,
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
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: args.repeat_last_n,
            },
            args.chat_template,
            args.tokenizer_json,
            Some(model_id),
        )
        .build(arch),
        TomlModelSelected::XLora {
            model_id,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            arch,
            dtype: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: args.repeat_last_n,
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
        .build(arch),
        TomlModelSelected::Lora {
            model_id,
            adapters_model_id,
            order,
            arch,
            dtype: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n: args.repeat_last_n,
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
        .build(arch),
        TomlModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: args.repeat_last_n,
            },
            args.chat_template,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
        )
        .build(),
        TomlModelSelected::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: args.repeat_last_n,
            },
            args.chat_template,
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
        TomlModelSelected::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig {
                repeat_last_n: args.repeat_last_n,
            },
            args.chat_template,
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
        TomlModelSelected::GGML {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: args.repeat_last_n,
                gqa,
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: args.repeat_last_n,
                gqa,
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
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                repeat_last_n: args.repeat_last_n,
                gqa,
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
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                repeat_last_n: args.repeat_last_n,
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
            repeat_last_n: selector.repeat_last_n,
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
        Ok(loader)
    }
}
