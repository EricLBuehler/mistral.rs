use std::fs::{self, File};

use crate::{
    get_toml_selected_model_dtype,
    pipeline::{
        GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, GGUFSpecificConfig,
        NormalSpecificConfig,
    },
    Loader, ModelDType, ModelSelected, NormalLoaderBuilder, TomlLoaderArgs, TomlSelector,
    VisionLoaderBuilder, VisionSpecificConfig,
};

/// A builder for a loader using the selected model.
pub struct LoaderBuilder {
    model: ModelSelected,
    no_kv_cache: bool,
    chat_template: Option<String>,
    use_flash_attn: bool,
}

impl LoaderBuilder {
    pub fn new(model: ModelSelected) -> Self {
        Self {
            model,
            no_kv_cache: false,
            chat_template: None,
            use_flash_attn: false,
        }
    }

    pub fn with_no_kv_cache(mut self, no_kv_cache: bool) -> Self {
        self.no_kv_cache = no_kv_cache;
        self
    }
    pub fn with_chat_template(mut self, chat_template: Option<String>) -> Self {
        self.chat_template = chat_template;
        self
    }
    pub fn with_use_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }

    pub fn build(self) -> anyhow::Result<Box<dyn Loader>> {
        loader_from_model_selected(self)
    }
}

pub fn get_tgt_non_granular_index(model: &ModelSelected) -> Option<usize> {
    match model {
        ModelSelected::Plain { .. }
        | ModelSelected::Lora { .. }
        | ModelSelected::GGUF { .. }
        | ModelSelected::LoraGGUF { .. }
        | ModelSelected::GGML { .. }
        | ModelSelected::LoraGGML { .. }
        | ModelSelected::Toml { .. }
        | ModelSelected::VisionPlain { .. } => None,
        ModelSelected::XLora {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraGGUF {
            tgt_non_granular_index,
            ..
        }
        | ModelSelected::XLoraGGML {
            tgt_non_granular_index,
            ..
        } => *tgt_non_granular_index,
    }
}

pub fn get_model_dtype(model: &ModelSelected) -> anyhow::Result<ModelDType> {
    match model {
        ModelSelected::Plain { dtype, .. }
        | ModelSelected::Lora { dtype, .. }
        | ModelSelected::XLora { dtype, .. }
        | ModelSelected::VisionPlain { dtype, .. } => Ok(*dtype),
        ModelSelected::GGUF { .. }
        | ModelSelected::LoraGGUF { .. }
        | ModelSelected::GGML { .. }
        | ModelSelected::LoraGGML { .. }
        | ModelSelected::XLoraGGUF { .. }
        | ModelSelected::XLoraGGML { .. } => Ok(ModelDType::Auto),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            Ok(get_toml_selected_model_dtype(&selector))
        }
    }
}

fn loader_from_model_selected(args: LoaderBuilder) -> anyhow::Result<Box<dyn Loader>> {
    let use_flash_attn = args.use_flash_attn;
    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            let args = TomlLoaderArgs {
                use_flash_attn,
                chat_template: args.chat_template,
                no_kv_cache: args.no_kv_cache,
            };
            (selector, args).try_into()?
        }
        ModelSelected::Plain {
            model_id,
            repeat_last_n,
            tokenizer_json,
            arch,
            dtype: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .build(arch),
        ModelSelected::XLora {
            model_id,
            xlora_model_id,
            repeat_last_n,
            order,
            tokenizer_json,
            tgt_non_granular_index,
            arch,
            dtype: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::Lora {
            model_id,
            tokenizer_json,
            adapters_model_id,
            repeat_last_n,
            order,
            arch,
            dtype: _,
        } => NormalLoaderBuilder::new(
            NormalSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig { repeat_last_n },
            args.chat_template,
            tok_model_id,
            quantized_model_id,
            quantized_filename,
        )
        .build(),
        ModelSelected::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tgt_non_granular_index,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig { repeat_last_n },
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
        ModelSelected::LoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            adapters_model_id,
            order,
        } => GGUFLoaderBuilder::new(
            GGUFSpecificConfig { repeat_last_n },
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
        ModelSelected::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig { repeat_last_n, gqa },
            args.chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
        )
        .build(),
        ModelSelected::XLoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig { repeat_last_n, gqa },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::LoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            repeat_last_n,
            adapters_model_id,
            order,
            gqa,
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig { repeat_last_n, gqa },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::VisionPlain {
            model_id,
            repeat_last_n,
            tokenizer_json,
            arch,
            dtype: _,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                repeat_last_n,
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
        )
        .build(arch),
    };
    Ok(loader)
}
