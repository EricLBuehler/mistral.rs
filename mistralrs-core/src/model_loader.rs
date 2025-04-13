use std::{
    fs::{self, File},
    num::NonZeroUsize,
    path::PathBuf,
    str::FromStr,
};

use mistralrs_quant::MULTI_LORA_DELIMITER;

use crate::{
    get_toml_selected_model_dtype,
    pipeline::{GGMLLoaderBuilder, GGMLSpecificConfig, GGUFLoaderBuilder, NormalSpecificConfig},
    toml_selector::get_toml_selected_model_device_map_params,
    AutoDeviceMapParams, DiffusionLoaderBuilder, DiffusionSpecificConfig, GGUFSpecificConfig,
    Loader, ModelDType, ModelSelected, NormalLoaderBuilder, TomlLoaderArgs, TomlSelector, Topology,
    VisionLoaderBuilder, VisionSpecificConfig, GGUF_MULTI_FILE_DELIMITER,
    UQFF_MULTI_FILE_DELIMITER,
};

/// A builder for a loader using the selected model.
pub struct LoaderBuilder {
    model: ModelSelected,
    no_kv_cache: bool,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
    use_flash_attn: bool,
    prompt_chunksize: Option<NonZeroUsize>,
}

impl LoaderBuilder {
    pub fn new(model: ModelSelected) -> Self {
        Self {
            model,
            no_kv_cache: false,
            chat_template: None,
            use_flash_attn: false,
            prompt_chunksize: None,
            jinja_explicit: None,
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
    pub fn with_jinja_explicit(mut self, jinja_explicit: Option<String>) -> Self {
        self.jinja_explicit = jinja_explicit;
        self
    }
    pub fn with_use_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = use_flash_attn;
        self
    }
    pub fn with_prompt_chunksize(mut self, prompt_chunksize: Option<NonZeroUsize>) -> Self {
        self.prompt_chunksize = prompt_chunksize;
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
        | ModelSelected::VisionPlain { .. }
        | ModelSelected::DiffusionPlain { .. } => None,
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
        | ModelSelected::VisionPlain { dtype, .. }
        | ModelSelected::DiffusionPlain { dtype, .. }
        | ModelSelected::GGML { dtype, .. }
        | ModelSelected::GGUF { dtype, .. }
        | ModelSelected::XLoraGGUF { dtype, .. }
        | ModelSelected::XLoraGGML { dtype, .. }
        | ModelSelected::LoraGGUF { dtype, .. }
        | ModelSelected::LoraGGML { dtype, .. } => Ok(*dtype),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            Ok(get_toml_selected_model_dtype(&selector))
        }
    }
}

pub fn get_auto_device_map_params(model: &ModelSelected) -> anyhow::Result<AutoDeviceMapParams> {
    match model {
        ModelSelected::Plain {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::Lora {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLora {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::GGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::GGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::XLoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::LoraGGUF {
            max_seq_len,
            max_batch_size,
            ..
        }
        | ModelSelected::LoraGGML {
            max_seq_len,
            max_batch_size,
            ..
        } => Ok(AutoDeviceMapParams::Text {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
        }),
        ModelSelected::VisionPlain {
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => Ok(AutoDeviceMapParams::Vision {
            max_seq_len: *max_seq_len,
            max_batch_size: *max_batch_size,
            max_image_shape: (*max_image_length, *max_image_length),
            max_num_images: *max_num_images,
        }),
        ModelSelected::DiffusionPlain { .. } => Ok(AutoDeviceMapParams::default_text()),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            get_toml_selected_model_device_map_params(&selector)
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
                prompt_chunksize: args.prompt_chunksize,
                jinja_explicit: args.jinja_explicit,
            };
            (selector, args).try_into()?
        }
        ModelSelected::Plain {
            model_id,
            tokenizer_json,
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
                use_flash_attn,
                prompt_chunksize: args.prompt_chunksize,
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
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(arch)?,
        ModelSelected::XLora {
            model_id,
            xlora_model_id,
            order,
            tokenizer_json,
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
                use_flash_attn,
                prompt_chunksize: args.prompt_chunksize,
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
            },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::Lora {
            model_id,
            tokenizer_json,
            adapter_model_id,
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
                use_flash_attn,
                prompt_chunksize: args.prompt_chunksize,
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
            },
            args.chat_template,
            tokenizer_json,
            model_id,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .with_lora(
            adapter_model_id
                .split(MULTI_LORA_DELIMITER)
                .map(ToString::to_string)
                .collect(),
        )
        .build(arch)?,
        ModelSelected::GGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
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
                prompt_chunksize: args.prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        ModelSelected::XLoraGGUF {
            tok_model_id,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
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
                prompt_chunksize: args.prompt_chunksize,
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
        ModelSelected::LoraGGUF {
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
                prompt_chunksize: args.prompt_chunksize,
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
        ModelSelected::GGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize: args.prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
            Some(tok_model_id),
            quantized_model_id,
            quantized_filename,
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(),
        ModelSelected::XLoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            xlora_model_id,
            order,
            tgt_non_granular_index,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize: args.prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::LoraGGML {
            tok_model_id,
            tokenizer_json,
            quantized_model_id,
            quantized_filename,
            adapters_model_id,
            order,
            gqa,
            topology,
            ..
        } => GGMLLoaderBuilder::new(
            GGMLSpecificConfig {
                gqa,
                prompt_chunksize: args.prompt_chunksize,
                topology: Topology::from_option_path(topology)?,
            },
            args.chat_template,
            tokenizer_json,
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
        ModelSelected::VisionPlain {
            model_id,
            tokenizer_json,
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
            hf_cache_path,
            imatrix,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
                use_flash_attn,
                prompt_chunksize: args.prompt_chunksize,
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
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
            args.jinja_explicit,
        )
        .build(arch),
        ModelSelected::DiffusionPlain {
            model_id,
            arch,
            dtype: _,
        } => {
            DiffusionLoaderBuilder::new(DiffusionSpecificConfig { use_flash_attn }, Some(model_id))
                .build(arch)
        }
    };
    Ok(loader)
}
