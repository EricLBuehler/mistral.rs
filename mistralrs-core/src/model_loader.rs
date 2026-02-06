use std::{
    fs::{self, File},
    path::PathBuf,
    str::FromStr,
};

use mistralrs_quant::MULTI_LORA_DELIMITER;

use crate::{
    get_toml_selected_model_dtype,
    pipeline::{
        AutoLoaderBuilder, DiffusionLoaderBuilder, GGMLLoaderBuilder, GGMLSpecificConfig,
        GGUFLoaderBuilder, GGUFSpecificConfig, NormalLoaderBuilder, NormalSpecificConfig,
        VisionLoaderBuilder, VisionSpecificConfig,
    },
    toml_selector::get_toml_selected_model_device_map_params,
    AutoDeviceMapParams, EmbeddingLoaderBuilder, EmbeddingSpecificConfig, Loader, ModelDType,
    ModelSelected, SpeechLoader, TomlLoaderArgs, TomlSelector, Topology, GGUF_MULTI_FILE_DELIMITER,
    UQFF_MULTI_FILE_DELIMITER,
};

/// A builder for a loader using the selected model.
pub struct LoaderBuilder {
    model: ModelSelected,
    no_kv_cache: bool,
    chat_template: Option<String>,
    jinja_explicit: Option<String>,
}

impl LoaderBuilder {
    pub fn new(model: ModelSelected) -> Self {
        Self {
            model,
            no_kv_cache: false,
            chat_template: None,
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

    pub fn build(self) -> anyhow::Result<Box<dyn Loader>> {
        loader_from_model_selected(self)
    }
}

pub fn get_tgt_non_granular_index(model: &ModelSelected) -> Option<usize> {
    match model {
        ModelSelected::Plain { .. }
        | ModelSelected::Run { .. }
        | ModelSelected::Lora { .. }
        | ModelSelected::GGUF { .. }
        | ModelSelected::LoraGGUF { .. }
        | ModelSelected::GGML { .. }
        | ModelSelected::LoraGGML { .. }
        | ModelSelected::Toml { .. }
        | ModelSelected::VisionPlain { .. }
        | ModelSelected::DiffusionPlain { .. }
        | ModelSelected::Speech { .. }
        | ModelSelected::Embedding { .. } => None,
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
        ModelSelected::MultiModel { .. } => {
            panic!("MultiModel variant should not be used in model loading functions")
        }
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
        | ModelSelected::LoraGGML { dtype, .. }
        | ModelSelected::Run { dtype, .. }
        | ModelSelected::Speech { dtype, .. }
        | ModelSelected::Embedding { dtype, .. } => Ok(*dtype),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            Ok(get_toml_selected_model_dtype(&selector))
        }
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
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
        ModelSelected::Run {
            max_seq_len,
            max_batch_size,
            max_image_length,
            max_num_images,
            ..
        } => {
            if max_num_images.is_some() || max_image_length.is_some() {
                let max_image_length =
                    max_image_length.unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_IMAGE_LENGTH);
                Ok(AutoDeviceMapParams::Vision {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                    max_image_shape: (max_image_length, max_image_length),
                    max_num_images: max_num_images
                        .unwrap_or(AutoDeviceMapParams::DEFAULT_MAX_NUM_IMAGES),
                })
            } else {
                Ok(AutoDeviceMapParams::Text {
                    max_seq_len: *max_seq_len,
                    max_batch_size: *max_batch_size,
                })
            }
        }
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
        ModelSelected::DiffusionPlain { .. }
        | ModelSelected::Speech { .. }
        | ModelSelected::Embedding { .. } => Ok(AutoDeviceMapParams::default_text()),
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            get_toml_selected_model_device_map_params(&selector)
        }
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    }
}

fn loader_from_model_selected(args: LoaderBuilder) -> anyhow::Result<Box<dyn Loader>> {
    let loader: Box<dyn Loader> = match args.model {
        ModelSelected::Toml { file } => {
            let selector: TomlSelector = toml::from_str(
                &fs::read_to_string(file.clone())
                    .unwrap_or_else(|_| panic!("Could not load toml selector file at {file}")),
            )?;
            let args = TomlLoaderArgs {
                chat_template: args.chat_template,
                no_kv_cache: args.no_kv_cache,
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
            matformer_config_path,
            matformer_slice_name,
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
                matformer_config_path,
                matformer_slice_name,
            },
            args.chat_template,
            tokenizer_json,
            Some(model_id),
            args.no_kv_cache,
            args.jinja_explicit,
        )
        .build(arch)?,
        ModelSelected::Run {
            model_id,
            tokenizer_json,
            dtype: _,
            topology,
            organization,
            write_uqff,
            from_uqff,
            imatrix,
            calibration_file,
            max_edge,
            max_seq_len: _,
            max_batch_size: _,
            max_num_images: _,
            max_image_length: _,
            hf_cache_path,
            matformer_config_path,
            matformer_slice_name,
        } => {
            let builder = AutoLoaderBuilder::new(
                NormalSpecificConfig {
                    topology: Topology::from_option_path(topology.clone())?,
                    organization: organization.unwrap_or_default(),
                    write_uqff: write_uqff.clone(),
                    from_uqff: from_uqff.clone().map(|x| {
                        x.split(UQFF_MULTI_FILE_DELIMITER)
                            .map(PathBuf::from_str)
                            .map(|x| x.unwrap())
                            .collect::<Vec<_>>()
                    }),
                    imatrix: imatrix.clone(),
                    calibration_file: calibration_file.clone(),
                    hf_cache_path: hf_cache_path.clone(),
                    matformer_config_path: matformer_config_path.clone(),
                    matformer_slice_name: matformer_slice_name.clone(),
                },
                VisionSpecificConfig {
                    topology: Topology::from_option_path(topology.clone())?,
                    write_uqff: write_uqff.clone(),
                    from_uqff: from_uqff.clone().map(|x| {
                        x.split(UQFF_MULTI_FILE_DELIMITER)
                            .map(PathBuf::from_str)
                            .map(|x| x.unwrap())
                            .collect::<Vec<_>>()
                    }),
                    max_edge,
                    calibration_file,
                    imatrix,
                    hf_cache_path: hf_cache_path.clone(),
                    matformer_config_path,
                    matformer_slice_name,
                    organization: organization.unwrap_or_default(),
                },
                EmbeddingSpecificConfig {
                    topology: Topology::from_option_path(topology)?,
                    write_uqff,
                    from_uqff: from_uqff.map(|x| {
                        x.split(UQFF_MULTI_FILE_DELIMITER)
                            .map(PathBuf::from_str)
                            .map(|x| x.unwrap())
                            .collect::<Vec<_>>()
                    }),
                    hf_cache_path: hf_cache_path.clone(),
                },
                args.chat_template,
                tokenizer_json,
                model_id,
                args.no_kv_cache,
                args.jinja_explicit,
            );
            let builder = if let Some(ref path) = hf_cache_path {
                builder.hf_cache_path(path.clone())
            } else {
                builder
            };
            builder.build()
        }
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
            matformer_config_path,
            matformer_slice_name,
            organization,
        } => VisionLoaderBuilder::new(
            VisionSpecificConfig {
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
                matformer_config_path,
                matformer_slice_name,
                organization: organization.unwrap_or_default(),
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
        } => DiffusionLoaderBuilder::new(Some(model_id)).build(arch),
        ModelSelected::Speech {
            model_id,
            dac_model_id,
            arch,
            ..
        } => Box::new(SpeechLoader {
            model_id,
            dac_model_id,
            arch,
            cfg: None,
        }),
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
        ModelSelected::Embedding {
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
                hf_cache_path,
            },
            tokenizer_json,
            Some(model_id),
        )
        .build(arch),
        ModelSelected::MultiModel { .. } => {
            anyhow::bail!("MultiModel variant should not be used in model loading functions")
        }
    };
    Ok(loader)
}
