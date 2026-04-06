//! Server command implementation — converts CLI args and delegates to mistralrs-serve.

use anyhow::{Context, Result};
use mistralrs_core::{DiffusionLoaderType, ModelSelected, PagedCacheType, SpeechLoaderType};

use crate::args::{
    AdapterOptions, DeviceOptions, FormatOptions, GlobalOptions, ModelFormat, ModelSourceOptions,
    ModelType, QuantizationOptions, RuntimeOptions, ServerOptions,
};

/// Run the HTTP server with the specified model
pub async fn run_server(
    model_type: ModelType,
    server: ServerOptions,
    runtime: RuntimeOptions,
    global: GlobalOptions,
) -> Result<()> {
    let model_selected = convert_to_model_selected(&model_type)?;

    let paged_attn_tuple = extract_paged_attn_settings(&model_type);
    let paged_attn = mistralrs_serve::PagedAttnConfig {
        paged_attn: paged_attn_tuple.0,
        paged_attn_gpu_mem: paged_attn_tuple.1,
        paged_attn_gpu_mem_usage: paged_attn_tuple.2,
        paged_ctxt_len: paged_attn_tuple.3,
        paged_attn_block_size: paged_attn_tuple.4,
        paged_cache_type: paged_attn_tuple.5,
    };

    let device_tuple = extract_device_settings(&model_type);
    let device = mistralrs_serve::DeviceConfig {
        cpu: device_tuple.0,
        device_layers: device_tuple.1,
    };

    let isq = extract_isq_setting(&model_type);

    mistralrs_serve::run_server(
        model_selected,
        paged_attn,
        device,
        isq,
        mistralrs_serve::ServerConfig {
            host: server.host,
            port: server.port,
            ui: server.ui,
            idle_timeout_secs: server.idle_timeout_secs,
        },
        mistralrs_serve::RuntimeConfig {
            max_seqs: runtime.max_seqs,
            no_kv_cache: runtime.no_kv_cache,
            prefix_cache_n: runtime.prefix_cache_n,
            enable_search: runtime.enable_search,
            search_embedding_model: runtime.search_embedding_model.map(|m| m.into()),
            chat_template: runtime.chat_template,
            jinja_explicit: runtime.jinja_explicit,
        },
        mistralrs_serve::GlobalConfig {
            token_source: global.token_source,
            seed: global.seed,
            log: global.log,
        },
    )
    .await
}

/// Convert our clean ModelType to the ModelSelected enum
pub(crate) fn convert_to_model_selected(model_type: &ModelType) -> Result<ModelSelected> {
    match model_type {
        ModelType::Auto {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
            multimodal,
        } => {
            let format_type = format.format.unwrap_or(ModelFormat::Plain);
            let has_lora = adapter.lora.is_some();
            let has_xlora = adapter.xlora.is_some();

            match format_type {
                ModelFormat::Gguf | ModelFormat::Ggml => {
                    if format.quantized_file.is_none() {
                        let format_name = match format_type {
                            ModelFormat::Gguf => "GGUF",
                            ModelFormat::Ggml => "GGML",
                            _ => unreachable!(),
                        };
                        anyhow::bail!(
                            "{format_name} format requires a quantized file.\n\n\
                            Usage: mistralrs run auto -m <hf model/local dir> --format {fmt} -f <filename>\n\n\
                            The -f/--quantized-file flag specifies which {format_name} file to load from the model repository/local dir.",
                            fmt = format_name.to_lowercase()
                        );
                    }
                    return convert_text_model(model, format, adapter, quantization, device);
                }
                ModelFormat::Plain => {
                    if has_lora || has_xlora {
                        return convert_text_model(model, format, adapter, quantization, device);
                    }
                }
            }

            Ok(ModelSelected::Run {
                model_id: model.model_id.clone(),
                tokenizer_json: model
                    .tokenizer
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                dtype: model.dtype,
                topology: device
                    .topology
                    .as_ref()
                    .map(|p| p.to_string_lossy().to_string()),
                organization: quantization.isq_organization,
                write_uqff: None,
                from_uqff: quantization.from_uqff.clone(),
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                max_edge: multimodal.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: multimodal.max_num_images,
                max_image_length: multimodal.max_image_length,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: None,
                matformer_slice_name: None,
            })
        }

        ModelType::Text {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
        } => convert_text_model(model, format, adapter, quantization, device),

        ModelType::Multimodal {
            model,
            format: _,
            adapter: _,
            quantization,
            device,
            cache: _,
            multimodal,
        } => Ok(ModelSelected::MultimodalPlain {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            arch: None,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            max_edge: multimodal.max_edge,
            calibration_file: quantization.calibration_file.clone(),
            imatrix: quantization.imatrix.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: multimodal.max_num_images.unwrap_or(1),
            max_image_length: multimodal.max_image_length.unwrap_or(1024),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: None,
            matformer_slice_name: None,
            organization: quantization.isq_organization,
        }),

        ModelType::Diffusion { model, device: _ } => Ok(ModelSelected::DiffusionPlain {
            model_id: model.model_id.clone(),
            arch: DiffusionLoaderType::Flux,
            dtype: model.dtype,
        }),

        ModelType::Speech { model, device: _ } => Ok(ModelSelected::Speech {
            model_id: model.model_id.clone(),
            dac_model_id: None,
            arch: SpeechLoaderType::Dia,
            dtype: model.dtype,
        }),

        ModelType::Embedding {
            model,
            format: _,
            quantization,
            device,
            cache: _,
        } => Ok(ModelSelected::Embedding {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            arch: None,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            hf_cache_path: device.hf_cache.clone(),
        }),
    }
}

fn convert_text_model(
    model: &ModelSourceOptions,
    format_opts: &FormatOptions,
    adapter: &AdapterOptions,
    quantization: &QuantizationOptions,
    device: &DeviceOptions,
) -> Result<ModelSelected> {
    let format_type = format_opts.format.unwrap_or(ModelFormat::Plain);
    let has_lora = adapter.lora.is_some();
    let has_xlora = adapter.xlora.is_some();

    match (format_type, has_lora, has_xlora) {
        (ModelFormat::Plain, false, false) => Ok(ModelSelected::Plain {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: None,
            matformer_slice_name: None,
        }),

        (ModelFormat::Plain, true, false) => Ok(ModelSelected::Lora {
            model_id: Some(model.model_id.clone()),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            adapter_model_id: adapter.lora.clone().unwrap_or_default(),
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
        }),

        (ModelFormat::Plain, false, true) => Ok(ModelSelected::XLora {
            model_id: Some(model.model_id.clone()),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            arch: model.arch.clone(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            write_uqff: None,
            from_uqff: quantization.from_uqff.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            hf_cache_path: device.hf_cache.clone(),
        }),

        (ModelFormat::Gguf, false, false) => Ok(ModelSelected::GGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `--quantized-filename`/`-f` to be specified")?,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Gguf, true, false) => Ok(ModelSelected::LoraGGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `quantized-filename` to be specified")?,
            adapters_model_id: adapter.lora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Gguf, false, true) => Ok(ModelSelected::XLoraGGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `quantized-filename` to be specified")?,
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, false, false) => Ok(ModelSelected::GGML {
            tok_model_id: format_opts
                .tok_model_id
                .clone()
                .unwrap_or_else(|| model.model_id.clone()),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `quantized-filename` to be specified")?,
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, true, false) => Ok(ModelSelected::LoraGGML {
            tok_model_id: Some(
                format_opts
                    .tok_model_id
                    .clone()
                    .unwrap_or_else(|| model.model_id.clone()),
            ),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `quantized-filename` to be specified")?,
            adapters_model_id: adapter.lora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, false, true) => Ok(ModelSelected::XLoraGGML {
            tok_model_id: Some(
                format_opts
                    .tok_model_id
                    .clone()
                    .unwrap_or_else(|| model.model_id.clone()),
            ),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `quantized-filename` to be specified")?,
            xlora_model_id: adapter.xlora.clone().unwrap_or_default(),
            order: adapter
                .xlora_order
                .as_ref()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default(),
            tgt_non_granular_index: adapter.tgt_non_granular_index,
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        _ => anyhow::bail!("Cannot use both --lora and --xlora simultaneously"),
    }
}

pub(crate) fn extract_paged_attn_settings(
    model_type: &ModelType,
) -> crate::args::PagedAttnBuilderFlags {
    let cache = match model_type {
        ModelType::Auto { cache, .. } => cache,
        ModelType::Text { cache, .. } => cache,
        ModelType::Multimodal { cache, .. } => cache,
        ModelType::Embedding { cache, .. } => cache,
        _ => return (None, None, None, None, None, PagedCacheType::Auto),
    };

    cache.paged_attn.clone().into_builder_flags()
}

pub(crate) fn extract_device_settings(model_type: &ModelType) -> (bool, Option<Vec<String>>) {
    let device = match model_type {
        ModelType::Auto { device, .. } => device,
        ModelType::Text { device, .. } => device,
        ModelType::Multimodal { device, .. } => device,
        ModelType::Diffusion { device, .. } => device,
        ModelType::Speech { device, .. } => device,
        ModelType::Embedding { device, .. } => device,
    };

    (device.cpu, device.device_layers.clone())
}

pub(crate) fn extract_isq_setting(model_type: &ModelType) -> Option<String> {
    match model_type {
        ModelType::Auto { quantization, .. } => quantization.in_situ_quant.clone(),
        ModelType::Text { quantization, .. } => quantization.in_situ_quant.clone(),
        ModelType::Multimodal { quantization, .. } => quantization.in_situ_quant.clone(),
        ModelType::Embedding { quantization, .. } => quantization.in_situ_quant.clone(),
        _ => None,
    }
}
