//! Server command implementation

use anyhow::{Context, Result};
use tracing::info;

use mistralrs_core::{
    initialize_logging, DiffusionLoaderType, ModelSelected, PagedCacheType, SpeechLoaderType,
};
use mistralrs_server_core::{
    mistralrs_for_server_builder::MistralRsForServerBuilder,
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

use crate::args::{
    AdapterOptions, DeviceOptions, FormatOptions, GlobalOptions, ModelFormat, ModelSourceOptions,
    ModelType, QuantizationOptions, RuntimeOptions, ServerOptions,
};
use crate::ui::build_ui_router;

/// Run the HTTP server with the specified model
pub async fn run_server(
    model_type: ModelType,
    server: ServerOptions,
    runtime: RuntimeOptions,
    global: GlobalOptions,
) -> Result<()> {
    initialize_logging();

    // Convert our clean args to ModelSelected for the existing loader infrastructure
    let model_selected = convert_to_model_selected(&model_type)?;

    // Extract paged attention settings
    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = extract_paged_attn_settings(&model_type);

    // Extract device settings
    let (cpu, device_layers) = extract_device_settings(&model_type);

    // Extract quantization settings
    let isq = extract_isq_setting(&model_type);

    // Build the MistralRs instance
    let mut builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(runtime.max_seqs)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(false)
        .with_prefix_cache_n(runtime.prefix_cache_n)
        .set_paged_attn(paged_attn)
        .with_cpu(cpu)
        .with_enable_search(runtime.enable_search)
        .with_seed_optional(global.seed)
        .with_log_optional(global.log.as_ref().map(|p| p.to_string_lossy().to_string()))
        .with_chat_template_optional(
            runtime
                .chat_template
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_jinja_explicit_optional(
            runtime
                .jinja_explicit
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
        )
        .with_num_device_layers_optional(device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type);

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mistralrs = builder.build().await?;
    let mistralrs_for_ui = mistralrs.clone();

    // Build and run the server
    let mut app = MistralRsServerRouterBuilder::new()
        .with_mistralrs(mistralrs)
        .build()
        .await?;

    if server.ui {
        let ui_router = build_ui_router(
            mistralrs_for_ui,
            runtime.enable_search,
            runtime.search_embedding_model.map(|m| m.into()),
        )
        .await?;
        app = app.nest("/ui", ui_router);
        info!("UI available at http://{}:{}/ui", server.host, server.port);
    }

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", server.host, server.port)).await?;

    info!("Server listening on http://{}:{}", server.host, server.port);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Convert our clean ModelType to the legacy ModelSelected enum
pub(crate) fn convert_to_model_selected(model_type: &ModelType) -> Result<ModelSelected> {
    match model_type {
        ModelType::Auto {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
            vision,
        } => {
            // If user explicitly specified a quantized format, handle it
            let format_type = format.format.unwrap_or(ModelFormat::Plain);
            let has_lora = adapter.lora.is_some();
            let has_xlora = adapter.xlora.is_some();

            // For GGUF/GGML formats, delegate to text model conversion which has proper validation
            match format_type {
                ModelFormat::Gguf | ModelFormat::Ggml => {
                    // Validate that required options are present
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
                    // Use the text model conversion which handles GGUF/GGML properly
                    return convert_text_model(model, format, adapter, quantization, device);
                }
                ModelFormat::Plain => {
                    // For plain format with adapters, also use text model conversion
                    if has_lora || has_xlora {
                        return convert_text_model(model, format, adapter, quantization, device);
                    }
                }
            }

            // Use Run (auto-loader) for auto mode without explicit quantized format
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
                max_edge: vision.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: vision.max_num_images,
                max_image_length: vision.max_image_length,
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

        ModelType::Vision {
            model,
            format: _,
            adapter: _,
            quantization,
            device,
            cache: _,
            vision,
        } => Ok(ModelSelected::VisionPlain {
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
            max_edge: vision.max_edge,
            calibration_file: quantization.calibration_file.clone(),
            imatrix: quantization.imatrix.clone(),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: vision.max_num_images.unwrap_or(1),
            max_image_length: vision.max_image_length.unwrap_or(1024),
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

/// Convert text model with orthogonal format/adapter flags
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
        // Plain format
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

        // GGUF format - quantized_filename is required String
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

        // GGML format
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
        ModelType::Vision { cache, .. } => cache,
        ModelType::Embedding { cache, .. } => cache,
        _ => return (None, None, None, None, None, PagedCacheType::Auto),
    };

    cache.paged_attn.clone().into_builder_flags()
}

pub(crate) fn extract_device_settings(model_type: &ModelType) -> (bool, Option<Vec<String>>) {
    let device = match model_type {
        ModelType::Auto { device, .. } => device,
        ModelType::Text { device, .. } => device,
        ModelType::Vision { device, .. } => device,
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
        ModelType::Vision { quantization, .. } => quantization.in_situ_quant.clone(),
        ModelType::Embedding { quantization, .. } => quantization.in_situ_quant.clone(),
        _ => None,
    }
}
