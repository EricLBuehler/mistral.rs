//! Server command implementation

use anyhow::{Context, Result};
use std::path::Path;
use tracing::info;

use mistralrs_core::{
    initialize_logging, DiffusionLoaderType, McpClientConfig, ModelSelected, PagedCacheType,
    SpeechLoaderType,
};
use mistralrs_server_core::{
    approvals::ApprovalBroker, mistralrs_for_server_builder::MistralRsForServerBuilder,
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

use crate::args::{
    AdapterOptions, AgentCliOptions, CodeExecPermissionArg, DeviceOptions, FormatOptions,
    GlobalOptions, MatformerSelection, ModelFormat, ModelSourceOptions, ModelType,
    QuantizationOptions, RuntimeOptions, SandboxMode, SandboxOptions, ServerOptions,
};
use crate::ui::build_ui_router;

/// Run the HTTP server with the specified model
#[allow(clippy::too_many_arguments)]
pub async fn run_server(
    mut model_type: ModelType,
    server: ServerOptions,
    mut runtime: RuntimeOptions,
    agent_options: AgentCliOptions,
    sandbox: SandboxOptions,
    global: GlobalOptions,
) -> Result<()> {
    initialize_logging();

    agent_options.apply_to(&mut runtime);
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, server.max_tool_rounds);

    // Convert our clean args to ModelSelected for the existing loader infrastructure
    let matformer = runtime.matformer_selection();
    let original_model_id = model_id_of(&model_type).to_string();
    apply_quant_resolution(&mut model_type, &global.token_source, &matformer).await?;
    let api_id_override =
        (model_id_of(&model_type) != original_model_id).then_some(original_model_id);
    let model_selected = convert_to_model_selected(&model_type, &matformer)?;

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
        .with_model_id_override_optional(api_id_override)
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_mtp_config_optional(runtime.mtp_config())
        .with_paged_attn_cache_type(paged_cache_type);

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mcp_client_config = load_mcp_config(runtime.mcp_config.as_deref())?;
    builder = builder.with_mcp_config_optional(mcp_client_config);

    let sandbox_policy = extract_sandbox_settings(sandbox);

    let approval_broker = ApprovalBroker::default();

    #[cfg(feature = "code-execution")]
    {
        let config = build_code_exec_config(&runtime, sandbox_policy);
        builder = builder.with_code_exec_config_optional(config);
    }
    #[cfg(not(feature = "code-execution"))]
    let _ = sandbox_policy;

    let mistralrs = builder.build().await?;
    let mistralrs_for_ui = mistralrs.clone();

    // Build and run the server
    let mut app = MistralRsServerRouterBuilder::new()
        .with_mistralrs(mistralrs)
        .with_max_tool_rounds_optional(server.max_tool_rounds)
        .with_tool_dispatch_url_optional(server.tool_dispatch_url.clone())
        .with_agent_permission(runtime.code_exec_permission.into())
        .with_approval_broker(approval_broker.clone())
        .build()
        .await?;

    if !server.no_ui {
        let enable_code_execution = {
            #[cfg(feature = "code-execution")]
            {
                runtime.enable_code_execution
            }
            #[cfg(not(feature = "code-execution"))]
            {
                false
            }
        };
        let ui_router = build_ui_router(
            mistralrs_for_ui,
            runtime.enable_search,
            runtime.search_embedding_model.map(|m| m.into()),
            enable_code_execution,
            server.tool_dispatch_url.clone(),
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
pub(crate) fn convert_to_model_selected(
    model_type: &ModelType,
    matformer: &MatformerSelection,
) -> Result<ModelSelected> {
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
                    return convert_text_model(
                        model,
                        format,
                        adapter,
                        quantization,
                        device,
                        matformer,
                    );
                }
                ModelFormat::Plain => {
                    // For plain format with adapters, also use text model conversion
                    if has_lora || has_xlora {
                        return convert_text_model(
                            model,
                            format,
                            adapter,
                            quantization,
                            device,
                            matformer,
                        );
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
                max_edge: multimodal.max_edge,
                max_seq_len: device.max_seq_len,
                max_batch_size: device.max_batch_size,
                max_num_images: multimodal.max_num_images,
                max_image_length: multimodal.max_image_length,
                hf_cache_path: device.hf_cache.clone(),
                matformer_config_path: matformer.config_path.clone(),
                matformer_slice_name: matformer.slice_name.clone(),
            })
        }

        ModelType::Text {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
        } => convert_text_model(model, format, adapter, quantization, device, matformer),

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
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
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
    matformer: &MatformerSelection,
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
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
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
    extract_quantization(model_type).and_then(|q| q.in_situ_quant.clone())
}

pub(crate) fn extract_quant_flag(model_type: &ModelType) -> Option<String> {
    extract_quantization(model_type).and_then(|q| q.quant.clone())
}

fn extract_quantization(model_type: &ModelType) -> Option<&crate::args::QuantizationOptions> {
    match model_type {
        ModelType::Auto { quantization, .. } => Some(quantization),
        ModelType::Text { quantization, .. } => Some(quantization),
        ModelType::Multimodal { quantization, .. } => Some(quantization),
        ModelType::Embedding { quantization, .. } => Some(quantization),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } => None,
    }
}

pub(crate) fn model_quantization_mut(
    model_type: &mut ModelType,
) -> Option<&mut crate::args::QuantizationOptions> {
    match model_type {
        ModelType::Auto { quantization, .. } => Some(quantization),
        ModelType::Text { quantization, .. } => Some(quantization),
        ModelType::Multimodal { quantization, .. } => Some(quantization),
        ModelType::Embedding { quantization, .. } => Some(quantization),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } => None,
    }
}

pub(crate) fn model_id_mut(model_type: &mut ModelType) -> &mut String {
    match model_type {
        ModelType::Auto { model, .. } => &mut model.model_id,
        ModelType::Text { model, .. } => &mut model.model_id,
        ModelType::Multimodal { model, .. } => &mut model.model_id,
        ModelType::Diffusion { model, .. } => &mut model.model_id,
        ModelType::Speech { model, .. } => &mut model.model_id,
        ModelType::Embedding { model, .. } => &mut model.model_id,
    }
}

pub(crate) fn model_id_of(model_type: &ModelType) -> &str {
    match model_type {
        ModelType::Auto { model, .. } => &model.model_id,
        ModelType::Text { model, .. } => &model.model_id,
        ModelType::Multimodal { model, .. } => &model.model_id,
        ModelType::Diffusion { model, .. } => &model.model_id,
        ModelType::Speech { model, .. } => &model.model_id,
        ModelType::Embedding { model, .. } => &model.model_id,
    }
}

pub(crate) async fn apply_quant_resolution(
    model_type: &mut ModelType,
    token_source: &mistralrs_core::TokenSource,
    matformer: &MatformerSelection,
) -> Result<()> {
    let raw = match model_quantization_mut(model_type).and_then(|q| q.quant.clone()) {
        Some(r) => r,
        None => return Ok(()),
    };

    let model_id = model_id_mut(model_type).clone();
    let force_cpu = extract_device_settings(model_type).0;
    let model_selected = convert_to_model_selected(model_type, matformer)?;

    let resolved = crate::commands::quant::resolve_quant(
        &raw,
        &model_id,
        token_source,
        &model_selected,
        force_cpu,
    )
    .await?;

    if let Some(swap) = resolved.model_id_swap {
        *model_id_mut(model_type) = swap;
    }
    if let Some(q) = model_quantization_mut(model_type) {
        q.quant = None;
        q.in_situ_quant = resolved.in_situ_quant;
        q.from_uqff = resolved.from_uqff;
    }
    Ok(())
}

/// Load an MCP client config from `--mcp-config` (or `MCP_CONFIG_PATH` if no path given).
pub(crate) fn load_mcp_config(path: Option<&Path>) -> Result<Option<McpClientConfig>> {
    let resolved = match path {
        Some(p) => Some(p.to_path_buf()),
        None => std::env::var("MCP_CONFIG_PATH").ok().map(Into::into),
    };
    let Some(path) = resolved else {
        return Ok(None);
    };
    let contents = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read MCP config {}", path.display()))?;
    let config: McpClientConfig = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse MCP config {}", path.display()))?;
    info!(
        "Loaded MCP configuration from {} ({} servers)",
        path.display(),
        config.servers.len()
    );
    Ok(Some(config))
}

/// Build a `CodeExecutionConfig` from runtime options. Returns `None` when code execution is off.
#[cfg(feature = "code-execution")]
pub(crate) fn build_code_exec_config(
    runtime: &RuntimeOptions,
    sandbox_policy: Option<mistralrs_sandbox::SandboxPolicy>,
) -> Option<mistralrs_core::CodeExecutionConfig> {
    if !runtime.enable_code_execution {
        return None;
    }
    let mut config = mistralrs_core::CodeExecutionConfig::default();
    if let Some(python) = runtime.code_exec_python.clone() {
        config.python_path = python;
    }
    if let Some(timeout) = runtime.code_exec_timeout {
        config.timeout_secs = timeout;
    }
    config.working_directory = runtime.code_exec_workdir.clone();
    config.sandbox_policy = sandbox_policy;
    Some(config)
}

pub(crate) fn extract_sandbox_settings(
    sandbox: SandboxOptions,
) -> Option<mistralrs_sandbox::SandboxPolicy> {
    let mode = match (
        sandbox.mode,
        std::env::var(mistralrs_sandbox::SANDBOX_ENV_VAR).ok(),
    ) {
        (SandboxMode::Auto, Some(v)) => match v.to_ascii_lowercase().as_str() {
            "auto" => SandboxMode::Auto,
            "on" => SandboxMode::On,
            "off" => SandboxMode::Off,
            other => {
                tracing::warn!(
                    "ignoring invalid {}={other} (expected auto/on/off)",
                    mistralrs_sandbox::SANDBOX_ENV_VAR
                );
                SandboxMode::Auto
            }
        },
        (mode, _) => mode,
    };

    match mode {
        SandboxMode::Off => None,
        SandboxMode::Auto | SandboxMode::On => {
            let mut policy = mistralrs_sandbox::SandboxPolicy::default();
            if let Some(v) = sandbox.max_memory_mb {
                policy.max_memory_mb = v;
            }
            if let Some(v) = sandbox.max_cpu_secs {
                policy.max_cpu_secs = v;
            }
            if let Some(v) = sandbox.max_procs {
                policy.max_procs = v;
            }
            policy.network = sandbox.network.into();
            policy.strict = matches!(mode, SandboxMode::On);
            Some(policy)
        }
    }
}

pub(crate) fn apply_agent_mode(runtime: &mut RuntimeOptions) {
    if !runtime.agent {
        return;
    }
    runtime.enable_search = true;
    #[cfg(feature = "code-execution")]
    {
        runtime.enable_code_execution = true;
    }
}

pub(crate) fn validate_agent_options(runtime: &RuntimeOptions) -> Result<()> {
    if runtime.search_embedding_model.is_some() && !runtime.enable_search {
        anyhow::bail!(
            "`--search-embedding-model` requires `--enable-search` (or `--agent`/`--agentic`)"
        );
    }
    #[cfg(feature = "code-execution")]
    {
        let touches_code_exec = runtime.code_exec_python.is_some()
            || runtime.code_exec_timeout.is_some()
            || runtime.code_exec_workdir.is_some();
        if touches_code_exec && !runtime.enable_code_execution {
            anyhow::bail!(
                "`--code-exec-*` options require `--enable-code-execution` (or `--agent`/`--agentic`)"
            );
        }
    }
    Ok(())
}

pub(crate) fn log_agent_runtime(runtime: &RuntimeOptions, max_tool_rounds: Option<usize>) {
    if !runtime.agent && !runtime.enable_search && !is_code_execution_enabled(runtime) {
        return;
    }

    let rounds = max_tool_rounds.unwrap_or(mistralrs_core::DEFAULT_MAX_TOOL_ROUNDS);
    let mode = if runtime.agent { "agent" } else { "tools" };
    tracing::info!(
        "{mode}: search {}, code execution {}, approvals {}, max tool rounds {rounds}",
        search_summary(runtime),
        code_execution_summary(runtime),
        agent_permission_summary(runtime.code_exec_permission)
    );
    log_agent_runtime_details(runtime);
}

fn search_summary(runtime: &RuntimeOptions) -> String {
    if !runtime.enable_search {
        return "off".to_string();
    }
    let model = runtime
        .search_embedding_model
        .map(mistralrs_core::SearchEmbeddingModel::from)
        .unwrap_or_default();
    format!("on (reranker {model})")
}

fn agent_permission_summary(permission: CodeExecPermissionArg) -> &'static str {
    match permission {
        CodeExecPermissionArg::Auto => "auto",
        CodeExecPermissionArg::Ask => "ask",
        CodeExecPermissionArg::Deny => "deny",
    }
}

#[cfg(feature = "code-execution")]
fn is_code_execution_enabled(runtime: &RuntimeOptions) -> bool {
    runtime.enable_code_execution
}
#[cfg(not(feature = "code-execution"))]
fn is_code_execution_enabled(_runtime: &RuntimeOptions) -> bool {
    false
}

#[cfg(feature = "code-execution")]
fn code_execution_summary(runtime: &RuntimeOptions) -> &'static str {
    if !runtime.enable_code_execution {
        "off"
    } else {
        "on"
    }
}

#[cfg(feature = "code-execution")]
fn log_agent_runtime_details(runtime: &RuntimeOptions) {
    if !runtime.enable_code_execution {
        return;
    }
    let python = runtime
        .code_exec_python
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "python3 (default)".to_string());
    let timeout = runtime
        .code_exec_timeout
        .map_or_else(|| "30s (default)".to_string(), |t| format!("{t}s"));
    let workdir = runtime
        .code_exec_workdir
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| "per-session temp dir".to_string());
    tracing::info!(
        "code-exec: python={python}, timeout={timeout}, workdir={workdir}, permission={}",
        agent_permission_summary(runtime.code_exec_permission)
    );
}
#[cfg(not(feature = "code-execution"))]
fn code_execution_summary(runtime: &RuntimeOptions) -> &'static str {
    if runtime.agent {
        "not compiled in"
    } else {
        "off"
    }
}

#[cfg(not(feature = "code-execution"))]
fn log_agent_runtime_details(runtime: &RuntimeOptions) {
    if runtime.agent {
        tracing::warn!(
            "code-exec: not compiled in (build with `--features code-execution`); --agent enabled search only"
        );
    }
}
