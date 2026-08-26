//! Server command implementation

use anyhow::{Context, Result};
use axum::middleware;
use std::path::Path;
use tracing::{debug, info, warn};

use mistralrs_core::{
    initialize_logging, DiffusionLoaderType, McpClientConfig, ModelSelected, PagedCacheType,
    SpeechLoaderType,
};
use mistralrs_server_core::{
    approvals::ApprovalBroker,
    lora_adapters::runtime_lora_updates_enabled,
    mcp_server::{create_mcp_router, MCP_PROTOCOL_VERSION, MCP_ROUTE},
    metrics::{install_prometheus_recorder, observe_http, ObservabilityState},
    mistralrs_for_server_builder::MistralRsForServerBuilder,
    mistralrs_server_router_builder::{MistralRsServerRouterBuilder, DEFAULT_MAX_BODY_LIMIT},
    route_registry::{RouteInfo, RouteKind, MISTRALRS_API_ROUTES, RUNTIME_LORA_API_ROUTES},
    types::SharedMistralRsState,
};

#[cfg(test)]
use crate::args::MultimodalAdapterOptions;
use crate::args::{
    AdapterOptions, AgentCliOptions, CodeExecPermissionArg, DeviceOptions, FormatOptions,
    GlobalOptions, MatformerSelection, ModelFormat, ModelSourceOptions, ModelType,
    MultimodalOptions, QuantizationOptions, RuntimeOptions, SandboxMode, SandboxOptions,
    ServerOptions,
};
use crate::ui::build_ui_router;

const MEBIBYTE_BYTES: usize = 1024 * 1024;

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
    if server.observability_config().metrics {
        install_prometheus_recorder();
    }

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
    let (max_model_len, hf_config_overrides) = extract_hf_config_settings(&model_type);

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
    let encoder_cache_memory_bytes = extract_encoder_cache_memory_bytes(&model_type)?;

    // Build the MistralRs instance
    let mut builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(runtime.max_seqs)
        .with_max_num_batched_tokens(runtime.max_num_batched_tokens)
        .with_max_prefill_chunk_tokens(runtime.max_prefill_chunk_tokens)
        .with_max_decode_steps_before_prefill(runtime.max_decode_steps_before_prefill)
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
        .with_max_model_len_optional(max_model_len)
        .with_hf_config_overrides_optional(hf_config_overrides)
        .with_paged_attn_cache_type(paged_cache_type);

    if let Some(max_bytes) = encoder_cache_memory_bytes {
        builder = builder.with_encoder_cache_memory_bytes(max_bytes);
    }

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mcp_client_config = load_mcp_config(runtime.mcp_config.as_deref())?;
    builder = builder.with_mcp_config_optional(mcp_client_config);

    let sandbox_policy = extract_sandbox_settings(sandbox, &runtime);

    let approval_broker = ApprovalBroker::default();

    #[cfg(feature = "code-execution")]
    {
        let config = build_code_exec_config(&runtime, sandbox_policy.clone());
        builder = builder.with_code_exec_config_optional(config);
        let shell_config = build_shell_config(&runtime, sandbox_policy);
        builder = builder.with_shell_config_optional(shell_config);
    }
    #[cfg(not(feature = "code-execution"))]
    let _ = sandbox_policy;

    let mistralrs = builder.build().await?;
    let mistralrs_for_ui = mistralrs.clone();
    let mistralrs_for_mcp = mistralrs.clone();

    // Build and run the server
    let mut app = MistralRsServerRouterBuilder::new()
        .with_mistralrs(mistralrs)
        .with_max_tool_rounds_optional(server.max_tool_rounds)
        .with_tool_dispatch_url_optional(server.tool_dispatch_url.clone())
        .with_observability_config(server.observability_config())
        .with_agent_permission(runtime.code_exec_permission.into())
        .with_approval_broker(approval_broker.clone())
        .with_skills_dir_optional({
            #[cfg(feature = "code-execution")]
            {
                runtime.skills_dir.clone()
            }
            #[cfg(not(feature = "code-execution"))]
            {
                None
            }
        })
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
        let enable_shell = {
            #[cfg(feature = "code-execution")]
            {
                runtime.enable_shell
            }
            #[cfg(not(feature = "code-execution"))]
            {
                false
            }
        };
        let ui_observability = ObservabilityState::with_max_body_bytes(
            server.observability_config(),
            mistralrs_for_ui.clone(),
            DEFAULT_MAX_BODY_LIMIT,
        );
        let ui_router = build_ui_router(
            mistralrs_for_ui,
            runtime.enable_search,
            runtime.search_embedding_model.map(|m| m.into()),
            enable_code_execution,
            enable_shell,
            server.tool_dispatch_url.clone(),
        )
        .await?
        .layer(middleware::from_fn_with_state(
            ui_observability,
            observe_http,
        ));
        app = app.nest("/ui", ui_router);
        info!("UI available at http://{}:{}/ui", server.host, server.port);
    }

    if let Some(mcp_port) = server.mcp_port {
        spawn_mcp_server(mistralrs_for_mcp, &server.host, mcp_port, server.port).await?;
    }

    let listener =
        tokio::net::TcpListener::bind(format!("{}:{}", server.host, server.port)).await?;
    let listener = tcp_nodelay_listener(listener);

    info!("Server listening on http://{}:{}", server.host, server.port);
    log_api_surfaces(&server.host, server.port);

    axum::serve(listener, app).await?;

    Ok(())
}

/// Bind and spawn the MCP server on its own port, alongside the main HTTP server.
pub(crate) async fn spawn_mcp_server(
    mistralrs: SharedMistralRsState,
    host: &str,
    mcp_port: u16,
    http_port: u16,
) -> Result<()> {
    if mcp_port == http_port {
        anyhow::bail!("--mcp-port must differ from the HTTP --port ({http_port})");
    }
    let listener = tokio::net::TcpListener::bind(format!("{host}:{mcp_port}"))
        .await
        .with_context(|| format!("Failed to bind MCP server to {host}:{mcp_port}"))?;
    let listener = tcp_nodelay_listener(listener);
    let router = create_mcp_router(mistralrs);

    info!("MCP server listening on http://{host}:{mcp_port}{MCP_ROUTE}");
    info!("MCP protocol version is {MCP_PROTOCOL_VERSION}");

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, router).await {
            tracing::error!("MCP server error: {e}");
        }
    });
    Ok(())
}

pub(crate) fn tcp_nodelay_listener(
    listener: tokio::net::TcpListener,
) -> impl axum::serve::Listener<Io = tokio::net::TcpStream, Addr = std::net::SocketAddr> {
    use axum::serve::ListenerExt;

    listener.tap_io(|stream| {
        if let Err(error) = stream.set_nodelay(true) {
            tracing::warn!("failed to set TCP_NODELAY on incoming connection: {error}");
        }
    })
}

pub(crate) fn log_api_surfaces(host: &str, port: u16) {
    let client_host = match host {
        "0.0.0.0" => "localhost",
        "::" => "[::1]",
        host => host,
    };
    let root = format!("http://{client_host}:{port}");

    info!("OpenAI-compatible API: {root}/v1");
    info!("Anthropic-compatible API: {root}");
    info!("Swagger UI docs: {root}/docs");

    debug!("Available OpenAI-compatible routes:");
    log_routes(MISTRALRS_API_ROUTES, RouteKind::OpenAi);
    debug!("Available Anthropic-compatible routes:");
    log_routes(MISTRALRS_API_ROUTES, RouteKind::Anthropic);
    debug!("Available additional mistral.rs routes:");
    log_routes(MISTRALRS_API_ROUTES, RouteKind::MistralRs);
    if runtime_lora_updates_enabled() {
        log_routes(RUNTIME_LORA_API_ROUTES, RouteKind::MistralRs);
    }
}

fn log_routes(routes: &[RouteInfo], kind: RouteKind) {
    for route in routes.iter().filter(|route| route.kind == kind) {
        log_route(route);
    }
}

fn log_route(route: &RouteInfo) {
    debug!("  Route: {}, Methods: {}", route.path, route.methods);
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
            validate_mmproj_format(format)?;
            adapter.validate().map_err(anyhow::Error::msg)?;
            let has_lora = adapter.dynamic_lora_enabled();
            let has_legacy_lora = adapter.legacy_lora.is_some();
            let has_xlora = adapter.xlora.is_some();

            // For GGUF/GGML formats, delegate to text model conversion which has proper validation
            match format_type {
                ModelFormat::Gguf | ModelFormat::Ggml => {
                    // Validate that required options are present
                    if format.quantized_file.is_none() {
                        match format_type {
                            ModelFormat::Gguf => anyhow::bail!(
                                "GGUF format requires a model file. Pass `-f <model.gguf>`, or use \
                                 `-m <GGUF-repo> --quant <level>` to select one automatically."
                            ),
                            ModelFormat::Ggml => anyhow::bail!(
                                "GGML format requires a model file. Pass \
                                 `-m <model-repo-or-directory> -f <model.ggml>`."
                            ),
                            ModelFormat::Plain => unreachable!(),
                        }
                    }
                    // Use the text model conversion which handles GGUF/GGML properly
                    return convert_text_model(
                        model,
                        format,
                        adapter,
                        quantization,
                        device,
                        matformer,
                        Some(multimodal),
                    );
                }
                ModelFormat::Plain => {
                    // For plain format with adapters, also use text model conversion
                    if has_lora || has_legacy_lora || has_xlora {
                        return convert_text_model(
                            model,
                            format,
                            adapter,
                            quantization,
                            device,
                            matformer,
                            Some(multimodal),
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
        } => convert_text_model(
            model,
            format,
            adapter,
            quantization,
            device,
            matformer,
            None,
        ),

        ModelType::Multimodal {
            model,
            format,
            adapter,
            quantization,
            device,
            cache: _,
            multimodal,
        } => {
            validate_mmproj_format(format)?;
            adapter.validate().map_err(anyhow::Error::msg)?;
            let adapter = adapter.as_adapter_options();
            let mut model = model.clone();
            model.arch = None;
            match format.format.unwrap_or(ModelFormat::Plain) {
                ModelFormat::Gguf => {
                    if format.mmproj.is_none() {
                        anyhow::bail!(
                            "No companion projector was found for this multimodal GGUF; pass \
                             `--mmproj <filename>` to select one explicitly"
                        );
                    }
                    convert_text_model(
                        &model,
                        format,
                        &adapter,
                        quantization,
                        device,
                        matformer,
                        Some(multimodal),
                    )
                }
                ModelFormat::Ggml => {
                    anyhow::bail!(
                        "GGML is not supported for multimodal models; use a GGUF model with \
                         `--mmproj`, or use plain safetensors"
                    )
                }
                ModelFormat::Plain if adapter.dynamic_lora_enabled() => convert_text_model(
                    &model,
                    format,
                    &adapter,
                    quantization,
                    device,
                    matformer,
                    Some(multimodal),
                ),
                ModelFormat::Plain => Ok(ModelSelected::MultimodalPlain {
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
            }
        }

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
            format,
            quantization,
            device,
            cache: _,
        } => {
            validate_mmproj_format(format)?;
            if !matches!(format.format, None | Some(ModelFormat::Plain)) {
                anyhow::bail!("Embedding models do not support GGUF or GGML format");
            }
            Ok(ModelSelected::Embedding {
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
                imatrix: quantization.imatrix.clone(),
                calibration_file: quantization.calibration_file.clone(),
                hf_cache_path: device.hf_cache.clone(),
            })
        }
    }
}

fn validate_mmproj_format(format_opts: &FormatOptions) -> Result<()> {
    if format_opts.mmproj.is_some() && !matches!(format_opts.format, Some(ModelFormat::Gguf)) {
        anyhow::bail!("`--mmproj` requires GGUF format");
    }
    Ok(())
}

/// Convert text model with orthogonal format/adapter flags
fn convert_text_model(
    model: &ModelSourceOptions,
    format_opts: &FormatOptions,
    adapter: &AdapterOptions,
    quantization: &QuantizationOptions,
    device: &DeviceOptions,
    matformer: &MatformerSelection,
    multimodal: Option<&MultimodalOptions>,
) -> Result<ModelSelected> {
    validate_mmproj_format(format_opts)?;
    adapter.validate().map_err(anyhow::Error::msg)?;
    let format_type = format_opts.format.unwrap_or(ModelFormat::Plain);
    let has_lora = adapter.dynamic_lora_enabled();
    let has_legacy_lora = adapter.legacy_lora.is_some();
    let has_xlora = adapter.xlora.is_some();
    if format_opts.mmproj.is_some() && (has_legacy_lora || has_xlora) {
        anyhow::bail!("Multimodal GGUF does not support legacy LoRA or X-LoRA adapters");
    }

    match (format_type, has_lora, has_legacy_lora, has_xlora) {
        // Plain format
        (ModelFormat::Plain, false, false, false) => Ok(ModelSelected::Plain {
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

        (ModelFormat::Plain, true, false, false) => Ok(ModelSelected::Lora {
            model_id: model.model_id.clone(),
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            adapters: adapter.lora.clone(),
            runtime_config: adapter.lora_runtime_config(),
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
            max_edge: multimodal.and_then(|options| options.max_edge),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: multimodal.and_then(|options| options.max_num_images),
            max_image_length: multimodal.and_then(|options| options.max_image_length),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Plain, false, false, true) => Ok(ModelSelected::XLora {
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
        (ModelFormat::Gguf, dynamic_lora, false, false) => Ok(ModelSelected::GGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `--quantized-file`/`-f` to be specified")?,
            tokenizer_json: model
                .tokenizer
                .as_ref()
                .map(|path| path.to_string_lossy().to_string()),
            mmproj_filename: format_opts.mmproj.clone(),
            lora_adapters: adapter.lora.clone(),
            lora_runtime_config: dynamic_lora.then(|| adapter.lora_runtime_config()),
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            organization: quantization.isq_organization,
            write_uqff: None,
            imatrix: quantization.imatrix.clone(),
            calibration_file: quantization.calibration_file.clone(),
            max_edge: multimodal.and_then(|options| options.max_edge),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
            max_num_images: multimodal.and_then(|options| options.max_num_images),
            max_image_length: multimodal.and_then(|options| options.max_image_length),
            hf_cache_path: device.hf_cache.clone(),
            matformer_config_path: matformer.config_path.clone(),
            matformer_slice_name: matformer.slice_name.clone(),
        }),

        (ModelFormat::Gguf, false, true, false) => Ok(ModelSelected::LoraGGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `--quantized-file`/`-f` to be specified")?,
            adapters_model_id: adapter.legacy_lora.clone().unwrap_or_default(),
            order: adapter
                .legacy_lora_order
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

        (ModelFormat::Gguf, false, false, true) => Ok(ModelSelected::XLoraGGUF {
            tok_model_id: format_opts.tok_model_id.clone(),
            quantized_model_id: model.model_id.clone(),
            quantized_filename: format_opts
                .quantized_file
                .clone()
                .context("GGUF model type requires `--quantized-file`/`-f` to be specified")?,
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
        (ModelFormat::Ggml, false, false, false) => Ok(ModelSelected::GGML {
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
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
            gqa: format_opts.gqa,
            dtype: model.dtype,
            topology: device
                .topology
                .as_ref()
                .map(|p| p.to_string_lossy().to_string()),
            max_seq_len: device.max_seq_len,
            max_batch_size: device.max_batch_size,
        }),

        (ModelFormat::Ggml, false, true, false) => Ok(ModelSelected::LoraGGML {
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
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
            adapters_model_id: adapter.legacy_lora.clone().unwrap_or_default(),
            order: adapter
                .legacy_lora_order
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

        (ModelFormat::Ggml, false, false, true) => Ok(ModelSelected::XLoraGGML {
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
                .context("GGML model type requires `--quantized-file`/`-f` to be specified")?,
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

        (ModelFormat::Plain, false, true, false) => {
            anyhow::bail!("--legacy-lora is only supported with raw GGUF or GGML models")
        }
        (ModelFormat::Ggml, true, false, false) => {
            anyhow::bail!(
                "dynamic --lora adapters are not supported with raw GGML models; use \
                 --legacy-lora with --legacy-lora-order"
            )
        }
        _ => anyhow::bail!("dynamic LoRA, legacy LoRA, and X-LoRA are mutually exclusive"),
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

pub(crate) fn extract_encoder_cache_memory_bytes(model_type: &ModelType) -> Result<Option<usize>> {
    let memory_mb = match model_type {
        ModelType::Auto { multimodal, .. } | ModelType::Multimodal { multimodal, .. } => {
            multimodal.encoder_cache_memory_mb
        }
        _ => None,
    };
    memory_mb
        .map(|memory_mb| {
            memory_mb
                .get()
                .checked_mul(MEBIBYTE_BYTES)
                .context("encoder cache memory capacity overflow")
        })
        .transpose()
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

fn model_format_mut(model_type: &mut ModelType) -> Option<&mut FormatOptions> {
    match model_type {
        ModelType::Auto { format, .. }
        | ModelType::Text { format, .. }
        | ModelType::Multimodal { format, .. }
        | ModelType::Embedding { format, .. } => Some(format),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } => None,
    }
}

fn model_format(model_type: &ModelType) -> Option<&FormatOptions> {
    match model_type {
        ModelType::Auto { format, .. }
        | ModelType::Text { format, .. }
        | ModelType::Multimodal { format, .. }
        | ModelType::Embedding { format, .. } => Some(format),
        ModelType::Diffusion { .. } | ModelType::Speech { .. } => None,
    }
}

fn model_dtype(model_type: &ModelType) -> mistralrs_core::ModelDType {
    match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model.dtype,
    }
}

fn device_options(model_type: &ModelType) -> &DeviceOptions {
    match model_type {
        ModelType::Auto { device, .. }
        | ModelType::Text { device, .. }
        | ModelType::Multimodal { device, .. }
        | ModelType::Diffusion { device, .. }
        | ModelType::Speech { device, .. }
        | ModelType::Embedding { device, .. } => device,
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

pub(crate) fn extract_hf_config_settings(
    model_type: &ModelType,
) -> (Option<usize>, Option<mistralrs_core::HfConfigOverrides>) {
    let model = match model_type {
        ModelType::Auto { model, .. }
        | ModelType::Text { model, .. }
        | ModelType::Multimodal { model, .. }
        | ModelType::Diffusion { model, .. }
        | ModelType::Speech { model, .. }
        | ModelType::Embedding { model, .. } => model,
    };
    (model.max_model_len, model.hf_overrides.clone())
}

pub(crate) async fn apply_quant_resolution(
    model_type: &mut ModelType,
    token_source: &mistralrs_core::TokenSource,
    matformer: &MatformerSelection,
) -> Result<()> {
    if let Some(path) = device_options(model_type).hf_cache.clone() {
        mistralrs_core::set_hf_cache_path(path);
    }
    if let Some(format) = model_format_mut(model_type) {
        format.normalize()?;
    }
    let model_id = model_id_mut(model_type).clone();
    let raw = model_quantization_mut(model_type).and_then(|q| q.quant.clone());
    let (explicit_format, exact_file) = model_format(model_type).map_or((None, None), |format| {
        (format.format, format.quantized_file.clone())
    });
    let is_explicit_gguf = matches!(explicit_format, Some(ModelFormat::Gguf));
    let is_explicit_multimodal = matches!(model_type, ModelType::Multimodal { .. });
    if raw.is_some() && exact_file.is_some() {
        anyhow::bail!("`--quant` and `--quantized-file` are mutually exclusive");
    }
    if raw.is_some() && matches!(explicit_format, Some(ModelFormat::Ggml)) {
        anyhow::bail!("`--quant` cannot select a GGML file; pass one explicitly with `-f`");
    }
    let should_inspect_files = raw.is_some() || (is_explicit_gguf && exact_file.is_some());
    let repo_files = if should_inspect_files {
        selected_model_files(&model_id, exact_file.as_deref(), token_source)?
    } else {
        None
    };
    let is_confident_gguf_repo = repo_files
        .as_ref()
        .is_some_and(|files| is_confident_gguf_artifact_repo(&model_id, files));
    let looks_like_gguf_repo = repo_files.as_ref().is_some_and(|files| {
        crate::commands::quant::has_gguf_model_files(files)
            && !matches!(
                explicit_format,
                Some(ModelFormat::Plain | ModelFormat::Ggml)
            )
            && (is_explicit_gguf || is_confident_gguf_repo)
    });

    if looks_like_gguf_repo {
        let files = repo_files
            .as_ref()
            .expect("GGUF repository detection requires a file listing");
        if let Some(raw) = raw.as_deref() {
            let artifact = crate::commands::quant::resolve_gguf_quant(files, raw)?;
            info!(
                "quant: --quant {raw} -> GGUF {} from `{model_id}`",
                artifact.label
            );
            let format = model_format_mut(model_type)
                .context("GGUF artifacts are not supported for this model type")?;
            format.format = Some(ModelFormat::Gguf);
            format.quantized_file = Some(artifact.file_spec());
            if let Some(quantization) = model_quantization_mut(model_type) {
                // The CLI rejects these alongside `--quant`, but a TOML config can still set both.
                if quantization.in_situ_quant.is_some() || quantization.from_uqff.is_some() {
                    warn!(
                        "quant: `--quant {raw}` selected a published GGUF artifact, ignoring the \
                         configured `isq`/`from_uqff` target"
                    );
                }
                quantization.quant = None;
                quantization.in_situ_quant = None;
                quantization.from_uqff = None;
            }
        }

        let dtype = model_dtype(model_type);
        let format = model_format_mut(model_type)
            .context("GGUF artifacts are not supported for this model type")?;
        if format.mmproj.is_none()
            && (is_confident_gguf_repo || is_explicit_multimodal || format.direct_file_only)
        {
            if let Some(projector) = crate::commands::quant::resolve_gguf_projector(files, dtype)? {
                info!(
                    "GGUF: selected {} projector `{}`",
                    projector.label,
                    projector.file_spec()
                );
                format.mmproj = Some(projector.file_spec());
            }
        }
        return Ok(());
    }

    let Some(raw) = raw else {
        return Ok(());
    };
    if is_explicit_gguf {
        anyhow::bail!(
            "Could not inspect GGUF artifacts for `{model_id}`. Pass `-f <filename.gguf>` \
             explicitly or check repository access."
        );
    }
    if model_name_looks_gguf(&model_id) {
        anyhow::bail!(
            "Model `{model_id}` appears to be a GGUF artifact repo, but its files could not be \
             inspected or no model GGUF was found. Pass `-f <filename.gguf>` explicitly or check \
             repository access."
        );
    }

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

fn selected_model_files(
    model_id: &str,
    exact_file: Option<&str>,
    token_source: &mistralrs_core::TokenSource,
) -> Result<Option<Vec<String>>> {
    let path = Path::new(model_id);
    if path.exists() {
        if let Some(exact_file) = exact_file {
            return crate::commands::quant::list_local_gguf_companions(path, exact_file).map(Some);
        }
        return crate::commands::quant::list_local_files_recursive(path).map(Some);
    }
    Ok(mistralrs_core::probe_hf_repo_files(
        model_id,
        "main",
        token_source,
    ))
}

fn model_name_looks_gguf(model_id: &str) -> bool {
    model_id
        .rsplit_once('/')
        .map_or(model_id, |(_, name)| name)
        .to_ascii_lowercase()
        .ends_with("-gguf")
}

fn is_confident_gguf_artifact_repo(model_id: &str, files: &[String]) -> bool {
    if model_name_looks_gguf(model_id) {
        return true;
    }
    !files.iter().any(|file| {
        let lower = file.to_ascii_lowercase();
        lower.ends_with(".uqff")
            || lower.ends_with(".safetensors")
            || lower.ends_with(".pth")
            || lower.ends_with(".pt")
            || lower.ends_with(".bin")
    })
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

/// Build a `ShellConfig` from runtime options. Returns `None` when shell execution is off.
#[cfg(feature = "code-execution")]
pub(crate) fn build_shell_config(
    runtime: &RuntimeOptions,
    sandbox_policy: Option<mistralrs_sandbox::SandboxPolicy>,
) -> Option<mistralrs_core::ShellConfig> {
    if !runtime.enable_shell {
        return None;
    }
    let mut config = mistralrs_core::ShellConfig::default();
    if let Some(shell_path) = runtime.shell_path.clone() {
        config.shell_path = shell_path;
    }
    if let Some(timeout) = runtime.shell_timeout {
        config.timeout_secs = timeout;
    }
    config.working_directory = runtime.shell_workdir.clone();
    config.sandbox_policy = sandbox_policy;
    config.permission = runtime.code_exec_permission.into();
    Some(config)
}

pub(crate) fn extract_sandbox_settings(
    sandbox: SandboxOptions,
    runtime: &RuntimeOptions,
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
            let profile = sandbox
                .profile
                .map(Into::into)
                .unwrap_or_else(|| default_sandbox_profile(runtime));
            let mut policy = profile.default_policy();
            if let Some(v) = sandbox.max_memory_mb {
                policy.max_memory_mb = v;
            }
            if let Some(v) = sandbox.max_cpu_secs {
                policy.max_cpu_secs = v;
            }
            if let Some(v) = sandbox.max_procs {
                policy.max_procs = v;
            }
            if let Some(network) = sandbox.network {
                policy.network = network.into();
            }
            policy.strict = matches!(mode, SandboxMode::On);
            Some(policy)
        }
    }
}

fn default_sandbox_profile(runtime: &RuntimeOptions) -> mistralrs_sandbox::SandboxProfile {
    #[cfg(feature = "code-execution")]
    {
        if runtime.agent || runtime.enable_code_execution || runtime.enable_shell {
            return mistralrs_sandbox::SandboxProfile::Developer;
        }
    }
    #[cfg(not(feature = "code-execution"))]
    {
        let _ = runtime;
    }
    mistralrs_sandbox::SandboxProfile::Restricted
}

pub(crate) fn apply_agent_mode(runtime: &mut RuntimeOptions) {
    if !runtime.agent {
        return;
    }
    runtime.enable_search = true;
    #[cfg(feature = "code-execution")]
    {
        runtime.enable_code_execution = true;
        runtime.enable_shell = true;
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
        let touches_shell = runtime.shell_path.is_some()
            || runtime.shell_timeout.is_some()
            || runtime.shell_workdir.is_some()
            || runtime.skills_dir.is_some();
        if touches_shell && !runtime.enable_shell {
            anyhow::bail!(
                "`--shell-*` and `--skills-dir` options require `--enable-shell` (or `--agent`/`--agentic`)"
            );
        }
    }
    Ok(())
}

pub(crate) fn log_agent_runtime(runtime: &RuntimeOptions, max_tool_rounds: Option<usize>) {
    if !runtime.agent
        && !runtime.enable_search
        && !is_code_execution_enabled(runtime)
        && !is_shell_enabled(runtime)
    {
        return;
    }

    let rounds = max_tool_rounds.unwrap_or(mistralrs_core::DEFAULT_MAX_TOOL_ROUNDS);
    let mode = if runtime.agent { "agent" } else { "tools" };
    tracing::info!(
        "{mode}: search {}, code execution {}, shell {}, approvals {}, max tool rounds {rounds}",
        search_summary(runtime),
        code_execution_summary(runtime),
        shell_summary(runtime),
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
fn is_shell_enabled(runtime: &RuntimeOptions) -> bool {
    runtime.enable_shell
}
#[cfg(not(feature = "code-execution"))]
fn is_shell_enabled(_runtime: &RuntimeOptions) -> bool {
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
fn shell_summary(runtime: &RuntimeOptions) -> &'static str {
    if runtime.enable_shell {
        "on"
    } else {
        "off"
    }
}

#[cfg(feature = "code-execution")]
fn log_agent_runtime_details(runtime: &RuntimeOptions) {
    if !runtime.enable_code_execution && !runtime.enable_shell {
        return;
    }
    if runtime.enable_code_execution {
        let python = runtime
            .code_exec_python
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "python3 (default)".to_string());
        let timeout = runtime.code_exec_timeout.map_or_else(
            || {
                format!(
                    "{}s (default)",
                    mistralrs_core::DEFAULT_CODE_EXEC_TIMEOUT_SECS
                )
            },
            |t| format!("{t}s"),
        );
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
    if runtime.enable_shell {
        let shell = runtime
            .shell_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "/bin/sh (default)".to_string());
        let timeout = runtime.shell_timeout.map_or_else(
            || format!("{}s (default)", mistralrs_core::DEFAULT_SHELL_TIMEOUT_SECS),
            |t| format!("{t}s"),
        );
        let workdir = runtime
            .shell_workdir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "per-session temp dir".to_string());
        let skills_dir = runtime
            .skills_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "system temp dir".to_string());
        tracing::info!(
            "shell: shell={shell}, timeout={timeout}, workdir={workdir}, skills_dir={skills_dir}, permission={}",
            agent_permission_summary(runtime.code_exec_permission)
        );
    }
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
fn shell_summary(runtime: &RuntimeOptions) -> &'static str {
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

#[cfg(test)]
mod tests {
    use axum::serve::Listener;
    use mistralrs_core::{
        AutoDeviceMapParams, IsqOrganization, LoraAdapterSpec, ModelDType, NormalLoaderType,
    };
    use mistralrs_sandbox::NetworkMode;
    use std::{fs, num::NonZeroUsize, path::PathBuf};

    use super::*;
    use crate::args::{SandboxNetworkMode, SandboxProfileArg};

    fn test_model() -> ModelSourceOptions {
        ModelSourceOptions {
            model_id: "org/base".to_string(),
            tokenizer: None,
            arch: None,
            dtype: ModelDType::Auto,
            hf_overrides: None,
            max_model_len: None,
        }
    }

    #[test]
    fn extracts_runtime_hf_config_settings() {
        let mut model = test_model();
        model.max_model_len = Some(131072);
        model.hf_overrides = Some(
            r#"{"text_config":{"max_position_embeddings":131072}}"#
                .parse()
                .unwrap(),
        );
        let model_type = ModelType::Auto {
            model,
            format: FormatOptions::default(),
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: Default::default(),
            multimodal: MultimodalOptions::default(),
        };

        let (max_model_len, overrides) = extract_hf_config_settings(&model_type);
        assert_eq!(max_model_len, Some(131072));
        assert_eq!(
            overrides.unwrap().as_value()["text_config"]["max_position_embeddings"],
            131072
        );
    }

    #[tokio::test]
    async fn local_gguf_quant_selects_model_and_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in [
            "model-Q4_K_S.gguf",
            "model-Q4_K_M.gguf",
            "mmproj-BF16.gguf",
            "mmproj-F16.gguf",
        ] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        model.dtype = ModelDType::F16;
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions::default(),
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto {
            format,
            quantization,
            ..
        } = model_type
        else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.quantized_file.as_deref(), Some("model-Q4_K_M.gguf"));
        assert_eq!(format.mmproj.as_deref(), Some("mmproj-F16.gguf"));
        assert!(quantization.quant.is_none());
        assert!(quantization.in_situ_quant.is_none());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn local_gguf_quant_selects_vision_and_audio_projectors() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in [
            "model-Q4_K_M.gguf",
            "model-vision-mmproj-BF16.gguf",
            "model-audio-mmproj-BF16.gguf",
        ] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions::default(),
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(
            format.mmproj.as_deref(),
            Some("model-vision-mmproj-BF16.gguf;model-audio-mmproj-BF16.gguf")
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn dynamic_lora_keeps_automatic_projector_selection() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("mmproj-BF16.gguf"), []).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions::default(),
            adapter: AdapterOptions {
                enable_lora: true,
                ..AdapterOptions::default()
            },
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.quantized_file.as_deref(), Some("model-Q4_K_M.gguf"));
        assert_eq!(format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exact_local_gguf_only_discovers_nearby_projectors() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("selected")).unwrap();
        fs::create_dir_all(root.join("unrelated")).unwrap();
        for file in [
            "selected/model.gguf",
            "selected/mmproj-BF16.gguf",
            "unrelated/model-Q4_K_M.gguf",
            "unrelated/mmproj-BF16.gguf",
        ] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                quantized_file: Some("selected/model.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.mmproj.as_deref(), Some("selected/mmproj-BF16.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn direct_file_shorthand_discovers_only_a_sibling_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(root.join("unrelated")).unwrap();
        for file in ["model.gguf", "mmproj-BF16.gguf", "model.safetensors"] {
            fs::write(root.join(file), []).unwrap();
        }
        fs::write(root.join("unrelated/mmproj-BF16.gguf"), []).unwrap();

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                direct_file_only: true,
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exact_gguf_in_a_source_repository_does_not_guess_a_projector() {
        let root = std::env::temp_dir().join(format!("mistralrs-source-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in ["model.gguf", "mmproj-BF16.gguf", "model.safetensors"] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert!(format.mmproj.is_none());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn explicit_multimodal_discovers_a_projector_in_a_source_repository() {
        let root = std::env::temp_dir().join(format!("mistralrs-source-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in ["model.gguf", "mmproj-BF16.gguf", "model.safetensors"] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Multimodal {
            model,
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: MultimodalAdapterOptions {
                enable_lora: true,
                ..MultimodalAdapterOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Multimodal {
            format, adapter, ..
        } = model_type
        else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Gguf));
        assert_eq!(format.mmproj.as_deref(), Some("mmproj-BF16.gguf"));
        assert!(adapter.dynamic_lora_enabled());

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn explicit_multimodal_projector_override_wins_in_a_source_repository() {
        let root = std::env::temp_dir().join(format!("mistralrs-source-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        for file in [
            "model.gguf",
            "mmproj-BF16.gguf",
            "chosen-mmproj-F16.gguf",
            "model.safetensors",
        ] {
            fs::write(root.join(file), []).unwrap();
        }

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Multimodal {
            model,
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("chosen-mmproj-F16.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: MultimodalAdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap();

        let ModelType::Multimodal { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.mmproj.as_deref(), Some("chosen-mmproj-F16.gguf"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn exact_file_and_quant_conflict_before_repository_access() {
        let mut model_type = ModelType::Auto {
            model: test_model(),
            format: FormatOptions {
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        let error = apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("mutually exclusive"));
    }

    #[tokio::test]
    async fn explicit_gguf_quant_does_not_fall_back_to_isq() {
        let root = std::env::temp_dir().join(format!("mistralrs-gguf-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                format: Some(ModelFormat::Gguf),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        let error = apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("Could not inspect GGUF artifacts"));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn quant_does_not_override_an_explicit_non_gguf_format() {
        let root = std::env::temp_dir().join(format!("mistralrs-{}-GGUF", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("model-Q4_K_M.gguf"), []).unwrap();

        let mut model = test_model();
        model.model_id = root.to_string_lossy().into_owned();
        let mut model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                format: Some(ModelFormat::Plain),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        let error = apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("appears to be a GGUF artifact repo"));
        let ModelType::Auto { format, .. } = model_type else {
            unreachable!()
        };
        assert_eq!(format.format, Some(ModelFormat::Plain));

        fs::remove_dir_all(root).unwrap();
    }

    #[tokio::test]
    async fn quant_rejects_explicit_ggml_without_repository_access() {
        let mut model_type = ModelType::Auto {
            model: test_model(),
            format: FormatOptions {
                format: Some(ModelFormat::Ggml),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                quant: Some("4".to_string()),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        };

        let error = apply_quant_resolution(
            &mut model_type,
            &mistralrs_core::TokenSource::None,
            &MatformerSelection::default(),
        )
        .await
        .unwrap_err();
        assert!(error.to_string().contains("cannot select a GGML file"));
    }

    fn test_multimodal_model(format: FormatOptions) -> ModelType {
        ModelType::Multimodal {
            model: test_model(),
            format,
            adapter: MultimodalAdapterOptions::default(),
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions::default(),
        }
    }

    #[test]
    fn encoder_cache_memory_converts_mib_to_bytes() {
        let mut model_type = test_multimodal_model(FormatOptions::default());
        let ModelType::Multimodal { multimodal, .. } = &mut model_type else {
            unreachable!()
        };
        multimodal.encoder_cache_memory_mb = NonZeroUsize::new(64);

        assert_eq!(
            extract_encoder_cache_memory_bytes(&model_type).unwrap(),
            Some(64 * MEBIBYTE_BYTES)
        );
    }

    #[tokio::test]
    async fn accepted_connections_enable_tcp_nodelay() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut listener = tcp_nodelay_listener(listener);
        let connect = tokio::net::TcpStream::connect(address);

        let ((stream, _), client) = tokio::join!(listener.accept(), connect);

        client.unwrap();
        assert!(stream.nodelay().unwrap());
    }

    #[test]
    fn enable_lora_builds_an_empty_dynamic_runtime() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions::default(),
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(
            mistralrs_core::get_auto_device_map_params(&selected).unwrap(),
            AutoDeviceMapParams::Text { .. }
        ));
        match selected {
            ModelSelected::Lora {
                adapters,
                runtime_config,
                organization,
                imatrix,
                calibration_file,
                max_edge,
                max_num_images,
                max_image_length,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert!(adapters.is_empty());
                assert_eq!(runtime_config, adapter.lora_runtime_config());
                assert!(organization.is_none());
                assert!(imatrix.is_none());
                assert!(calibration_file.is_none());
                assert!(max_edge.is_none());
                assert!(max_num_images.is_none());
                assert!(max_image_length.is_none());
                assert!(matformer_config_path.is_none());
                assert!(matformer_slice_name.is_none());
            }
            _ => panic!("expected dynamic LoRA model"),
        }
    }

    #[test]
    fn auto_lora_preserves_multimodal_and_loading_options() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let quantization = QuantizationOptions {
            from_uqff: Some("q4k-0.uqff".to_string()),
            isq_organization: Some(IsqOrganization::MoeExpertsOnly),
            imatrix: Some(PathBuf::from("model.imatrix")),
            ..QuantizationOptions::default()
        };
        let device = DeviceOptions {
            max_seq_len: 4096,
            max_batch_size: 7,
            ..DeviceOptions::default()
        };
        let matformer = MatformerSelection {
            config_path: Some(PathBuf::from("matformer.csv")),
            slice_name: Some("slice".to_string()),
        };
        let multimodal = MultimodalOptions {
            encoder_cache_memory_mb: None,
            max_edge: Some(2048),
            max_num_images: Some(5),
            max_image_length: Some(1536),
        };
        let model_type = ModelType::Auto {
            model: test_model(),
            format: FormatOptions::default(),
            adapter,
            quantization,
            device,
            cache: crate::args::CacheOptions::default(),
            multimodal,
        };
        let selected = convert_to_model_selected(&model_type, &matformer).unwrap();

        match mistralrs_core::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 4096);
                assert_eq!(max_batch_size, 7);
                assert_eq!(max_image_shape, (1536, 1536));
                assert_eq!(max_num_images, 5);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
        match selected {
            ModelSelected::Lora {
                organization,
                from_uqff,
                imatrix,
                max_edge,
                max_num_images,
                max_image_length,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert!(matches!(
                    organization,
                    Some(IsqOrganization::MoeExpertsOnly)
                ));
                assert_eq!(from_uqff.as_deref(), Some("q4k-0.uqff"));
                assert_eq!(imatrix, Some(PathBuf::from("model.imatrix")));
                assert_eq!(max_edge, Some(2048));
                assert_eq!(max_num_images, Some(5));
                assert_eq!(max_image_length, Some(1536));
                assert_eq!(matformer_config_path, Some(PathBuf::from("matformer.csv")));
                assert_eq!(matformer_slice_name.as_deref(), Some("slice"));
            }
            _ => panic!("expected dynamic LoRA model"),
        }
    }

    #[test]
    fn explicit_multimodal_lora_preserves_multimodal_device_mapping() {
        let mut model = test_model();
        model.arch = Some(NormalLoaderType::Qwen3);
        let model_type = ModelType::Multimodal {
            model,
            format: FormatOptions::default(),
            adapter: MultimodalAdapterOptions {
                enable_lora: true,
                ..MultimodalAdapterOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions {
                max_seq_len: 8192,
                max_batch_size: 3,
                ..DeviceOptions::default()
            },
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions {
                encoder_cache_memory_mb: None,
                max_edge: Some(1280),
                max_num_images: Some(4),
                max_image_length: Some(1024),
            },
        };
        let selected =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap();

        let ModelSelected::Lora { arch, .. } = &selected else {
            panic!("expected dynamic LoRA model")
        };
        assert!(arch.is_none());
        match mistralrs_core::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1024, 1024));
                assert_eq!(max_num_images, 4);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
    }

    #[test]
    fn dynamic_lora_routes_native_text_gguf() {
        let preload = LoraAdapterSpec::new("code", "org/code-lora");
        let adapter = AdapterOptions {
            lora: vec![preload.clone()],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            ..FormatOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                mmproj_filename,
                ..
            } => {
                assert_eq!(lora_adapters, vec![preload]);
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
                assert!(mmproj_filename.is_none());
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn dynamic_lora_routes_native_multimodal_gguf() {
        let preload = LoraAdapterSpec::new("vision-chat", "org/language-lora");
        let adapter = AdapterOptions {
            lora: vec![preload.clone()],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            mmproj: Some("mmproj-BF16.gguf".to_string()),
            ..FormatOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            Some(&MultimodalOptions::default()),
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                mmproj_filename,
                ..
            } => {
                assert_eq!(lora_adapters, vec![preload]);
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj-BF16.gguf"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn enable_lora_routes_empty_native_text_gguf_runtime() {
        let adapter = AdapterOptions {
            enable_lora: true,
            ..AdapterOptions::default()
        };
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                lora_adapters,
                lora_runtime_config,
                ..
            } => {
                assert!(lora_adapters.is_empty());
                assert_eq!(lora_runtime_config, Some(adapter.lora_runtime_config()));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn dynamic_lora_is_rejected_for_ggml() {
        let adapter = AdapterOptions {
            lora: vec![LoraAdapterSpec::new("code", "org/code-lora")],
            ..AdapterOptions::default()
        };
        let format = FormatOptions {
            format: Some(ModelFormat::Ggml),
            quantized_file: Some("model.ggml".to_string()),
            ..FormatOptions::default()
        };
        let error = convert_text_model(
            &test_model(),
            &format,
            &adapter,
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap_err();
        assert!(error.to_string().contains("raw GGML"));
    }

    #[test]
    fn legacy_lora_keeps_legacy_gguf_selection() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                legacy_lora: Some("org/legacy-lora".to_string()),
                legacy_lora_order: Some(PathBuf::from("order.json")),
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(selected, ModelSelected::LoraGGUF { .. }));
    }

    #[test]
    fn xlora_keeps_legacy_gguf_selection() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                xlora: Some("org/xlora".to_string()),
                xlora_order: Some(PathBuf::from("order.json")),
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(selected, ModelSelected::XLoraGGUF { .. }));
    }

    #[test]
    fn auto_gguf_preserves_mmproj_files() {
        let mut model = test_model();
        model.tokenizer = Some(PathBuf::from("tokenizer.json"));
        let model_type = ModelType::Auto {
            model,
            format: FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj-0.gguf;mmproj-1.gguf".to_string()),
                tok_model_id: Some("org/tokenizer".to_string()),
                ..FormatOptions::default()
            },
            adapter: AdapterOptions::default(),
            quantization: QuantizationOptions {
                isq_organization: Some(IsqOrganization::MoeExpertsOnly),
                imatrix: Some(PathBuf::from("model.imatrix")),
                ..QuantizationOptions::default()
            },
            device: DeviceOptions {
                hf_cache: Some(PathBuf::from("hf-cache")),
                max_seq_len: 8192,
                max_batch_size: 3,
                ..DeviceOptions::default()
            },
            cache: crate::args::CacheOptions::default(),
            multimodal: MultimodalOptions {
                encoder_cache_memory_mb: None,
                max_edge: Some(1280),
                max_num_images: Some(4),
                max_image_length: Some(1152),
            },
        };
        let selected = convert_to_model_selected(
            &model_type,
            &MatformerSelection {
                config_path: Some(PathBuf::from("matformer.csv")),
                slice_name: Some("small".to_string()),
            },
        )
        .unwrap();

        match mistralrs_core::get_auto_device_map_params(&selected).unwrap() {
            AutoDeviceMapParams::Multimodal {
                max_seq_len,
                max_batch_size,
                max_image_shape,
                max_num_images,
            } => {
                assert_eq!(max_seq_len, 8192);
                assert_eq!(max_batch_size, 3);
                assert_eq!(max_image_shape, (1152, 1152));
                assert_eq!(max_num_images, 4);
            }
            _ => panic!("expected multimodal device-map parameters"),
        }
        match selected {
            ModelSelected::GGUF {
                tok_model_id,
                quantized_filename,
                tokenizer_json,
                mmproj_filename,
                organization,
                imatrix,
                max_edge,
                hf_cache_path,
                matformer_config_path,
                matformer_slice_name,
                ..
            } => {
                assert_eq!(tok_model_id.as_deref(), Some("org/tokenizer"));
                assert_eq!(quantized_filename, "model.gguf");
                assert_eq!(tokenizer_json.as_deref(), Some("tokenizer.json"));
                assert_eq!(
                    mmproj_filename.as_deref(),
                    Some("mmproj-0.gguf;mmproj-1.gguf")
                );
                assert!(matches!(
                    organization,
                    Some(IsqOrganization::MoeExpertsOnly)
                ));
                assert_eq!(imatrix, Some(PathBuf::from("model.imatrix")));
                assert_eq!(max_edge, Some(1280));
                assert_eq!(hf_cache_path, Some(PathBuf::from("hf-cache")));
                assert_eq!(matformer_config_path, Some(PathBuf::from("matformer.csv")));
                assert_eq!(matformer_slice_name.as_deref(), Some("small"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn multimodal_gguf_preserves_dynamic_lora() {
        let adapter = LoraAdapterSpec::new("code", "org/code-lora");
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions {
                lora: vec![adapter.clone()],
                ..AdapterOptions::default()
            },
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            Some(&MultimodalOptions::default()),
        )
        .unwrap();

        match selected {
            ModelSelected::GGUF {
                mmproj_filename,
                lora_adapters,
                lora_runtime_config,
                ..
            } => {
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj.gguf"));
                assert_eq!(lora_adapters, vec![adapter]);
                assert!(lora_runtime_config.is_some());
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn embedding_rejects_gguf_format() {
        let model_type = ModelType::Embedding {
            model: test_model(),
            format: FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                mmproj: Some("mmproj.gguf".to_string()),
                ..FormatOptions::default()
            },
            quantization: QuantizationOptions::default(),
            device: DeviceOptions::default(),
            cache: crate::args::CacheOptions::default(),
        };
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("Embedding models"));
    }

    #[test]
    fn text_gguf_keeps_text_device_map() {
        let selected = convert_text_model(
            &test_model(),
            &FormatOptions {
                format: Some(ModelFormat::Gguf),
                quantized_file: Some("model.gguf".to_string()),
                ..FormatOptions::default()
            },
            &AdapterOptions::default(),
            &QuantizationOptions::default(),
            &DeviceOptions::default(),
            &MatformerSelection::default(),
            None,
        )
        .unwrap();

        assert!(matches!(
            mistralrs_core::get_auto_device_map_params(&selected).unwrap(),
            AutoDeviceMapParams::Text { .. }
        ));
    }

    #[test]
    fn explicit_multimodal_gguf_routes_to_gguf() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            mmproj: Some("mmproj.gguf".to_string()),
            ..FormatOptions::default()
        });
        let selected =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap();

        match selected {
            ModelSelected::GGUF {
                quantized_model_id,
                quantized_filename,
                mmproj_filename,
                ..
            } => {
                assert_eq!(quantized_model_id, "org/base");
                assert_eq!(quantized_filename, "model.gguf");
                assert_eq!(mmproj_filename.as_deref(), Some("mmproj.gguf"));
            }
            _ => panic!("expected GGUF model"),
        }
    }

    #[test]
    fn explicit_multimodal_gguf_requires_model_file() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Gguf),
            mmproj: Some("mmproj.gguf".to_string()),
            ..FormatOptions::default()
        });
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("--quantized-file"));
    }

    #[test]
    fn explicit_multimodal_gguf_requires_mmproj() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Gguf),
            quantized_file: Some("model.gguf".to_string()),
            ..FormatOptions::default()
        });
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("--mmproj"));
    }

    #[test]
    fn explicit_multimodal_ggml_is_rejected() {
        let model_type = test_multimodal_model(FormatOptions {
            format: Some(ModelFormat::Ggml),
            quantized_file: Some("model.ggml".to_string()),
            ..FormatOptions::default()
        });
        let error =
            convert_to_model_selected(&model_type, &MatformerSelection::default()).unwrap_err();

        assert!(error.to_string().contains("GGML is not supported"));
    }

    #[test]
    fn sandbox_off_returns_none() {
        let runtime = RuntimeOptions::default();
        let sandbox = SandboxOptions {
            mode: SandboxMode::Off,
            ..SandboxOptions::default()
        };

        assert!(extract_sandbox_settings(sandbox, &runtime).is_none());
    }

    #[test]
    fn sandbox_on_sets_strict() {
        let runtime = RuntimeOptions::default();
        let sandbox = SandboxOptions {
            mode: SandboxMode::On,
            ..SandboxOptions::default()
        };

        let policy = extract_sandbox_settings(sandbox, &runtime).unwrap();
        assert!(policy.strict);
    }

    #[test]
    fn restricted_profile_uses_loopback_by_default() {
        let runtime = RuntimeOptions::default();
        let sandbox = SandboxOptions {
            mode: SandboxMode::On,
            profile: Some(SandboxProfileArg::Restricted),
            ..SandboxOptions::default()
        };

        let policy = extract_sandbox_settings(sandbox, &runtime).unwrap();
        assert_eq!(policy.network, NetworkMode::Loopback);
        assert!(policy.extra_env.is_empty());
    }

    #[test]
    #[cfg(feature = "code-execution")]
    fn agent_defaults_to_developer_profile() {
        let runtime = RuntimeOptions {
            agent: true,
            ..RuntimeOptions::default()
        };
        let sandbox = SandboxOptions {
            mode: SandboxMode::On,
            ..SandboxOptions::default()
        };

        let policy = extract_sandbox_settings(sandbox, &runtime).unwrap();
        assert_eq!(policy.network, NetworkMode::Full);
        assert!(policy.extra_env.iter().any(|v| v == "RUSTUP_HOME"));
    }

    #[test]
    fn explicit_network_overrides_profile_default() {
        let runtime = RuntimeOptions {
            agent: true,
            ..RuntimeOptions::default()
        };
        let sandbox = SandboxOptions {
            mode: SandboxMode::On,
            network: Some(SandboxNetworkMode::Loopback),
            ..SandboxOptions::default()
        };

        let policy = extract_sandbox_settings(sandbox, &runtime).unwrap();
        assert_eq!(policy.network, NetworkMode::Loopback);
    }
}
