//! Run mistralrs-cli from a full TOML configuration.

use anyhow::Result;
use tracing::info;

use mistralrs_core::initialize_logging;
use mistralrs_server_core::{
    mistralrs_for_server_builder::{MistralRsForServerBuilder, ModelConfig},
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

use crate::args::{MatformerSelection, RuntimeOptions};
use crate::commands::run::interactive_mode;
#[cfg(feature = "code-execution")]
use crate::commands::serve::build_code_exec_config;
use crate::commands::serve::{
    apply_agent_mode, convert_to_model_selected, extract_sandbox_settings, load_mcp_config,
    log_agent_runtime, validate_agent_options,
};
use crate::config::{load_cli_config, CliConfig};
use crate::ui::build_ui_router;

/// Execute the CLI using a TOML configuration file.
pub async fn run_from_config(path: std::path::PathBuf) -> Result<()> {
    initialize_logging();

    let config = load_cli_config(&path)?;

    match config {
        CliConfig::Serve(cfg) => run_serve_config(cfg).await,
        CliConfig::Run(cfg) => run_run_config(cfg).await,
    }
}

async fn run_serve_config(cfg: crate::config::ServeConfig) -> Result<()> {
    let crate::config::ServeConfig {
        global,
        mut runtime,
        server,
        paged_attn,
        sandbox,
        models,
        default_model_id,
    } = cfg;

    let global = global.to_global_options()?;
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, server.max_tool_rounds);

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = paged_attn.into_builder_flags();

    let (model_configs, cpu) = build_model_configs(&models, &runtime, &global.token_source).await?;

    let mut builder = MistralRsForServerBuilder::new()
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
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type);

    for config in model_configs {
        builder = builder.add_model_config(config);
    }

    if let Some(default_model_id) = default_model_id {
        builder = builder.with_default_model_id(default_model_id);
    }

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mcp_client_config = load_mcp_config(runtime.mcp_config.as_deref())?;
    builder = builder.with_mcp_config_optional(mcp_client_config);

    let sandbox_policy = extract_sandbox_settings(sandbox);

    #[cfg(feature = "code-execution")]
    {
        builder = builder
            .with_code_exec_config_optional(build_code_exec_config(&runtime, sandbox_policy));
    }
    #[cfg(not(feature = "code-execution"))]
    let _ = sandbox_policy;

    let mistralrs = builder.build().await?;
    let mistralrs_for_ui = mistralrs.clone();

    let mut app = MistralRsServerRouterBuilder::new()
        .with_mistralrs(mistralrs)
        .with_max_tool_rounds_optional(server.max_tool_rounds)
        .with_tool_dispatch_url_optional(server.tool_dispatch_url.clone())
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

async fn run_run_config(cfg: crate::config::RunConfig) -> Result<()> {
    let crate::config::RunConfig {
        global,
        mut runtime,
        paged_attn,
        sandbox,
        models,
        thinking,
    } = cfg;

    let global = global.to_global_options()?;
    apply_agent_mode(&mut runtime);
    validate_agent_options(&runtime)?;
    log_agent_runtime(&runtime, None);

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = paged_attn.into_builder_flags();

    let (model_configs, cpu) = build_model_configs(&models, &runtime, &global.token_source).await?;

    let mut builder = MistralRsForServerBuilder::new()
        .with_max_seqs(runtime.max_seqs)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(true)
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
        .with_paged_attn_gpu_mem_optional(paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn_block_size)
        .with_paged_attn_cache_type(paged_cache_type);

    for config in model_configs {
        builder = builder.add_model_config(config);
    }

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model.into());
    }

    let mcp_client_config = load_mcp_config(runtime.mcp_config.as_deref())?;
    builder = builder.with_mcp_config_optional(mcp_client_config);

    let sandbox_policy = extract_sandbox_settings(sandbox);

    #[cfg(feature = "code-execution")]
    {
        builder = builder
            .with_code_exec_config_optional(build_code_exec_config(&runtime, sandbox_policy));
    }
    #[cfg(not(feature = "code-execution"))]
    let _ = sandbox_policy;

    let mistralrs = builder.build().await?;

    #[cfg(feature = "code-execution")]
    let do_code_exec = runtime.enable_code_execution;
    #[cfg(not(feature = "code-execution"))]
    let do_code_exec = false;

    info!("Model(s) loaded, starting interactive mode...");

    interactive_mode(
        mistralrs.clone(),
        runtime.enable_search,
        do_code_exec,
        runtime.code_exec_permission.into(),
        thinking,
    )
    .await;

    Ok(())
}

async fn build_model_configs(
    models: &[crate::config::ModelEntry],
    runtime: &RuntimeOptions,
    token_source: &mistralrs_core::TokenSource,
) -> Result<(Vec<ModelConfig>, bool)> {
    let mut cpu_setting: Option<bool> = None;
    let mut configs = Vec::new();

    for entry in models {
        if let Some(cpu) = entry.device.cpu {
            match cpu_setting {
                None => cpu_setting = Some(cpu),
                Some(existing) if existing != cpu => {
                    anyhow::bail!(
                        "cpu must be consistent across all models (found both true and false)"
                    );
                }
                _ => {}
            }
        }
    }

    let cpu = cpu_setting.unwrap_or(false);

    for entry in models {
        let mut model_type = entry.to_model_type(cpu);
        let matformer = MatformerSelection {
            config_path: entry
                .matformer_config_path
                .clone()
                .or_else(|| runtime.matformer_config_path.clone()),
            slice_name: entry
                .matformer_slice_name
                .clone()
                .or_else(|| runtime.matformer_slice_name.clone()),
        };
        crate::commands::serve::apply_quant_resolution(&mut model_type, token_source, &matformer)
            .await?;
        let model_selected = convert_to_model_selected(&model_type, &matformer)?;

        let resolved_loader_id = crate::commands::serve::model_id_of(&model_type);
        let mut config = ModelConfig::new(entry.model_id.clone(), model_selected);
        if resolved_loader_id != entry.model_id {
            config = config.with_alias(entry.model_id.clone());
        }

        if let Some(chat_template) = entry.chat_template.as_ref() {
            config = config.with_chat_template(chat_template.to_string_lossy().to_string());
        }

        if let Some(jinja_explicit) = entry.jinja_explicit.as_ref() {
            config = config.with_jinja_explicit(jinja_explicit.to_string_lossy().to_string());
        }

        if let Some(device_layers) = entry.device.device_layers.clone() {
            config = config.with_num_device_layers(device_layers);
        }

        if let Some(isq) = crate::commands::serve::extract_isq_setting(&model_type) {
            config = config.with_in_situ_quant(isq);
        }

        configs.push(config);
    }

    Ok((configs, cpu))
}
