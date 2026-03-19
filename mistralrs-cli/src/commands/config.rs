//! Run mistralrs-cli from a full TOML configuration.

use anyhow::Result;
use tracing::info;

use mistralrs_core::initialize_logging;
use mistralrs_server_core::{
    mistralrs_for_server_builder::{MistralRsForServerBuilder, ModelConfig},
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

use crate::commands::run::interactive_mode;
use crate::commands::serve::convert_to_model_selected;
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
        runtime,
        server,
        paged_attn,
        models,
        default_model_id,
    } = cfg;

    let global = global.to_global_options()?;

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = paged_attn.into_builder_flags();

    let (model_configs, cpu) = build_model_configs(&models)?;

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

    let mistralrs = builder.build().await?;
    let mistralrs_for_ui = mistralrs.clone();

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

async fn run_run_config(cfg: crate::config::RunConfig) -> Result<()> {
    let crate::config::RunConfig {
        global,
        runtime,
        paged_attn,
        models,
        enable_thinking,
    } = cfg;

    let global = global.to_global_options()?;

    let (
        paged_attn,
        paged_attn_gpu_mem,
        paged_attn_gpu_mem_usage,
        paged_ctxt_len,
        paged_attn_block_size,
        paged_cache_type,
    ) = paged_attn.into_builder_flags();

    let (model_configs, cpu) = build_model_configs(&models)?;

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

    let mistralrs = builder.build().await?;

    info!("Model(s) loaded, starting interactive mode...");

    interactive_mode(
        mistralrs.clone(),
        runtime.enable_search,
        if enable_thinking { Some(true) } else { None },
    )
    .await;

    Ok(())
}

fn build_model_configs(models: &[crate::config::ModelEntry]) -> Result<(Vec<ModelConfig>, bool)> {
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
        let model_type = entry.to_model_type(cpu);
        let model_selected = convert_to_model_selected(&model_type)?;

        let mut config = ModelConfig::new(entry.model_id.clone(), model_selected);

        if let Some(chat_template) = entry.chat_template.as_ref() {
            config = config.with_chat_template(chat_template.to_string_lossy().to_string());
        }

        if let Some(jinja_explicit) = entry.jinja_explicit.as_ref() {
            config = config.with_jinja_explicit(jinja_explicit.to_string_lossy().to_string());
        }

        if let Some(device_layers) = entry.device.device_layers.clone() {
            config = config.with_num_device_layers(device_layers);
        }

        if let Some(isq) = entry.quantization.in_situ_quant.clone() {
            config = config.with_in_situ_quant(isq);
        }

        configs.push(config);
    }

    Ok((configs, cpu))
}
