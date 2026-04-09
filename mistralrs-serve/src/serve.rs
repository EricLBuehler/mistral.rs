//! Server command implementation — builds MistralRs and runs the HTTP server.

use anyhow::Result;
use tracing::info;

use mistralrs_core::initialize_logging;
use mistralrs_core::ModelSelected;
use mistralrs_server_core::mistralrs_for_server_builder::MistralRsForServerBuilder;
use mistralrs_server_core::mistralrs_server_router_builder::MistralRsServerRouterBuilder;

use crate::config::{DeviceConfig, GlobalConfig, PagedAttnConfig, RuntimeConfig, ServerConfig};
use crate::ui::build_ui_router;

/// Run the HTTP server with the specified model
pub async fn run_server(
    model_selected: ModelSelected,
    paged_attn: PagedAttnConfig,
    device: DeviceConfig,
    isq: Option<String>,
    server: ServerConfig,
    runtime: RuntimeConfig,
    global: GlobalConfig,
) -> Result<()> {
    initialize_logging();

    let mut builder = MistralRsForServerBuilder::new()
        .with_model(model_selected)
        .with_max_seqs(runtime.max_seqs)
        .with_no_kv_cache(runtime.no_kv_cache)
        .with_token_source(global.token_source)
        .with_interactive_mode(false)
        .with_prefix_cache_n(runtime.prefix_cache_n)
        .set_paged_attn(paged_attn.paged_attn)
        .with_cpu(device.cpu)
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
        .with_num_device_layers_optional(device.device_layers)
        .with_in_situ_quant_optional(isq)
        .with_paged_attn_gpu_mem_optional(paged_attn.paged_attn_gpu_mem)
        .with_paged_attn_gpu_mem_usage_optional(paged_attn.paged_attn_gpu_mem_usage)
        .with_paged_ctxt_len_optional(paged_attn.paged_ctxt_len)
        .with_paged_attn_block_size_optional(paged_attn.paged_attn_block_size)
        .with_paged_attn_cache_type(paged_attn.paged_cache_type)
        .with_idle_timeout_secs(server.idle_timeout_secs);

    if let Some(model) = runtime.search_embedding_model {
        builder = builder.with_search_embedding_model(model);
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
            runtime.search_embedding_model,
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
