---
title: Embed mistralrs inside an Axum application
description: Mount the HTTP API inside an existing Axum router.
sidebar:
  order: 2
---

To add mistral.rs to an existing Axum app, mount the mistralrs router under a sub-path. The pattern uses two builders from `mistralrs-server-core`:

- `MistralRsForServerBuilder` constructs the engine state (`SharedMistralRsState = Arc<MistralRs>`).
- `MistralRsServerRouterBuilder` produces an Axum `Router` from that state.

## Dependencies

```toml
[dependencies]
mistralrs = "0.8"
mistralrs-core = "0.8"
mistralrs-server-core = "0.8"
axum = "0.8"
tokio = { version = "1", features = ["full"] }
```

## Mount under a sub-path

```rust
use axum::{Router, routing::get};
use mistralrs_core::{AutoDeviceMapParams, ModelDType, ModelSelected};
use mistralrs_server_core::{
    mistralrs_for_server_builder::MistralRsForServerBuilder,
    mistralrs_server_router_builder::MistralRsServerRouterBuilder,
};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = ModelSelected::Plain {
        model_id: "Qwen/Qwen3-4B".into(),
        tokenizer_json: None,
        arch: None,
        dtype: ModelDType::Auto,
        topology: None,
        organization: None,
        write_uqff: None,
        from_uqff: None,
        imatrix: None,
        calibration_file: None,
        max_seq_len: AutoDeviceMapParams::DEFAULT_MAX_SEQ_LEN,
        max_batch_size: AutoDeviceMapParams::DEFAULT_MAX_BATCH_SIZE,
        hf_cache_path: None,
        matformer_config_path: None,
        matformer_slice_name: None,
    };

    let shared_mistralrs = MistralRsForServerBuilder::new()
        .with_model(model)
        .with_in_situ_quant("4".to_string())
        .build()
        .await?;

    let mistralrs_router = MistralRsServerRouterBuilder::new()
        .with_mistralrs(shared_mistralrs)
        .build()
        .await?;

    let app = Router::new()
        .route("/", get(|| async { "My app" }))
        .nest("/ai", mistralrs_router);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`POST /ai/v1/chat/completions` then behaves identically to the standalone server, as do the other routes.

## Builder options

`MistralRsServerRouterBuilder` exposes:

- `with_include_swagger_routes(bool)`
- `with_base_path(&str)`
- `with_allowed_origins(Vec<String>)`
- `with_max_body_limit(usize)`
- `with_max_tool_rounds(usize)`
- `with_tool_dispatch_url(String)`

`MistralRsForServerBuilder` exposes engine-level options (`with_model`, `with_in_situ_quant`, `set_paged_attn`, etc.).

## Calling the model directly from a handler

For custom request shapes, share the `SharedMistralRsState` directly with Axum handlers and use the lower-level helpers exposed by `mistralrs-server-core`.

A complete example (with custom OpenAPI integration) is in the `mistralrs-server-core` crate-level documentation.
