---
title: Embed mistralrs inside an Axum application
description: Mount the full HTTP API inside an existing Axum router, or expose just the pieces you need.
sidebar:
  order: 2
---

To add mistral.rs to an existing Axum application, two options exist: mount the full mistralrs HTTP API under a sub-path, or write custom handlers calling a loaded `Model` directly.

The full-mount option is fastest — OpenAI-compatible endpoints under any path with no request parsing required. The direct option exposes only the desired parts in any request shape.

## Full mount under a sub-path

The `mistralrs-server-core` crate produces a ready-to-mount Axum router. Dependencies:

```toml
[dependencies]
mistralrs = "0.8"
mistralrs-server-core = "0.8"
axum = "0.7"
tokio = { version = "1", features = ["full"] }
```

Then:

```rust
use std::sync::Arc;
use mistralrs::{IsqBits, ModelBuilder};
use mistralrs_server_core::MistralRsServerRouterBuilder;
use axum::Router;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = ModelBuilder::new("Qwen/Qwen3-4B")
        .with_auto_isq(IsqBits::Four)
        .build()
        .await?;

    let mistralrs_router = MistralRsServerRouterBuilder::new()
        .with_mistralrs(Arc::new(model.into_mistralrs_state()))
        .build()
        .await?;

    let app = Router::new()
        .route("/", axum::routing::get(|| async { "My app" }))
        .nest("/ai", mistralrs_router);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`POST /ai/v1/chat/completions` now behaves identically to the standalone server. So do `POST /ai/v1/embeddings`, `GET /ai/v1/models`, and the rest of the surface.

`MistralRsServerRouterBuilder` exposes the same options as the standalone CLI: allowed origins, body limit, agent defaults.

## Hitting the model directly

For custom endpoint shapes (e.g., a simpler request format without the full OpenAI body), bypass the router builder and call the `Model` directly:

```rust
use axum::{extract::State, Json, Router};
use mistralrs::{IsqBits, Model, ModelBuilder, TextMessageRole, TextMessages};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Deserialize)]
struct ChatRequest { message: String }

#[derive(Serialize)]
struct ChatResponse { reply: String }

async fn handle_chat(
    State(model): State<Arc<Model>>,
    Json(req): Json<ChatRequest>,
) -> Json<ChatResponse> {
    let messages = TextMessages::new()
        .add_message(TextMessageRole::User, &req.message);
    let response = model.send_chat_request(messages).await.unwrap();
    let reply = response.choices[0].message.content.clone().unwrap_or_default();
    Json(ChatResponse { reply })
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = Arc::new(
        ModelBuilder::new("Qwen/Qwen3-4B")
            .with_auto_isq(IsqBits::Four)
            .build()
            .await?,
    );

    let app = Router::new()
        .route("/chat", axum::routing::post(handle_chat))
        .with_state(model);

    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await?;
    axum::serve(listener, app).await?;
    Ok(())
}
```

`Model` is reference-counted; sharing `Arc<Model>` across handlers is cheap. Each handler runs concurrent requests against the same loaded model.

## Which option to pick

Full mount when:

- OpenAI compatibility is desired by default.
- The app adds features around mistralrs rather than exposing custom request shapes.
- The web UI is needed (mount it separately or include in the router).

Direct when:

- A specific application-driven request shape is required.
- The full OpenAI surface is not needed (e.g., chat only, no embeddings or speech).
- mistralrs is wired into an existing non-HTTP code path with HTTP as one of several entry points.

Both options can coexist: full router under one path, custom endpoints under others.
