---
title: Embed mistralrs inside an Axum application
description: Mount the full HTTP API inside an existing Axum router, or expose just the pieces you need.
sidebar:
  order: 2
---

If you already have an Axum-based web application and want to add mistral.rs to it, you have two options: mount the full mistralrs HTTP API under a sub-path of your existing router, or write your own handlers that call a loaded `Model` directly.

The full-mount option is the fastest way. Your users get OpenAI-compatible endpoints under `/ai` or wherever you mount them, and you did not have to write any of the request parsing. The direct option lets you expose only the parts you want, with whatever request shape suits your application.

## Full mount under a sub-path

The `mistralrs-server-core` crate has a builder that produces a ready-to-mount Axum router. Add it as a dependency:

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

Now `POST /ai/v1/chat/completions` works exactly like it would against the standalone server. So does `POST /ai/v1/embeddings`, `GET /ai/v1/models`, and the rest of the surface.

The `MistralRsServerRouterBuilder` has the same options as the standalone CLI: allowed origins, body limit, agent defaults. Chain them the same way.

## Hitting the model directly

If you want to expose your own endpoint shape (maybe a simpler "send a message, get a response" endpoint without the full OpenAI request body), bypass the router builder and talk to the `Model` directly:

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

`Model` is internally reference-counted, so sharing an `Arc<Model>` across handlers is cheap. Every handler can spawn its own request concurrently against the same loaded model.

## Which option to pick

Full mount when:

- You want OpenAI compatibility for free.
- You are building an app that adds features around mistralrs rather than exposing custom request shapes.
- You want the web UI available (mount it separately or include it in the router).

Direct when:

- You have a specific request shape driven by your application's needs.
- You do not want the full OpenAI surface (you only need chat, not embeddings or speech).
- You are wiring mistralrs into an existing non-HTTP code path and just happen to be adding an HTTP endpoint as one of several entry points.

Both options can coexist: mount the full router under one path and add your own custom endpoints under others.
