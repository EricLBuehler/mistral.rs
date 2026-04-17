---
title: Rust API reference
description: Public surface of the mistralrs crate. Canonical docs are on docs.rs; this page covers the common patterns.
sidebar:
  order: 7
---

Authoritative Rust API documentation: [docs.rs/mistralrs](https://docs.rs/mistralrs). This page covers commonly used types and methods.

## Cargo dependency

```toml
[dependencies]
mistralrs = "0.8"
tokio = { version = "1", features = ["full"] }
```

Add accelerator feature flags as needed. See the [cargo features reference](/mistral.rs/reference/cargo-features/).

## ModelBuilder

The fluent builder for loading a model.

```rust
use mistralrs::{IsqType, ModelBuilder, PagedAttentionMetaBuilder};

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_isq(IsqType::Q4K)
    .with_logging()
    .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
    .with_chat_template("path.jinja")
    .with_topology(/* ... */)
    .with_hf_revision("abc123")
    .build()
    .await?;
```

`ModelBuilder::new` takes the Hugging Face repo id. `.build().await?` produces a `Model`.

`ModelBuilder` auto-detects the model type. Type-specific builders also exist:

- `TextModelBuilder` — text-only models.
- `MultimodalModelBuilder` — vision, audio, video.
- `GgufModelBuilder` — GGUF-quantized models.
- `GgufLoraModelBuilder`, `GgufXLoraModelBuilder` — GGUF + adapter combinations.
- `LoraModelBuilder`, `XLoraModelBuilder` — adapter models.
- `SpeechModelBuilder`, `DiffusionModelBuilder`, `EmbeddingModelBuilder` — dedicated model types.
- `UqffTextModelBuilder`, `UqffMultimodalModelBuilder`, `UqffEmbeddingModelBuilder` — UQFF wrappers.
- `AnyMoeModelBuilder` — AnyMoE composition.
- `TextSpeculativeBuilder` — speculative decoding.

## MultiModelBuilder

For multi-model loading in one process:

```rust
use mistralrs::{MultiModelBuilder, TextModelBuilder, MultimodalModelBuilder};

let model = MultiModelBuilder::new()
    .add_model(TextModelBuilder::new("Qwen/Qwen3-4B"))
    .add_model_with_alias("gemma", MultimodalModelBuilder::new("google/gemma-4-E4B-it"))
    .with_default_model("Qwen/Qwen3-4B")
    .build()
    .await?;
```

## Model

The loaded model handle. Cheap to clone (internally Arc-based).

### Request methods

```rust
async fn send_chat_request<R: RequestLike>(&self, request: R)
    -> Result<ChatCompletionResponse>

async fn stream_chat_request<R: RequestLike>(&self, request: R)
    -> Result<impl Stream<Item = Response>>

async fn chat(&self, message: impl ToString) -> Result<String>
```

### `_with_model` variants for multi-model

When `Model` holds multiple models, use `_with_model` variants to target a specific one:

```rust
async fn send_chat_request_with_model<R: RequestLike>(
    &self, request: R, model_id: Option<&str>
) -> Result<ChatCompletionResponse>
```

`None` selects the default. Variants exist for streaming, image generation, speech, embeddings, tokenization, and config queries.

### Model management

```rust
fn list_models(&self) -> Result<Vec<String>>
fn list_models_with_status(&self) -> Result<Vec<(String, ModelStatus)>>
fn get_default_model_id(&self) -> Result<String>
fn set_default_model_id(&self, id: &str) -> Result<()>
fn is_model_loaded(&self, id: &str) -> Result<bool>
fn unload_model(&self, id: &str) -> Result<()>
async fn reload_model(&self, id: &str) -> Result<()>
```

## Request builders

### TextMessages

Conversation assembly:

```rust
use mistralrs::{TextMessages, TextMessageRole};

let messages = TextMessages::new()
    .add_message(TextMessageRole::System, "You are concise.")
    .add_message(TextMessageRole::User, "What is 2 + 2?");
```

### RequestBuilder

Full control over sampling, tools, logprobs:

```rust
use mistralrs::RequestBuilder;

let request = RequestBuilder::new()
    .add_message(TextMessageRole::User, "Hello")
    .set_sampler_temperature(0.7)
    .set_sampler_max_len(200)
    .return_logprobs(true)
    .with_session_id("user-42");
```

Sampling methods use the `set_sampler_*` prefix.

Both `TextMessages` and `RequestBuilder` implement `RequestLike` and can be passed to `send_chat_request` / `stream_chat_request`.

## Response and Stream types

```rust
pub enum Response {
    Chunk(ChatCompletionChunkResponse),
    Done(ChatCompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    CompletionDone(CompletionResponse),
    InternalError(Error),
    ModelError(Error, CompletionResponse),
    CompletionModelError(Error, CompletionResponse),
    ValidationError(Error),
    ImageGeneration(ImageGenerationResponse),
    Speech(SpeechResponse),
    Embeddings(EmbeddingResponse),
    AgenticToolCallProgress(AgenticToolCallProgress),
    Raw { /* ... */ },
}
```

## IsqBits

```rust
pub enum IsqBits {
    Two, Three, Four, Five, Six, Eight,
}
```

Resolves to an `IsqType` based on target device. Use `with_auto_isq(IsqBits::Four)` or pass a specific `with_isq(IsqType::Q4K)`.

## Speculative decoding and AnyMoE

`TextSpeculativeBuilder` and `AnyMoeModelBuilder` are SDK-only and not exposed via the CLI or TOML config. See `examples/advanced/` for usage.

## Feature flags for the crate

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

Available features: `cuda`, `flash-attn`, `flash-attn-v3`, `cudnn`, `metal`, `accelerate`, `mkl`, `code-execution`, `ring`.

## Sessions

`Model` does not expose session management methods directly. Use `RequestBuilder::with_session_id` per request, or run the HTTP server and call the `/v1/sessions/{id}` endpoints.

## Integration patterns

For Axum mounting, see the [embed-in-axum guide](/mistral.rs/guides/rust/embed-in-axum/). The `mistralrs-server-core` crate provides `MistralRsServerRouterBuilder`, which takes a `SharedMistralRsState`.

For the full method, type, and feature-gated API list: [docs.rs/mistralrs](https://docs.rs/mistralrs).
