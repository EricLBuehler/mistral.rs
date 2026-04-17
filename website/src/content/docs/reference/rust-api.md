---
title: Rust API reference
description: Public surface of the mistralrs crate. Canonical docs are on docs.rs; this page covers the common patterns.
sidebar:
  order: 7
---

Authoritative Rust API documentation: [docs.rs/mistralrs](https://docs.rs/mistralrs). This page covers commonly used types and methods, with pointers into the full reference for the rest.

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
use mistralrs::{IsqBits, ModelBuilder, PagedAttentionMetaBuilder};

let model = ModelBuilder::new("Qwen/Qwen3-4B")
    .with_auto_isq(IsqBits::Four)          // ISQ bit width
    .with_isq(IsqType::Q4K)                // or a specific ISQ type
    .with_logging()                        // enable tracing subscriber
    .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
    .with_chat_template("path.jinja")
    .with_topology("topology.yaml")
    .with_hf_revision("abc123")
    .build()
    .await?;
```

`ModelBuilder::new` takes the Hugging Face repo id. Each `with_*` method returns `self`. `.build().await?` produces a `Model`.

Variants for specific model kinds:

- `TextModelBuilder` — alias for `ModelBuilder`, emphasizing text-only use.
- `MultimodalModelBuilder` — vision, audio, video models.
- `GgufModelBuilder` — GGUF-quantized models.
- `LoraModelBuilder`, `XLoraModelBuilder` — adapter models.
- `SpeechModelBuilder`, `DiffusionModelBuilder`, `EmbeddingModelBuilder` — dedicated model types.

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
    -> Result<Stream<'_>>

async fn chat(&self, message: impl ToString) -> Result<String>
```

The `chat` convenience method wraps `send_chat_request` for one-shot "user message in, string out" use.

### With-model variants for multi-model

When `Model` holds multiple models, use `_with_model` variants to target a specific one:

```rust
async fn send_chat_request_with_model<R: RequestLike>(
    &self, request: R, model_id: Option<&str>
) -> Result<ChatCompletionResponse>
```

`None` targets the default model.

Other `_with_model` variants:

- `stream_chat_request_with_model`
- `generate_image_with_model`
- `generate_speech_with_model`
- `generate_embeddings_with_model`
- `tokenize_with_model`, `detokenize_with_model`
- `config_with_model`, `max_sequence_length_with_model`

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

### Session management

```rust
fn export_session(&self, model: Option<&str>, id: &str) -> Result<SerializedSession>
fn import_session(&self, model: Option<&str>, id: String, s: SerializedSession) -> Result<()>
fn delete_session(&self, model: Option<&str>, id: &str) -> Result<bool>
fn list_session_ids(&self, model: Option<&str>) -> Result<Vec<String>>
```

### Code execution (feature-gated)

When built with the `code-execution` feature (default):

```rust
async fn exec_in_session(&self, session_id: &str, code: &str) -> Result<CodeExecResult>
async fn reset_session_python(&self, session_id: &str) -> Result<()>
```

## Request builders

Two options for building requests:

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
    .with_temperature(0.7)
    .with_max_tokens(200)
    .return_logprobs(true)
    .with_session_id("user-42");
```

Both implement `RequestLike` and can be passed to `send_chat_request` / `stream_chat_request`.

## Response and Stream types

```rust
pub enum Response {
    Chunk(ChatCompletionChunkResponse),
    Done(ChatCompletionResponse),
    CompletionChunk(CompletionChunkResponse),
    CompletionDone(CompletionResponse),
    InternalError(Error),
    ModelError(Error, CompletionResponse),
    ValidationError(Error),
    ImageGeneration(ImageGenerationResponse),
    Speech(SpeechResponse),
    Embedding(EmbeddingResponse),
    AgenticToolCallProgress(AgenticToolCallProgress),
    Raw { .. },
}
```

## IsqBits

```rust
pub enum IsqBits {
    Two, Three, Four, Five, Six, Eight,
}
```

Resolves to an `IsqType` based on target device. Use `with_auto_isq(IsqBits::Four)` or pass a specific `with_isq(IsqType::Q4K)`.

## Feature flags for the crate

The `mistralrs` crate has feature flags mirroring the CLI:

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

Available features: `cuda`, `flash-attn`, `flash-attn-v3`, `cudnn`, `metal`, `accelerate`, `mkl`, `code-execution`, `ring`.

## Integration patterns

For Axum mounting, see the [embed-in-axum guide](/mistral.rs/guides/rust/embed-in-axum/). The `mistralrs-server-core` crate provides a pre-built router.

For the full method, type, and feature-gated API list: [docs.rs/mistralrs](https://docs.rs/mistralrs).
