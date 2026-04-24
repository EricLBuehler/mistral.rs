---
title: Rust API reference
description: Public surface of the mistralrs crate. Canonical signatures are on docs.rs; this page is a map.
sidebar:
  order: 7
---

Canonical Rust API documentation: [docs.rs/mistralrs](https://docs.rs/mistralrs). This page names the types and describes when to use each, without duplicating signatures.

## Cargo dependency

```toml
[dependencies]
mistralrs = "0.8"
tokio = { version = "1", features = ["full"] }
```

Add accelerator feature flags as needed. See the [cargo features reference](/mistral.rs/reference/cargo-features/).

## Builders

The fluent entry point for loading a model.

- [`ModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.ModelBuilder.html): auto-detects the model type.
- [`TextModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.TextModelBuilder.html): text-only models.
- [`MultimodalModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.MultimodalModelBuilder.html): vision, audio, video.
- [`GgufModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.GgufModelBuilder.html), [`GgufLoraModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.GgufLoraModelBuilder.html), [`GgufXLoraModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.GgufXLoraModelBuilder.html): GGUF-quantized variants.
- [`LoraModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.LoraModelBuilder.html), [`XLoraModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.XLoraModelBuilder.html): adapter models.
- [`SpeechModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.SpeechModelBuilder.html), [`DiffusionModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.DiffusionModelBuilder.html), [`EmbeddingModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.EmbeddingModelBuilder.html): dedicated model types.
- [`UqffTextModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.UqffTextModelBuilder.html), [`UqffMultimodalModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.UqffMultimodalModelBuilder.html), [`UqffEmbeddingModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.UqffEmbeddingModelBuilder.html): UQFF wrappers.
- [`AnyMoeModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.AnyMoeModelBuilder.html): AnyMoE composition.
- [`TextSpeculativeBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.TextSpeculativeBuilder.html): speculative decoding.
- [`MultiModelBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.MultiModelBuilder.html): load several models in one process.

Each builder's `.build().await?` produces a [`Model`](https://docs.rs/mistralrs/latest/mistralrs/struct.Model.html).

## Model

[`Model`](https://docs.rs/mistralrs/latest/mistralrs/struct.Model.html) is the loaded handle. Cheap to clone (internally `Arc`-based).

Groups of methods:

- **Requests:** `send_chat_request`, `stream_chat_request`, `chat`, `send_completion_request`, `send_embedding_request`, `generate_image`, `generate_audio`.
- **Multi-model routing:** `_with_model` variants of the above take an `Option<&str>` model id.
- **Model management:** `list_models`, `list_models_with_status`, `get_default_model_id`, `set_default_model_id`, `is_model_loaded`, `unload_model`, `reload_model`, `remove_model`.
- **Sessions:** `export_session`, `import_session`, `delete_session`, `list_session_ids`. See [persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- **Tokenization:** `tokenize_with_model`, `detokenize_with_model`.
- **MCP:** `list_mcp_tools`.

Full signatures: [docs.rs/mistralrs/latest/mistralrs/struct.Model.html](https://docs.rs/mistralrs/latest/mistralrs/struct.Model.html).

## Request builders

- [`TextMessages`](https://docs.rs/mistralrs/latest/mistralrs/struct.TextMessages.html): conversation assembly.
- [`RequestBuilder`](https://docs.rs/mistralrs/latest/mistralrs/struct.RequestBuilder.html): full control over sampling, tools, logprobs, and the session id.

Both implement [`RequestLike`](https://docs.rs/mistralrs/latest/mistralrs/trait.RequestLike.html) and can be passed to `send_chat_request` / `stream_chat_request`. Sampling methods on `RequestBuilder` use the `set_sampler_*` prefix (`set_sampler_temperature`, `set_sampler_max_len`, ...).

## Responses

[`Response`](https://docs.rs/mistralrs/latest/mistralrs/enum.Response.html) is the streamed enum. Variants include `Chunk`, `Done`, `CompletionChunk`, `CompletionDone`, `InternalError`, `ModelError`, `ValidationError`, `ImageGeneration`, `Speech`, `Embeddings`, `AgenticToolCallProgress`, and `Raw`. Non-exhaustive.

For agentic applications, `stream_chat_request` can yield `Response::AgenticToolCallProgress` between model chunks. Match that variant to render code execution, search, MCP, callback, or dispatch progress. See [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/) for the current event model and SDK boundaries.

## Quantization

- [`IsqType`](https://docs.rs/mistralrs/latest/mistralrs/enum.IsqType.html): explicit format (`Q4K`, `AFQ4`, `FP8E4M3`, ...).
- [`IsqBits`](https://docs.rs/mistralrs/latest/mistralrs/enum.IsqBits.html): numeric shorthand (`Two`, `Three`, `Four`, `Five`, `Six`, `Eight`). Resolves to an `IsqType` for the target device.

Call `.with_isq(IsqType::Q4K)` for an explicit format or `.with_auto_isq(IsqBits::Four)` for per-device resolution.

## Server integration

Embedding mistralrs inside an Axum app: see the [embed-in-axum guide](/mistral.rs/guides/rust/embed-in-axum/) and the [`mistralrs-server-core`](https://docs.rs/mistralrs-server-core) crate. `MistralRsServerRouterBuilder` produces an Axum `Router` from a `SharedMistralRsState`.

## Feature flags

```toml
[dependencies]
mistralrs = { version = "0.8", features = ["cuda", "flash-attn", "cudnn"] }
```

Available features: `cuda`, `flash-attn`, `flash-attn-v3`, `cudnn`, `metal`, `accelerate`, `mkl`, `code-execution`, `ring`, `nccl`. Full list: [cargo features reference](/mistral.rs/reference/cargo-features/).
