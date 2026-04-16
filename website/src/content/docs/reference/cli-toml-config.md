---
title: CLI TOML configuration
description: Schema for the config file that mistralrs from-config reads.
sidebar:
  order: 2
---

`mistralrs from-config -f <path>` reads a TOML file that describes everything a normal `mistralrs run` or `mistralrs serve` invocation would set on the command line. This page is the complete schema.

## Minimal example

```toml
model = "Qwen/Qwen3-4B"
isq = "4"

[server]
host = "0.0.0.0"
port = 1234
```

Running `mistralrs from-config -f this.toml` starts the server with those settings.

## Top-level fields

| Field | Type | Purpose |
|---|---|---|
| `model` | string | Hugging Face repo id (single-model mode). |
| `filename` | string | File within the repo (GGUF, UQFF). |
| `hf_revision` | string | Pin a model revision. |
| `isq` | string | ISQ level (`4`, `q4k`, `afq8`, etc.). |
| `chat_template` | path | Override the chat template. |
| `jinja_explicit` | string | Inline chat template. |
| `topology` | path | Per-layer topology YAML. |
| `dtype` | string | Weight dtype (`auto`, `f16`, `bf16`, `f32`). |
| `seed` | int | Sampling seed. |

## `[server]` section

| Field | Type | Default | Purpose |
|---|---|---|---|
| `host` | string | `0.0.0.0` | Bind address. |
| `port` | int | `1234` | Bind port. |
| `allowed_origins` | list | `[]` | CORS allowed origins. |
| `max_body_limit` | int | 50 MB | Maximum request body size in bytes. |
| `ui` | bool | false | Enable the built-in web UI. |

## `[features]` section

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enable_search` | bool | false | Enable web search tool. |
| `enable_code_execution` | bool | false | Enable Python code execution. |
| `max_tool_rounds` | int | 10 | Cap on agentic tool loop rounds. |
| `tool_dispatch_url` | string | | External URL for tool execution. |
| `search_embedding_model` | string | embeddinggemma | Reranker model. |
| `code_working_dir` | path | | Working directory for code execution. |
| `code_timeout_secs` | int | 30 | Code execution timeout. |

## `[paged_attention]` section

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | bool | auto | Force paged attention on or off. |
| `gpu_memory_mb` | int | auto | Memory budget for KV blocks. |
| `block_size` | int | auto | Block size in tokens. |

## `[sampling]` section

Defaults applied to requests that do not override each field:

| Field | Type | Purpose |
|---|---|---|
| `temperature` | float | Sampling temperature. |
| `top_p` | float | Nucleus sampling threshold. |
| `top_k` | int | Hard candidate cap. |
| `min_p` | float | Min-p threshold. |
| `presence_penalty` | float | Flat repetition penalty. |
| `frequency_penalty` | float | Frequency-weighted repetition penalty. |

## `[mcp]` section

| Field | Type | Default | Purpose |
|---|---|---|---|
| `enabled` | bool | false | Enable the MCP server endpoint. |
| `transport` | string | `http` | `stdio`, `http`, or `ws`. |
| `port` | int | | Separate port for MCP over HTTP. |
| `client_config` | path | | Path to MCP client config (outbound servers). |

## Multi-model: `[models]`

A table of `[models.<name>]` sections, each describing one model to load:

```toml
[models.qwen]
alias = "qwen"
in_situ_quant = "4"

[models.qwen.Plain]
model_id = "Qwen/Qwen3-4B"

[models.gemma]
alias = "gemma"
in_situ_quant = "4"

[models.gemma.MultimodalPlain]
model_id = "google/gemma-4-E4B-it"
```

The per-model fields:

| Field | Type | Purpose |
|---|---|---|
| `alias` | string | Name clients use in the `model` field. |
| `in_situ_quant` | string | ISQ level for this model. |
| `chat_template` | path | Override the chat template. |
| `jinja_explicit` | string | Inline chat template. |
| `num_device_layers` | list | Per-GPU layer counts. |

The nested `[models.<name>.<kind>]` section declares the model type. Options:

- `Plain`: standard text model.
- `MultimodalPlain`: multimodal model (vision, audio, video).
- `GGUF`: pre-quantized GGUF format.
- `GGML`: legacy GGML format.
- `Lora`: LoRA-adapted model.
- `XLora`: X-LoRA-adapted model.
- `Speech`: dedicated speech-to-text or text-to-speech.
- `DiffusionPlain`: image-generation model.
- `Embedding`: embedding model.

Each kind takes its own subfields. `Plain` and `MultimodalPlain` both take `model_id`. The other kinds have more specific fields; see the [supported models reference](/mistral.rs/reference/supported-models/) for each model's expected shape.

## `default_model_id`

At the top level:

```toml
default_model_id = "qwen"
```

Sets which model responds to requests that ask for `"default"` or omit the `model` field. Without this, the first model in the file's order is the default.

## Validation

On startup mistralrs validates the whole file before doing any loading. Unknown fields produce errors, not warnings. Type mismatches produce errors with the offending key named. If loading fails partway through (one model's weights cannot be found), the failure is reported and the remaining models are skipped.
