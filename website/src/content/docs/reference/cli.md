---
title: CLI reference
description: Every subcommand and flag of the mistralrs binary.
sidebar:
  order: 1
---

Subcommands and flags. For conceptual coverage, see the [tutorials](/mistral.rs/tutorials/) and [guides](/mistral.rs/guides/).

## Subcommands

| Subcommand | Purpose |
|---|---|
| `mistralrs run` | Load a model and open an interactive chat. |
| `mistralrs serve` | Load a model and expose an OpenAI-compatible HTTP server. |
| `mistralrs bench` | Benchmark a model's throughput. |
| `mistralrs tune` | Measure quantization tradeoffs and recommend a configuration. |
| `mistralrs quantize` | Produce a UQFF file from an unquantized model. |
| `mistralrs from-config` | Load and run from a TOML configuration file. |
| `mistralrs login` | Save a Hugging Face authentication token. |
| `mistralrs doctor` | Report system, hardware, and build information. |
| `mistralrs cache` | List or delete Hugging Face cache entries. |
| `mistralrs completions` | Generate shell completions. |

Subcommands accept a `model_type` positional argument (`plain`, `multimodal`, `gguf`, etc.) followed by subcommand-specific options. The positional defaults to the model's required type and is rarely needed explicitly.

## Global flags

Apply to any subcommand that loads a model:

| Flag | Takes | Purpose |
|---|---|---|
| `-m`, `--model-id` | repo id | Hugging Face repository to load. |
| `--hf-revision` | revision | Specific revision (branch, tag, or commit SHA). |
| `-f`, `--filename` | path | File inside the repo (for GGUF, UQFF). |
| `--isq` | level | In-situ quantization. Numeric (`4`, `8`) or format (`q4k`, `afq4`). |
| `--from-uqff` | path | Load pre-quantized UQFF weights. |
| `--chat-template` | path | Override the auto-detected chat template. |
| `--jinja-explicit` | string | Inline chat template. |
| `--topology` | path | Per-layer placement and quantization YAML. |
| `--num-device-layers` | list | Manual per-GPU layer counts. |
| `--paged-attn` / `--no-paged-attn` | | Force paged attention on or off. |
| `--paged-attn-gpu-mem` | MB | Memory budget for paged attention blocks. |
| `--paged-attn-block-size` | tokens | Block size for paged attention. |
| `--dtype` | dtype | Model weight dtype: `auto`, `f16`, `bf16`, `f32`. |
| `--cpu` | | Force CPU-only inference. |
| `--seed` | int | Random seed for sampling. |

## `mistralrs run` flags

Interactive chat-specific:

| Flag | Takes | Purpose |
|---|---|---|
| `-i`, `--interactive-prompt` | string | Send the prompt non-interactively and exit. |
| `--image` | path | Attach an image (multimodal models only). |
| `--audio` | path | Attach audio. |
| `--video` | path | Attach video. |
| `--enable-search` | | Enable the web search tool. |
| `--enable-code-execution` | | Enable Python code execution. |
| `--code-working-dir` | path | Directory for code execution output. |
| `--code-timeout-secs` | int | Code execution timeout. Default 30. |

## `mistralrs serve` flags

HTTP server-specific (all `run` flags also apply):

| Flag | Takes | Default | Purpose |
|---|---|---|---|
| `--host` | ip | `0.0.0.0` | Bind address. |
| `--port`, `-p` | port | `1234` | Bind port. |
| `--allowed-origin` | origin | any | CORS allowed origin (repeatable). |
| `--max-body-limit` | bytes | 50 MB | Maximum request body size. |
| `--ui` | | off | Enable the built-in web UI at `/ui`. |
| `--mcp` | | off | Enable the MCP server endpoint. |
| `--mcp-transport` | stdio/http/ws | http | MCP transport when `--mcp` is on. |
| `--mcp-port` | port | | Separate port for MCP over HTTP. |
| `--mcp-config` | path | | MCP client configuration (outbound). |
| `--max-tool-rounds` | int | 10 | Cap on agentic tool loop rounds. |
| `--tool-dispatch-url` | url | | External URL for tool execution. |
| `--search-embedding-model` | repo id | embeddinggemma | Reranker model. |
| `--default-model-id` | model | | Default model for the `default` alias. |

## `mistralrs tune` flags

| Flag | Takes | Default | Purpose |
|---|---|---|---|
| `--profile` | quality/balanced/fast | balanced | Tuning profile. |
| `--emit-config` | path | | Write recommended settings as TOML. |
| `--json` | | | Machine-readable output. |

## `mistralrs bench` flags

| Flag | Takes | Default | Purpose |
|---|---|---|---|
| `--prompt-len` | tokens | 512 | Prompt length per iteration. |
| `--gen-len` | tokens | 128 | Generation length per iteration. |
| `--iterations` | count | 3 | Number of runs to average. |
| `--concurrency` | int | 1 | Concurrent requests per iteration. |

## `mistralrs quantize` flags

| Flag | Takes | Purpose |
|---|---|---|
| `--isq` | list | Comma-separated ISQ types (one file per type). |
| `--output` | path | Output file (single type) or directory (multiple types). |
| `--isq-organization` | default/moqe | Organization strategy. |
| `--imatrix` | path | Importance matrix for better quantization. |

## `mistralrs from-config` flags

| Flag | Takes | Purpose |
|---|---|---|
| `-f`, `--file` | path | TOML configuration file. Required. |
| `--selector` | path | Optional selector file for one-shot requests. |

## `mistralrs login` flags

| Flag | Takes | Purpose |
|---|---|---|
| `--token` | string | Provide the token non-interactively. |

Without `--token`, the command prompts for one.

## `mistralrs cache` subcommands

| Subcommand | Purpose |
|---|---|
| `mistralrs cache list` | List all cached model entries. |
| `mistralrs cache delete <id>` | Remove a specific cache entry. |

## Environment variables

| Variable | Purpose |
|---|---|
| `RUST_LOG` | tracing-based log level filter. |
| `HF_HOME` | Base directory for the Hugging Face cache. |
| `HF_TOKEN` | Override the cached token at runtime. |
| `CUDA_VISIBLE_DEVICES` | Restrict visible GPUs. |
| `CUDA_ROOT` | Non-standard CUDA toolkit location (build time). |

## Exit codes

| Code | Meaning |
|---|---|
| `0` | Success. |
| `1` | General error (flag parsing, runtime failure). |
| `2` | Model loading failure. |
| `130` | Killed by Ctrl+C. |
