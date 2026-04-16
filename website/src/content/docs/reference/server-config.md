---
title: Server configuration
description: Every server-level flag and config field, with defaults.
sidebar:
  order: 3
---

This is a short reference covering just the server-level settings. For the full TOML schema, see the [CLI TOML config reference](/mistral.rs/reference/cli-toml-config/). For prose on what each setting does, see the [HTTP server guide](/mistral.rs/guides/serve/http-server/) and the [production checklist](/mistral.rs/guides/deploy/production-checklist/).

## Binding

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--host` | `server.host` | `0.0.0.0` | Interface to bind. Use `127.0.0.1` for local-only. |
| `--port`, `-p` | `server.port` | `1234` | TCP port. |

## CORS

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--allowed-origin` (repeatable) | `server.allowed_origins` | `[]` (any) | CORS allow-list. Empty means allow any origin. |

Allowed methods and headers are not configurable; the server always allows `GET`, `POST`, `PUT`, `DELETE`, and `Content-Type` plus `Authorization` headers.

## Request size

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--max-body-limit` | `server.max_body_limit` | `52428800` (50 MB) | Maximum request body size in bytes. |

## Web UI

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--ui` | `server.ui` | false | Mount the web UI at `/ui`. |

## MCP

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--mcp` | `mcp.enabled` | false | Enable the MCP server endpoint. |
| `--mcp-transport` | `mcp.transport` | `http` | `stdio`, `http`, or `ws`. |
| `--mcp-port` | `mcp.port` | | Separate port for MCP over HTTP. |
| `--mcp-config` | `mcp.client_config` | | Path to MCP client config (outbound connections). |

## Agentic features

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--enable-search` | `features.enable_search` | false | Enable web search tool. |
| `--enable-code-execution` | `features.enable_code_execution` | false | Enable Python code execution. |
| `--max-tool-rounds` | `features.max_tool_rounds` | 10 | Cap on agentic tool loop rounds. |
| `--tool-dispatch-url` | `features.tool_dispatch_url` | | External URL for tool execution. |
| `--search-embedding-model` | `features.search_embedding_model` | embeddinggemma | Reranker model for web search. |
| `--code-working-dir` | `features.code_working_dir` | temp dir | Working directory for code execution. |
| `--code-timeout-secs` | `features.code_timeout_secs` | 30 | Code execution timeout. |

## Paged attention

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--paged-attn` | `paged_attention.enabled = true` | auto | Force paged attention on. |
| `--no-paged-attn` | `paged_attention.enabled = false` | auto | Force paged attention off. |
| `--paged-attn-gpu-mem` | `paged_attention.gpu_memory_mb` | auto | KV cache memory in MB. |
| `--paged-attn-block-size` | `paged_attention.block_size` | 16 (CUDA), 32 (Metal) | Block size in tokens. |

## Environment variables

| Variable | Meaning |
|---|---|
| `RUST_LOG` | `tracing` log filter. E.g. `debug`, `mistralrs_core=info`. |
| `HF_HOME` | Hugging Face cache root. |
| `HF_TOKEN` | Override cached auth token. |
| `CUDA_VISIBLE_DEVICES` | GPU selection. |
| `PORT` | Overridden by `--port`; used by some container images as a default. |
