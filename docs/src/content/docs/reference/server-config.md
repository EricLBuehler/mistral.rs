---
title: Server configuration
description: Server-level flags and config fields with defaults.
sidebar:
  order: 3
---

For the full TOML schema, see the [CLI TOML config reference](/mistral.rs/reference/cli-toml-config/). For prose, see the [HTTP server guide](/mistral.rs/guides/serve/http-server/).

## Binding

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--host` | `server.host` | `0.0.0.0` | Bind interface. |
| `-p`, `--port` | `server.port` | `1234` | TCP port. |

## Web UI

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--no-ui` | `server.no_ui` | false | Disable the built-in web UI (mounted at `/ui` by default). |

## MCP

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--mcp-port` | `server.mcp_port` | not set | Enable the MCP server on this port. |
| `--mcp-config` | `server.mcp_config` | not set | Path to MCP client config (outbound servers). |

## Agentic features

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--agent` (alias `--agentic`) | `runtime.agent` | false | Build a local agent: enables search and code execution with a per-session temp workdir. |
| `--enable-search` | `runtime.enable_search` | false | Enable web search tool. |
| `--enable-code-execution` | `runtime.enable_code_execution` | false | Enable Python code execution. |
| `--max-tool-rounds` | `server.max_tool_rounds` | 256 | Cap on agentic tool loop rounds. |
| `--tool-dispatch-url` | `server.tool_dispatch_url` | not set | External URL for tool execution. |
| `--search-embedding-model` | `runtime.search_embedding_model` | not set | Reranker for web search. Only `embedding-gemma` accepted. |
| `--code-exec-python` | `runtime.code_exec_python` | `python` on Windows, `python3` elsewhere | Python interpreter for code execution. |
| `--code-exec-workdir` | `runtime.code_exec_workdir` | per-session temp dir | Code execution working directory. |
| `--code-exec-timeout` | `runtime.code_exec_timeout` | 30 | Code execution timeout (seconds). |
| `--agent-permission` | `runtime.agent_permission` | `auto` | `auto`, `ask`, or `deny`. Controls whether agent actions run automatically, require approval, or are denied. See [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions). `--code-exec-permission` and `runtime.code_exec_permission` are accepted as aliases. |

## Paged attention

| CLI flag | TOML key | Default | Meaning |
|---|---|---|---|
| `--paged-attn` | `paged_attn.mode` | `auto` | `auto`, `on`, or `off`. |
| `--pa-context-len` | `paged_attn.context_len` | not set | KV cache context length. |
| `--pa-memory-mb` | `paged_attn.memory_mb` | not set | KV cache budget in MB. |
| `--pa-memory-fraction` | `paged_attn.memory_fraction` | not set | KV cache budget as a fraction of VRAM. |
| `--pa-block-size` | `paged_attn.block_size` | not set | Tokens per block. |
| `--pa-cache-type` | `paged_attn.cache_type` | `auto` | KV cache quantization type. |

## Not exposed via CLI

CORS allowed origins and the request body limit (default 50 MB) are configurable only programmatically through `MistralRsServerRouterBuilder` in `mistralrs-server-core`.

## Environment variables

| Variable | Meaning |
|---|---|
| `RUST_LOG` | Override the `tracing` log filter. CLI users can usually use `-v` or `-vv` instead. |
| `HF_HOME` | Hugging Face cache root. |
| `HF_TOKEN` | Override cached auth token. |
| `HF_HUB_OFFLINE` | `HF_HUB_OFFLINE=1` runs fully offline; only the local Hugging Face cache is consulted and no network calls are made. |
| `MCP_CONFIG_PATH` | Alternative to `--mcp-config`. |

Full list: [environment variables](/mistral.rs/reference/environment-variables/).
