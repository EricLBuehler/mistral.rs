---
title: CLI TOML configuration
description: Schema for the config file that mistralrs from-config reads.
sidebar:
  order: 2
---

`mistralrs from-config -f <path>` reads a TOML file. The top level is tagged by the `command` field and selects either `serve` or `run`.

## Minimal example

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234

[[models]]
kind = "text"
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"
```

`mistralrs from-config -f this.toml` runs the server.

## Top-level fields

| Field | Type | Required | Purpose |
|---|---|---|---|
| `command` | string | yes | `"serve"` or `"run"`. |
| `default_model_id` | string | no (serve only) | Model id treated as the default. Must match one of the `[[models]]` entries. |

## `[global]` section

| Field | Type | Default | Purpose |
|---|---|---|---|
| `seed` | int | not set | Sampling seed. |
| `log` | path | not set | Log file for requests/responses. |
| `token_source` | string | `cache` | Token source string (`literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`). |

## `[runtime]` section

Most CLI runtime flags map to fields here. Notable ones:

| Field | Default | Purpose |
|---|---|---|
| `enable_search` | false | Enable web search tool. |
| `search_embedding_model` | not set | `embedding-gemma`. Requires `enable_search = true`. |
| `enable_code_execution` | false | Enable Python code execution. |
| `code_exec_python` | `python` on Windows, `python3` elsewhere | Python interpreter for code execution. |
| `code_exec_workdir` | per-session temp dir | Code execution working directory. |
| `code_exec_timeout` | 30 | Code execution timeout (seconds). |
| `max_seqs` | 32 | Max concurrent sequences. |
| `prefix_cache_n` | 16 | Prefix caches retained. |

## `[server]` section (serve only)

| Field | Type | Default | Purpose |
|---|---|---|---|
| `host` | string | `0.0.0.0` | Bind address. |
| `port` | u16 | 1234 | TCP port. |
| `ui` | bool | false | Mount the web UI at `/ui`. |
| `mcp_port` | u16 | not set | Enable MCP server on this port. |
| `mcp_config` | path | not set | MCP client configuration (outbound). |
| `max_tool_rounds` | int | not set | Cap on tool loop rounds. |
| `tool_dispatch_url` | string | not set | External URL for tool execution. |

## `[paged_attn]` section

| Field | Default | Purpose |
|---|---|---|
| `mode` | `auto` | `auto`, `on`, or `off`. |
| `context_len` | not set | KV cache context length. |
| `memory_mb` | not set | KV cache budget in MB. |
| `memory_fraction` | not set | KV cache budget as fraction of VRAM. |
| `block_size` | not set | Tokens per block. |
| `cache_type` | `auto` | KV cache quantization type. |

## `[[models]]` array

Each entry defines one loaded model.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `kind` | enum | yes | `auto`, `text`, `multimodal`, `diffusion`, `speech`, or `embedding`. |
| `model_id` | string | yes | Hugging Face id or local path. |
| `tokenizer` | path | no | Local tokenizer.json. |
| `arch` | enum | no | Architecture override (text models). |
| `dtype` | enum | no | `auto`, `f16`, `bf16`, `f32`. |
| `chat_template` | path | no | Chat template override. |
| `jinja_explicit` | path | no | Inline Jinja override. |
| `matformer_config_path` | path | no | Path to a MatFormer slice config (CSV/JSON). |
| `matformer_slice_name` | string | no | MatFormer slice to load. |

Per-model nested sections: `[models.format]`, `[models.adapter]`, `[models.quantization]`, `[models.device]`, `[models.multimodal]`. Field shapes mirror the corresponding CLI flags. `cpu` in `[models.device]` must be consistent across every entry.

## Multi-model example

```toml
command = "serve"
default_model_id = "Qwen/Qwen3-4B"

[server]
host = "0.0.0.0"
port = 1234

[runtime]
enable_search = true
search_embedding_model = "embedding-gemma"

[[models]]
kind = "text"
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"

[[models]]
kind = "multimodal"
model_id = "google/gemma-4-E4B-it"

[models.quantization]
in_situ_quant = "4"
```

## Validation

Configs are validated at startup. Invalid configs abort the run with a message identifying the problem. Validation includes:

- At least one entry in `[[models]]`.
- `default_model_id` matches a `model_id` in `[[models]]`.
- `cpu` is consistent across all models when set.
- `search_embedding_model` requires `enable_search = true`.
