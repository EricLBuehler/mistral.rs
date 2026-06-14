---
title: TOML configuration
description: Schema for the config file mistralrs from-config reads, with the CLI flag each key maps to.
---

`mistralrs from-config -f <path>` reads a TOML file. The top-level `command` field selects `serve` or `run`. Every key maps to a CLI flag of the same subcommand; the mapping is listed per table below. For per-flag semantics, see the [generated CLI reference](/mistral.rs/reference/cli/).

## Minimal example

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234

[[models]]
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"
```

`mistralrs from-config -f this.toml` runs the server.

## Top-level fields

| Field | Type | Applies to | Purpose |
|---|---|---|---|
| `command` | string | both | `"serve"` or `"run"`. |
| `default_model_id` | string | serve | Model id treated as the default. Must match one of the `[[models]]` entries. |
| `thinking` | bool | run | Force thinking mode on or off for models that support it (alias: `enable_thinking`). Omit to defer to the chat template default. Maps to `--thinking` on `mistralrs run`. |

## `[global]` section

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `seed` | `--seed` | not set | Sampling seed. |
| `log` | `-l`, `--log` | not set | Log file for requests/responses. |
| `token_source` | `--token-source` | `cache` | Token source string (`literal:<token>`, `env:<var>`, `path:<file>`, `cache`, `none`). |

`-v`/`--verbose` has no TOML equivalent; use `RUST_LOG` instead.

## `[runtime]` section

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `max_seqs` | `--max-seqs` | 32 | Max concurrent sequences. |
| `no_kv_cache` | `--no-kv-cache` | false | Disable KV cache entirely. |
| `prefix_cache_n` | `--prefix-cache-n` | 16 | Prefix caches retained (0 to disable). |
| `chat_template` | `-c`, `--chat-template` | not set | Custom chat template file (`.json` or `.jinja`), applied to every model. Per-model `chat_template` in `[[models]]` overrides it. |
| `jinja_explicit` | `-j`, `--jinja-explicit` | not set | Explicit Jinja template override. Per-model `jinja_explicit` also exists. |
| `matformer_config_path` | `--matformer-config-path` | not set | MatFormer (nested-submodel) slice config (CSV/JSON). |
| `matformer_slice_name` | `--matformer-slice-name` | not set | MatFormer slice to load. Requires `matformer_config_path`. |
| `mtp_model` | `--mtp-model` | not set | [MTP (multi-token prediction)](/mistral.rs/guides/perf/speculative-decoding/) assistant model id or path. |
| `mtp_n_predict` | `--mtp-n-predict` | not set | MTP draft tokens proposed per target step. |
| `mcp_config` | `--mcp-config` | not set | [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) client configuration JSON for outbound servers. Also reads `MCP_CONFIG_PATH` if unset. |
| `agent` | `--agent` (alias `--agentic`) | false | Shortcut for `enable_search = true` + `enable_code_execution = true`. |
| `enable_search` | `--enable-search` | false | Enable the built-in web search tool. |
| `search_embedding_model` | `--search-embedding-model` | not set | Search reranker; `embedding-gemma` is the only accepted value. Requires `enable_search` (or `agent`). |
| `enable_code_execution` | `--enable-code-execution` | false | Enable Python code execution. |
| `code_exec_python` | `--code-exec-python` | `python` on Windows, `python3` elsewhere | Python interpreter. Requires `enable_code_execution` (or `agent`). |
| `code_exec_timeout` | `--code-exec-timeout` | 30 | Per-call timeout in seconds. Requires `enable_code_execution` (or `agent`). |
| `code_exec_workdir` | `--code-exec-workdir` | per-session temp dir | Code execution working directory. Requires `enable_code_execution` (or `agent`). |
| `agent_permission` | `--agent-permission` | `auto` | `auto`, `ask`, or `deny`: whether model-requested agent actions run automatically, require approval, or are denied. `code_exec_permission` / `--code-exec-permission` are accepted as aliases. |

## `[server]` section (serve only)

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `host` | `--host` | `0.0.0.0` | Bind address. |
| `port` | `-p`, `--port` | 1234 | TCP port. |
| `no_ui` | `--no-ui` | false | Disable the built-in web UI (mounted at `/ui` by default). |
| `mcp_port` | `--mcp-port` | not set | Also expose the loaded model as an MCP server on this port (JSON-RPC 2.0 at `POST /mcp`). See [serve over MCP](/mistral.rs/guides/agents/expose-as-mcp/). |
| `max_tool_rounds` | `--max-tool-rounds` | not set | Default cap on agentic tool loop rounds. Per-request values from the HTTP API override it; the safety cap is 256 when unset. |
| `tool_dispatch_url` | `--tool-dispatch-url` | not set | URL to POST tool calls to for server-side execution. Only configurable server-side, never per-request. |

:::caution
The default `host = "0.0.0.0"` binds on all interfaces, exposing the server to your network. There is no built-in authentication. Set `host = "127.0.0.1"` for local-only access, or put an authenticating reverse proxy in front before exposing it.
:::

The MCP *client* configuration (`mcp_config`) lives under `[runtime]`, not `[server]`: it applies to `run` as well as `serve`.

## `[paged_attn]` section

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `mode` | `--paged-attn` | `auto` | `auto` (on for CUDA, off for Metal/CPU), `on`, or `off`. |
| `context_len` | `--pa-context-len` | not set | Allocate KV cache for this context length. |
| `memory_mb` | `--pa-memory-mb` | not set | KV cache budget in MB. Conflicts with `context_len`. |
| `memory_fraction` | `--pa-memory-fraction` | not set | KV cache budget as fraction of VRAM (0.0 to 1.0). Conflicts with `context_len` and `memory_mb`. |
| `block_size` | `--pa-block-size` | not set | Tokens per block. |
| `cache_type` | `--pa-cache-type` | `auto` | KV cache quantization type. |

## `[sandbox]` section

OS-level isolation for the code-execution subprocess. Mechanics and threat model: [sandbox reference](/mistral.rs/reference/sandbox/).

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `mode` | `--sandbox` | `auto` | `auto` (on for Linux/macOS, no-op elsewhere), `on` (missing isolation is a hard error), or `off`. |
| `max_memory_mb` | `--sb-max-memory-mb` | 2048 | Per-session memory cap in MiB. |
| `max_cpu_secs` | `--sb-max-cpu-secs` | 300 | Per-session CPU time cap in seconds. |
| `max_procs` | `--sb-max-procs` | 64 | Per-session process/thread cap. |
| `network` | `--sandbox-network` | `loopback` | `none`, `loopback`, or `full`. |

## `[[models]]` array

Each entry defines one loaded model.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `kind` | enum | no | Defaults to `auto`. Set to `text`, `multimodal`, `diffusion`, `speech`, or `embedding` only to force a loader. |
| `model_id` | string | yes | Hugging Face id or local path. |
| `tokenizer` | path | no | Local tokenizer.json. |
| `arch` | enum | no | Architecture override (text models). |
| `dtype` | enum | no | `auto`, `f16`, `bf16`, `f32`. |
| `chat_template` | path | no | Chat template override for this model. |
| `jinja_explicit` | path | no | Jinja override for this model. |
| `matformer_config_path` | path | no | MatFormer slice config (CSV/JSON). |
| `matformer_slice_name` | string | no | MatFormer slice to load. |

Each `[[models]]` entry can carry nested sections whose field shapes mirror the corresponding CLI flags:

| Section | Purpose |
|---|---|
| `[models.format]` | Weight format selection (e.g. GGUF file/repo). |
| `[models.adapter]` | LoRA/X-LoRA adapter configuration. |
| `[models.quantization]` | In-situ quantization and UQFF options (e.g. `in_situ_quant`). |
| `[models.device]` | Device placement: `cpu`, `device_layers`, `topology`, `hf_cache`, `max_seq_len`, `max_batch_size`. `cpu` must be consistent across every entry. |
| `[models.multimodal]` | Multimodal load-time caps (image/video/audio limits). |

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
model_id = "Qwen/Qwen3-4B"

[models.quantization]
in_situ_quant = "4"

[[models]]
model_id = "google/gemma-4-E4B-it"

[models.quantization]
in_situ_quant = "4"
```

## Validation

Invalid configs abort startup with a message identifying the problem:

- At least one entry in `[[models]]`.
- `default_model_id` matches a `model_id` in `[[models]]`.
- `cpu` is consistent across all models when set.
- `search_embedding_model` requires `enable_search = true` (or `agent = true`).
- `code_exec_python`, `code_exec_timeout`, and `code_exec_workdir` each require `enable_code_execution = true` (or `agent = true`).

## CLI usage notes

Flag interactions that hold on the command line and as TOML keys:

- `--quant` conflicts with `--isq` and `--from-uqff`; it is the front door that tries a prebuilt [UQFF (Universal Quantized File Format)](/mistral.rs/reference/uqff-format/) first and falls back to [ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/). `mistralrs tune` rejects `--quant auto` because `tune` is the recommender.
- `--calibration-file` conflicts with `--imatrix`.
- `--xlora` conflicts with `--lora`. `--xlora-order` and `--tgt-non-granular-index` require `--xlora`; `--xlora` alone is accepted.
- `--matformer-slice-name` requires `--matformer-config-path`.
- `mistralrs run`: `--image`, `--video`, and `--audio` require `-i`/`--input`.
- `mistralrs bench`: `--prompt-len` and `--depth` accept comma-separated values for sweeps.
  - Each `--prompt-len` value produces a prefill measurement at that prompt length.
  - Each `--depth` value produces a decode measurement that prefills `depth` tokens and then generates `--gen-len` tokens.
  - `--depth` must be greater than 0 when `--gen-len` is greater than 0.

## Server behavior notes

- **CORS and body limit.** Not exposed as CLI flags or TOML keys. Defaults: any origin; methods `GET`, `POST`, `PUT`, `DELETE`; allowed headers `Content-Type`, `Authorization`, `x-api-key`, `anthropic-version`, `anthropic-beta`; 50 MB request body limit. Configure programmatically through `MistralRsServerRouterBuilder` in `mistralrs-server-core`.
- **Authentication.** mistral.rs does not implement authentication. Put a reverse proxy (nginx, Caddy, Traefik) in front for auth and TLS. OpenAI-protocol clients always send `Authorization: Bearer ...` because the OpenAI SDK requires an API key; mistral.rs does not validate the header.
- **Logging.** `-v` enables debug detail and `-vv` trace-level file/cache internals; `RUST_LOG` module filters (e.g. `RUST_LOG=mistralrs_core=debug,tower_http=info`) override both. `-l <path>` logs all requests and responses to a file.
