---
title: TOML configuration
description: Schema for the config file mistralrs from-config reads, with the CLI flag each key maps to.
---

`mistralrs from-config -f <path>` reads a TOML file. The top-level `command` field selects `serve` or `run`. Every key maps to a CLI flag of the same subcommand; the mapping is listed per table below. For per-flag semantics, see the [generated CLI reference](/reference/cli/).

## Minimal example

```toml
command = "serve"

[server]
host = "0.0.0.0"
port = 1234

[[models]]
model_id = "Qwen/Qwen3-4B"

[models.quantization]
quant = "4"
```

`mistralrs from-config -f this.toml` runs the server.

Memory sizing and device placement are automatic. For optional overrides, see [paged attention and KV caching](/guides/perf/paged-attention/#memory-budget).

## Top-level fields

| Field | Type | Applies to | Purpose |
|---|---|---|---|
| `command` | string | both | `"serve"` or `"run"`. |
| `default_model_id` | string | serve | Model id treated as the default. Must match one of the `[[models]]` entries. |
| `thinking` | bool | run | Turns thinking on or off (alias `enable_thinking`). If this and `reasoning_effort` are both omitted, thinking is on. Maps to `--thinking`. |
| `reasoning_effort` | string | run | `off` (or `none`), `low`, `medium`, `high`, or `xhigh`. Omit to leave it unset. Maps to `--reasoning-effort`. |
| `adapter` | string | run | Alias of the LoRA adapter to start with, from `[models.adapter]`. Omit to run the base model. See [adapters](#adapters). |

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
| `max_seqs` | `--max-seqs` | 1 for `run`, 32 for `serve` | Max concurrent sequences. An explicit value overrides the command default. |
| `max_num_batched_tokens` | `--max-num-batched-tokens` | 4096 | Most tokens processed in one step, across all requests. Must be nonzero. |
| `max_prefill_chunk_tokens` | `--max-prefill-chunk-tokens` | 512 | Largest piece of a new prompt read in one step while other requests are generating. Models with recurrent layers, such as Qwen3.5, read every prompt in pieces up to this size. CUDA and Metal only. Must be nonzero. |
| `max_decode_steps_before_prefill` | `--max-decode-steps-before-prefill` | 8 | Most generation steps to run before a waiting prompt gets a turn. Must be nonzero. |
| `no_kv_cache` | `--no-kv-cache` | false | Disable KV cache entirely. |
| `prefix_cache_n` | `--prefix-cache-n` | up to 16 | How many prefixes the prefix cache keeps. Unset, mistral.rs may keep fewer on models with recurrent layers to fit memory. `0` turns prefix caching off. See [prefix caching](/guides/perf/throughput-tuning/#prefix-caching). |
| `chat_template` | `-c`, `--chat-template` | not set | Custom chat template file (`.json` or `.jinja`), applied to every model. Per-model `chat_template` in `[[models]]` overrides it. |
| `jinja_explicit` | `-j`, `--jinja-explicit` | not set | Explicit Jinja template override. Per-model `jinja_explicit` also exists. |
| `matformer_config_path` | `--matformer-config-path` | not set | MatFormer (nested-submodel) slice config (CSV/JSON). |
| `matformer_slice_name` | `--matformer-slice-name` | not set | MatFormer slice to load. Requires `matformer_config_path`. |
| `mtp_model` | `--mtp-model` | not set | [MTP (multi-token prediction)](/guides/perf/speculative-decoding/) assistant model id or path. |
| `mtp_n_predict` | `--mtp-n-predict` | not set | MTP draft tokens proposed per target step. |
| `mtp_draft_sampling` | `--mtp-draft-sampling` | `auto` | MTP draft policy: `auto`, `greedy`, or `probabilistic`. |
| `mcp_config` | `--mcp-config` | not set | [MCP (Model Context Protocol)](/guides/agents/connect-mcp-server/) client configuration JSON for outbound servers. Also reads `MCP_CONFIG_PATH` if unset. |
| `agent` | `--agent` (alias `--agentic`) | false | Shortcut for `enable_search = true` + `enable_code_execution = true` + `enable_shell = true`. |
| `enable_search` | `--enable-search` | false | Enable the built-in web search tool. |
| `search_embedding_model` | `--search-embedding-model` | not set | Search reranker; `embedding-gemma` is the only accepted value. Requires `enable_search` (or `agent`). |
| `enable_code_execution` | `--enable-code-execution` | false | Enable Python code execution. |
| `code_exec_python` | `--code-exec-python` | `python` on Windows, `python3` elsewhere | Python interpreter. Requires `enable_code_execution` (or `agent`). |
| `code_exec_timeout` | `--code-exec-timeout` | 60 | Per-call timeout in seconds. Requires `enable_code_execution` (or `agent`). |
| `code_exec_workdir` | `--code-exec-workdir` | per-session temp dir | Code execution working directory. Requires `enable_code_execution` (or `agent`). |
| `enable_shell` | `--enable-shell` | false | Enable the built-in shell tool for Responses `tools[*].type="shell"`. |
| `shell_path` | `--shell-path` | `/bin/sh` on Unix, `cmd` on Windows | Shell executable. Requires `enable_shell` (or `agent`). |
| `shell_timeout` | `--shell-timeout` | 600 | Per-call shell timeout in seconds. Requires `enable_shell` (or `agent`). |
| `shell_workdir` | `--shell-workdir` | per-session temp dir | Root directory for per-session shell working directories. Requires `enable_shell` (or `agent`). |
| `skills_dir` | `--skills-dir` | system temp dir | Directory for uploaded OpenAI-compatible Skills. Requires `enable_shell` (or `agent`). |
| `agent_permission` | `--agent-permission` | `auto` | `auto`, `ask`, or `deny`: whether model-requested agent actions run automatically, require approval, or are denied. `code_exec_permission` / `--code-exec-permission` are accepted as aliases. |

To tune `max_num_batched_tokens`, `max_prefill_chunk_tokens`, and `max_decode_steps_before_prefill`, see [prefill and decode scheduling](/guides/perf/throughput-tuning/#prefill-and-decode-scheduling).

## `[server]` section (serve only)

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `host` | `--host` | `0.0.0.0` | Bind address. |
| `port` | `-p`, `--port` | 1234 | TCP port. |
| `no_ui` | `--no-ui` | false | Disable the built-in web UI (mounted at `/ui` by default). |
| `mcp_port` | `--mcp-port` | not set | Also expose the loaded model as an MCP server on this port (JSON-RPC 2.0 at `POST /mcp`). See [serve over MCP](/guides/agents/expose-as-mcp/). |
| `max_tool_rounds` | `--max-tool-rounds` | not set | Default cap on agentic tool loop rounds. Per-request values from the HTTP API override it; the safety cap is 256 when unset. |
| `tool_dispatch_url` | `--tool-dispatch-url` | not set | URL to POST tool calls to for server-side execution. Only configurable server-side, never per-request. |
| `disable_access_log` | `--disable-access-log` | false | Disable info-level HTTP access logs. |
| `access_log_format` | `--access-log-format` | `text` | Access-log format: `text` or `json`. |
| `access_log_health` | `--access-log-health` | false | Include health, metrics, docs, and UI requests in HTTP access logs. |
| `disable_request_id_header` | `--disable-request-id-header` | false | Stop echoing `x-request-id` on responses. |
| `disable_metrics` | `--disable-metrics` | false | Disable Prometheus HTTP metrics and recorder installation. |

:::caution
The default `host = "0.0.0.0"` binds on all interfaces, exposing the server to your network. There is no built-in authentication. Set `host = "127.0.0.1"` for local-only access, or put an authenticating reverse proxy in front before exposing it.
:::

The MCP *client* configuration (`mcp_config`) lives under `[runtime]`, not `[server]`: it applies to `run` as well as `serve`.

## `[paged_attn]` section

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `mode` | `--paged-attn` | `auto` | `auto` (on for CUDA and Metal, off for CPU), `on`, or `off`. |
| `gpu_memory_utilization` | `--gpu-memory-utilization` | not set | Cap on the fraction of GPU memory used by weights, working memory, and KV cache together, such as `0.9`. Conflicts with `kv_cache`. |
| `kv_cache` | `--kv-cache` | not set | Fixed KV cache size shared by all requests, such as `"4GiB"` or `"65536tokens"`. Conflicts with `gpu_memory_utilization`. |
| `block_size` | `--pa-block-size` | 32 on CUDA and Metal | Tokens per block. |
| `cache_type` | `--pa-cache-type` | `auto` | KV cache data type: `auto`, or `f8e4m3` for an FP8 cache that uses half the memory of BF16. |

For automatic sizing, omit both `gpu_memory_utilization` and `kv_cache` (`kv_cache = "auto"` isn't accepted). Setting either one needs paging, so it can't be combined with `mode = "off"`.

To give the model a fixed cache budget, add this to the minimal example:

```toml
[paged_attn]
kv_cache = "4GiB"
```

The budget is shared by all requests, and it's the total across the model's GPUs. Use `max_model_len` in `[[models]]` to limit each request's context. See [memory budgets](/guides/perf/paged-attention/#memory-budget) for help choosing a setting.

## `[sandbox]` section

OS-level isolation for the code-execution subprocess. Mechanics and threat model: [sandbox reference](/reference/sandbox/).

| Field | CLI flag | Default | Purpose |
|---|---|---|---|
| `mode` | `--sandbox` | `auto` | `auto` (on for Linux/macOS, no-op elsewhere), `on` (missing isolation is a hard error), or `off`. |
| `profile` | `--sandbox-profile` | profile-dependent | `developer` for agent/code/shell tools, otherwise `restricted`. |
| `max_memory_mb` | `--sb-max-memory-mb` | 2048 | Per-session memory cap in MiB. |
| `max_cpu_secs` | `--sb-max-cpu-secs` | 600 | Per-session CPU time cap in seconds. When rlimits apply, this is raised before execution to at least the enabled code or shell timeout. |
| `max_procs` | `--sb-max-procs` | 64 | Per-session process/thread cap. |
| `network` | `--sandbox-network` | profile-dependent | `none`, `loopback`, or `full`. Defaults to `full` for `developer`, `loopback` for `restricted`. |

## `[[models]]` array

Each entry defines one loaded model.

| Field | Type | Required | Purpose |
|---|---|---|---|
| `kind` | enum | no | Defaults to `auto`. Set to `text`, `multimodal`, `diffusion`, `speech`, or `embedding` only to force a loader. |
| `model_id` | string | yes | Hugging Face id or local path. |
| `tokenizer` | path | no | Local tokenizer.json. |
| `arch` | enum | no | Architecture override (text models). |
| `dtype` | enum | no | `auto`, `f16`, `bf16`, `f32`. |
| `hf_overrides` | JSON object | no | Recursively merged Hugging Face config overrides. |
| `max_model_len` | integer | no | Runtime prompt-plus-output context limit. |
| `chat_template` | path | no | Chat template override for this model. |
| `jinja_explicit` | path | no | Jinja override for this model. |
| `matformer_config_path` | path | no | MatFormer slice config (CSV/JSON). |
| `matformer_slice_name` | string | no | MatFormer slice to load. |

Each `[[models]]` entry can carry nested sections whose field shapes mirror the corresponding CLI flags:

| Section | Purpose |
|---|---|
| `[models.format]` | Weight format and file overrides: `format`, `quantized_file`, `mmproj`, `tok_model_id`, and GGML `gqa`. |
| `[models.adapter]` | LoRA, legacy LoRA, or X-LoRA adapters. See [adapters](#adapters). |
| `[models.quantization]` | `quant` (same as `--quant`), `isq`, `from_uqff`, `isq_organization`, `imatrix`, `calibration_file`. See [quantize a model](/guides/quantization/quantize-a-model/). |
| `[models.device]` | Device placement: `cpu`, `device_layers`, `topology`, `hf_cache`. `cpu` must be consistent across every entry. |
| `[models.multimodal]` | Multimodal load-time caps (image/video/audio limits). |

### Adapters

`[models.adapter]` takes one of three adapter modes per model:

| Mode | Keys | Use it for |
|---|---|---|
| Dynamic LoRA | `enable_lora`, `lora`, `lora_max_adapters`, `lora_max_rank`, `lora_max_bytes` | Adapters chosen per request, on text models and on the language model of multimodal models, including GGUF. |
| Legacy LoRA | `legacy_lora`, `legacy_lora_order` | One fixed adapter on a GGML model or Phi-3 GGUF. |
| X-LoRA | `xlora`, `xlora_order`, `tgt_non_granular_index` | An X-LoRA checkpoint on a plain or GGML text model, or Phi-3 GGUF. |

The modes can't be combined, and each `*_order` key is required with its adapter key. Multimodal models take only dynamic LoRA, on the language model: vision, audio, and projector adapters aren't supported. See [LoRA adapters](/guides/customize/lora-adapters/) for details.

Dynamic LoRA adapters are picked per request. With `command = "run"`, the top-level `adapter` key picks the one to start with:

```toml
command = "run"
adapter = "code"

[[models]]
model_id = "Qwen/Qwen3-4B"

[models.adapter]
enable_lora = true
lora = [
  { alias = "code", source = "org/code-lora", revision = "refs/pr/7" },
  { alias = "math", source = "./math-adapter" },
]
lora_max_adapters = 16
lora_max_rank = 256
lora_max_bytes = 8589934592
```

- Each `lora` entry needs a unique, nonempty `alias` and a `source`: a Hugging Face repo or a local directory.
- `revision` sets the Hugging Face revision of a remote adapter, such as `refs/pr/7`. It defaults to `main`, not the base model's revision, and is ignored for local directories.
- `enable_lora = true` turns on dynamic LoRA with no adapter preloaded. Any `lora` entry turns it on too.
- `lora_max_adapters` (default 16), `lora_max_rank` (default 256), and `lora_max_bytes` (default 8 GiB) limit loaded adapters. See [adapter limits](/guides/customize/lora-adapters/#adapter-limits).

GGUF models take the same keys:

```toml
command = "serve"

[[models]]
model_id = "Qwen/Qwen2.5-0.5B-Instruct-GGUF"

[models.quantization]
quant = "4"

[models.adapter]
lora = [
  { alias = "philosophy", source = "closestfriend/brie-qwen2.5-0.5b" },
]
```

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
quant = "4"

[[models]]
model_id = "google/gemma-4-E4B-it"

[models.quantization]
quant = "4"
```

For a multimodal GGUF repo, `model_id` and `quant` are enough when the repo has a single compatible projector:

```toml
[[models]]
model_id = "unsloth/gemma-4-E4B-it-GGUF"

[models.quantization]
quant = "4"
```

To choose exact files, set them in `[models.format]`. `mmproj` picks the projector, and `tok_model_id` sets where the config, tokenizer, and processor come from. A `.gguf` filename is enough to select the GGUF format:

```toml
[[models]]
model_id = "unsloth/gemma-4-E4B-it-GGUF"

[models.format]
quantized_file = "gemma-4-E4B-it-Q4_K_M.gguf"
mmproj = "mmproj-BF16.gguf"
tok_model_id = "google/gemma-4-E4B-it"
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

- `quant` (`--quant`) sets a bit width such as `"4"`, a format name, or `"auto"`, and loads a prebuilt quantized copy when one exists. See [how mistral.rs chooses what to load](/guides/quantization/quantize-a-model/#how-mistralrs-chooses-what-to-load).
  - `"auto"` always quantizes: it picks the highest bit width that fits, from 6 bits down to 2. Omit `quant` for full precision.
  - `quant` conflicts with `isq` (`--isq`) and `from_uqff` (`--from-uqff`).
  - `mistralrs tune` needs an explicit level and rejects `"auto"`.
- `calibration_file` (`--calibration-file`) conflicts with `imatrix` (`--imatrix`).
- Dynamic LoRA, legacy LoRA, and X-LoRA are mutually exclusive. See [adapters](#adapters).
- `--matformer-slice-name` requires `--matformer-config-path`.
- `mistralrs run`: `--image`, `--video`, and `--audio` require `-i`/`--input`.
- `mistralrs bench`: `--prompt-len` and `--depth` accept comma-separated values for sweeps.
  - Each `--prompt-len` value produces a prefill measurement at that prompt length.
  - Each `--depth` value produces a decode measurement that prefills `depth` tokens and then generates `--gen-len` tokens.
  - `--depth` must be greater than 0 when `--gen-len` is greater than 0.

## Server behavior notes

- **Authentication.** mistral.rs does not implement authentication. Put a reverse proxy (nginx, Caddy, Traefik) in front for auth and TLS. OpenAI SDK clients send an `Authorization: Bearer ...` header because the SDK requires an API key. mistral.rs doesn't check it, so any value works.
- **Logging and metrics.** Access logs go to the server's stdout/stderr by default, with the request id, route, status, and latency. `GET /metrics` exposes [Prometheus metrics](/reference/metrics/) by default. See [observability](/guides/deploy/observability/).
- **Payload logging.** `-v` turns on debug logging and `-vv` trace logging. A `RUST_LOG` filter such as `RUST_LOG=mistralrs_core=debug,tower_http=info` overrides both. `-l <path>` logs all requests and responses to a file.

### CORS and request body limit

These have no CLI flag or TOML key. To change the allowed origins or the body limit, build the server yourself with `MistralRsServerRouterBuilder` from `mistralrs-server-core`.

| Setting | Default | Builder method |
|---|---|---|
| Allowed origins | Any | `with_allowed_origins` |
| Request body limit | 50 MiB | `with_max_body_limit` |
| Allowed methods | `GET`, `POST`, `PUT`, `DELETE` | fixed |
| Allowed headers | `Content-Type`, `Authorization`, `x-api-key`, `anthropic-version`, `anthropic-beta`, `x-request-id`, `request-id` | fixed |
| Exposed headers | `x-request-id`, `request-id` | fixed |
