---
title: CLI reference
description: Subcommands and flags of the mistralrs binary.
sidebar:
  order: 1
---

This page documents what the binary actually exposes. For complete and current help, run `mistralrs --help` and `mistralrs <subcommand> --help`.

## Subcommands

| Subcommand | Purpose |
|---|---|
| `mistralrs run` | Load a model and open an interactive chat (or one-shot with `-i`). |
| `mistralrs serve` | Load a model and expose OpenAI-compatible and Anthropic-compatible HTTP APIs. |
| `mistralrs bench` | Benchmark a model. |
| `mistralrs tune` | Recommend a quantization and device-mapping configuration. |
| `mistralrs quantize` | Generate UQFF files from a model. |
| `mistralrs from-config` | Load and run from a TOML configuration file. |
| `mistralrs login` | Save a Hugging Face token to `~/.cache/huggingface/token`. |
| `mistralrs doctor` | Report system, hardware, and build information. |
| `mistralrs cache` | List or delete Hugging Face cache entries. |
| `mistralrs completions` | Generate shell completions (bash, zsh, fish, elvish, powershell). |

## Global flags

| Flag | Default | Purpose |
|---|---|---|
| `--seed <int>` | not set | Sampling seed. |
| `-l`, `--log <path>` | not set | Log all requests/responses to a file. |
| `--token-source <source>` | `cache` | Token source: `literal:<token>`, `env:<var>`, `path:<file>`, `cache`, or `none`. |
| `-v`, `--verbose` | `0` | Increase startup logging. Use `-v` for debug details and `-vv` for trace-level file/cache internals. `RUST_LOG` overrides this. |

## Common model flags

Apply to subcommands that load or inspect a model (`serve`, `run`, `bench`, `tune`).

| Flag | Default | Purpose |
|---|---|---|
| `-m`, `--model-id <id>` | required | Hugging Face repo id or local path. |
| `-t`, `--tokenizer <path>` | not set | Local `tokenizer.json`. |
| `-a`, `--arch <arch>` | auto-detect | Model architecture. |
| `--dtype <dtype>` | `auto` | `auto`, `f16`, `bf16`, `f32`. |
| `--cpu` | off | Force CPU-only inference. |
| `-n`, `--device-layers <list>` | auto | Per-device layer counts. Format: `ORD:NUM;...` (e.g. `0:32;1:32`). |
| `--topology <path>` | not set | Topology YAML for per-layer placement and quantization. |
| `--hf-cache <path>` | not set | Custom Hugging Face cache directory. |
| `--max-seq-len <n>` | 4096 | Max sequence length used for automatic device mapping. |
| `--max-batch-size <n>` | 1 | Max batch size used for automatic device mapping. |

## Shared generation runtime flags

Accepted by `serve`, `run`, and `bench`; `tune` rejects them.

| Flag | Default | Purpose |
|---|---|---|
| `--no-kv-cache` | off | Disable KV cache. |
| `--matformer-config-path <path>` | not set | Path to a MatFormer slice config (CSV/JSON). |
| `--matformer-slice-name <name>` | not set | MatFormer slice to load. Requires `--matformer-config-path`. |

## Serve and run runtime flags

Accepted by `serve` and `run`; `bench` rejects them at startup because it measures plain model generation.

| Flag | Default | Purpose |
|---|---|---|
| `--max-seqs <n>` | 32 | Max concurrent sequences. |
| `--prefix-cache-n <n>` | 16 | Number of prefix caches to hold (0 to disable). |
| `-c`, `--chat-template <path>` | not set | Custom chat template (`.json` or `.jinja`). |
| `-j`, `--jinja-explicit <path>` | not set | Explicit Jinja template override. |
| `--mcp-config <path>` | not set | MCP client configuration for outbound servers. Also reads `MCP_CONFIG_PATH` if unset. |

## Format flags (Plain / GGUF / GGML)

| Flag | Default | Purpose |
|---|---|---|
| `--format <fmt>` | auto-detect | `plain`, `gguf`, or `ggml`. |
| `-f`, `--quantized-file <path>` | not set | Quantized filename(s) for GGUF/GGML. Semicolon-separated for multiple. |
| `--tok-model-id <id>` | not set | Tokenizer source for quantized formats. |
| `--gqa <n>` | 1 | GQA value for GGML. |

## Quantization flags

| Flag | Purpose |
|---|---|
| `--quant <value>` | Quantization front-door. Numeric (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names (e.g. `q4k`, `afq8`, `fp8`, `mxfp4`) prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is accepted by `serve`, `run`, and `bench`; `tune` rejects `auto` because it is the recommender. Conflicts with `--isq` and `--from-uqff`. |
| `--isq <type>` | Lower-level in-situ quantization knob (no UQFF lookup). Numeric (`2`, `3`, `4`, `5`, `6`, `8`) or format name (`q4k`, `afq4`, `q8_0`, etc.). |
| `--from-uqff <path>` | Load a pre-quantized UQFF file. |
| `--isq-organization <org>` | `default` or `moqe`. |
| `--imatrix <path>` | imatrix file. |
| `--calibration-file <path>` | Calibration data for imatrix generation. Conflicts with `--imatrix`. |

## Adapter flags

| Flag | Purpose |
|---|---|
| `--lora <ids>` | LoRA adapter id(s), semicolon-separated. |
| `--xlora <id>` | X-LoRA adapter id. Requires `--xlora-order`. Conflicts with `--lora`. |
| `--xlora-order <path>` | X-LoRA ordering JSON. |
| `--tgt-non-granular-index <n>` | X-LoRA target non-granular index. |

## Search and code execution

Accepted by `serve` and `run`. `bench` rejects these flags at startup.

| Flag | Default | Purpose |
|---|---|---|
| `--agent` (alias `--agentic`) | off | One-flag agent: equivalent to `--enable-search --enable-code-execution` with a per-session temp workdir. The agentic loop runs up to 256 tool rounds by default. |
| `--enable-search` | off | Enable the built-in web search tool. |
| `--search-embedding-model <name>` | not set | Reranker model. Only `embedding-gemma` is accepted. |
| `--enable-code-execution` | off | Enable Python code execution (compiled in by default). |
| `--code-exec-python <path>` | `python` on Windows, `python3` elsewhere | Python interpreter for code execution. |
| `--code-exec-timeout <secs>` | 30 | Code execution timeout in seconds. |
| `--code-exec-workdir <path>` | per-session temp dir | Code execution working directory. |
| `--agent-permission <mode>` | `auto` | `auto`, `ask`, or `deny`. Controls whether agent actions run automatically, require approval, or are denied. See [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions). `--code-exec-permission` is accepted as an alias. |

## Sandbox

OS-level isolation applied to the code-execution subprocess. See [sandbox reference](/mistral.rs/reference/sandbox/) for the layering and threat model.

| Flag | Default | Purpose |
|---|---|---|
| `--sandbox <mode>` | `auto` | `auto`, `on`, or `off`. `auto` enables on Linux/macOS, no-op on Windows. |
| `--sb-max-memory-mb <mb>` | 2048 | Per-session memory cap (`RLIMIT_AS`, plus cgroup `memory.max` when available). |
| `--sb-max-cpu-secs <secs>` | 300 | Per-session CPU time cap (`RLIMIT_CPU`). |
| `--sb-max-procs <n>` | 64 | Per-session process/thread cap (`RLIMIT_NPROC`, plus cgroup `pids.max`). |
| `--sandbox-network <mode>` | `loopback` | `none`, `loopback`, or `full`. `none` denies `socket(2)` outright. |

## Paged attention flags

| Flag | Default | Purpose |
|---|---|---|
| `--paged-attn <mode>` | `auto` | `auto`, `on`, or `off`. |
| `--pa-context-len <n>` | not set | Allocate KV cache for this context length. |
| `--pa-memory-mb <mb>` | not set | GPU memory in MB for KV cache. Conflicts with `--pa-context-len`. |
| `--pa-memory-fraction <f>` | not set | GPU memory utilization fraction (0.0 to 1.0). |
| `--pa-block-size <n>` | not set | Tokens per block. |
| `--pa-cache-type <type>` | `auto` | KV cache quantization type. |

## Multimodal flags

| Flag | Purpose |
|---|---|
| `--max-edge <px>` | Max edge length for image resizing (aspect ratio preserved). |
| `--max-num-images <n>` | Max images per request. |
| `--max-image-length <px>` | Max image dimension for device mapping. |

## `mistralrs serve` flags

| Flag | Default | Purpose |
|---|---|---|
| `--host <ip>` | `0.0.0.0` | Bind address. |
| `-p`, `--port <port>` | 1234 | TCP port. |
| `--no-ui` | off | Disable the built-in web UI (mounted at `/ui` by default). |
| `--mcp-port <port>` | not set | Enable MCP server on a separate port. |
| `--max-tool-rounds <n>` | not set | Cap on agentic tool loop rounds. |
| `--tool-dispatch-url <url>` | not set | External URL for tool execution. |

CORS allowed origins and the request body limit (default 50 MB) are not exposed as CLI flags. They can be configured programmatically through `MistralRsServerRouterBuilder` in `mistralrs-server-core`.

## `mistralrs run` flags

| Flag | Purpose |
|---|---|
| `-i`, `--input <text>` | Send a single prompt non-interactively and exit. |
| `--image <path>` | Attach an image (repeatable, requires `-i`). |
| `--audio <path>` | Attach audio (repeatable, requires `-i`). |
| `--video <path>` | Attach video (repeatable, requires `-i`). |
| `--thinking [bool]` | Control thinking mode for models that support it. |

## `mistralrs bench` flags

| Flag | Default | Purpose |
|---|---|---|
| `--prompt-len <n>` | 512 | Prompt length per iteration. |
| `--gen-len <n>` | 128 | Generation length per iteration. |
| `--iterations <n>` | 3 | Number of measured runs to average. |
| `--warmup <n>` | 1 | Number of warmup runs (discarded). |

## `mistralrs tune` flags

| Flag | Default | Purpose |
|---|---|---|
| `--profile <p>` | `balanced` | `quality`, `balanced`, or `fast`. |
| `--json` | off | Emit JSON instead of a human-readable table. |
| `--emit-config <path>` | not set | Write the recommended settings as TOML. |

## `mistralrs quantize` flags

| Flag | Purpose |
|---|---|
| `--isq <types>` | ISQ levels to produce. Repeatable or comma-separated. |
| `-o`, `--output <path>` | Output file (single ISQ) or directory (multiple). |
| `--no-readme` | Skip README generation. |
| `--uqff-base-model <id>` | Base model id for the README. |
| `--uqff-repo-id <id>` | Hugging Face repo id for the README. |
| `--isq-organization <org>` | `default` or `moqe`. |
| `--imatrix <path>` / `--calibration-file <path>` | Quantization enhancement options. |

## `mistralrs from-config` flags

| Flag | Purpose |
|---|---|
| `-f`, `--file <path>` | TOML configuration file (required). |

## `mistralrs login` flags

| Flag | Purpose |
|---|---|
| `--token <token>` | Provide the token non-interactively. Must start with `hf_`. |

Without `--token`, the command prompts interactively. The token is saved to `~/.cache/huggingface/token` (or `$HF_HOME/token` if set).

## `mistralrs cache` subcommands

| Subcommand | Purpose |
|---|---|
| `mistralrs cache list` | List cached model entries. |
| `mistralrs cache delete -m <id>` | Remove a cache entry. |

## `mistralrs doctor` flags

Run `mistralrs doctor` after installation or when GPU acceleration, build features, or Hugging Face connectivity look wrong.

| Flag | Purpose |
|---|---|
| `--json` | Emit JSON instead of human-readable output. |

## Environment variables

Common ones:

| Variable | Purpose |
|---|---|
| `RUST_LOG` | Override the `tracing` log filter (e.g. `mistralrs_core=debug,tower_http=info`). |
| `HF_HOME` | Hugging Face cache root. |
| `HF_TOKEN` | Override the cached token at runtime. |
| `HF_HUB_OFFLINE` | `HF_HUB_OFFLINE=1` disables all network calls to the Hugging Face Hub; files are loaded from the local cache only. |
| `MCP_CONFIG_PATH` | MCP config path (alternative to `--mcp-config`). |
| `MISTRALRS_SANDBOX` | `auto`/`on`/`off`. Overrides the sandbox only when the resolved mode is `auto`; `on` and `off` win. |
| `MISTRALRS_CUDA_GRAPHS` | Set to `0` to disable CUDA decode graphs for supported paged-attention decode steps. |

See [environment variables](/mistral.rs/reference/environment-variables/) for the full list.
