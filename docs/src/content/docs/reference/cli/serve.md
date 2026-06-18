---
title: "mistralrs serve"
description: "Start HTTP/MCP server and (optionally) the UI at /ui"
sidebar:
  order: 2
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Start HTTP/MCP server and (optionally) the UI at /ui

```
mistralrs serve [OPTIONS] [COMMAND]
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` |  | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--format <FORMAT>` |  | Model format: plain (safetensors), gguf, or ggml Auto-detected if not specified Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID for tokenizer when using quantized format |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--lora <LORA>` |  | LoRA adapter model ID(s), semicolon-separated for multiple |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization front-door. Numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is for `serve`, `run`, and `bench`; `tune` rejects it because `tune` is the recommender. Use `--isq` for the explicit knob |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.) |
| `--from-uqff <FROM_UQFF>` |  | UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8) to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8). Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons to separate different quantizations |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache quantization type |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |
| `-p, --port <PORT>` | `1234` | HTTP server port |
| `--host <HOST>` | `0.0.0.0` | Bind address |
| `--no-ui` | `false` | Disable the built-in web UI (served at /ui by default) |
| `--mcp-port <MCP_PORT>` |  | Also expose the loaded model as an MCP server on this port (JSON-RPC 2.0 at POST /mcp) |
| `--max-tool-rounds <MAX_TOOL_ROUNDS>` |  | Default maximum tool-call rounds for the agentic loop. Per-request values from the HTTP API override this. Safety cap: 256 if unset |
| `--tool-dispatch-url <TOOL_DISPATCH_URL>` |  | URL to POST tool calls to for server-side execution. For security, this is only configurable server-side (not per-request via HTTP API) |
| `--disable-access-log` | `false` | Disable per-request HTTP access logs |
| `--access-log-format <ACCESS_LOG_FORMAT>` | `text` | Format for HTTP access logs Possible values: `text`, `json`. |
| `--access-log-health` | `false` | Include health, metrics, docs, and UI requests in HTTP access logs |
| `--disable-request-id-header` | `false` | Disable the x-request-id response header |
| `--disable-metrics` | `false` | Disable Prometheus HTTP metrics and the metrics recorder |
| `--max-seqs <MAX_SEQS>` | `32` | Maximum concurrent sequences |
| `--no-kv-cache` | `false` | Disable KV cache entirely |
| `--prefix-cache-n <PREFIX_CACHE_N>` | `16` | Number of prefix caches to hold (0 to disable) |
| `-c, --chat-template <CHAT_TEMPLATE>` |  | Custom chat template file (.json or .jinja) |
| `-j, --jinja-explicit <JINJA_EXPLICIT>` |  | Explicit JINJA template override |
| `--matformer-config-path <MATFORMER_CONFIG_PATH>` |  | Path to a MatFormer config (CSV/JSON describing available slices). See model card |
| `--matformer-slice-name <MATFORMER_SLICE_NAME>` |  | MatFormer slice to load (must match a slice name in the config file) |
| `--mtp-model <MTP_MODEL>` |  | MTP assistant model id or path |
| `--mtp-n-predict <MTP_N_PREDICT>` |  | Number of MTP draft tokens to propose per target step |
| `--mcp-config <MCP_CONFIG>` |  | Path to an MCP client configuration JSON. Also reads `MCP_CONFIG_PATH` if unset |
| `--agent` | `false` | Build a local agent: enables web search, Python code execution, and shell execution, runs the agentic tool loop with a per-session temp workdir. Equivalent to passing `--enable-search --enable-code-execution --enable-shell` together |
| `--enable-search` | `false` | Enable web search (requires embedding model) |
| `--search-embedding-model <SEARCH_EMBEDDING_MODEL>` |  | Search embedding model to use. Requires `--enable-search` or `--agent` Possible values: `embedding-gemma`. |
| `--enable-code-execution` | `false` | Enable Python code execution tool (WARNING: allows arbitrary code execution) |
| `--enable-shell` | `false` | Enable shell execution tool (WARNING: allows arbitrary command execution) |
| `--code-exec-python <CODE_EXEC_PYTHON>` |  | Python interpreter path for code execution. Requires code execution to be on (via `--enable-code-execution` or `--agent`). Defaults to `python3` |
| `--code-exec-timeout <CODE_EXEC_TIMEOUT>` |  | Code execution timeout in seconds (default: 30). Requires code execution to be on |
| `--code-exec-workdir <CODE_EXEC_WORKDIR>` |  | Working directory for code execution. Defaults to a temp dir; use "." for cwd. Requires code execution to be on |
| `--shell-path <SHELL_PATH>` |  | Shell executable path. Requires shell execution to be on. Defaults to /bin/sh |
| `--shell-timeout <SHELL_TIMEOUT>` |  | Shell execution timeout in seconds (default: 30). Requires shell execution to be on |
| `--shell-workdir <SHELL_WORKDIR>` |  | Root directory for per-session shell working directories. Defaults to temp dirs |
| `--skills-dir <SKILLS_DIR>` |  | Directory for uploaded OpenAI-compatible Skills. Defaults to the system temp directory |
| `--agent-permission <PERMISSION>` | `auto` | Agent action permission mode Possible values: `auto`, `ask`, `deny`. |
| `--sandbox <MODE>` | `auto` | Sandbox mode Possible values: `auto`, `on`, `off`. |
| `--sandbox-profile <PROFILE>` |  | Sandbox policy profile Possible values: `restricted`, `developer`. |
| `--sb-max-memory-mb <MEMORY_MB>` |  | Per-session memory cap in MiB (default: 2048) |
| `--sb-max-cpu-secs <CPU_SECS>` |  | Per-session CPU time cap in seconds (default: 300) |
| `--sb-max-procs <PROCS>` |  | Per-session process/thread cap (default: 64) |
| `--sandbox-network <NETWORK>` |  | Network access permitted to the sandboxed session Possible values: `none`, `loopback`, `full`. |

## mistralrs serve auto

Auto-detect model type (recommended)

```
mistralrs serve auto [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--format <FORMAT>` |  | Model format: plain (safetensors), gguf, or ggml Auto-detected if not specified Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID for tokenizer when using quantized format |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--lora <LORA>` |  | LoRA adapter model ID(s), semicolon-separated for multiple |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization front-door. Numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is for `serve`, `run`, and `bench`; `tune` rejects it because `tune` is the recommender. Use `--isq` for the explicit knob |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.) |
| `--from-uqff <FROM_UQFF>` |  | UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8) to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8). Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons to separate different quantizations |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache quantization type |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |

## mistralrs serve text

Text generation model with explicit configuration

```
mistralrs serve text [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--format <FORMAT>` |  | Model format: plain (safetensors), gguf, or ggml Auto-detected if not specified Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID for tokenizer when using quantized format |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--lora <LORA>` |  | LoRA adapter model ID(s), semicolon-separated for multiple |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization front-door. Numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is for `serve`, `run`, and `bench`; `tune` rejects it because `tune` is the recommender. Use `--isq` for the explicit knob |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.) |
| `--from-uqff <FROM_UQFF>` |  | UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8) to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8). Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons to separate different quantizations |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache quantization type |

## mistralrs serve multimodal

Multimodal model

```
mistralrs serve multimodal [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--format <FORMAT>` |  | Model format: plain (safetensors), gguf, or ggml Auto-detected if not specified Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID for tokenizer when using quantized format |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--lora <LORA>` |  | LoRA adapter model ID(s), semicolon-separated for multiple |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization front-door. Numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is for `serve`, `run`, and `bench`; `tune` rejects it because `tune` is the recommender. Use `--isq` for the explicit knob |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.) |
| `--from-uqff <FROM_UQFF>` |  | UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8) to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8). Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons to separate different quantizations |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache quantization type |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Maximum image dimension for device mapping |

## mistralrs serve diffusion

Image generation model (diffusion)

```
mistralrs serve diffusion [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |

## mistralrs serve speech

Speech synthesis model

```
mistralrs serve speech [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |

## mistralrs serve embedding

Embedding model

```
mistralrs serve embedding [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | HuggingFace model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Model data type |
| `--format <FORMAT>` |  | Model format: plain (safetensors), gguf, or ggml Auto-detected if not specified Possible values: `plain`, `gguf`, `ggml`. |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | Quantized model filename(s) for GGUF/GGML (semicolon-separated for multiple) |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID for tokenizer when using quantized format |
| `--gqa <GQA>` | `1` | GQA value for GGML models |
| `--quant <QUANT>` |  | Quantization front-door. Numeric levels (`2`, `3`, `4`, `5`, `6`, `8`) and ISQ names prefer a prebuilt UQFF from `mistralrs-community/<model>-UQFF`, then fall back to ISQ. `auto` is for `serve`, `run`, and `bench`; `tune` rejects it because `tune` is the recommender. Use `--isq` for the explicit knob |
| `--isq <IN_SITU_QUANT>` |  | In-situ quantization level (e.g., "4", "8", "q4_0", "q4_1", etc.) |
| `--from-uqff <FROM_UQFF>` |  | UQFF file(s) to load from. Accepts numeric shorthands (2, 3, 4, 5, 6, 8) to auto-detect the appropriate UQFF file (e.g., `--from-uqff 8` finds q8_0-0.uqff or afq8-0.uqff). Also accepts ISQ type names (e.g., q4k, afq8). Shards are auto-discovered: specifying the first shard (e.g., q4k-0.uqff) automatically finds q4k-1.uqff, etc. Use semicolons to separate different quantizations |
| `--isq-organization <ISQ_ORGANIZATION>` |  | ISQ organization strategy: default or moqe |
| `--imatrix <IMATRIX>` |  | imatrix file for enhanced quantization |
| `--calibration-file <CALIBRATION_FILE>` |  | Calibration file for imatrix generation |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Device layer mapping (format: ORD:NUM;... e.g., "0:10;1:20") Omit for automatic device mapping |
| `--topology <TOPOLOGY>` |  | Topology YAML file for device mapping |
| `--hf-cache <HF_CACHE>` |  | Custom HuggingFace cache directory |
| `--max-seq-len <MAX_SEQ_LEN>` | `4096` | Max sequence length for automatic device mapping |
| `--max-batch-size <MAX_BATCH_SIZE>` | `1` | Max batch size for automatic device mapping |
| `--paged-attn <MODE>` | `auto` | PagedAttention mode - auto: enabled on CUDA, disabled on Metal/CPU (default) - on: force enable (fails if unsupported) - off: force disable Possible values: `auto`, `on`, `off`. |
| `--pa-context-len <CONTEXT_LEN>` |  | Allocate KV cache for this context length. If not specified, defaults to using 90% of available VRAM |
| `--pa-memory-mb <MEMORY_MB>` |  | GPU memory to allocate in MBs (alternative to context-len) |
| `--pa-memory-fraction <MEMORY_FRACTION>` |  | GPU memory utilization fraction 0.0-1.0 (alternative to context-len/memory-mb) |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per block (default: 32 on CUDA) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache quantization type |

