---
title: "mistralrs run"
description: "Chat with a model interactively, or send a single prompt with `-i`"
sidebar:
  order: 3
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Chat with a model interactively, or send a single prompt with `-i`

```
mistralrs run [OPTIONS] [COMMAND]
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` |  | Hugging Face model ID or local model directory; optional when `-f` names local files |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--format <FORMAT>` |  | Model file format. Auto-detected from `-f` when omitted<br>`plain`: Safetensors weights<br>`gguf`: GGUF file<br>`ggml`: GGML file |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF vision or audio projector file(s), separated by semicolons. Auto-selected when there is only one |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID to take the config, tokenizer, and processor files from, for a GGUF or GGML model |
| `--gqa <GQA>` | `1` | Grouped-query attention (GQA) factor for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum LoRA adapters loaded at once. Separately caps adapter versions in memory, counting old versions still used by running requests |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--legacy-lora <SOURCE>` |  | Static LoRA adapter source for GGML or a Phi3 GGUF model |
| `--legacy-lora-order <LEGACY_LORA_ORDER>` |  | Ordering JSON file for a legacy raw GGUF or GGML LoRA adapter |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization level: a bit width (`2`, `3`, `4`, `5`, `6`, `8`), a named type such as `q4k` or `fp8`, or `auto` to pick one that fits your hardware. Uses a matching prebuilt file when one exists, otherwise quantizes at load time |
| `--isq <IN_SITU_QUANT>` |  | Quantize the weights at load time (in-situ quantization, ISQ): a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k` or `q8_0`. Unlike `--quant`, never swaps in a prebuilt file |
| `--from-uqff <FROM_UQFF>` |  | Prebuilt UQFF file to load: a filename, a bit width (`2`, `3`, `4`, `5`, `6`, `8`), or a type such as `q4k` or `afq8`. Sharded files load every shard; separate names with `;` to pick shards yourself |
| `--isq-organization <ISQ_ORGANIZATION>` |  | Which layers to quantize: `default` for all supported layers, or `moqe` for only the experts of a mixture-of-experts model, leaving the rest unquantized |
| `--imatrix <IMATRIX>` |  | Importance matrix (imatrix) file, which tells the quantizer which weights matter most and improves quality at low bit widths |
| `--calibration-file <CALIBRATION_FILE>` |  | Text file to compute an importance matrix from, instead of passing `--imatrix` |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--paged-attn <MODE>` | `auto` | Store the KV cache in PagedAttention blocks so more requests fit in memory<br>`auto`: On for CUDA and Metal, off for CPU<br>`on`: Always on; fails if the device does not support it<br>`off`: Always off |
| `--gpu-memory-utilization <GPU_MEMORY_UTILIZATION>` |  | Fraction of GPU memory (e.g. 0.9) to use for weights, activations, and the KV cache |
| `--kv-cache <KV_CACHE>` |  | Explicit KV cache capacity, such as 4GiB or 65536tokens |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per PagedAttention block (default: 32) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache data type: `auto` or `f8e4m3` (8-bit float, about half the memory) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Memory cap, in MiB, for cached image and audio encoder outputs |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Largest image side, in pixels, to reserve memory for when placing layers on devices |
| `--max-seqs <MAX_SEQS>` |  | Maximum concurrent sequences (defaults to 1 for run and 32 for serve) |
| `--max-num-batched-tokens <MAX_NUM_BATCHED_TOKENS>` | `4096` | Maximum tokens processed per step when PagedAttention is on |
| `--max-prefill-chunk-tokens <MAX_PREFILL_CHUNK_TOKENS>` | `512` | Largest prompt chunk processed per step while other requests are generating (CUDA and Metal). Lower values keep running generations smooth; higher values give new requests a faster first token. Recurrent models always prefill in chunks of this size |
| `--max-decode-steps-before-prefill <MAX_DECODE_STEPS_BEFORE_PREFILL>` | `8` | Generation steps to run before pausing to process a waiting prompt. Higher values keep running generations smooth; lower values give new requests a faster first token |
| `--no-kv-cache` | `false` | Disable KV cache entirely |
| `--prefix-cache-n <PREFIX_CACHE_N>` |  | Number of prefix caches to hold (0 to disable; defaults to up to 16 within the memory budget) |
| `-c, --chat-template <CHAT_TEMPLATE>` |  | Custom chat template file (.json or .jinja) |
| `-j, --jinja-explicit <JINJA_EXPLICIT>` |  | `.jinja` chat template file that always replaces the model's own template |
| `--matformer-config-path <MATFORMER_CONFIG_PATH>` |  | Path to a MatFormer config (CSV/JSON describing available slices). See model card |
| `--matformer-slice-name <MATFORMER_SLICE_NAME>` |  | MatFormer slice to load (must match a slice name in the config file) |
| `--mtp` | `false` | Speed up generation with multi-token prediction (MTP), using the draft head built into the checkpoint |
| `--mtp-model <MTP_MODEL>` |  | Separate MTP draft model, as a Hugging Face ID or local path |
| `--mtp-n-predict <MTP_N_PREDICT>` |  | Fixed MTP draft tokens per step; omit to adapt the depth automatically |
| `--mtp-draft-sampling <MTP_DRAFT_SAMPLING>` | `auto` | How the MTP drafter picks the tokens it proposes<br>`auto`: Probabilistic when the drafter supports it, otherwise greedy<br>`greedy`: Always propose the most likely token<br>`probabilistic`: Sample proposals from the draft distribution; fails if unsupported |
| `--mcp-config <MCP_CONFIG>` |  | Path to an MCP client configuration JSON. Also reads `MCP_CONFIG_PATH` if unset |
| `--agent` | `false` | Run as a local agent that can search the web and run Python and shell commands, each session in its own temp directory. Same as `--enable-search --enable-code-execution --enable-shell` |
| `--enable-search` | `false` | Give the model a web search tool. Also loads a small embedding model to rank results |
| `--search-embedding-model <SEARCH_EMBEDDING_MODEL>` |  | Embedding model that ranks search results. Requires `--enable-search` or `--agent`. Possible values: `embedding-gemma`. |
| `--enable-code-execution` | `false` | Enable Python code execution tool (WARNING: allows arbitrary code execution) |
| `--enable-shell` | `false` | Enable shell execution tool (WARNING: allows arbitrary command execution) |
| `--code-exec-python <CODE_EXEC_PYTHON>` |  | Python interpreter path for code execution. Requires code execution to be on (via `--enable-code-execution` or `--agent`). Defaults to `python3` |
| `--code-exec-timeout <CODE_EXEC_TIMEOUT>` |  | Code execution timeout in seconds (default: 60). Requires code execution to be on |
| `--code-exec-workdir <CODE_EXEC_WORKDIR>` |  | Working directory for code execution. Defaults to a temp dir; use "." for cwd. Requires code execution to be on |
| `--shell-path <SHELL_PATH>` |  | Shell executable path. Requires shell execution to be on. Defaults to /bin/sh |
| `--shell-timeout <SHELL_TIMEOUT>` |  | Shell execution timeout in seconds (default: 600). Requires shell execution to be on |
| `--shell-workdir <SHELL_WORKDIR>` |  | Root directory for per-session shell working directories. Defaults to temp dirs |
| `--skills-dir <SKILLS_DIR>` |  | Directory for uploaded OpenAI-compatible Skills. Defaults to the system temp directory |
| `--agent-permission <PERMISSION>` | `auto` | Approval policy for the agent's tool calls<br>`auto`: Run tool calls without asking<br>`ask`: Ask for approval before each tool call<br>`deny`: Block all tool calls |
| `--sandbox <MODE>` | `auto` | Sandbox for the processes started by the code and shell tools<br>`auto`: Sandbox on Linux and macOS, skipping protections the OS lacks; no sandbox on Windows<br>`on`: Always sandbox; fail if any protection is unavailable<br>`off`: No sandbox |
| `--sandbox-profile <PROFILE>` |  | Files and network the sandbox allows. Defaults to `developer` when code or shell tools are on, otherwise `restricted`<br>`restricted`: Session directory and system files only<br>`developer`: Also reads installed toolchains (PATH, Homebrew, nvm, conda, and similar) with full network |
| `--sb-max-memory-mb <MEMORY_MB>` |  | Per-session memory cap in MiB (default: 2048) |
| `--sb-max-cpu-secs <CPU_SECS>` |  | Per-session CPU time cap in seconds (default: 600). Raised to at least enabled code/shell timeouts |
| `--sb-max-procs <PROCS>` |  | Per-session process/thread cap (default: 64) |
| `--sandbox-network <NETWORK>` |  | Network access inside the sandbox. Defaults to `loopback` for the restricted profile and `full` for developer<br>`none`: No network access<br>`loopback`: Localhost only<br>`full`: Unrestricted network access |
| `--thinking <THINKING>` |  | Turn thinking on or off for models that support it; `--thinking` alone means `true`. Thinking is on unless this or `--reasoning-effort` turns it off. Possible values: `true`, `false`. |
| `--reasoning-effort <REASONING_EFFORT>` |  | Set reasoning effort without changing the model's sampling parameters. Values are off, low, medium, high, or xhigh. "none" is an alias for off |
| `-i, --input <INPUT>` |  | One-shot text prompt. When provided, sends a single request and exits instead of entering interactive mode. Combine with --image, --video, or --audio for multimodal requests |
| `--image <IMAGE>` |  | Image URL(s) or file path(s) to include in the request (requires -i). Can be specified multiple times: --image img1.jpg --image img2.png |
| `--video <VIDEO>` |  | Video URL(s) or file path(s) to include in the request (requires -i). Can be specified multiple times: --video vid1.mp4 --video vid2.webm |
| `--audio <AUDIO>` |  | Audio URL(s) or file path(s) to include in the request (requires -i). Can be specified multiple times: --audio audio1.wav --audio audio2.mp3 |
| `--adapter <ADAPTER>` |  | LoRA adapter alias to use for requests. Omit to run the base model |

## mistralrs run auto

Auto-detect model type (recommended)

```
mistralrs run auto [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--format <FORMAT>` |  | Model file format. Auto-detected from `-f` when omitted<br>`plain`: Safetensors weights<br>`gguf`: GGUF file<br>`ggml`: GGML file |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF vision or audio projector file(s), separated by semicolons. Auto-selected when there is only one |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID to take the config, tokenizer, and processor files from, for a GGUF or GGML model |
| `--gqa <GQA>` | `1` | Grouped-query attention (GQA) factor for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum LoRA adapters loaded at once. Separately caps adapter versions in memory, counting old versions still used by running requests |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--legacy-lora <SOURCE>` |  | Static LoRA adapter source for GGML or a Phi3 GGUF model |
| `--legacy-lora-order <LEGACY_LORA_ORDER>` |  | Ordering JSON file for a legacy raw GGUF or GGML LoRA adapter |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization level: a bit width (`2`, `3`, `4`, `5`, `6`, `8`), a named type such as `q4k` or `fp8`, or `auto` to pick one that fits your hardware. Uses a matching prebuilt file when one exists, otherwise quantizes at load time |
| `--isq <IN_SITU_QUANT>` |  | Quantize the weights at load time (in-situ quantization, ISQ): a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k` or `q8_0`. Unlike `--quant`, never swaps in a prebuilt file |
| `--from-uqff <FROM_UQFF>` |  | Prebuilt UQFF file to load: a filename, a bit width (`2`, `3`, `4`, `5`, `6`, `8`), or a type such as `q4k` or `afq8`. Sharded files load every shard; separate names with `;` to pick shards yourself |
| `--isq-organization <ISQ_ORGANIZATION>` |  | Which layers to quantize: `default` for all supported layers, or `moqe` for only the experts of a mixture-of-experts model, leaving the rest unquantized |
| `--imatrix <IMATRIX>` |  | Importance matrix (imatrix) file, which tells the quantizer which weights matter most and improves quality at low bit widths |
| `--calibration-file <CALIBRATION_FILE>` |  | Text file to compute an importance matrix from, instead of passing `--imatrix` |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--paged-attn <MODE>` | `auto` | Store the KV cache in PagedAttention blocks so more requests fit in memory<br>`auto`: On for CUDA and Metal, off for CPU<br>`on`: Always on; fails if the device does not support it<br>`off`: Always off |
| `--gpu-memory-utilization <GPU_MEMORY_UTILIZATION>` |  | Fraction of GPU memory (e.g. 0.9) to use for weights, activations, and the KV cache |
| `--kv-cache <KV_CACHE>` |  | Explicit KV cache capacity, such as 4GiB or 65536tokens |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per PagedAttention block (default: 32) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache data type: `auto` or `f8e4m3` (8-bit float, about half the memory) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Memory cap, in MiB, for cached image and audio encoder outputs |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Largest image side, in pixels, to reserve memory for when placing layers on devices |

## mistralrs run text

Text generation model with explicit configuration

```
mistralrs run text [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--format <FORMAT>` |  | Model file format. Auto-detected from `-f` when omitted<br>`plain`: Safetensors weights<br>`gguf`: GGUF file<br>`ggml`: GGML file |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF vision or audio projector file(s), separated by semicolons. Auto-selected when there is only one |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID to take the config, tokenizer, and processor files from, for a GGUF or GGML model |
| `--gqa <GQA>` | `1` | Grouped-query attention (GQA) factor for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA without preloading an adapter. Supports compatible text and multimodal language models, including GGUF. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Supports compatible text and multimodal language models, including GGUF. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum LoRA adapters loaded at once. Separately caps adapter versions in memory, counting old versions still used by running requests |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--legacy-lora <SOURCE>` |  | Static LoRA adapter source for GGML or a Phi3 GGUF model |
| `--legacy-lora-order <LEGACY_LORA_ORDER>` |  | Ordering JSON file for a legacy raw GGUF or GGML LoRA adapter |
| `--xlora <XLORA>` |  | X-LoRA adapter model ID |
| `--xlora-order <XLORA_ORDER>` |  | X-LoRA ordering JSON file |
| `--tgt-non-granular-index <TGT_NON_GRANULAR_INDEX>` |  | Target non-granular index for X-LoRA |
| `--quant <QUANT>` |  | Quantization level: a bit width (`2`, `3`, `4`, `5`, `6`, `8`), a named type such as `q4k` or `fp8`, or `auto` to pick one that fits your hardware. Uses a matching prebuilt file when one exists, otherwise quantizes at load time |
| `--isq <IN_SITU_QUANT>` |  | Quantize the weights at load time (in-situ quantization, ISQ): a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k` or `q8_0`. Unlike `--quant`, never swaps in a prebuilt file |
| `--from-uqff <FROM_UQFF>` |  | Prebuilt UQFF file to load: a filename, a bit width (`2`, `3`, `4`, `5`, `6`, `8`), or a type such as `q4k` or `afq8`. Sharded files load every shard; separate names with `;` to pick shards yourself |
| `--isq-organization <ISQ_ORGANIZATION>` |  | Which layers to quantize: `default` for all supported layers, or `moqe` for only the experts of a mixture-of-experts model, leaving the rest unquantized |
| `--imatrix <IMATRIX>` |  | Importance matrix (imatrix) file, which tells the quantizer which weights matter most and improves quality at low bit widths |
| `--calibration-file <CALIBRATION_FILE>` |  | Text file to compute an importance matrix from, instead of passing `--imatrix` |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--paged-attn <MODE>` | `auto` | Store the KV cache in PagedAttention blocks so more requests fit in memory<br>`auto`: On for CUDA and Metal, off for CPU<br>`on`: Always on; fails if the device does not support it<br>`off`: Always off |
| `--gpu-memory-utilization <GPU_MEMORY_UTILIZATION>` |  | Fraction of GPU memory (e.g. 0.9) to use for weights, activations, and the KV cache |
| `--kv-cache <KV_CACHE>` |  | Explicit KV cache capacity, such as 4GiB or 65536tokens |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per PagedAttention block (default: 32) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache data type: `auto` or `f8e4m3` (8-bit float, about half the memory) |

## mistralrs run multimodal

Multimodal model

```
mistralrs run multimodal [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--format <FORMAT>` |  | Model file format. Auto-detected from `-f` when omitted<br>`plain`: Safetensors weights<br>`gguf`: GGUF file<br>`ggml`: GGML file |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF vision or audio projector file(s), separated by semicolons. Auto-selected when there is only one |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID to take the config, tokenizer, and processor files from, for a GGUF or GGML model |
| `--gqa <GQA>` | `1` | Grouped-query attention (GQA) factor for GGML models |
| `--enable-lora` | `false` | Enable dynamic LoRA for the language model without preloading an adapter. Vision, audio, and projector adapters are unsupported |
| `--lora <ALIAS=SOURCE\|JSON>` |  | Preload a language-model LoRA adapter as ALIAS=SOURCE. Remote adapters use revision main. May be repeated. Vision, audio, and projector adapters are unsupported |
| `--lora-max-adapters <LORA_MAX_ADAPTERS>` | `16` | Maximum LoRA adapters loaded at once. Separately caps adapter versions in memory, counting old versions still used by running requests |
| `--lora-max-rank <LORA_MAX_RANK>` | `256` | Maximum rank accepted for a LoRA adapter |
| `--lora-max-bytes <BYTES>` | `8589934592` | Maximum memory used by loaded adapters |
| `--quant <QUANT>` |  | Quantization level: a bit width (`2`, `3`, `4`, `5`, `6`, `8`), a named type such as `q4k` or `fp8`, or `auto` to pick one that fits your hardware. Uses a matching prebuilt file when one exists, otherwise quantizes at load time |
| `--isq <IN_SITU_QUANT>` |  | Quantize the weights at load time (in-situ quantization, ISQ): a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k` or `q8_0`. Unlike `--quant`, never swaps in a prebuilt file |
| `--from-uqff <FROM_UQFF>` |  | Prebuilt UQFF file to load: a filename, a bit width (`2`, `3`, `4`, `5`, `6`, `8`), or a type such as `q4k` or `afq8`. Sharded files load every shard; separate names with `;` to pick shards yourself |
| `--isq-organization <ISQ_ORGANIZATION>` |  | Which layers to quantize: `default` for all supported layers, or `moqe` for only the experts of a mixture-of-experts model, leaving the rest unquantized |
| `--imatrix <IMATRIX>` |  | Importance matrix (imatrix) file, which tells the quantizer which weights matter most and improves quality at low bit widths |
| `--calibration-file <CALIBRATION_FILE>` |  | Text file to compute an importance matrix from, instead of passing `--imatrix` |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--paged-attn <MODE>` | `auto` | Store the KV cache in PagedAttention blocks so more requests fit in memory<br>`auto`: On for CUDA and Metal, off for CPU<br>`on`: Always on; fails if the device does not support it<br>`off`: Always off |
| `--gpu-memory-utilization <GPU_MEMORY_UTILIZATION>` |  | Fraction of GPU memory (e.g. 0.9) to use for weights, activations, and the KV cache |
| `--kv-cache <KV_CACHE>` |  | Explicit KV cache capacity, such as 4GiB or 65536tokens |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per PagedAttention block (default: 32) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache data type: `auto` or `f8e4m3` (8-bit float, about half the memory) |
| `--encoder-cache-memory-mb <ENCODER_CACHE_MEMORY_MB>` |  | Memory cap, in MiB, for cached image and audio encoder outputs |
| `--max-edge <MAX_EDGE>` |  | Maximum edge length for image resizing (aspect ratio preserved) |
| `--max-num-images <MAX_NUM_IMAGES>` |  | Maximum number of images per request |
| `--max-image-length <MAX_IMAGE_LENGTH>` |  | Largest image side, in pixels, to reserve memory for when placing layers on devices |

## mistralrs run diffusion

Image generation model (diffusion)

```
mistralrs run diffusion [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |

## mistralrs run speech

Speech synthesis model

```
mistralrs run speech [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |

## mistralrs run embedding

Embedding model

```
mistralrs run embedding [OPTIONS] --model-id <MODEL_ID>
```

| Option | Default | Description |
|---|---|---|
| `-m, --model-id <MODEL_ID>` | required | Hugging Face model ID or local path to model directory |
| `-t, --tokenizer <TOKENIZER>` |  | Path to local tokenizer.json file |
| `-a, --arch <ARCH>` |  | Model architecture (auto-detected if not specified) |
| `--dtype <DTYPE>` | `auto` | Weight data type: `auto`, `bf16`, `f16`, or `f32`. `auto` picks the best one for the device |
| `--hf-overrides <HF_OVERRIDES>` |  | JSON to merge into the model's `config.json`; nested objects merge key by key |
| `--max-model-len <MAX_MODEL_LEN>` |  | Maximum context length, in tokens |
| `--format <FORMAT>` |  | Model file format. Auto-detected from `-f` when omitted<br>`plain`: Safetensors weights<br>`gguf`: GGUF file<br>`ggml`: GGML file |
| `-f, --quantized-file <QUANTIZED_FILE>` |  | GGUF/GGML filename(s); the suffix selects the format (semicolon-separated for multiple) |
| `--mmproj <MMPROJ>` |  | GGUF vision or audio projector file(s), separated by semicolons. Auto-selected when there is only one |
| `--tok-model-id <TOK_MODEL_ID>` |  | Model ID to take the config, tokenizer, and processor files from, for a GGUF or GGML model |
| `--gqa <GQA>` | `1` | Grouped-query attention (GQA) factor for GGML models |
| `--quant <QUANT>` |  | Quantization level: a bit width (`2`, `3`, `4`, `5`, `6`, `8`), a named type such as `q4k` or `fp8`, or `auto` to pick one that fits your hardware. Uses a matching prebuilt file when one exists, otherwise quantizes at load time |
| `--isq <IN_SITU_QUANT>` |  | Quantize the weights at load time (in-situ quantization, ISQ): a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k` or `q8_0`. Unlike `--quant`, never swaps in a prebuilt file |
| `--from-uqff <FROM_UQFF>` |  | Prebuilt UQFF file to load: a filename, a bit width (`2`, `3`, `4`, `5`, `6`, `8`), or a type such as `q4k` or `afq8`. Sharded files load every shard; separate names with `;` to pick shards yourself |
| `--isq-organization <ISQ_ORGANIZATION>` |  | Which layers to quantize: `default` for all supported layers, or `moqe` for only the experts of a mixture-of-experts model, leaving the rest unquantized |
| `--imatrix <IMATRIX>` |  | Importance matrix (imatrix) file, which tells the quantizer which weights matter most and improves quality at low bit widths |
| `--calibration-file <CALIBRATION_FILE>` |  | Text file to compute an importance matrix from, instead of passing `--imatrix` |
| `--cpu` | `false` | Force CPU-only execution |
| `-n, --device-layers <DEVICE_LAYERS>` |  | Layers per device as `ORD:NUM;...`, e.g. `0:10;1:20` puts 10 layers on GPU 0 and 20 on GPU 1. Omit for automatic mapping |
| `--topology <TOPOLOGY>` |  | YAML file setting the device and quantization for each layer range |
| `--hf-cache <HF_CACHE>` |  | Custom Hugging Face cache directory |
| `--paged-attn <MODE>` | `auto` | Store the KV cache in PagedAttention blocks so more requests fit in memory<br>`auto`: On for CUDA and Metal, off for CPU<br>`on`: Always on; fails if the device does not support it<br>`off`: Always off |
| `--gpu-memory-utilization <GPU_MEMORY_UTILIZATION>` |  | Fraction of GPU memory (e.g. 0.9) to use for weights, activations, and the KV cache |
| `--kv-cache <KV_CACHE>` |  | Explicit KV cache capacity, such as 4GiB or 65536tokens |
| `--pa-block-size <BLOCK_SIZE>` |  | Tokens per PagedAttention block (default: 32) |
| `--pa-cache-type <CACHE_TYPE>` | `auto` | KV cache data type: `auto` or `f8e4m3` (8-bit float, about half the memory) |
