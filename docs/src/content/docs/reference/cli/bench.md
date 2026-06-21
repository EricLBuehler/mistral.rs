---
title: "mistralrs bench"
description: "Run performance benchmarks for plain model generation"
sidebar:
  order: 11
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Run performance benchmarks for plain model generation

```
mistralrs bench [OPTIONS] [COMMAND]
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
| `--no-kv-cache` | `false` | Disable KV cache entirely |
| `--matformer-config-path <MATFORMER_CONFIG_PATH>` |  | Path to a MatFormer config (CSV/JSON describing available slices). See model card |
| `--matformer-slice-name <MATFORMER_SLICE_NAME>` |  | MatFormer slice to load (must match a slice name in the config file) |
| `--mtp-model <MTP_MODEL>` |  | MTP assistant model id or path |
| `--mtp-n-predict <MTP_N_PREDICT>` |  | Number of MTP draft tokens to propose per target step |
| `--prompt-len <PROMPT_LEN>` | `512` | Number of tokens in prompt. Accepts comma-separated values for sweeps |
| `--gen-len <GEN_LEN>` | `128` | Number of tokens to generate |
| `--depth <DEPTH>` | `4` | Number of prompt tokens to prefill before measuring decode. Accepts comma-separated values for sweeps |
| `--iterations <ITERATIONS>` | `3` | Number of benchmark iterations |
| `--warmup <WARMUP>` | `1` | Number of warmup runs (discarded) |

## mistralrs bench auto

Auto-detect model type (recommended)

```
mistralrs bench auto [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs bench text

Text generation model with explicit configuration

```
mistralrs bench text [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs bench multimodal

Multimodal model

```
mistralrs bench multimodal [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs bench diffusion

Image generation model (diffusion)

```
mistralrs bench diffusion [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs bench speech

Speech synthesis model

```
mistralrs bench speech [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs bench embedding

Embedding model

```
mistralrs bench embedding [OPTIONS] --model-id <MODEL_ID>
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

