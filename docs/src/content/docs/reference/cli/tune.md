---
title: "mistralrs tune"
description: "Recommend quantization + device mapping for a model. Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias the recommendation toward a specific quantization target"
sidebar:
  order: 7
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Recommend quantization + device mapping for a model. Rejects `--quant auto`; pass `--quant <level>` or `--isq <level>` to bias the recommendation toward a specific quantization target

```
mistralrs tune [OPTIONS] [COMMAND]
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
| `--profile <PROFILE>` | `balanced` | Tuning profile (quality, balanced, fast) Possible values: `quality`, `balanced`, `fast`. |
| `--json` | `false` | Output JSON instead of human-readable text |
| `--emit-config <EMIT_CONFIG>` |  | Emit a TOML config file with the recommended settings |

## mistralrs tune auto

Auto-detect model type (recommended)

```
mistralrs tune auto [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune text

Text generation model with explicit configuration

```
mistralrs tune text [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune multimodal

Multimodal model

```
mistralrs tune multimodal [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune diffusion

Image generation model (diffusion)

```
mistralrs tune diffusion [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune speech

Speech synthesis model

```
mistralrs tune speech [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune embedding

Embedding model

```
mistralrs tune embedding [OPTIONS] --model-id <MODEL_ID>
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

