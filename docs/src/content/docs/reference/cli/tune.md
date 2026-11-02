---
title: "mistralrs tune"
description: "Recommend a quantization level and device mapping that fit your hardware"
sidebar:
  order: 8
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Recommend a quantization level and device mapping that fit your hardware

```
mistralrs tune [OPTIONS] [COMMAND]
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
| `--quant <QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
| `--isq <IN_SITU_QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
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
| `--profile <PROFILE>` | `balanced` | What to favor when picking a quantization level<br>`quality`: Most precise level that fits, starting from 8-bit<br>`balanced`: Start from 6-bit, trading a little quality for memory and speed<br>`fast`: Start from 4-bit for the smallest, fastest model |
| `--json` | `false` | Output JSON instead of human-readable text |
| `--emit-config <EMIT_CONFIG>` |  | Write the recommended settings to this TOML file, for use with `mistralrs from-config` |

## mistralrs tune auto

Auto-detect model type (recommended)

```
mistralrs tune auto [OPTIONS] --model-id <MODEL_ID>
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
| `--quant <QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
| `--isq <IN_SITU_QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
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

## mistralrs tune text

Text generation model with explicit configuration

```
mistralrs tune text [OPTIONS] --model-id <MODEL_ID>
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
| `--quant <QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
| `--isq <IN_SITU_QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
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

## mistralrs tune multimodal

Multimodal model

```
mistralrs tune multimodal [OPTIONS] --model-id <MODEL_ID>
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
| `--quant <QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
| `--isq <IN_SITU_QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
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

## mistralrs tune diffusion

Image generation model (diffusion)

```
mistralrs tune diffusion [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune speech

Speech synthesis model

```
mistralrs tune speech [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs tune embedding

Embedding model

```
mistralrs tune embedding [OPTIONS] --model-id <MODEL_ID>
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
| `--quant <QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
| `--isq <IN_SITU_QUANT>` |  | Level to recommend if it fits: a bit width (`2`, `3`, `4`, `5`, `6`, `8`) or a named type such as `q4k`. `auto` is not accepted |
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
