---
title: "mistralrs download"
description: "Download the files needed to run a model without loading it"
sidebar:
  order: 9
---

<!-- Generated from clap definitions by mistralrs-cli docgen. Do not edit. -->

Download the files needed to run a model without loading it or initializing a GPU.

Use an explicit --quant level; --quant auto is not supported. In-situ quantization happens when the model is loaded.

```
mistralrs download [OPTIONS] [COMMAND]
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
| `--revision <REVISION>` |  | Hugging Face revision to download (defaults to main) |

## mistralrs download auto

Auto-detect model type (recommended)

```
mistralrs download auto [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs download text

Text generation model with explicit configuration

```
mistralrs download text [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs download multimodal

Multimodal model

```
mistralrs download multimodal [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs download diffusion

Image generation model (diffusion)

```
mistralrs download diffusion [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs download speech

Speech synthesis model

```
mistralrs download speech [OPTIONS] --model-id <MODEL_ID>
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

## mistralrs download embedding

Embedding model

```
mistralrs download embedding [OPTIONS] --model-id <MODEL_ID>
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
