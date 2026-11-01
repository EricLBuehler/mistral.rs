---
title: Choose a model
description: Find the model capability, loading format, and request API for your task.
---

Choose by the input and output you need, then check that the checkpoint fits your hardware.

| Task | Supported examples | Guide / API |
| --- | --- | --- |
| Chat, code, or tools | Qwen3, Flint, GPT-OSS | [Run models](/guides/models/run-any-model/): chat, Responses, native chat |
| Understand media | Qwen3-VL (images/video), LFM3-Omni (also audio) | [Multimodal input](/guides/models/multimodal-input/): chat content parts |
| Classify, score, extract, or locate objects | LiquidAI/d1-2B, LiquidAI/d1-Omni-2B | [Decision models](/guides/models/use-decision-models/): SystemOne / native decisions |
| Transcribe or generate speech | Voxtral Realtime (transcribe), LFM3-Audio (both), Dia (synthesize) | [Speech models](/guides/models/use-speech-models/): audio endpoints / chat audio |
| Search with text vectors | google/embeddinggemma-300m, Qwen/Qwen3-Embedding-0.6B | [Embedding models](/guides/models/use-embeddings/): embeddings |
| Generate images | black-forest-labs/FLUX.1-schnell | [Image generation](/guides/models/use-image-generation/): image generations |

The [supported-models inventory](/reference/supported-models/) groups supported architectures by model category. [Family notes](/guides/models/model-family-notes/) cover exceptions such as reasoning controls and template overrides.

## Choose a loading format

- **Original checkpoint:** pass its Hugging Face ID or local directory with `-m`. Add `--quant 4` to reduce memory through a matching UQFF or runtime quantization.
- **GGUF:** select a published artifact with `--quant`, or an exact file with `-f`. Check [GGUF compatibility](/reference/gguf-support/); it differs from safetensors support.
- **Saved UQFF:** reuse a quantized model without repeating conversion. See [quantization](/guides/quantization/quantize-a-model/).

Start with automatic memory sizing. Weights, context, concurrent requests, and media processing all consume memory; a model's parameter count alone does not determine whether it fits. Use [hardware tuning](/reference/cli/tune/) and [memory budgeting](/guides/perf/paged-attention/#memory-budget) when selecting a larger model.
