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

- **Original checkpoint:** pass its Hugging Face ID or a local directory with `-m`. To use less memory, add `--quant` with a bit width of `2`, `3`, `4`, `5`, `6`, or `8`. Lower is smaller: `--quant 8` needs about half the memory of the original weights and `--quant 4` about a quarter. `--quant auto` picks the highest bit width, up to 6, that fits your hardware. To run at full precision, leave out `--quant`. Specific formats like `q4k` and `fp8` are covered in [quantize a model](/guides/quantization/quantize-a-model/).
- **GGUF:** pass a bit width with `--quant` to pick the matching file from the repo, or name an exact file with `-f`. GGUF supports a different set of models than safetensors; see [GGUF compatibility](/reference/gguf-support/).
- **UQFF:** a model that's already been quantized and saved, so it loads without quantizing again. See [UQFF](/guides/quantization/uqff/).

Not sure a model will fit? `mistralrs tune -m <model>` estimates memory use and recommends a `--quant` level before you download the weights.
