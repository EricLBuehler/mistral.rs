---
title: Supported models
description: Architectures supported by mistral.rs.
sidebar:
  order: 9
---

Supported model architectures. Specific model sizes within each family are on Hugging Face. Architecture names below match the SDK enum variants (Python `Architecture` / `MultimodalArchitecture` / `EmbeddingArchitecture` / `DiffusionArchitecture`). The text-only `--arch` CLI flag accepts the lowercase form (`mistral`, `gpt_oss`, `glm4moe`, ...); multimodal, speech, and diffusion architectures are auto-detected and not selectable via `--arch`.

To run:

```bash
mistralrs run -m <model>
mistralrs serve -m <model>
```

Passing `--arch` is only necessary in rare cases.

## Text models

| Architecture | Example repo |
|---|---|
| `Mistral` | `mistralai/Mistral-7B-Instruct-v0.3` |
| `Gemma` | `google/gemma-7b-it` |
| `Mixtral` | `mistralai/Mixtral-8x7B-Instruct-v0.1` |
| `Llama` | `meta-llama/Llama-3.1-8B-Instruct` |
| `Phi2` | `microsoft/phi-2` |
| `Phi3` | `microsoft/Phi-3-medium-4k-instruct` |
| `Qwen2` | `Qwen/Qwen2-7B-Instruct` |
| `Gemma2` | `google/gemma-2-9b-it` |
| `Starcoder2` | `bigcode/starcoder2-7b` |
| `Phi3_5MoE` | `microsoft/Phi-3.5-MoE-instruct` |
| `DeepSeekV2` | `deepseek-ai/DeepSeek-V2-Chat` |
| `DeepSeekV3` | `deepseek-ai/DeepSeek-V3` |
| `Qwen3` | `Qwen/Qwen3-4B` |
| `GLM4` | `zai-org/GLM-4-32B-0414` |
| `GLM4Moe` | `zai-org/GLM-4.7` |
| `GLM4MoeLite` | `zai-org/GLM-4.7-Flash` |
| `Qwen3Moe` | `Qwen/Qwen3-30B-A3B` |
| `SmolLm3` | `HuggingFaceTB/SmolLM3-3B` |
| `GraniteMoeHybrid` | `ibm-granite/granite-4.0-micro` |
| `GptOss` | `openai/gpt-oss-20b` |
| `Qwen3Next` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |

## Multimodal models

| Architecture | Example repo | Modalities |
|---|---|---|
| `Phi3V` | `microsoft/Phi-3.5-vision-instruct` | Text, image |
| `Idefics2` | `HuggingFaceM4/idefics2-8b` | Text, image |
| `LLaVANext` | `llava-hf/llava-v1.6-mistral-7b-hf` | Text, image |
| `LLaVA` | `llava-hf/llava-1.5-7b-hf` | Text, image |
| `VLlama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Text, image |
| `Qwen2VL` | `Qwen/Qwen2-VL-7B-Instruct` | Text, image, video |
| `Idefics3` | `HuggingFaceM4/Idefics3-8B-Llama3` | Text, image |
| `MiniCpmO` | `openbmb/MiniCPM-o-2_6` | Text, image, audio |
| `Phi4MM` | `microsoft/Phi-4-multimodal-instruct` | Text, image, audio |
| `Qwen2_5VL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Text, image, video |
| `Gemma3` | `google/gemma-3-12b-it` | Text, image |
| `Mistral3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Text, image |
| `Llama4` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Text, image |
| `Gemma3n` | `google/gemma-3n-E4B-it` | Text, image, audio, video |
| `Qwen3VL` | `Qwen/Qwen3-VL-4B-Instruct` | Text, image, video |
| `Qwen3VLMoE` | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Text, image, video |
| `Qwen3_5` | `Qwen/Qwen3.5-27B` | Text, image |
| `Qwen3_5Moe` | `Qwen/Qwen3.5-35B-A3B` | Text, image |
| `Voxtral` | `mistralai/Voxtral-Mini-3B-2507` | Text, audio |
| `Gemma4` | `google/gemma-4-E4B-it` | Text, image, audio, video |

## Image generation

| Architecture | Example repo |
|---|---|
| `Flux` | `black-forest-labs/FLUX.1-schnell` |
| `FluxOffloaded` | `black-forest-labs/FLUX.1-schnell` |

`FluxOffloaded` loads the same FLUX checkpoints as `Flux` with CPU offload enabled for memory-constrained hosts.

## Speech

| Architecture | Example repo | Direction |
|---|---|---|
| `Dia` | `nari-labs/Dia-1.6B` | Text to speech |

## Embedding

| Architecture | Example repo |
|---|---|
| `EmbeddingGemma` | `google/embeddinggemma-300m` |
| `Qwen3Embedding` | `Qwen/Qwen3-Embedding-0.6B` |

## Format and quantization notes

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. Pre-quantized format availability (GGUF, UQFF, GPTQ, AWQ) is per-model on Hugging Face.

## Speculative decoding

| Mode | Target architecture | Assistant checkpoint family | Guide |
|---|---|---|---|
| MTP | `Gemma4` | Gemma 4 assistant checkpoints, PagedAttention required | [Gemma 4 MTP](/mistral.rs/guides/perf/gemma4-mtp/) |

## Model notes

For non-standard behavior, see [model notes](/mistral.rs/reference/model-notes/).
