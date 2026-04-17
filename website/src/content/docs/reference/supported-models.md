---
title: Supported models
description: Architectures supported by mistralrs.
sidebar:
  order: 9
---

Architectures registered in the loader enums in `mistralrs-core`. Specific model sizes within each family are on Hugging Face.

## Text models

`NormalLoaderType` variants:

| Variant | Example repo |
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
| `GLM4Moe` | `zai-org/GLM-4.5` |
| `GLM4MoeLite` | |
| `Qwen3Moe` | `Qwen/Qwen3-30B-A3B` |
| `SmolLm3` | `HuggingFaceTB/SmolLM3-3B` |
| `GraniteMoeHybrid` | |
| `GptOss` | `openai/gpt-oss-20b` |
| `Qwen3Next` | `Qwen/Qwen3-Next-80B-A3B-Instruct` |

## Multimodal models

`MultimodalLoaderType` variants:

| Variant | Example repo | Modalities |
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
| `Mistral3` | | |
| `Llama4` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Text, image |
| `Gemma3n` | `google/gemma-3n-E4B-it` | Text, image, audio, video |
| `Qwen3VL` | `Qwen/Qwen3-VL-4B-Instruct` | Text, image, video |
| `Qwen3VLMoE` | | Text, image, video |
| `Qwen3_5` | | |
| `Qwen3_5Moe` | | |
| `Voxtral` | `mistralai/Voxtral-Mini-3B-2507` | Text, audio |
| `Gemma4` | `google/gemma-4-E4B-it` | Text, image, audio, video |

## Image generation

`DiffusionLoaderType` variants:

| Variant | Example repo |
|---|---|
| `Flux` | `black-forest-labs/FLUX.1-schnell` |
| `FluxOffloaded` | |

## Speech

`SpeechLoaderType` variants:

| Variant | Example repo | Direction |
|---|---|---|
| `Dia` | `nari-labs/Dia-1.6B` | Text to speech |

## Embedding

`EmbeddingLoaderType` variants:

| Variant | Example repo |
|---|---|
| `EmbeddingGemma` | `google/embeddinggemma-300m` |
| `Qwen3Embedding` | `Qwen/Qwen3-Embedding-0.6B` |

## Quantization compatibility

All supported models work with ISQ at load time. Pre-quantized format availability (GGUF, UQFF, GPTQ, AWQ) is per-model on Hugging Face.

## Model notes

For non-standard behavior, see [model notes](/mistral.rs/reference/model-notes/).
