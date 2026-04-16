---
title: Supported models
description: Every model architecture mistralrs supports, what modalities each accepts, and which quantizations work.
sidebar:
  order: 9
---

This table lists the architectures mistralrs supports as of the current release. Specific model sizes within each family are available on Hugging Face; the repo ids in the "Examples" column are representative choices that we test regularly.

If a model you want to run is not listed here, it may still work if its architecture matches one we support. Try it with `mistralrs run -m <repo-id>`; if detection fails you will see a clear error.

## Text models

| Architecture | Example repo | Modalities | Notes |
|---|---|---|---|
| DeepSeek V2 | `deepseek-ai/DeepSeek-V2-Chat` | Text | MLA attention. |
| DeepSeek V3 | `deepseek-ai/DeepSeek-V3` | Text | MLA attention. Large MoE. |
| Gemma 2 | `google/gemma-2-9b-it` | Text | |
| Gemma 3 | `google/gemma-3-12b-it` | Text | |
| Gemma 3n | `google/gemma-3n-E4B-it` | Text | MatFormer. |
| GLM-4 | `zai-org/GLM-4-32B-0414` | Text | |
| GLM-4 MoE | `zai-org/GLM-4.5` | Text | |
| GLM-4.7 Flash (MoE Lite) | `zai-org/GLM-4.7-Flash-Air` | Text | |
| GPT-OSS | `openai/gpt-oss-20b` | Text | |
| LLaMA 3.2 | `meta-llama/Llama-3.2-3B-Instruct` | Text | |
| Mistral | `mistralai/Mistral-7B-Instruct-v0.3` | Text | |
| Phi 3, Phi 3.5 | `microsoft/Phi-3.5-mini-instruct` | Text | |
| Phi 3.5 MoE | `microsoft/Phi-3.5-MoE-instruct` | Text | |
| Qwen 3 | `Qwen/Qwen3-4B` | Text | Thinking-capable. |
| Qwen 3 Next | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Text | Hybrid linear + softmax attention. |
| Qwen 3.5 | `Qwen/Qwen3.5-4B` | Text | |
| SmolLM3 | `HuggingFaceTB/SmolLM3-3B` | Text | |

## Multimodal models

| Architecture | Example repo | Modalities | Notes |
|---|---|---|---|
| Gemma 4 | `google/gemma-4-E4B-it` | Text, image, audio, video | Full multimodal. |
| Idefics 2 | `HuggingFaceM4/idefics2-8b` | Text, image | |
| Idefics 3 | `HuggingFaceM4/Idefics3-8B-Llama3` | Text, image | |
| Llama 3.2 Vision | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Text, image | |
| Llama 4 | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Text, image | |
| LLaVA | `liuhaotian/llava-v1.6-mistral-7b` | Text, image | |
| MiniCPM-O 2.6 | `openbmb/MiniCPM-o-2_6` | Text, image, audio | |
| Mistral 3 | `mistralai/Mistral-3-24B` | Text, image | |
| Phi 3.5 Vision | `microsoft/Phi-3.5-vision-instruct` | Text, image | |
| Phi 4 Multimodal | `microsoft/Phi-4-multimodal-instruct` | Text, image, audio | |
| Qwen 2-VL | `Qwen/Qwen2-VL-7B-Instruct` | Text, image | |
| Qwen 3-VL | `Qwen/Qwen3-VL-4B-Instruct` | Text, image, video | |
| Qwen 3.5 | `Qwen/Qwen3.5-VL-7B` | Text, image | |

## Image generation

| Architecture | Example repo | Notes |
|---|---|---|
| FLUX | `black-forest-labs/FLUX.1-schnell` | Schnell is fastest; dev is higher quality. |

## Speech models

| Architecture | Example repo | Direction | Notes |
|---|---|---|---|
| Voxtral | `mistralai/Voxtral-Mini-3B` | Speech to text | |
| Dia | `nari-labs/Dia-1.6B` | Text to speech | |

## Embedding models

| Architecture | Example repo | Dimensions | Notes |
|---|---|---|---|
| EmbeddingGemma | `google/embeddinggemma-300m` | 768 | Matryoshka truncation supported. |
| Qwen 3 Embedding | `Qwen/Qwen3-Embedding-0.6B` | 4096 | |

## Quantization compatibility

Every supported model works with ISQ at load time. For pre-quantized formats (GGUF, UQFF, GPTQ, AWQ), availability is per-model on the Hugging Face hub. To check whether a model is available in a pre-quantized form, search the hub for the model name plus the format name.

## What is not supported

- Vision models that use encoder architectures not in the list above.
- Non-English TTS models.
- Real-time streaming audio APIs.
- Models with proprietary tokenizers that Hugging Face does not host.

If you run into a model that you think should work but does not, an issue on the GitHub repo is the right place.

## Model notes

A handful of models have genuinely surprising behavior that is worth knowing about. See [model notes](/mistral.rs/reference/model-notes/) for the list. Everything not in that list behaves like its peers.
