---
title: Supported models
description: Supported model families, example checkpoints, and architecture identifiers.
---

<!-- Generated from the loader registry by mistralrs-core model_metadata. Do not edit by hand. -->

Browse by family below, or [choose a model by task](/guides/models/choose-a-model/). Expand an example for a command. For model-specific setup and limits, see [model family notes](/guides/models/model-family-notes/).

```bash
mistralrs run -m <model>     # interactive
mistralrs serve -m <model>   # OpenAI-compatible server
```

## Text models

| Model families | Example | Architecture |
|---|---|---|
| DeepSeek-V2 | <details><summary><code>deepseek-ai/DeepSeek-V2-Chat</code></summary><code>mistralrs run -m deepseek-ai/DeepSeek-V2-Chat</code></details> | `DeepseekV2ForCausalLM` |
| DeepSeek-V3, DeepSeek-R1 | <details><summary><code>deepseek-ai/DeepSeek-V3</code> (V3), <code>deepseek-ai/DeepSeek-R1</code> (R1)</summary><code>mistralrs run -m deepseek-ai/DeepSeek-V3</code><br><code>mistralrs run -m deepseek-ai/DeepSeek-R1</code></details> | `DeepseekV3ForCausalLM` |
| Flint | <details><summary><code>RecursiveSI/flint</code></summary><code>mistralrs run -m RecursiveSI/flint</code></details> | `FlintForCausalLM` |
| Gemma | <details><summary><code>google/gemma-7b-it</code></summary><code>mistralrs run -m google/gemma-7b-it</code></details> | `GemmaForCausalLM` |
| Gemma 2 | <details><summary><code>google/gemma-2-9b-it</code></summary><code>mistralrs run -m google/gemma-2-9b-it</code></details> | `Gemma2ForCausalLM` |
| GLM-4 | <details><summary><code>zai-org/GLM-4-32B-0414</code></summary><code>mistralrs run -m zai-org/GLM-4-32B-0414</code></details> | `Glm4ForCausalLM` |
| GLM-4.7 | <details><summary><code>zai-org/GLM-4.7</code></summary><code>mistralrs run -m zai-org/GLM-4.7</code></details> | `Glm4MoeForCausalLM` |
| GLM-4.7-Flash | <details><summary><code>zai-org/GLM-4.7-Flash</code></summary><code>mistralrs run -m zai-org/GLM-4.7-Flash</code></details> | `Glm4MoeLiteForCausalLM` |
| GPT-OSS | <details><summary><code>openai/gpt-oss-20b</code> (20b), <code>openai/gpt-oss-120b</code> (120b)</summary><code>mistralrs run -m openai/gpt-oss-20b</code><br><code>mistralrs run -m openai/gpt-oss-120b</code></details> | `GptOssForCausalLM` |
| Granite 4.0 | <details><summary><code>ibm-granite/granite-4.0-micro</code></summary><code>mistralrs run -m ibm-granite/granite-4.0-micro</code></details> | `GraniteMoeHybridForCausalLM` |
| HunYuan | <details><summary><code>tencent/Hunyuan-7B-Instruct</code></summary><code>mistralrs run -m tencent/Hunyuan-7B-Instruct</code></details> | `HunYuanDenseV1ForCausalLM` |
| HunYuan MoE | <details><summary><code>tencent/Hunyuan-A13B-Instruct</code></summary><code>mistralrs run -m tencent/Hunyuan-A13B-Instruct</code></details> | `HunYuanMoEV1ForCausalLM` |
| LFM2, LFM2.5 | <details><summary><code>LiquidAI/LFM2.5-1.2B-Instruct</code> (LFM2.5), <code>LiquidAI/LFM2-1.2B</code> (LFM2)</summary><code>mistralrs run -m LiquidAI/LFM2.5-1.2B-Instruct</code><br><code>mistralrs run -m LiquidAI/LFM2-1.2B</code></details> | `Lfm2ForCausalLM` |
| LFM2 MoE, LFM2.5 MoE | <details><summary><code>LiquidAI/LFM2.5-8B-A1B</code> (LFM2.5), <code>LiquidAI/LFM2-8B-A1B</code> (LFM2)</summary><code>mistralrs run -m LiquidAI/LFM2.5-8B-A1B</code><br><code>mistralrs run -m LiquidAI/LFM2-8B-A1B</code></details> | `Lfm2MoeForCausalLM` |
| LFM3 | <details><summary><code>LiquidAI/LFM3-1.7B</code></summary><code>mistralrs run -m LiquidAI/LFM3-1.7B</code></details> | `Lfm3ForCausalLM` |
| Llama 2, Llama 3.x | <details><summary><code>meta-llama/Llama-3.1-8B-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-3.1-8B-Instruct</code></details> | `LlamaForCausalLM` |
| Mistral | <details><summary><code>mistralai/Mistral-7B-Instruct-v0.3</code></summary><code>mistralrs run -m mistralai/Mistral-7B-Instruct-v0.3</code></details> | `MistralForCausalLM` |
| Mixtral | <details><summary><code>mistralai/Mixtral-8x7B-Instruct-v0.1</code></summary><code>mistralrs run -m mistralai/Mixtral-8x7B-Instruct-v0.1</code></details> | `MixtralForCausalLM` |
| Phi-2 | <details><summary><code>microsoft/phi-2</code></summary><code>mistralrs run -m microsoft/phi-2</code></details> | `PhiForCausalLM` |
| Phi-3, Phi-3.5 | <details><summary><code>microsoft/Phi-3-medium-4k-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3-medium-4k-instruct</code></details> | `Phi3ForCausalLM` |
| Phi-3.5-MoE | <details><summary><code>microsoft/Phi-3.5-MoE-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3.5-MoE-instruct</code></details> | `PhiMoEForCausalLM` |
| Qwen2, Qwen2.5 | <details><summary><code>Qwen/Qwen2.5-7B-Instruct</code> (2.5), <code>Qwen/Qwen2-7B-Instruct</code> (2)</summary><code>mistralrs run -m Qwen/Qwen2.5-7B-Instruct</code><br><code>mistralrs run -m Qwen/Qwen2-7B-Instruct</code></details> | `Qwen2ForCausalLM` |
| Qwen3 | <details><summary><code>Qwen/Qwen3-4B</code></summary><code>mistralrs run -m Qwen/Qwen3-4B</code></details> | `Qwen3ForCausalLM` |
| Qwen3 MoE | <details><summary><code>Qwen/Qwen3-30B-A3B</code></summary><code>mistralrs run -m Qwen/Qwen3-30B-A3B</code></details> | `Qwen3MoeForCausalLM` |
| Qwen3-Next, Qwen3-Coder-Next | <details><summary><code>Qwen/Qwen3-Next-80B-A3B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-Next-80B-A3B-Instruct</code></details> | `Qwen3NextForCausalLM` |
| Qwen3.5 | No published example | `Qwen3_5ForCausalLM` |
| SmolLM3 | <details><summary><code>HuggingFaceTB/SmolLM3-3B</code></summary><code>mistralrs run -m HuggingFaceTB/SmolLM3-3B</code></details> | `SmolLM3ForCausalLM` |
| Starcoder2 | <details><summary><code>bigcode/starcoder2-7b</code></summary><code>mistralrs run -m bigcode/starcoder2-7b</code></details> | `Starcoder2ForCausalLM` |

## Multimodal models

| Model families | Example | Architecture |
|---|---|---|
| DiffusionGemma | <details><summary><code>google/diffusiongemma-26B-A4B-it</code></summary><code>mistralrs run -m google/diffusiongemma-26B-A4B-it</code></details> | `DiffusionGemmaForBlockDiffusion` |
| Gemma 3 | <details><summary><code>google/gemma-3-12b-it</code></summary><code>mistralrs run -m google/gemma-3-12b-it</code></details> | `Gemma3ForConditionalGeneration` |
| Gemma 3n | <details><summary><code>google/gemma-3n-E4B-it</code></summary><code>mistralrs run -m google/gemma-3n-E4B-it</code></details> | `Gemma3nForConditionalGeneration` |
| Gemma 4 | <details><summary><code>google/gemma-4-E4B-it</code> (E4B), <code>google/gemma-4-26B-A4B-it</code> (26B-A4B MoE), <code>google/gemma-4-31B-it</code> (31B dense)</summary><code>mistralrs run -m google/gemma-4-E4B-it</code><br><code>mistralrs run -m google/gemma-4-26B-A4B-it</code><br><code>mistralrs run -m google/gemma-4-31B-it</code></details> | `Gemma4ForConditionalGeneration` |
| Idefics2 | <details><summary><code>HuggingFaceM4/idefics2-8b</code></summary><code>mistralrs run -m HuggingFaceM4/idefics2-8b</code></details> | `Idefics2ForConditionalGeneration` |
| Idefics3, SmolVLM | <details><summary><code>HuggingFaceM4/Idefics3-8B-Llama3</code></summary><code>mistralrs run -m HuggingFaceM4/Idefics3-8B-Llama3</code></details> | `Idefics3ForConditionalGeneration` |
| LFM2-VL, LFM2.5-VL | <details><summary><code>LiquidAI/LFM2.5-VL-1.6B</code> (1.6B), <code>LiquidAI/LFM2.5-VL-450M</code> (450M)</summary><code>mistralrs run -m LiquidAI/LFM2.5-VL-1.6B</code><br><code>mistralrs run -m LiquidAI/LFM2.5-VL-450M</code></details> | `Lfm2VlForConditionalGeneration` |
| LFM3-Audio | <details><summary><code>LiquidAI/LFM3-Audio-2B</code></summary><code>mistralrs run -m LiquidAI/LFM3-Audio-2B</code></details> | `Lfm3AudioForConditionalGeneration` |
| LFM3-Omni | <details><summary><code>LiquidAI/LFM3-Omni-2B</code></summary><code>mistralrs run -m LiquidAI/LFM3-Omni-2B</code></details> | `Lfm3OmniForConditionalGeneration` |
| LFM3-VL | <details><summary><code>LiquidAI/LFM3-VL-2B</code></summary><code>mistralrs run -m LiquidAI/LFM3-VL-2B</code></details> | `Lfm3VlForConditionalGeneration` |
| Llama 3.2 Vision | <details><summary><code>meta-llama/Llama-3.2-11B-Vision-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-3.2-11B-Vision-Instruct</code></details> | `MllamaForConditionalGeneration` |
| Llama 4 | <details><summary><code>meta-llama/Llama-4-Scout-17B-16E-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-4-Scout-17B-16E-Instruct</code></details> | `Llama4ForConditionalGeneration` |
| LLaVA 1.5 | <details><summary><code>llava-hf/llava-1.5-7b-hf</code></summary><code>mistralrs run -m llava-hf/llava-1.5-7b-hf</code></details> | `LlavaForConditionalGeneration` |
| LLaVA-NeXT | <details><summary><code>llava-hf/llava-v1.6-mistral-7b-hf</code></summary><code>mistralrs run -m llava-hf/llava-v1.6-mistral-7b-hf</code></details> | `LlavaNextForConditionalGeneration` |
| MiniCPM-o | <details><summary><code>openbmb/MiniCPM-o-2_6</code></summary><code>mistralrs run -m openbmb/MiniCPM-o-2_6</code></details> | `MiniCPMO` |
| Mistral Small 3 | <details><summary><code>mistralai/Mistral-Small-3.2-24B-Instruct-2506</code></summary><code>mistralrs run -m mistralai/Mistral-Small-3.2-24B-Instruct-2506</code></details> | `Mistral3ForConditionalGeneration` |
| Muse Glimmer | <details><summary><code>meta-models/Muse-Glimmer-30B</code></summary><code>mistralrs run -m meta-models/Muse-Glimmer-30B</code></details> | `MuseGlimmerForConditionalGeneration` |
| Phi-3.5-Vision | <details><summary><code>microsoft/Phi-3.5-vision-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3.5-vision-instruct</code></details> | `Phi3VForCausalLM` |
| Phi-4-multimodal | <details><summary><code>microsoft/Phi-4-multimodal-instruct</code></summary><code>mistralrs run -m microsoft/Phi-4-multimodal-instruct</code></details> | `Phi4MMForCausalLM` |
| Qwen 3.5, Qwen 3.6 | <details><summary><code>Qwen/Qwen3.5-27B</code> (3.5), <code>Qwen/Qwen3.6-27B</code> (3.6)</summary><code>mistralrs run -m Qwen/Qwen3.5-27B</code><br><code>mistralrs run -m Qwen/Qwen3.6-27B</code></details> | `Qwen3_5ForConditionalGeneration` |
| Qwen 3.5 MoE, Qwen 3.6 MoE | <details><summary><code>Qwen/Qwen3.5-35B-A3B</code> (3.5), <code>Qwen/Qwen3.6-35B-A3B</code> (3.6)</summary><code>mistralrs run -m Qwen/Qwen3.5-35B-A3B</code><br><code>mistralrs run -m Qwen/Qwen3.6-35B-A3B</code></details> | `Qwen3_5MoeForConditionalGeneration` |
| Qwen2-VL | <details><summary><code>Qwen/Qwen2-VL-7B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen2-VL-7B-Instruct</code></details> | `Qwen2VLForConditionalGeneration` |
| Qwen2.5-VL | <details><summary><code>Qwen/Qwen2.5-VL-7B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen2.5-VL-7B-Instruct</code></details> | `Qwen2_5_VLForConditionalGeneration` |
| Qwen3-VL | <details><summary><code>Qwen/Qwen3-VL-4B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-VL-4B-Instruct</code></details> | `Qwen3VLForConditionalGeneration` |
| Qwen3-VL MoE | <details><summary><code>Qwen/Qwen3-VL-235B-A22B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-VL-235B-A22B-Instruct</code></details> | `Qwen3VLMoeForConditionalGeneration` |
| Qwen3.8-Flash-Next | <details><summary><code>Qwen/Qwen3.8-Flash-Next</code></summary><code>mistralrs run -m Qwen/Qwen3.8-Flash-Next</code></details> | `Qwen4ExpForConditionalGeneration` |
| Voxtral | <details><summary><code>mistralai/Voxtral-Mini-4B-Realtime-2602</code></summary><code>mistralrs run -m mistralai/Voxtral-Mini-4B-Realtime-2602</code></details> | `VoxtralRealtimeForConditionalGeneration` |

## Decision models

Use the [decision API](/guides/models/use-decision-models/) to classify, score, or extract values. These models do not generate chat responses.

| Model families | Example | Architecture |
|---|---|---|
| d1, d1-Omni | <details><summary><code>LiquidAI/d1-2B</code>, <code>LiquidAI/d1-Omni-2B</code></summary><code>mistralrs serve -m LiquidAI/d1-2B</code><br><code>mistralrs serve -m LiquidAI/d1-Omni-2B</code></details> | `D1Model` |

## Image generation

| Model families | Example | Architecture |
|---|---|---|
| FLUX.1 | <details><summary><code>black-forest-labs/FLUX.1-schnell</code></summary><code>mistralrs run -m black-forest-labs/FLUX.1-schnell</code></details> | `Flux` |
| FLUX.1 (offloaded) | <details><summary><code>black-forest-labs/FLUX.1-schnell</code></summary><code>mistralrs run -m black-forest-labs/FLUX.1-schnell</code></details> | `FluxOffloaded` |

## Speech

| Model families | Example | Architecture |
|---|---|---|
| Dia | <details><summary><code>nari-labs/Dia-1.6B</code></summary><code>mistralrs run -m nari-labs/Dia-1.6B</code></details> | `Dia` |

## Embedding

| Model families | Example | Architecture |
|---|---|---|
| EmbeddingGemma | <details><summary><code>google/embeddinggemma-300m</code></summary><code>mistralrs run -m google/embeddinggemma-300m</code></details> | `Gemma3TextModel` |
| Qwen3 Embedding | <details><summary><code>Qwen/Qwen3-Embedding-0.6B</code></summary><code>mistralrs run -m Qwen/Qwen3-Embedding-0.6B</code></details> | `Qwen3ForCausalLM` |

## Is my model supported?

For a checkpoint not shown here, match its `config.json` `architectures` value to the Architecture column. The examples are not an exhaustive list of sizes or fine-tunes. GGUF has a separate [compatibility table](/reference/gguf-support/). If neither lists your architecture, [request model support](https://github.com/EricLBuehler/mistral.rs/issues/156).

## Format and quantization notes

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. See [GGUF support](/reference/gguf-support/) for GGUF compatibility; availability of [UQFF](/reference/uqff-format/), GPTQ, and AWQ artifacts varies by model on Hugging Face.

## Speculative decoding

See [speculative decoding](/guides/perf/speculative-decoding/) for supported models and setup.
