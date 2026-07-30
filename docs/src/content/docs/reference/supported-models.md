---
title: Supported models
description: Architectures supported by mistral.rs, and how to tell if yours is one of them.
---

<!-- Generated from the loader registry by mistralrs-core model_metadata. Do not edit by hand. -->

## Is my model supported?

mistral.rs auto-detects the architecture from a repo's `config.json`. To check yours:

1. Open the model's `config.json` on Hugging Face and read the `architectures` field (e.g. `"Qwen3ForCausalLM"`, `"Gemma4ForConditionalGeneration"`).
2. Find the matching row below. Each architecture covers every checkpoint that reports that class, including future fine-tunes and sizes, so the families and examples here are a sample, not the full list.
3. Not listed? You can still try it: force a known architecture with `--arch`, load a [GGUF](/mistral.rs/guides/models/run-any-model/) build, or [request the model](https://github.com/EricLBuehler/mistral.rs/issues/156).

```bash
mistralrs run -m <model>     # interactive
mistralrs serve -m <model>   # OpenAI-compatible server
```

Expand the example in any row to copy a ready-to-run command. One loader often serves several brand names (Qwen 3.5 and 3.6 share `Qwen3_5`; LFM2 and LFM2.5 share `Lfm2`) - the `Model families` column lists them. Behavior that differs from the defaults is collected in [model family notes](/mistral.rs/guides/models/model-family-notes/).

The `Architecture` column is the `config.json` `architectures` value. Per-family quantization, thinking, gated-repo, and tool-calling details live in [model family notes](/mistral.rs/guides/models/model-family-notes/).

## Text models

| Architecture | Model families | Example |
|---|---|---|
| `MistralForCausalLM` | Mistral | <details><summary><code>mistralai/Mistral-7B-Instruct-v0.3</code></summary><code>mistralrs run -m mistralai/Mistral-7B-Instruct-v0.3</code></details> |
| `GemmaForCausalLM` | Gemma | <details><summary><code>google/gemma-7b-it</code></summary><code>mistralrs run -m google/gemma-7b-it</code></details> |
| `MixtralForCausalLM` | Mixtral | <details><summary><code>mistralai/Mixtral-8x7B-Instruct-v0.1</code></summary><code>mistralrs run -m mistralai/Mixtral-8x7B-Instruct-v0.1</code></details> |
| `LlamaForCausalLM` | Llama 2, Llama 3.x | <details><summary><code>meta-llama/Llama-3.1-8B-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-3.1-8B-Instruct</code></details> |
| `PhiForCausalLM` | Phi-2 | <details><summary><code>microsoft/phi-2</code></summary><code>mistralrs run -m microsoft/phi-2</code></details> |
| `Phi3ForCausalLM` | Phi-3, Phi-3.5 | <details><summary><code>microsoft/Phi-3-medium-4k-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3-medium-4k-instruct</code></details> |
| `Qwen2ForCausalLM` | Qwen2, Qwen2.5 | <details><summary><code>Qwen/Qwen2.5-7B-Instruct</code> (2.5), <code>Qwen/Qwen2-7B-Instruct</code> (2)</summary><code>mistralrs run -m Qwen/Qwen2.5-7B-Instruct</code><br><code>mistralrs run -m Qwen/Qwen2-7B-Instruct</code></details> |
| `Gemma2ForCausalLM` | Gemma 2 | <details><summary><code>google/gemma-2-9b-it</code></summary><code>mistralrs run -m google/gemma-2-9b-it</code></details> |
| `Starcoder2ForCausalLM` | Starcoder2 | <details><summary><code>bigcode/starcoder2-7b</code></summary><code>mistralrs run -m bigcode/starcoder2-7b</code></details> |
| `PhiMoEForCausalLM` | Phi-3.5-MoE | <details><summary><code>microsoft/Phi-3.5-MoE-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3.5-MoE-instruct</code></details> |
| `DeepseekV2ForCausalLM` | DeepSeek-V2 | <details><summary><code>deepseek-ai/DeepSeek-V2-Chat</code></summary><code>mistralrs run -m deepseek-ai/DeepSeek-V2-Chat</code></details> |
| `DeepseekV3ForCausalLM` | DeepSeek-V3, DeepSeek-R1 | <details><summary><code>deepseek-ai/DeepSeek-V3</code> (V3), <code>deepseek-ai/DeepSeek-R1</code> (R1)</summary><code>mistralrs run -m deepseek-ai/DeepSeek-V3</code><br><code>mistralrs run -m deepseek-ai/DeepSeek-R1</code></details> |
| `Qwen3ForCausalLM` | Qwen3 | <details><summary><code>Qwen/Qwen3-4B</code></summary><code>mistralrs run -m Qwen/Qwen3-4B</code></details> |
| `Glm4ForCausalLM` | GLM-4 | <details><summary><code>zai-org/GLM-4-32B-0414</code></summary><code>mistralrs run -m zai-org/GLM-4-32B-0414</code></details> |
| `Glm4MoeLiteForCausalLM` | GLM-4.7-Flash | <details><summary><code>zai-org/GLM-4.7-Flash</code></summary><code>mistralrs run -m zai-org/GLM-4.7-Flash</code></details> |
| `Glm4MoeForCausalLM` | GLM-4.7 | <details><summary><code>zai-org/GLM-4.7</code></summary><code>mistralrs run -m zai-org/GLM-4.7</code></details> |
| `Qwen3MoeForCausalLM` | Qwen3 MoE | <details><summary><code>Qwen/Qwen3-30B-A3B</code></summary><code>mistralrs run -m Qwen/Qwen3-30B-A3B</code></details> |
| `SmolLM3ForCausalLM` | SmolLM3 | <details><summary><code>HuggingFaceTB/SmolLM3-3B</code></summary><code>mistralrs run -m HuggingFaceTB/SmolLM3-3B</code></details> |
| `GraniteMoeHybridForCausalLM` | Granite 4.0 | <details><summary><code>ibm-granite/granite-4.0-micro</code></summary><code>mistralrs run -m ibm-granite/granite-4.0-micro</code></details> |
| `GptOssForCausalLM` | GPT-OSS | <details><summary><code>openai/gpt-oss-20b</code> (20b), <code>openai/gpt-oss-120b</code> (120b)</summary><code>mistralrs run -m openai/gpt-oss-20b</code><br><code>mistralrs run -m openai/gpt-oss-120b</code></details> |
| `HunYuanDenseV1ForCausalLM` | HunYuan | <details><summary><code>tencent/Hunyuan-7B-Instruct</code></summary><code>mistralrs run -m tencent/Hunyuan-7B-Instruct</code></details> |
| `HunYuanMoEV1ForCausalLM` | HunYuan MoE | <details><summary><code>tencent/Hunyuan-A13B-Instruct</code></summary><code>mistralrs run -m tencent/Hunyuan-A13B-Instruct</code></details> |
| `Qwen3NextForCausalLM` | Qwen3-Next, Qwen3-Coder-Next | <details><summary><code>Qwen/Qwen3-Next-80B-A3B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-Next-80B-A3B-Instruct</code></details> |
| `Lfm2ForCausalLM` | LFM2, LFM2.5 | <details><summary><code>LiquidAI/LFM2.5-1.2B-Instruct</code> (LFM2.5), <code>LiquidAI/LFM2-1.2B</code> (LFM2)</summary><code>mistralrs run -m LiquidAI/LFM2.5-1.2B-Instruct</code><br><code>mistralrs run -m LiquidAI/LFM2-1.2B</code></details> |
| `Lfm2MoeForCausalLM` | LFM2 MoE, LFM2.5 MoE | <details><summary><code>LiquidAI/LFM2.5-8B-A1B</code> (LFM2.5), <code>LiquidAI/LFM2-8B-A1B</code> (LFM2)</summary><code>mistralrs run -m LiquidAI/LFM2.5-8B-A1B</code><br><code>mistralrs run -m LiquidAI/LFM2-8B-A1B</code></details> |

## Multimodal models

| Architecture | Model families | Example |
|---|---|---|
| `Phi3VForCausalLM` | Phi-3.5-Vision | <details><summary><code>microsoft/Phi-3.5-vision-instruct</code></summary><code>mistralrs run -m microsoft/Phi-3.5-vision-instruct</code></details> |
| `Idefics2ForConditionalGeneration` | Idefics2 | <details><summary><code>HuggingFaceM4/idefics2-8b</code></summary><code>mistralrs run -m HuggingFaceM4/idefics2-8b</code></details> |
| `LlavaNextForConditionalGeneration` | LLaVA-NeXT | <details><summary><code>llava-hf/llava-v1.6-mistral-7b-hf</code></summary><code>mistralrs run -m llava-hf/llava-v1.6-mistral-7b-hf</code></details> |
| `LlavaForConditionalGeneration` | LLaVA 1.5 | <details><summary><code>llava-hf/llava-1.5-7b-hf</code></summary><code>mistralrs run -m llava-hf/llava-1.5-7b-hf</code></details> |
| `Lfm2VlForConditionalGeneration` | LFM2-VL, LFM2.5-VL | <details><summary><code>LiquidAI/LFM2.5-VL-1.6B</code> (1.6B), <code>LiquidAI/LFM2.5-VL-450M</code> (450M)</summary><code>mistralrs run -m LiquidAI/LFM2.5-VL-1.6B</code><br><code>mistralrs run -m LiquidAI/LFM2.5-VL-450M</code></details> |
| `MllamaForConditionalGeneration` | Llama 3.2 Vision | <details><summary><code>meta-llama/Llama-3.2-11B-Vision-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-3.2-11B-Vision-Instruct</code></details> |
| `Qwen2VLForConditionalGeneration` | Qwen2-VL | <details><summary><code>Qwen/Qwen2-VL-7B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen2-VL-7B-Instruct</code></details> |
| `Idefics3ForConditionalGeneration` | Idefics3, SmolVLM | <details><summary><code>HuggingFaceM4/Idefics3-8B-Llama3</code></summary><code>mistralrs run -m HuggingFaceM4/Idefics3-8B-Llama3</code></details> |
| `MiniCPMO` | MiniCPM-o | <details><summary><code>openbmb/MiniCPM-o-2_6</code></summary><code>mistralrs run -m openbmb/MiniCPM-o-2_6</code></details> |
| `Phi4MMForCausalLM` | Phi-4-multimodal | <details><summary><code>microsoft/Phi-4-multimodal-instruct</code></summary><code>mistralrs run -m microsoft/Phi-4-multimodal-instruct</code></details> |
| `Qwen2_5_VLForConditionalGeneration` | Qwen2.5-VL | <details><summary><code>Qwen/Qwen2.5-VL-7B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen2.5-VL-7B-Instruct</code></details> |
| `Gemma3ForConditionalGeneration` | Gemma 3 | <details><summary><code>google/gemma-3-12b-it</code></summary><code>mistralrs run -m google/gemma-3-12b-it</code></details> |
| `Mistral3ForConditionalGeneration` | Mistral Small 3 | <details><summary><code>mistralai/Mistral-Small-3.2-24B-Instruct-2506</code></summary><code>mistralrs run -m mistralai/Mistral-Small-3.2-24B-Instruct-2506</code></details> |
| `Llama4ForConditionalGeneration` | Llama 4 | <details><summary><code>meta-llama/Llama-4-Scout-17B-16E-Instruct</code></summary><code>mistralrs run -m meta-llama/Llama-4-Scout-17B-16E-Instruct</code></details> |
| `Gemma3nForConditionalGeneration` | Gemma 3n | <details><summary><code>google/gemma-3n-E4B-it</code></summary><code>mistralrs run -m google/gemma-3n-E4B-it</code></details> |
| `Qwen3VLForConditionalGeneration` | Qwen3-VL | <details><summary><code>Qwen/Qwen3-VL-4B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-VL-4B-Instruct</code></details> |
| `Qwen3VLMoeForConditionalGeneration` | Qwen3-VL MoE | <details><summary><code>Qwen/Qwen3-VL-235B-A22B-Instruct</code></summary><code>mistralrs run -m Qwen/Qwen3-VL-235B-A22B-Instruct</code></details> |
| `Qwen3_5ForConditionalGeneration` | Qwen 3.5, Qwen 3.6 | <details><summary><code>Qwen/Qwen3.5-27B</code> (3.5), <code>Qwen/Qwen3.6-27B</code> (3.6)</summary><code>mistralrs run -m Qwen/Qwen3.5-27B</code><br><code>mistralrs run -m Qwen/Qwen3.6-27B</code></details> |
| `Qwen3_5MoeForConditionalGeneration` | Qwen 3.5 MoE, Qwen 3.6 MoE | <details><summary><code>Qwen/Qwen3.5-35B-A3B</code> (3.5), <code>Qwen/Qwen3.6-35B-A3B</code> (3.6)</summary><code>mistralrs run -m Qwen/Qwen3.5-35B-A3B</code><br><code>mistralrs run -m Qwen/Qwen3.6-35B-A3B</code></details> |
| `VoxtralForConditionalGeneration` | Voxtral | <details><summary><code>mistralai/Voxtral-Mini-3B-2507</code></summary><code>mistralrs run -m mistralai/Voxtral-Mini-3B-2507</code></details> |
| `Gemma4ForConditionalGeneration` | Gemma 4 | <details><summary><code>google/gemma-4-E4B-it</code> (E4B), <code>google/gemma-4-26B-A4B-it</code> (26B-A4B MoE), <code>google/gemma-4-31B-it</code> (31B dense)</summary><code>mistralrs run -m google/gemma-4-E4B-it</code><br><code>mistralrs run -m google/gemma-4-26B-A4B-it</code><br><code>mistralrs run -m google/gemma-4-31B-it</code></details> |
| `DiffusionGemmaForBlockDiffusion` | DiffusionGemma | <details><summary><code>google/diffusiongemma-26B-A4B-it</code></summary><code>mistralrs run -m google/diffusiongemma-26B-A4B-it</code></details> |

## Image generation

| Architecture | Model families | Example |
|---|---|---|
| `Flux` | FLUX.1 | <details><summary><code>black-forest-labs/FLUX.1-schnell</code></summary><code>mistralrs run -m black-forest-labs/FLUX.1-schnell</code></details> |
| `FluxOffloaded` | FLUX.1 (offloaded) | <details><summary><code>black-forest-labs/FLUX.1-schnell</code></summary><code>mistralrs run -m black-forest-labs/FLUX.1-schnell</code></details> |

## Speech

| Architecture | Model families | Example |
|---|---|---|
| `Dia` | Dia | <details><summary><code>nari-labs/Dia-1.6B</code></summary><code>mistralrs run -m nari-labs/Dia-1.6B</code></details> |

## Embedding

| Architecture | Model families | Example |
|---|---|---|
| `Gemma3TextModel` | EmbeddingGemma | <details><summary><code>google/embeddinggemma-300m</code></summary><code>mistralrs run -m google/embeddinggemma-300m</code></details> |
| `Qwen3ForCausalLM` | Qwen3 Embedding | <details><summary><code>Qwen/Qwen3-Embedding-0.6B</code></summary><code>mistralrs run -m Qwen/Qwen3-Embedding-0.6B</code></details> |

## Format and quantization notes

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. Pre-quantized format availability (GGUF, [UQFF](/mistral.rs/reference/uqff-format/), GPTQ, AWQ) is per-model on Hugging Face.

## Speculative decoding

| Mode | Target architecture | Assistant checkpoint family | Guide |
|---|---|---|---|
| MTP | `Gemma4` | Gemma 4 assistant checkpoints, PagedAttention required | [Speculative decoding (MTP)](/mistral.rs/guides/perf/speculative-decoding/) |
