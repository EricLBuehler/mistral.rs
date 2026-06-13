---
title: Supported models
description: Architectures supported by mistral.rs.
---

Supported model architectures. Specific model sizes within each family are on Hugging Face. You usually do not need `--arch`; the model type is auto-detected.

To run:

```bash
mistralrs run -m <model>
mistralrs serve -m <model>
```

- Architecture names below match the SDK enum variants (Python `Architecture` / `MultimodalArchitecture` / `EmbeddingArchitecture` / `DiffusionArchitecture`).
- The text-only `--arch` CLI flag takes the serde token for each architecture (`mistral`, `gpt_oss`, `phi3.5moe`, `glm4moe`, ...); these are mostly the lowercased enum name but some are irregular. The full accepted list is in the "Unknown architecture" parse error. Multimodal, speech, and diffusion architectures are auto-detected and not selectable via `--arch`.

Pass `--arch` only when a checkpoint's config does not identify its architecture (custom or converted repos). Behavior that differs from the defaults is collected in [model family notes](/mistral.rs/guides/models/model-family-notes/).

Acronyms in the Notes columns: [ISQ (in-situ quantization)](/mistral.rs/reference/quantization-types/), MoE (Mixture of Experts), MoQE (Mixture of Quantized Experts), MLA (Multi-head Latent Attention), GQA (grouped-query attention), GDN (gated delta net), [MXFP4](/mistral.rs/reference/quantization-types/), [MTP (multi-token prediction)](/mistral.rs/guides/perf/speculative-decoding/), [paged attention](/mistral.rs/guides/perf/paged-attention/), TP (tensor parallelism).

## Text models

| Architecture | Example repo | Notes |
|---|---|---|
| `Mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | |
| `Gemma` | `google/gemma-7b-it` | Gated repo: accept the HF license, then `mistralrs login`. |
| `Mixtral` | `mistralai/Mixtral-8x7B-Instruct-v0.1` | MoE. |
| `Llama` | `meta-llama/Llama-3.1-8B-Instruct` | |
| `Phi2` | `microsoft/phi-2` | |
| `Phi3` | `microsoft/Phi-3-medium-4k-instruct` | |
| `Qwen2` | `Qwen/Qwen2-7B-Instruct` | |
| `Gemma2` | `google/gemma-2-9b-it` | Gated repo. |
| `Starcoder2` | `bigcode/starcoder2-7b` | |
| `Phi3_5MoE` | `microsoft/Phi-3.5-MoE-instruct` | 16-expert MoE; supports MoQE. |
| `DeepseekV2` | `deepseek-ai/DeepSeek-V2-Chat` | MoE with MLA; supports MoQE. |
| `DeepseekV3` | `deepseek-ai/DeepSeek-V3` | MoE with MLA; non-distill DeepSeek R1 uses this architecture. |
| `Qwen3` | `Qwen/Qwen3-4B` | Hybrid reasoning; thinking on by default. |
| `GLM4` | `zai-org/GLM-4-32B-0414` | |
| `GLM4Moe` | `zai-org/GLM-4.7` | MoE with GQA attention and partial RoPE; supports MoQE. |
| `GLM4MoeLite` | `zai-org/GLM-4.7-Flash` | MoE with MLA; supports MoQE. |
| `Qwen3Moe` | `Qwen/Qwen3-30B-A3B` | Same thinking controls as dense Qwen3; supports MoQE. |
| `SmolLm3` | `HuggingFaceTB/SmolLM3-3B` | Hybrid reasoning; thinking controls match Qwen3. |
| `GraniteMoeHybrid` | `ibm-granite/granite-4.0-micro` | Hybrid Mamba-2 plus attention layers. |
| `GptOss` | `openai/gpt-oss-20b` | MXFP4 experts; ISQ applies to attention layers only. |
| `Qwen3Next` | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Hybrid GDN plus full attention; Qwen3-Coder-Next uses the same loader. |

## Multimodal models

| Architecture | Example repo | Modalities | Notes |
|---|---|---|---|
| `Phi3V` | `microsoft/Phi-3.5-vision-instruct` | Text, image | Best with a single image. |
| `Idefics2` | `HuggingFaceM4/idefics2-8b` | Text, image | |
| `LLaVANext` | `llava-hf/llava-v1.6-mistral-7b-hf` | Text, image | Vicuna-backed checkpoints need the Vicuna chat template. |
| `LLaVA` | `llava-hf/llava-1.5-7b-hf` | Text, image | |
| `VLlama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` | Text, image | |
| `Qwen2VL` | `Qwen/Qwen2-VL-7B-Instruct` | Text, image, video | |
| `Idefics3` | `HuggingFaceM4/Idefics3-8B-Llama3` | Text, image | SmolVLM uses the same loader path. |
| `MiniCpmO` | `openbmb/MiniCPM-o-2_6` | Text, image, audio | |
| `Phi4MM` | `microsoft/Phi-4-multimodal-instruct` | Text, image, audio | Audio and image can share one message. |
| `Qwen2_5VL` | `Qwen/Qwen2.5-VL-7B-Instruct` | Text, image, video | |
| `Gemma3` | `google/gemma-3-12b-it` | Text, image | Gated repo. |
| `Mistral3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | Text, image | Tool calling needs the bundled template. |
| `Llama4` | `meta-llama/Llama-4-Scout-17B-16E-Instruct` | Text, image | Up to 10M context with paged attention and TP. |
| `Gemma3n` | `google/gemma-3n-E4B-it` | Text, image, audio, video | Pick a MatFormer slice (model size) at load time. |
| `Qwen3VL` | `Qwen/Qwen3-VL-4B-Instruct` | Text, image, video | |
| `Qwen3VLMoE` | `Qwen/Qwen3-VL-235B-A22B-Instruct` | Text, image, video | Supports MoQE. |
| `Qwen3_5` | `Qwen/Qwen3.5-27B` | Text, image | |
| `Qwen3_5Moe` | `Qwen/Qwen3.5-35B-A3B` | Text, image | Supports MoQE. |
| `Voxtral` | `mistralai/Voxtral-Mini-3B-2507` | Text, audio | Mistral-native repo layout; auto-detected. |
| `Gemma4` | `google/gemma-4-E4B-it` | Text, image, audio, video | Mixed media in one message; strict tool grammar by default. |
| `DiffusionGemma` | `google/diffusiongemma-26B-A4B-it` | Text, image | Block-diffusion text generation. |

## Image generation

| Architecture | Example repo |
|---|---|
| `Flux` | `black-forest-labs/FLUX.1-schnell` |
| `FluxOffloaded` | `black-forest-labs/FLUX.1-schnell` |

`FluxOffloaded` loads the same FLUX checkpoints as `Flux` with CPU offload enabled for memory-constrained hosts. `FLUX.1-dev` requires HF license acceptance.

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

Text, multimodal, speech, and embedding models support ISQ at load time. Diffusion models (FLUX) do not; they load at native precision. Pre-quantized format availability (GGUF, [UQFF (Universal Quantized File Format)](/mistral.rs/reference/uqff-format/), GPTQ, AWQ) is per-model on Hugging Face.

## Speculative decoding

| Mode | Target architecture | Assistant checkpoint family | Guide |
|---|---|---|---|
| MTP | `Gemma4` | Gemma 4 assistant checkpoints, PagedAttention required | [Speculative decoding (MTP)](/mistral.rs/guides/perf/speculative-decoding/) |
