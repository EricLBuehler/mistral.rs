---
title: GGUF compatibility
description: Supported GGUF architectures, projectors, storage formats, and feature limits.
---

For loading commands, see [Run GGUF models](/guides/models/run-gguf/). The architecture column below is the file's `general.architecture` value. Fine-tunes share their base family's entry; checkpoint capabilities and chat-template requirements still apply.

## Text model families

| Model family | GGUF architecture |
| --- | --- |
| Llama, Mistral, Mixtral | `llama` |
| Mistral 3 text weights | `mistral3` |
| Gemma | `gemma` |
| Gemma 2 | `gemma2` |
| Gemma 3 text weights | `gemma3` |
| Phi-2 | `phi2` |
| Phi-3 and Phi-3.5 | `phi3` |
| Phi-3.5 MoE | `phimoe` |
| Qwen2 and Qwen2.5 | `qwen2` |
| Qwen3 | `qwen3` |
| Qwen3 MoE | `qwen3moe` |
| Flint | `flint` |
| Qwen3-Next and Qwen3-Coder-Next | `qwen3next` |
| Qwen3.5 and Qwen3.6 dense | `qwen35` |
| Qwen3.5 and Qwen3.6 MoE | `qwen35moe` |
| Qwen3.8-Flash-Next | `qwen4exp` |
| StarCoder2 | `starcoder2` |
| DeepSeek-V2/V3, non-distill R1, GLM-4 MoE Lite | `deepseek2` |
| GLM-4 dense | `glm4` |
| GLM-4 MoE | `glm4moe` |
| SmolLM3 | `smollm3` |
| Granite dense | `granite` |
| Granite MoE | `granitemoe` |
| Granite hybrid | `granitehybrid` |
| GPT-OSS | `gpt-oss` |
| Hunyuan dense | `hunyuan-dense` |
| Hunyuan MoE | `hunyuan-moe` |
| LFM2 and LFM2.5 dense | `lfm2` |
| LFM2 and LFM2.5 MoE | `lfm2moe` |
| LFM3 | `lfm3` |

Shared architecture names may need original assets to identify the family; use `--tok-model-id` when identification is ambiguous. GLM files configured for multimodal input cannot load as text-only models.

## Multimodal model families

These models need a matching companion projector. The CLI selects one from an unambiguous repository or beside a direct local `-f` file. Some families also need original config/processor assets.

| Model family | GGUF architecture |
| --- | --- |
| Gemma 3 | `gemma3` |
| Gemma 3n | `gemma3n` |
| Gemma 4 dense and MoE | `gemma4` |
| Idefics3 and SmolVLM | `llama` |
| Mistral 3 and Pixtral | `mistral3` |
| Llama 4 | `llama4` |
| LFM2-VL and LFM2.5-VL | `lfm2` |
| LFM3-VL, LFM3-Audio, LFM3-Omni | `lfm3` |
| d1 and d1-Omni | `lfm3` |
| Muse Glimmer | `muse-glimmer` |
| Qwen2-VL and Qwen2.5-VL | `qwen2vl` |
| Qwen3-VL | `qwen3vl` |
| Qwen3-VL MoE | `qwen3vlmoe` |
| Qwen3.5 and Qwen3.6 dense | `qwen35` |
| Qwen3.5 and Qwen3.6 MoE | `qwen35moe` |
| Qwen3.8-Flash-Next | `qwen4exp` |

Check the checkpoint's input modalities; a multimodal entry does not imply audio or video support.

- **LFM3:** configuration and preprocessing are reconstructed from main/projector files. VL accepts images/video; Audio accepts audio and generates speech; Omni combines those capabilities. No original model assets are required.
- **d1:** standalone files support `noul`, single-selection `choice`, and `score`. Optional matching `config.json` and `calibration.json` supplied through `--tok-model-id` enable the [full decision API](/guides/models/use-decision-models/#use-gguf).
- **Muse Glimmer:** published GGUFs need `--tok-model-id meta-models/Muse-Glimmer-30B`. Images work; video is rejected because conversion merges temporal patch weights irreversibly.
- **Qwen3.5/3.6:** images and video work; audio is unsupported. **Gemma 4** additionally supports audio when the matching components are present.

`gemma3n`, `gemma4`, `llama4`, `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, and `muse-glimmer` always require a projector. Dual text/multimodal architectures can load text-only without one, except d1 decision models, which require their projector even for text questions.

## Storage formats

A file may mix supported types, as `_K_M` and `_K_S` artifacts commonly do.

| Category | Types |
| --- | --- |
| Floating point | F32, F16, BF16 |
| Legacy block quants | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
| GPT-OSS | Its MXFP4 representation |

IQ types are unsupported for matrix projections. IQ4_NL is supported by embedding gather, including Qwen3.8-Flash-Next's PLE table. Choose Q/K artifacts for other weights. Unlisted formats, including NVFP4, and big-endian files are unsupported.

## Feature compatibility

| Capability | Support |
| --- | --- |
| Exact local / Hub files | `-f`, with `-m` for a Hub repository |
| Artifact selection | `-m` with `--quant` |
| Tokenizer/template | Embedded metadata or supplied model assets |
| Projector discovery | Unambiguous repository or adjacent direct-local file |
| Serving and tools | Normal model APIs; model/template restrictions still apply |
| Dynamic LoRA | Compatible language-model weights; see rotary-layout restriction below |
| Multimodal adapters | Language model only; no media encoder/projector adapters |
| Legacy static LoRA / X-LoRA | Text `phi3` only; no multimodal GGUF |
| ISQ requantization | Compatible weights in an exact `-f` file |
| Offline loading | All required files must be local or cached |

Dynamic LoRA rejects adjacent rotary layouts: `llama`, `mistral3`, `deepseek2`, `glm4`, `smollm3`, `granite`, `granitemoe`, `granitehybrid`, `llama4`, and `muse-glimmer`. This includes multimodal families routed through those architectures. Base loading is unaffected.

LFM3-Audio/Omni speech output works through their multimodal GGUF loaders. Dedicated speech models such as Dia, embedding models, and image-generation/diffusion pipelines do not load from GGUF.
