---
title: GGUF compatibility
description: Supported GGUF architectures, projectors, storage formats, and feature limits.
---

Which GGUF files mistral.rs can load. For loading commands, see [Run GGUF models](/guides/models/run-gguf/).

To find a file's row, match the architecture column against the file's `general.architecture` metadata value. Fine-tunes use their base family's row.

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

Some architecture names, such as `llama`, cover several families. If mistral.rs can't tell which family a standalone file belongs to, pass the original model with `--tok-model-id`.

GLM files configured for multimodal input can't load as text-only models.

## Multimodal model families

A multimodal GGUF model comes as two files: the language model and a projector, usually named `mmproj-*.gguf`, which holds the vision or audio encoder. The CLI finds the projector in the Hub repository, or in the same folder as a local `-f` file. If there are several and it can't tell which to use, pass `--mmproj <file>`. Some families also need the original model's config or processor files; the notes below the table say which.

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

A listed family doesn't necessarily take every input type. Check the model card for whether it accepts images, audio, or video. Family notes:

- **LFM3:** loads from the GGUF files alone. LFM3-VL takes images and video, LFM3-Audio takes audio and generates speech, and LFM3-Omni does both.
- **d1:** the GGUF files alone answer `noul` (yes/no), single-answer `choice`, and `score` questions. For every [question type](/reference/decisions/#question-types), also pass the original model with `--tok-model-id`. See [d1 with GGUF](/guides/models/use-decision-models/#use-gguf).
- **Muse Glimmer:** published GGUFs need `--tok-model-id meta-models/Muse-Glimmer-30B`. Images work. Video doesn't, because the llama.cpp conversion irreversibly merges the projector weights that video needs.
- **Qwen3.5 and Qwen3.6:** images and video, no audio.
- **Gemma 4:** images and video, plus audio when the model and projector files include the audio components.

These architectures always need a projector: `gemma3n`, `gemma4`, `llama4`, `qwen2vl`, `qwen3vl`, `qwen3vlmoe`, and `muse-glimmer`. Architectures in both tables load as text-only models when there's no projector, except d1, which always needs its projector.

## Storage formats

A file can mix the types below, as `_K_M` and `_K_S` files usually do.

| Category | Types |
| --- | --- |
| Floating point | F32, F16, BF16 |
| Legacy block quants | Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1 |
| K-quants | Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K |
| GPT-OSS | GPT-OSS's MXFP4 weights |

Not supported:

- IQ types, such as IQ4_XS, so pick a Q or K file instead. The one exception is IQ4_NL in embedding tables, such as Qwen3.8-Flash-Next's per-layer embedding table.
- Other types not listed above, including NVFP4.
- Big-endian files.

## Feature compatibility

| Capability | Support |
| --- | --- |
| Load a specific file | `-f <file>` for a local file, or `-m <repo> -f <file>` from the Hub. |
| Pick a quantization from a repo | `-m <repo> --quant <bits or name>`, such as `--quant 4` or `--quant q8_0`. `--quant auto` isn't supported. |
| Tokenizer and chat template | Read from the GGUF metadata, or from the original model passed with `--tok-model-id`. |
| Projector for multimodal models | Found in the repo or next to a local `-f` file. Pass `--mmproj` to choose one yourself. |
| Serving, tool calling, agents | Same as other formats. The model and its chat template still need to support tool calling. |
| Dynamic LoRA | Language-model adapters, except on the architectures listed below. |
| LoRA on multimodal models | Language-model adapters only. Vision, audio, and projector adapters aren't supported. |
| Static LoRA and X-LoRA | Text-only `phi3` files only. |
| ISQ requantization | Pass `--isq` with an exact `-f` file. |
| Offline loading | Works when every required file is local or cached. |

Dynamic LoRA doesn't work with GGUF files for these architectures: `llama`, `mistral3`, `deepseek2`, `glm4`, `smollm3`, `granite`, `granitemoe`, `granitehybrid`, `llama4`, and `muse-glimmer`. That includes multimodal models built on them, such as Idefics3, Mistral 3, Pixtral, and Llama 4. The base model still loads; use the original safetensors checkpoint if you need adapters.

LFM3-Audio and LFM3-Omni can generate speech from GGUF. Dedicated speech models such as Dia, embedding models, and image-generation models don't load from GGUF.
