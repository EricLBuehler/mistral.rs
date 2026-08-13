---
title: GGUF compatibility
description: Model families, multimodal projectors, and storage formats supported for GGUF files.
---

This page is the compatibility reference for GGUF models. For commands and workflows, see
[Run GGUF models](/mistral.rs/guides/models/run-gguf/).

## Text model families

The `GGUF architecture` column is the value stored in the file's `general.architecture` metadata.
Fine-tunes and model sizes within a family use the same entry.

| Model family | GGUF architecture |
|---|---|
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
| Qwen3-Next and Qwen3-Coder-Next | `qwen3next` |
| Qwen3.5 and Qwen3.6 dense | `qwen35` |
| Qwen3.5 and Qwen3.6 MoE | `qwen35moe` |
| StarCoder2 | `starcoder2` |
| DeepSeek-V2, DeepSeek-V3, DeepSeek-R1 (non-distill), GLM-4 MoE Lite | `deepseek2` |
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

Some GGUF architectures cover more than one model family. Repository files and GGUF metadata are
used together to select the family. For a standalone file that cannot be identified unambiguously,
pass the original model with `--tok-model-id`.

GLM GGUFs configured for multimodal input are not supported as text-only models.

## Multimodal model families

Multimodal GGUF requires a compatible companion projector. Depending on the family, the loader may
also need the original configuration or processor assets. Repository loading selects supporting
files when the repository and GGUF metadata identify them unambiguously. The direct local
`-f /path/model.gguf` shorthand also selects an unambiguous projector stored beside the model.

| Model family | GGUF architecture |
|---|---|
| Gemma 3 | `gemma3` |
| Gemma 3n | `gemma3n` |
| Gemma 4 dense and MoE | `gemma4` |
| Idefics3 and SmolVLM | `llama` |
| Mistral 3 and Pixtral | `mistral3` |
| Llama 4 | `llama4` |
| LFM2-VL and LFM2.5-VL | `lfm2` |
| Qwen2-VL and Qwen2.5-VL | `qwen2vl` |
| Qwen3-VL | `qwen3vl` |
| Qwen3-VL MoE | `qwen3vlmoe` |
| Qwen3.5 and Qwen3.6 multimodal dense | `qwen35` |
| Qwen3.5 and Qwen3.6 multimodal MoE | `qwen35moe` |

Input modalities depend on the model. A listed family does not imply that every checkpoint accepts
image, audio, and video. Use the model card and [multimodal input guide](/mistral.rs/guides/models/multimodal-input/)
for its supported request types.

Multimodal Qwen3.5 and Qwen3.6 accept image and video input, but not audio. Gemma 4 accepts
image/video and audio when the matching components are present in the model configuration and
projector files.

The `gemma3n`, `gemma4`, `llama4`, `qwen2vl`, `qwen3vl`, and `qwen3vlmoe` architectures always need a
projector. Architectures that appear in both tables above, including `gemma3`, `llama`, `mistral3`,
`lfm2`, `qwen35`, and `qwen35moe`, load as text models when no projector is supplied.

## Storage formats

mistral.rs accepts GGUF files using the following storage types. A file can mix types, as common
`_K_M` and `_K_S` artifacts do.

| Category | Supported storage types |
|---|---|
| Floating point | `F32`, `F16`, `BF16` |
| Legacy block quants | `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1` |
| K-quants | `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`, `Q8_K` |
| GPT-OSS | The GPT-OSS MXFP4 representation |

IQ storage types, including IQ1, IQ2, IQ3, and IQ4 variants, are not supported for GGUF files yet.
Select a supported Q/K artifact instead. Other storage types not listed above are also unsupported.
Big-endian GGUF files are not supported.

## Feature compatibility

| Capability | GGUF support |
|---|---|
| Local exact-file loading | Yes, with `-f` |
| Hugging Face exact-file loading | Yes, with `-m` and `-f` |
| Automatic artifact selection | Yes, with `-m` and `--quant` |
| Tokenizer and chat-template discovery | From embedded GGUF metadata or supplied model assets |
| Multimodal projector discovery | From an unambiguous GGUF repository, or an adjacent projector with the direct local `-f` shorthand |
| Serving, tool calling, and agents | Same runtime paths as other loads; checkpoint and chat-template support still apply |
| Dynamic LoRA | Language-model adapters for compatible rotary layouts; adjacent-RoPE layouts are rejected |
| Multimodal LoRA | Language-model adapters only; projector, vision, and audio adapters are not supported |
| Legacy static LoRA | Text GGUF with the `phi3` architecture; not supported with multimodal GGUF |
| X-LoRA | Text GGUF with the `phi3` architecture; not supported with multimodal GGUF |
| ISQ requantization | Yes, for compatible weights selected with `-f` |
| Offline loading | Yes, when every required file is local or cached |

Dynamic LoRA is currently rejected for native GGUF architectures that store Q/K features in
adjacent rotary order: `llama`, `mistral3`, `deepseek2`, `glm4`, `smollm3`, `granite`,
`granitemoe`, `granitehybrid`, and `llama4`. This also covers multimodal models routed through those
architectures, including Idefics3, Mistral 3/Pixtral, and Llama 4. Base-model loading is unaffected.

GGUF support covers text generation and the multimodal families listed above. GGUF is not a
loading format for embedding, speech, diffusion, or image-generation pipelines.
