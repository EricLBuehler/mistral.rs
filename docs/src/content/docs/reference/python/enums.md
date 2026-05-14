---
title: Enums
description: "Architecture, dtype, and option enums."
sidebar:
  order: 6
---
## `Architecture`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `Architecture.Mistral` | `'mistral'` |
| `Architecture.Gemma` | `'gemma'` |
| `Architecture.Mixtral` | `'mixtral'` |
| `Architecture.Llama` | `'llama'` |
| `Architecture.Phi2` | `'phi2'` |
| `Architecture.Phi3` | `'phi3'` |
| `Architecture.Qwen2` | `'qwen2'` |
| `Architecture.Gemma2` | `'gemma2'` |
| `Architecture.Starcoder2` | `'starcoder2'` |
| `Architecture.Phi3_5MoE` | `'phi3.5moe'` |
| `Architecture.DeepseekV2` | `'deepseekv2'` |
| `Architecture.DeepseekV3` | `'deepseekv3'` |
| `Architecture.Qwen3` | `'qwen3'` |
| `Architecture.GLM4` | `'glm4'` |
| `Architecture.GLM4Moe` | `'glm4moe'` |
| `Architecture.GLM4MoeLite` | `'glm4moelite'` |
| `Architecture.Qwen3Moe` | `'qwen3moe'` |
| `Architecture.SmolLm3` | `'smollm3'` |
| `Architecture.GraniteMoeHybrid` | `'granitemoehybrid'` |
| `Architecture.GptOss` | `'gptoss'` |
| `Architecture.Qwen3Next` | `'qwen3next'` |


## `EmbeddingArchitecture`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `EmbeddingArchitecture.EmbeddingGemma` | `'embeddinggemma'` |
| `EmbeddingArchitecture.Qwen3Embedding` | `'qwen3embedding'` |


## `MultimodalArchitecture`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `MultimodalArchitecture.Phi3V` | `'phi3v'` |
| `MultimodalArchitecture.Idefics2` | `'idefics2'` |
| `MultimodalArchitecture.LLaVANext` | `'llava-next'` |
| `MultimodalArchitecture.LLaVA` | `'llava'` |
| `MultimodalArchitecture.VLlama` | `'vllama'` |
| `MultimodalArchitecture.Qwen2VL` | `'qwen2vl'` |
| `MultimodalArchitecture.Idefics3` | `'idefics3'` |
| `MultimodalArchitecture.MiniCpmO` | `'minicpmo'` |
| `MultimodalArchitecture.Phi4MM` | `'phi4mm'` |
| `MultimodalArchitecture.Qwen2_5VL` | `'qwen2_5vl'` |
| `MultimodalArchitecture.Gemma3` | `'gemma3'` |
| `MultimodalArchitecture.Mistral3` | `'mistral3'` |
| `MultimodalArchitecture.Llama4` | `'llama4'` |
| `MultimodalArchitecture.Gemma3n` | `'Gemma3n'` |
| `MultimodalArchitecture.Qwen3VL` | `'Qwen3VL'` |
| `MultimodalArchitecture.Qwen3VLMoE` | `'Qwen3VLMoE'` |
| `MultimodalArchitecture.Qwen3_5` | `'Qwen3_5'` |
| `MultimodalArchitecture.Qwen3_5Moe` | `'Qwen3_5Moe'` |
| `MultimodalArchitecture.Voxtral` | `'Voxtral'` |
| `MultimodalArchitecture.Gemma4` | `'Gemma4'` |


## `DiffusionArchitecture`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `DiffusionArchitecture.Flux` | `'flux'` |
| `DiffusionArchitecture.FluxOffloaded` | `'flux-offloaded'` |


## `SpeechLoaderType`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `SpeechLoaderType.Dia` | `'Dia'` |


## `ModelDType`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `ModelDType.Auto` | `'auto'` |
| `ModelDType.BF16` | `'bf16'` |
| `ModelDType.F16` | `'f16'` |
| `ModelDType.F32` | `'f32'` |


## `IsqOrganization`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `IsqOrganization.Default` | `'default'` |
| `IsqOrganization.MoQE` | `'moqe'` |


## `ImageGenerationResponseFormat`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `ImageGenerationResponseFormat.Url` | `'Url'` |
| `ImageGenerationResponseFormat.B64Json` | `'B64Json'` |


## `ToolChoice`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `ToolChoice.NoTools` | `'None'` |
| `ToolChoice.Auto` | `'Auto'` |


## `SearchContextSize`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `SearchContextSize.Low` | `'low'` |
| `SearchContextSize.Medium` | `'medium'` |
| `SearchContextSize.High` | `'high'` |


## `PagedCacheType`

Members and the corresponding config / serde names (used in TOML and the `arch` HTTP field). The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Config name |
| --- | --- |
| `PagedCacheType.Auto` | `0` |
| `PagedCacheType.F8E4M3` | `1` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
