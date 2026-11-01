---
title: Enums
description: "Architecture, dtype, and option enums."
sidebar:
  order: 6
---
## `Architecture`

| Member |
| --- |
| `Architecture.Mistral` |
| `Architecture.Gemma` |
| `Architecture.Mixtral` |
| `Architecture.Llama` |
| `Architecture.Phi2` |
| `Architecture.Phi3` |
| `Architecture.Qwen2` |
| `Architecture.Gemma2` |
| `Architecture.Starcoder2` |
| `Architecture.Phi3_5MoE` |
| `Architecture.DeepseekV2` |
| `Architecture.DeepseekV3` |
| `Architecture.Qwen3` |
| `Architecture.GLM4` |
| `Architecture.GLM4Moe` |
| `Architecture.GLM4MoeLite` |
| `Architecture.Qwen3Moe` |
| `Architecture.Flint` |
| `Architecture.SmolLm3` |
| `Architecture.GraniteMoeHybrid` |
| `Architecture.GptOss` |
| `Architecture.HunYuanDenseV1` |
| `Architecture.HunYuanMoEV1` |
| `Architecture.Qwen3Next` |
| `Architecture.Lfm2` |
| `Architecture.Lfm3` |
| `Architecture.Lfm2Moe` |


## `EmbeddingArchitecture`

| Member |
| --- |
| `EmbeddingArchitecture.EmbeddingGemma` |
| `EmbeddingArchitecture.Qwen3Embedding` |


## `MultimodalArchitecture`

| Member |
| --- |
| `MultimodalArchitecture.Phi3V` |
| `MultimodalArchitecture.Idefics2` |
| `MultimodalArchitecture.LLaVANext` |
| `MultimodalArchitecture.LLaVA` |
| `MultimodalArchitecture.Lfm2Vl` |
| `MultimodalArchitecture.Lfm3Vl` |
| `MultimodalArchitecture.Lfm3Audio` |
| `MultimodalArchitecture.Lfm3Omni` |
| `MultimodalArchitecture.Lfm3D1` |
| `MultimodalArchitecture.VLlama` |
| `MultimodalArchitecture.Qwen2VL` |
| `MultimodalArchitecture.Idefics3` |
| `MultimodalArchitecture.MiniCpmO` |
| `MultimodalArchitecture.Phi4MM` |
| `MultimodalArchitecture.Qwen2_5VL` |
| `MultimodalArchitecture.Gemma3` |
| `MultimodalArchitecture.Mistral3` |
| `MultimodalArchitecture.Llama4` |
| `MultimodalArchitecture.Gemma3n` |
| `MultimodalArchitecture.Qwen3VL` |
| `MultimodalArchitecture.Qwen3VLMoE` |
| `MultimodalArchitecture.Qwen3_5` |
| `MultimodalArchitecture.Qwen3_5Moe` |
| `MultimodalArchitecture.Qwen4Exp` |
| `MultimodalArchitecture.Voxtral` |
| `MultimodalArchitecture.Gemma4` |
| `MultimodalArchitecture.MuseGlimmer` |
| `MultimodalArchitecture.DiffusionGemma` |


## `DiffusionArchitecture`

| Member |
| --- |
| `DiffusionArchitecture.Flux` |
| `DiffusionArchitecture.FluxOffloaded` |


## `SpeechLoaderType`

| Member |
| --- |
| `SpeechLoaderType.Dia` |


## `ModelDType`

| Member | Description |
| --- | --- |
| `ModelDType.Auto` | Pick a type the device supports. |
| `ModelDType.BF16` |  |
| `ModelDType.F16` |  |
| `ModelDType.F32` |  |


## `IsqOrganization`

| Member | Description |
| --- | --- |
| `IsqOrganization.Default` | Quantize every layer that in-situ quantization can quantize. |
| `IsqOrganization.MoQE` | Quantize only the MoE expert weights and leave the other layers as they are. |


## `ImageGenerationResponseFormat`

| Member |
| --- |
| `ImageGenerationResponseFormat.Url` |
| `ImageGenerationResponseFormat.B64Json` |


## `ToolChoice`

| Member | Description |
| --- | --- |
| `ToolChoice.NoTools` | The model can't call tools. |
| `ToolChoice.Auto` | The model decides whether to call a tool. |


## `SearchContextSize`

| Member |
| --- |
| `SearchContextSize.Low` |
| `SearchContextSize.Medium` |
| `SearchContextSize.High` |


## `AgentPermission`

| Member |
| --- |
| `AgentPermission.Auto` |
| `AgentPermission.Ask` |
| `AgentPermission.Deny` |


## `CodeExecutionPermission`

| Member |
| --- |
| `CodeExecutionPermission.Auto` |
| `CodeExecutionPermission.Ask` |
| `CodeExecutionPermission.Deny` |


## `NetworkMode`

| Member | Description |
| --- | --- |
| `NetworkMode.NoNetwork` | No network access. On Linux, opening a socket fails. |
| `NetworkMode.Loopback` | Only `localhost` is reachable. |
| `NetworkMode.Full` | No network restrictions. |


## `AgentToolSource`

| Member |
| --- |
| `AgentToolSource.BuiltIn` |
| `AgentToolSource.User` |
| `AgentToolSource.Mcp` |
| `AgentToolSource.External` |


## `AgentToolKind`

| Member |
| --- |
| `AgentToolKind.CodeExecution` |
| `AgentToolKind.WebSearch` |
| `AgentToolKind.Shell` |
| `AgentToolKind.File` |
| `AgentToolKind.Custom` |
| `AgentToolKind.External` |


## `AgentToolApprovalDecisionKind`

| Member |
| --- |
| `AgentToolApprovalDecisionKind.Approve` |
| `AgentToolApprovalDecisionKind.Deny` |


## `PagedCacheType`

| Member | Description |
| --- | --- |
| `PagedCacheType.Auto` | Store the KV cache in the model's activation type. |
| `PagedCacheType.F8E4M3` | Store the KV cache in FP8 (E4M3), half the memory of BF16. Needs F16, BF16, or F32 activations. |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
