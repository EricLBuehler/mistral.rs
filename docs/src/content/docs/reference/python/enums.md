---
title: Enums
description: "Architecture, dtype, and option enums."
sidebar:
  order: 6
---
## `Architecture`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
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

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `EmbeddingArchitecture.EmbeddingGemma` | `'embeddinggemma'` |
| `EmbeddingArchitecture.Qwen3Embedding` | `'qwen3embedding'` |


## `MultimodalArchitecture`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
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

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `DiffusionArchitecture.Flux` | `'flux'` |
| `DiffusionArchitecture.FluxOffloaded` | `'flux-offloaded'` |


## `SpeechLoaderType`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `SpeechLoaderType.Dia` | `'Dia'` |


## `ModelDType`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `ModelDType.Auto` | `'auto'` |
| `ModelDType.BF16` | `'bf16'` |
| `ModelDType.F16` | `'f16'` |
| `ModelDType.F32` | `'f32'` |


## `IsqOrganization`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `IsqOrganization.Default` | `'default'` |
| `IsqOrganization.MoQE` | `'moqe'` |


## `ImageGenerationResponseFormat`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `ImageGenerationResponseFormat.Url` | `'Url'` |
| `ImageGenerationResponseFormat.B64Json` | `'B64Json'` |


## `ToolChoice`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `ToolChoice.NoTools` | `'None'` |
| `ToolChoice.Auto` | `'Auto'` |


## `SearchContextSize`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `SearchContextSize.Low` | `'low'` |
| `SearchContextSize.Medium` | `'medium'` |
| `SearchContextSize.High` | `'high'` |


## `AgentPermission`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `AgentPermission.Auto` | `'auto'` |
| `AgentPermission.Ask` | `'ask'` |
| `AgentPermission.Deny` | `'deny'` |


## `CodeExecutionPermission`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `CodeExecutionPermission.Auto` | `'auto'` |
| `CodeExecutionPermission.Ask` | `'ask'` |
| `CodeExecutionPermission.Deny` | `'deny'` |


## `NetworkMode`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `NetworkMode.NoNetwork` | `'none'` |
| `NetworkMode.Loopback` | `'loopback'` |
| `NetworkMode.Full` | `'full'` |


## `AgentToolSource`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `AgentToolSource.BuiltIn` | `'built_in'` |
| `AgentToolSource.User` | `'user'` |
| `AgentToolSource.Mcp` | `'mcp'` |
| `AgentToolSource.External` | `'external'` |


## `AgentToolKind`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `AgentToolKind.CodeExecution` | `'code_execution'` |
| `AgentToolKind.WebSearch` | `'web_search'` |
| `AgentToolKind.File` | `'file'` |
| `AgentToolKind.Custom` | `'custom'` |
| `AgentToolKind.External` | `'external'` |


## `AgentToolApprovalDecisionKind`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `AgentToolApprovalDecisionKind.Approve` | `'approve'` |
| `AgentToolApprovalDecisionKind.Deny` | `'deny'` |


## `PagedCacheType`

Members and their wire/config names where relevant. The members are fieldless PyO3 enum variants and do not expose `.value`.

| Member | Wire/config name |
| --- | --- |
| `PagedCacheType.Auto` | `0` |
| `PagedCacheType.F8E4M3` | `1` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
