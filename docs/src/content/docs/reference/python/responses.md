---
title: Responses
description: "Response and streaming types returned by the engine."
sidebar:
  order: 5
---
## `DecisionResponse`

| Field | Type |
| --- | --- |
| `answers` | `dict[str, DecisionAnswer]` |
| `usage` | `DecisionUsage` |

### `DecisionResponse.to_dict`

```text
to_dict() -> dict[str, Any]
```


## `DecisionAnswer`

Fields depend on type; absent optional fields return None. to_dict preserves the wire object.

| Field | Type |
| --- | --- |
| `type` | `Literal['noul', 'choice', 'score', 'text', 'number', 'date', 'box', 'boxes']` |
| `noul` | `float` |
| `choice` | `str \| None` |
| `choices` | `list[str] \| None` |
| `confidence` | `float \| None` |
| `probabilities` | `dict[str, float]` |
| `score` | `float` |
| `legend` | `dict[str, str]` |
| `text` | `str \| None` |
| `span` | `list[int] \| None` |
| `found` | `float \| None` |
| `path` | `str \| None` |
| `number` | `float \| None` |
| `unit` | `str \| None` |
| `interval` | `list[float] \| None` |
| `date` | `str \| None` |
| `box` | `dict[str, float] \| None` |
| `boxes` | `list[dict[str, Any]]` |
| `count` | `int` |
| `at_least` | `bool` |
| `presence` | `list[float]` |

### `DecisionAnswer.to_dict`

```text
to_dict() -> dict[str, Any]
```


## `DecisionUsage`

| Field | Type |
| --- | --- |
| `input_tokens` | `int` |
| `output_tokens` | `int` |


## `ChatCompletionResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `choices` | `list[Choice]` | required |
| `created` | `int` | required |
| `model` | `str` | required |
| `system_fingerprint` | `str` | required |
| `object` | `str` | required |
| `usage` | `Usage` | required |
| `adapter_generation` | `str \| None` | `None` |
| `agentic_tool_calls` | `list[AgenticToolCallRecord] \| None` | `None` |
| `files` | `list[File] \| None` | `None` |
| `session_id` | `str \| None` | `None` |


## `ChatCompletionChunkResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `choices` | `list[ChunkChoice]` | required |
| `created` | `int` | required |
| `model` | `str` | required |
| `system_fingerprint` | `str` | required |
| `object` | `str` | required |
| `usage` | `Usage \| None` | `None` |
| `adapter_generation` | `str \| None` | `None` |
| `session_id` | `str \| None` | `None` |


## `AgenticToolCallRecord`

| Field | Type |
| --- | --- |
| `round` | `int` |
| `name` | `str` |
| `arguments` | `str` |
| `result_content` | `str` |
| `result_images_base64` | `list[str]` |
| `file_ids` | `list[str]` |


## `Choice`

| Field | Type | Default |
| --- | --- | --- |
| `finish_reason` | `str` | required |
| `index` | `int` | required |
| `message` | `ResponseMessage` | required |
| `logprobs` | `Logprobs \| None` | `None` |


## `ChunkChoice`

| Field | Type |
| --- | --- |
| `finish_reason` | `str \| None` |
| `index` | `int` |
| `delta` | `Delta` |
| `logprobs` | `ResponseLogprob \| None` |


## `Delta`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | required |
| `role` | `str` | required |
| `tool_calls` | `list[ToolCallResponse] \| None` | `None` |
| `reasoning_content` | `str \| None` | `None` |
| `audio` | `ChatAudioResponse \| None` | `None` |


## `ResponseMessage`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | required |
| `role` | `str` | required |
| `tool_calls` | `list[ToolCallResponse] \| None` | required |
| `reasoning_content` | `str \| None` | `None` |
| `audio` | `ChatAudioResponse \| None` | `None` |


## `ChatAudioResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str \| None` | `None` |
| `data` | `str \| None` | `None` |
| `expires_at` | `int \| None` | `None` |
| `transcript` | `str \| None` | `None` |


## `CompletionResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `choices` | `list[CompletionChoice]` | required |
| `created` | `int` | required |
| `model` | `str` | required |
| `system_fingerprint` | `str` | required |
| `object` | `str` | required |
| `usage` | `Usage` | required |
| `adapter_generation` | `str \| None` | `None` |


## `CompletionChoice`

| Field | Type | Default |
| --- | --- | --- |
| `finish_reason` | `str` | required |
| `index` | `int` | required |
| `text` | `str` | required |
| `logprobs` | `Logprobs \| None` | `None` |


## `Usage`

| Field | Type |
| --- | --- |
| `completion_tokens` | `int` |
| `prompt_tokens` | `int` |
| `total_tokens` | `int` |
| `avg_tok_per_sec` | `float` |
| `avg_prompt_tok_per_sec` | `float` |
| `avg_compl_tok_per_sec` | `float` |
| `total_time_sec` | `float` |
| `total_prompt_time_sec` | `float` |
| `total_completion_time_sec` | `float` |


## `Logprobs`

| Field | Type |
| --- | --- |
| `content` | `list[ResponseLogprob] \| None` |


## `ResponseLogprob`

| Field | Type |
| --- | --- |
| `token` | `str` |
| `logprob` | `float` |
| `bytes` | `list[int] \| None` |
| `top_logprobs` | `list[TopLogprob]` |


## `TopLogprob`

| Field | Type |
| --- | --- |
| `token` | `int` |
| `logprob` | `float` |
| `bytes` | `str \| None` |


## `ImageGenerationResponse`

| Field | Type |
| --- | --- |
| `data` | `list[ImageChoice]` |
| `created` | `int` |


## `ImageChoice`

| Field | Type |
| --- | --- |
| `url` | `str \| None` |
| `b64_json` | `str \| None` |


## `SpeechGenerationResponse`

This wraps PCM values, sampling rate and the number of channels.

| Field | Type |
| --- | --- |
| `pcm` | `list[float]` |
| `rate` | `int` |
| `channels` | `int` |


## `ToolCallResponse`

| Field | Type |
| --- | --- |
| `index` | `int` |
| `id` | `str` |
| `tp` | `ToolCallType` |
| `function` | `CalledFunction` |


## `ToolCallType`

| Member |
| --- |
| `ToolCallType.Function` |


## `CalledFunction`

| Field | Type |
| --- | --- |
| `name` | `str` |
| `arguments` | `str` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
