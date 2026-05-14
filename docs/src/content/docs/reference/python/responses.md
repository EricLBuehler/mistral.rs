---
title: Responses
description: "Response and streaming types returned by the engine."
sidebar:
  order: 5
---
### `ChatCompletionResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `choices` | `list[Choice]` | required |
| `created` | `int` | required |
| `model` | `str` | required |
| `system_fingerprint` | `str` | required |
| `object` | `str` | required |
| `usage` | `Usage` | required |
| `files` | `list[File] \| None` | `None` |
| `session_id` | `str \| None` | `None` |


### `ChatCompletionChunkResponse`

| Field | Type | Default |
| --- | --- | --- |
| `id` | `str` | required |
| `choices` | `list[ChunkChoice]` | required |
| `created` | `int` | required |
| `model` | `str` | required |
| `system_fingerprint` | `str` | required |
| `object` | `str` | required |
| `usage` | `Usage \| None` | `None` |
| `session_id` | `str \| None` | `None` |


### `Choice`

| Field | Type |
| --- | --- |
| `finish_reason` | `str` |
| `index` | `int` |
| `message` | `ResponseMessage` |
| `logprobs` | `Logprobs` |


### `ChunkChoice`

| Field | Type |
| --- | --- |
| `finish_reason` | `str \| None` |
| `index` | `int` |
| `delta` | `Delta` |
| `logprobs` | `ResponseLogprob \| None` |


### `Delta`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | required |
| `role` | `str` | required |
| `tool_calls` | `list[ToolCallResponse] \| None` | `None` |
| `reasoning_content` | `str \| None` | `None` |


### `ResponseMessage`

| Field | Type | Default |
| --- | --- | --- |
| `content` | `str \| None` | required |
| `role` | `str` | required |
| `tool_calls` | `list[ToolCallResponse] \| None` | required |
| `reasoning_content` | `str \| None` | `None` |


### `CompletionResponse`

| Field | Type |
| --- | --- |
| `id` | `str` |
| `choices` | `list[CompletionChoice]` |
| `created` | `int` |
| `model` | `str` |
| `system_fingerprint` | `str` |
| `object` | `str` |
| `usage` | `Usage` |


### `CompletionChoice`

| Field | Type |
| --- | --- |
| `finish_reason` | `str` |
| `index` | `int` |
| `text` | `str` |


### `Usage`

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


### `Logprobs`

| Field | Type |
| --- | --- |
| `content` | `list[ResponseLogprob] \| None` |


### `ResponseLogprob`

| Field | Type |
| --- | --- |
| `token` | `str` |
| `logprob` | `float` |
| `bytes` | `list[int]` |
| `top_logprobs` | `list[TopLogprob]` |


### `TopLogprob`

| Field | Type |
| --- | --- |
| `token` | `int` |
| `logprob` | `float` |
| `bytes` | `str` |


### `ImageGenerationResponse`

| Field | Type |
| --- | --- |
| `data` | `list[ImageChoice]` |
| `created` | `int` |


### `ImageChoice`

| Field | Type |
| --- | --- |
| `url` | `str \| None` |
| `b64_json` | `str \| None` |


### `SpeechGenerationResponse`

This wraps PCM values, sampling rate and the number of channels.

| Field | Type |
| --- | --- |
| `pcm` | `list[float]` |
| `rate` | `int` |
| `channels` | `int` |


### `ToolCallResponse`

| Field | Type |
| --- | --- |
| `id` | `str` |
| `type` | `ToolCallType` |
| `function` | `CalledFunction` |


### `ToolCallType`

| Member | Value |
| --- | --- |
| `ToolCallType.Function` | `'function'` |


### `CalledFunction`

| Field | Type |
| --- | --- |
| `name` | `str` |
| `arguments` | `str` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
