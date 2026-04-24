---
title: Responses
description: "Response and streaming types returned by the engine."
sidebar:
  order: 5
---
### `ChatCompletionResponse`

| Field | Type |
| --- | --- |
| `id` | `str` |
| `choices` | `list[Choice]` |
| `created` | `int` |
| `model` | `str` |
| `system_fingerprint` | `str` |
| `object` | `str` |
| `usage` | `Usage` |


### `ChatCompletionChunkResponse`

| Field | Type |
| --- | --- |
| `id` | `str` |
| `choices` | `list[ChunkChoice]` |
| `created` | `int` |
| `model` | `str` |
| `system_fingerprint` | `str` |
| `object` | `str` |


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

| Field | Type |
| --- | --- |
| `content` | `str` |
| `role` | `str` |


### `ResponseMessage`

| Field | Type |
| --- | --- |
| `content` | `str` |
| `role` | `str` |
| `tool_calls` | `list[ToolCallResponse]` |


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
| `choices` | `list[ImageChoice]` |
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
