---
title: Requests
description: "Request dataclasses passed to Runner methods."
sidebar:
  order: 4
---
### `ChatCompletionRequest`

A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
about input data, sampling, and how to return the response.

The messages type is as follows: (for normal chat completion, for chat completion with images, pretemplated prompt)

| Field | Type | Default |
| --- | --- | --- |
| `messages` | `list[dict[str, str]] \| list[dict[str, list[dict[str, str \| dict[str, str]]]]] \| str` | required |
| `model` | `str` | required |
| `logit_bias` | `dict[int, float] \| None` | `None` |
| `logprobs` | `bool` | `False` |
| `top_logprobs` | `int \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `n_choices` | `int` | `1` |
| `presence_penalty` | `float \| None` | `None` |
| `frequency_penalty` | `float \| None` | `None` |
| `stop_seqs` | `list[str] \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `stream` | `bool` | `False` |
| `top_k` | `int \| None` | `None` |
| `grammar` | `str \| None` | `None` |
| `grammar_type` | `str \| None` | `None` |
| `min_p` | `float \| None` | `None` |
| `min_p` | `float \| None` | `None` |
| `tool_schemas` | `list[str] \| None` | `None` |
| `tool_choice` | `ToolChoice \| None` | `None` |
| `web_search_options` | `WebSearchOptions \| None` | `None` |
| `enable_thinking` | `bool \| None` | `None` |
| `truncate_sequence` | `bool` | `False` |
| `enable_code_execution` | `bool` | `False` |
| `session_id` | `str \| None` | `None` |


### `CompletionRequest`

A CompletionRequest represents a request sent to the mistral.rs engine. It encodes information
about input data, sampling, and how to return the response.

| Field | Type | Default |
| --- | --- | --- |
| `prompt` | `str` | required |
| `model` | `str` | required |
| `echo_prompt` | `bool` | `False` |
| `logit_bias` | `dict[int, float] \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `n_choices` | `int` | `1` |
| `best_of` | `int` | `1` |
| `presence_penalty` | `float \| None` | `None` |
| `frequency_penalty` | `float \| None` | `None` |
| `stop_seqs` | `list[str] \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `top_k` | `int \| None` | `None` |
| `suffix` | `str \| None` | `None` |
| `grammar` | `str \| None` | `None` |
| `grammar_type` | `str \| None` | `None` |
| `min_p` | `float \| None` | `None` |
| `truncate_sequence` | `bool` | `False` |
| `tool_schemas` | `list[str] \| None` | `None` |
| `tool_choice` | `ToolChoice \| None` | `None` |


### `EmbeddingRequest`

An EmbeddingRequest represents a request to compute embeddings for the provided input text.

| Field | Type | Default |
| --- | --- | --- |
| `input` | `str \| list[str] \| list[int] \| list[list[int]]` | required |
| `truncate_sequence` | `bool` | `False` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
