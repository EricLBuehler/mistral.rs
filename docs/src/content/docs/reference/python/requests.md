---
title: Requests
description: "Request dataclasses passed to Runner methods."
sidebar:
  order: 4
---
## `LoraAdapterGeneration`

Select one exact immutable LoRA adapter generation by its 64-character ID.

| Field | Type |
| --- | --- |
| `generation` | `str` |


## `SynthIdTextWatermarkConfig`

SynthID-Text configuration; key is 64 hexadecimal characters and has no default.

### `SynthIdTextWatermarkConfig.__init__`

```text
__init__(key: str, *, ngram_len: int = 5, depth: int = 30) -> None
```

### `SynthIdTextWatermarkConfig.detect`

```text
detect(
    tokens: list[int],
    prompt_len: int = 0,
    eos_token_ids: list[int] | None = None,
) -> tuple[float | None, int]
```

Return (mean g-value, tokens scored), without a calibrated detection threshold.


## `WatermarkConfig`

Select a watermark scheme; key is 64 hexadecimal characters and has no default.

### `WatermarkConfig.__init__`

```text
__init__(
    key: str,
    *,
    scheme: str = 'synthid',
    **parameters: Any,
) -> None
```

Schemes: synthid, kgw, unigram, exponential, inverse_transform, mpac, textgrain, semstamp.

Parameters match the [watermark configuration](/guides/customize/watermarking/).
SemStamp requires sentence embeddings and cannot be used for token generation.

### `WatermarkConfig.scheme`

```text
scheme() -> str
```

### `WatermarkConfig.detect`

```text
detect(
    tokens: list[int],
    prompt_len: int = 0,
    eos_token_ids: list[int] | None = None,
) -> dict[str, Any]
```

Return scheme-specific evidence without a calibrated detection threshold.

### `WatermarkConfig.detect_embeddings`

```text
detect_embeddings(
    embeddings: list[list[float]],
    prompt_len: int = 0,
    *,
    device_name: Literal['cpu', 'cuda', 'metal'] = 'cpu',
) -> dict[str, Any]
```

Return SemStamp evidence using a fixed sentence encoder and the chosen backend.

### `WatermarkConfig.accepts_embedding`

```text
accepts_embedding(
    previous: list[float],
    candidate: list[float],
) -> bool
```

Check a SemStamp sentence transition on the CPU.


## `ChatCompletionRequest`

A ChatCompletionRequest represents a request sent to the mistral.rs engine. It encodes information
about input data, sampling, and how to return the response.

The messages type is as follows: (for normal chat completion, for chat completion with images, pretemplated prompt)

Agent permission fields:

- `agent_permission`: `AgentPermission.Auto`, `.Ask`, or `.Deny`. Applies to server-executed
  agent actions such as code execution, shell, web search, file tools, callbacks,
  and external tool dispatch.
- `agent_approval_callback`: called when `agent_permission=AgentPermission.Ask` with an
  `AgentToolApproval`. Return `True`, `False`, or
  `AgentToolApprovalDecision`.

See [agent permissions](/guides/agents/permissions-and-approvals/)
for the shared CLI, HTTP, Python, and Rust behavior.

`reasoning_effort` accepts `off`, `low`, `medium`, `high`, and `xhigh`; `none`
aliases `off`. Values are trimmed and case-insensitive. If both reasoning controls
are omitted, thinking is enabled with no selected effort. Contradictory
`enable_thinking` and `reasoning_effort` values raise `ValueError`.

| Field | Type | Default |
| --- | --- | --- |
| `messages` | `list[dict[str, str]] \| list[dict[str, list[dict[str, str \| dict[str, str]]]]] \| str` | required |
| `model` | `str` | required |
| `logprobs` | `bool` | `False` |
| `n_choices` | `int` | `1` |
| `logit_bias` | `dict[int, float] \| None` | `None` |
| `top_logprobs` | `int \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `presence_penalty` | `float \| None` | `None` |
| `frequency_penalty` | `float \| None` | `None` |
| `repetition_penalty` | `float \| None` | `None` |
| `stop_seqs` | `list[str] \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `top_k` | `int \| None` | `None` |
| `stream` | `bool` | `False` |
| `grammar` | `str \| None` | `None` |
| `grammar_type` | `str \| None` | `None` |
| `min_p` | `float \| None` | `None` |
| `tool_schemas` | `list[str] \| None` | `None` |
| `tool_choice` | `ToolChoice \| None` | `None` |
| `dry_multiplier` | `float \| None` | `None` |
| `dry_base` | `float \| None` | `None` |
| `dry_allowed_length` | `int \| None` | `None` |
| `dry_sequence_breakers` | `list[str] \| None` | `None` |
| `web_search_options` | `WebSearchOptions \| None` | `None` |
| `enable_thinking` | `bool \| None` | `None` |
| `truncate_sequence` | `bool` | `False` |
| `reasoning_effort` | `Literal['off', 'none', 'low', 'medium', 'high', 'xhigh'] \| None` | `None` |
| `max_tool_rounds` | `int \| None` | `None` |
| `tool_dispatch_url` | `str \| None` | `None` |
| `enable_code_execution` | `bool` | `False` |
| `enable_shell` | `bool` | `False` |
| `shell_skills` | `list[ShellSkillMount] \| None` | `None` |
| `agent_permission` | `AgentPermission \| None` | `None` |
| `agent_approval_callback` | `Callable[[AgentToolApproval], bool \| AgentToolApprovalDecision] \| None` | `None` |
| `code_execution_permission` | `CodeExecutionPermission \| None` | `None` |
| `session_id` | `str \| None` | `None` |
| `files` | `list[RequestedFile] \| None` | `None` |
| `input_files` | `list[InputFile] \| None` | `None` |
| `ignore_eos` | `bool` | `False` |
| `adapter` | `str \| LoraAdapterGeneration \| None` | `None (keyword-only)` |
| `watermark` | `WatermarkConfig \| SynthIdTextWatermarkConfig \| None` | `None (keyword-only)` |


## `CompletionRequest`

A CompletionRequest represents a request sent to the mistral.rs engine. It encodes information
about input data, sampling, and how to return the response.

| Field | Type | Default |
| --- | --- | --- |
| `prompt` | `str` | required |
| `model` | `str` | required |
| `best_of` | `int` | `1` |
| `echo_prompt` | `bool` | `False` |
| `presence_penalty` | `float \| None` | `None` |
| `frequency_penalty` | `float \| None` | `None` |
| `repetition_penalty` | `float \| None` | `None` |
| `logit_bias` | `dict[int, float] \| None` | `None` |
| `max_tokens` | `int \| None` | `None` |
| `n_choices` | `int` | `1` |
| `stop_seqs` | `list[str] \| None` | `None` |
| `temperature` | `float \| None` | `None` |
| `top_p` | `float \| None` | `None` |
| `suffix` | `str \| None` | `None` |
| `top_k` | `int \| None` | `None` |
| `grammar` | `str \| None` | `None` |
| `grammar_type` | `str \| None` | `None` |
| `min_p` | `float \| None` | `None` |
| `tool_schemas` | `list[str] \| None` | `None` |
| `tool_choice` | `ToolChoice \| None` | `None` |
| `dry_multiplier` | `float \| None` | `None` |
| `dry_base` | `float \| None` | `None` |
| `dry_allowed_length` | `int \| None` | `None` |
| `dry_sequence_breakers` | `list[str] \| None` | `None` |
| `truncate_sequence` | `bool` | `False` |
| `ignore_eos` | `bool` | `False` |
| `adapter` | `str \| LoraAdapterGeneration \| None` | `None (keyword-only)` |
| `watermark` | `WatermarkConfig \| SynthIdTextWatermarkConfig \| None` | `None (keyword-only)` |


## `EmbeddingRequest`

An EmbeddingRequest represents a request to compute embeddings for the provided input text.

| Field | Type | Default |
| --- | --- | --- |
| `input` | `str \| list[str] \| list[int] \| list[list[int]]` | required |
| `truncate_sequence` | `bool` | `False` |

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
