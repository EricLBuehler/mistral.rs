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
are omitted, thinking is enabled for text output and disabled for speech output,
with no selected effort. Contradictory
`enable_thinking` and `reasoning_effort` values raise `ValueError`.

Audio input content parts accept `{"type": "input_audio", "input_audio":
{"data": "<base64>", "format": "wav"}}` for WAV or MP3 clips. Other decodable
formats can use the `audio_url` extension. For models with speech output,
`modalities` selects audio or text plus audio, and `audio` accepts `format`
(`wav`, `pcm16`, `mp3`, `flac`, `opus`, or `aac`) and optional `voice="default"`.
Compressed output requires FFmpeg. Streaming requires PCM16 and returns
incremental `delta.audio.data` chunks. HTTP `audio.id` history replay is not
available through the native SDK.
Text sampling controls do not configure LFM3's separate speech-code sampler.

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
| `modalities` | `list[Literal['text', 'audio']] \| None` | `None (keyword-only)` |
| `audio` | `dict[str, str] \| None` | `None (keyword-only)` |


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


## `EmbeddingRequest`

An EmbeddingRequest represents a request to compute embeddings for the provided input text.

| Field | Type | Default |
| --- | --- | --- |
| `input` | `str \| list[str] \| list[int] \| list[list[int]]` | required |
| `truncate_sequence` | `bool` | `False` |


## `DecisionQuestion`

A typed decision question; spoken instructions accept audio_url/input_audio content parts.

### `DecisionQuestion.__init__`

```text
__init__(
    kind: Literal['noul', 'choice', 'score', 'text', 'number', 'date', 'box', 'boxes'],
    instructions: str | dict[str, Any],
    *,
    criteria: dict[str, Any] | list[str] | None = None,
    multiple: bool = False,
    range: tuple[float, float] | None = None,
    step: float | None = None,
    unit: str | None = None,
    max: int | None = None,
) -> None
```

### `DecisionQuestion.to_dict`

```text
to_dict() -> dict[str, Any]
```


## `DecisionRequest`

State is any JSON value. Media accepts image_url, audio_url, input_audio, and video_url parts.

### `DecisionRequest.__init__`

```text
__init__(
    state: Any,
    questions: dict[str, DecisionQuestion | dict[str, Any]],
    *,
    images: list[str] | None = None,
    media: list[dict[str, Any]] | None = None,
    set_size: int | None = None,
) -> None
```

---

<small>Generated from [`mistralrs-pyo3/mistralrs.pyi`](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs-pyo3/mistralrs.pyi).</small>
