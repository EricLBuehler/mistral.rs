---
title: Requests
description: "Request dataclasses passed to Runner methods."
sidebar:
  order: 4
---
## `LoraAdapterGeneration`

Pins a request to one exact version of a LoRA adapter, instead of whatever its alias points to now.

Each set of adapter files you load is a generation, identified by a 64-character hash of its config
and weights. Loading different files under an alias gives it a new generation. Loading the same
files again gives the same one. See
[pin an exact adapter version](/guides/customize/lora-adapters/#pin-an-exact-adapter-version).

Get one from `LoraAdapterInfo.exact()`, or pass a generation ID. A malformed ID raises `ValueError`.

| Field | Type | Description |
| --- | --- | --- |
| `generation` | `str` | The 64-character generation ID. |


## `ChatCompletionRequest`

A chat request. Send it with `Runner.send_chat_completion_request`.

`messages` takes one of three forms:

- A list of `{"role": ..., "content": "..."}` dicts.
- A list of dicts whose `content` is a list of parts, for images, audio, or video. Part types are
  `text`, `image_url`, `input_audio`, `audio_url`, and `video_url`. Only `user` messages can hold media.
- A plain string, sent as a single user message.

Sampling fields left as `None` use the model's `generation_config.json` values when it has them.

Reasoning controls:

- `reasoning_effort` accepts `off`, `low`, `medium`, `high`, and `xhigh`. `none` means `off` and
  `max` means `xhigh`. Case and surrounding spaces are ignored.
- If you set neither control, thinking is on for text output and off for speech output.
- Contradictory values, such as `enable_thinking=False` with `reasoning_effort="high"`, raise `ValueError`.

Audio:

- Input: add a part `{"type": "input_audio", "input_audio": {"data": "<base64>", "format": "wav"}}`.
  `format` is `wav` or `mp3`. For other formats, use an `audio_url` part.
- Speech output: set `modalities=["audio"]` or `["text", "audio"]`, and `audio={"format": ...}`.
  Formats are `wav`, `pcm16`, `mp3`, `flac`, `opus`, and `aac`. All but `wav` and `pcm16` need FFmpeg.
  `voice` can be omitted or set to `"default"`.
- Streaming speech requires `pcm16` and arrives as `delta.audio.data` chunks.
- The HTTP API's `audio.id` history replay isn't available here.
- Text sampling fields don't change LFM3's separate speech sampler.

For the agent fields, see [agent permissions](/guides/agents/permissions-and-approvals/).

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `messages` | `list[dict[str, str]] \| list[dict[str, list[dict[str, str \| dict[str, str]]]]] \| str` | required | The conversation, in one of the forms above. |
| `model` | `str` | required | Not used to choose a model. Pass `model_id` to the `Runner` method instead. |
| `logprobs` | `bool` | `False` | Return the log probability of each output token. |
| `n_choices` | `int` | `1` | Number of completions to generate. |
| `logit_bias` | `dict[int, float] \| None` | `None` | Map of token ID to a bias added to that token's logit before sampling. |
| `top_logprobs` | `int \| None` | `None` | With `logprobs`, how many of the most likely tokens to return at each step. Defaults to 1. |
| `max_tokens` | `int \| None` | `None` | Maximum number of tokens to generate. |
| `presence_penalty` | `float \| None` | `None` | Flat penalty on any token already generated. 0 turns it off. |
| `frequency_penalty` | `float \| None` | `None` | Penalty that grows each time a token is generated again. 0 turns it off. |
| `repetition_penalty` | `float \| None` | `None` | Penalty on tokens already in the prompt or output. 1.0 turns it off. |
| `stop_seqs` | `list[str] \| None` | `None` | Stop generating when any of these strings appears. |
| `temperature` | `float \| None` | `None` | Sampling temperature. 0 picks the most likely token every time. |
| `top_p` | `float \| None` | `None` | Sample only from the most likely tokens whose probabilities add up to `top_p`. |
| `top_k` | `int \| None` | `None` | Sample only from the `top_k` most likely tokens. |
| `stream` | `bool` | `False` | Return an iterator of `ChatCompletionChunkResponse` instead of one response. |
| `grammar` | `str \| None` | `None` | Constraint that the output must match. Set `grammar_type` with it. |
| `grammar_type` | `str \| None` | `None` | How to read `grammar`: `"regex"`, `"lark"`, `"json_schema"`, or `"llguidance"`. |
| `min_p` | `float \| None` | `None` | Drop tokens less likely than `min_p` times the most likely token. |
| `tool_schemas` | `list[str] \| None` | `None` | Tools the model may call, each a JSON string in the OpenAI format (`{"type": "function", "function": {...}}`). |
| `tool_choice` | `ToolChoice \| None` | `None` | `ToolChoice.Auto` lets the model decide whether to call a tool. `ToolChoice.NoTools` turns tool calls off. |
| `dry_multiplier` | `float \| None` | `None` | Turns on DRY ("don't repeat yourself") sampling, which penalizes continuing a sequence that already appeared. Higher is stronger. |
| `dry_base` | `float \| None` | `None` | How fast the DRY penalty grows with the length of the repeat. Defaults to 1.75. |
| `dry_allowed_length` | `int \| None` | `None` | Repeats up to this many tokens long aren't penalized. Defaults to 2. |
| `dry_sequence_breakers` | `list[str] \| None` | `None` | Strings that end a repeat for DRY. Defaults to newline, `:`, `"`, and `*`. |
| `web_search_options` | `WebSearchOptions \| None` | `None` | Turns on web search for this request. |
| `enable_thinking` | `bool \| None` | `None` | Turn the model's thinking on or off. See the reasoning controls above. |
| `truncate_sequence` | `bool` | `False` | If the prompt is too long for the model, drop tokens from its start instead of failing. |
| `reasoning_effort` | `Literal['off', 'none', 'low', 'medium', 'high', 'xhigh', 'max'] \| None` | `None` | How much a reasoning model thinks. See the reasoning controls above. |
| `max_tool_rounds` | `int \| None` | `None` | Most tool-call rounds the server runs by itself. Setting it turns on the agent loop. Defaults to 256. |
| `tool_dispatch_url` | `str \| None` | `None` | URL that receives, as a POST, any tool call that no built-in tool or `tool_callbacks` entry handles. |
| `enable_code_execution` | `bool` | `False` | Let the model run Python. The `Runner` needs a `code_execution_config`. |
| `enable_shell` | `bool` | `False` | Let the model run shell commands. The `Runner` needs a `shell_config`. Non-empty `shell_skills` also turns it on. |
| `shell_skills` | `list[ShellSkillMount] \| None` | `None` | Skill directories to mount for the shell tool. |
| `agent_permission` | `AgentPermission \| None` | `None` | Whether agent actions run (`Auto`), wait for approval (`Ask`), or are refused (`Deny`). |
| `agent_approval_callback` | `Callable[[AgentToolApproval], bool \| AgentToolApprovalDecision] \| None` | `None` | Called when `agent_permission` is `Ask`. Return `True`, `False`, or an `AgentToolApprovalDecision`. Raising or returning anything else denies the action. |
| `code_execution_permission` | `CodeExecutionPermission \| None` | `None` | Older, code-only form of `agent_permission`. Used when `agent_permission` is unset. |
| `session_id` | `str \| None` | `None` | Agent session to continue. If unset, the server matches an earlier session by its messages or starts a new one. |
| `files` | `list[RequestedFile] \| None` | `None` | Output files the model must produce. |
| `input_files` | `list[InputFile] \| None` | `None` | Files to give the model. |
| `ignore_eos` | `bool` | `False` | Keep generating past the end-of-sequence token, until `max_tokens` or a stop string. |
| `adapter` | `str \| LoraAdapterGeneration \| None` | `None (keyword-only)` | LoRA adapter to use: an alias, or a `LoraAdapterGeneration` for one exact version. |
| `modalities` | `list[Literal['text', 'audio']] \| None` | `None (keyword-only)` | Output types for a speech model: `["text"]`, `["audio"]`, or `["text", "audio"]`. |
| `audio` | `dict[str, str] \| None` | `None (keyword-only)` | Speech output settings: `format`, and optionally `voice`. Requires `"audio"` in `modalities`. |


## `CompletionRequest`

A raw text completion request, with no chat template. Send it with `Runner.send_completion_request`.

Fields shared with `ChatCompletionRequest` work the same way.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `prompt` | `str` | required | Text to continue. |
| `model` | `str` | required | Not used to choose a model. Pass `model_id` to the `Runner` method instead. |
| `best_of` | `int` | `1` | How many of the `n_choices` completions to return, keeping those with the highest log probability. |
| `echo_prompt` | `bool` | `False` | Put the prompt at the start of each returned completion. |
| `presence_penalty` | `float \| None` | `None` | Flat penalty on any token already generated. 0 turns it off. |
| `frequency_penalty` | `float \| None` | `None` | Penalty that grows each time a token is generated again. 0 turns it off. |
| `repetition_penalty` | `float \| None` | `None` | Penalty on tokens already in the prompt or output. 1.0 turns it off. |
| `logit_bias` | `dict[int, float] \| None` | `None` | Map of token ID to a bias added to that token's logit before sampling. |
| `max_tokens` | `int \| None` | `None` | Maximum number of tokens to generate. |
| `n_choices` | `int` | `1` | Number of completions to generate. |
| `stop_seqs` | `list[str] \| None` | `None` | Stop generating when any of these strings appears. |
| `temperature` | `float \| None` | `None` | Sampling temperature. 0 picks the most likely token every time. |
| `top_p` | `float \| None` | `None` | Sample only from the most likely tokens whose probabilities add up to `top_p`. |
| `suffix` | `str \| None` | `None` | Text appended to each returned completion. |
| `top_k` | `int \| None` | `None` | Sample only from the `top_k` most likely tokens. |
| `grammar` | `str \| None` | `None` | Constraint that the output must match. Set `grammar_type` with it. |
| `grammar_type` | `str \| None` | `None` | How to read `grammar`: `"regex"`, `"lark"`, `"json_schema"`, or `"llguidance"`. |
| `min_p` | `float \| None` | `None` | Drop tokens less likely than `min_p` times the most likely token. |
| `tool_schemas` | `list[str] \| None` | `None` | Tools the model may call, each a JSON string in the OpenAI format. |
| `tool_choice` | `ToolChoice \| None` | `None` | `ToolChoice.Auto` lets the model decide whether to call a tool. `ToolChoice.NoTools` turns tool calls off. |
| `dry_multiplier` | `float \| None` | `None` | Turns on DRY ("don't repeat yourself") sampling. Higher is stronger. |
| `dry_base` | `float \| None` | `None` | How fast the DRY penalty grows with the length of the repeat. Defaults to 1.75. |
| `dry_allowed_length` | `int \| None` | `None` | Repeats up to this many tokens long aren't penalized. Defaults to 2. |
| `dry_sequence_breakers` | `list[str] \| None` | `None` | Strings that end a repeat for DRY. Defaults to newline, `:`, `"`, and `*`. |
| `truncate_sequence` | `bool` | `False` | If the prompt is too long for the model, drop tokens from its start instead of failing. |
| `ignore_eos` | `bool` | `False` | Keep generating past the end-of-sequence token, until `max_tokens` or a stop string. |
| `adapter` | `str \| LoraAdapterGeneration \| None` | `None (keyword-only)` | LoRA adapter to use: an alias, or a `LoraAdapterGeneration` for one exact version. |


## `EmbeddingRequest`

Text or token IDs to embed. Send it with `Runner.send_embedding_request`, which returns one vector per input.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `input` | `str \| list[str] \| list[int] \| list[list[int]]` | required | A string, a list of strings, a list of token IDs, or a list of token-ID lists. |
| `truncate_sequence` | `bool` | `False` | If an input is too long for the model, drop tokens from its end instead of failing. |


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
