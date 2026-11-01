---
title: OpenAI compatibility
description: Which OpenAI API fields mistralrs implements, which it extends, and which it does not support.
---

Use OpenAI clients with a local base URL; capabilities depend on the loaded model and endpoint. This page lists supported fields and deviations. Start with the [serving guide](/guides/serve/openai-compatible-apis/); use [OpenAPI](/reference/http-api-generated/) for schemas and [HTTP semantics](/reference/http-api/) for streaming and history.

Decision models use the [SystemOne API at `POST /v1/systemone`](/reference/decisions/), a separate endpoint outside the OpenAI API.

## Chat Completions fields

### Implemented

`model`, `messages` (including multimodal content), `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `stream`, `stop`, `tools`, `tool_choice`, `response_format` (`text`, `json_schema`), `logit_bias`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, and `n`.

Loaded LoRA adapters can be selected by their exact model-card ID from `GET /v1/models`, a unique short alias, or the `adapter` extension. See [adapter routing](/reference/http-api/#lora-adapter-routing-and-management).

### Implemented with deviation

| Field | Support and limits |
| --- | --- |
| `modalities`, `audio` | Chat speech models accept `["audio"]` or `["text", "audio"]`. Formats: `wav`, `pcm16`, `mp3`, `flac`, `opus`, `aac`; compressed output requires FFmpeg. Voice must be omitted or `"default"`. Streaming requires PCM16. Audio output cannot be combined with tools, constrained decoding, custom logits processors, or logprobs. |
| `input_audio` content parts | WAV/MP3 base64 file bytes in `input_audio: {"data":"...","format":"wav"}`. `audio_url` is an extension for URLs and other decodable formats. |
| Assistant `audio.id` | Restores the server-cached transcript as text context. IDs can expire or be evicted; see [audio history](/reference/http-api/#audio-requests-and-history). |
| `tool_choice` | `auto`, `none`, `required`, specific function objects in Chat or Responses form, and function-only `allowed_tools` subsets. `required` needs available tools. |
| Function `strict: true` | Constrains arguments to the function's `parameters` JSON Schema. |
| `code_interpreter` | Requires server-side code execution enabled. Only `container: {"type":"auto"}` is supported; container IDs, `file_ids`, `memory_limit`, and container lifecycle APIs are rejected. |
| `file` content parts | Accept `file_id`, or `filename` plus base64/data-URL `file_data`. File URLs require Responses. |
| `response_format` | `json_schema` constrains output; `json_object` is rejected. Ambiguous schemas can produce different shapes from OpenAI. |
| `seed` | Seeds request-scoped sampling, with a distinct stream per choice when `n > 1`. |

See [tool calling](/guides/agents/tool-calling-basics/), [structured output](/guides/serve/structured-output/), and [audio streaming](/reference/http-api/#streaming-chat-completions) for examples and wire formats.

### Silently ignored

`user`, `stream_options`, `metadata`, `service_tier`, `parallel_tool_calls`, `store`. These fields have no effect on Chat Completions. Use `session_id` for agent session persistence.

### Additional request controls

| Fields | Purpose |
| --- | --- |
| `top_k`, `min_p`, `repetition_penalty` | Additional sampling controls. |
| `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` | DRY sampling. |
| `grammar` | Constraints beyond JSON Schema. |
| `reasoning_effort`, `enable_thinking` | Reasoning controls described below. |
| `web_search_options` | Server-side search configuration. |
| `session_id` | Agent session persistence. |
| `files` | Requested output files. |
| `agent_permission`, `code_execution_permission` | Tighten tool permissions for a request. |
| `max_tool_rounds` | Cap server-side tool-loop rounds. |
| `truncate_sequence` | Truncate an oversized prompt instead of rejecting it. |
| `adapter` | Loaded LoRA alias or `{"generation":"..."}`; omitted/null selects the base model. |

`reasoning_effort` accepts `off`, `low`, `medium`, `high`, and `xhigh`, case-insensitively; `none` aliases `off`. Omission enables thinking for text output and disables it for speech output without selecting an effort. Positive effort enables thinking; `off` disables it. Contradictory explicit values of `reasoning_effort` and the legacy `enable_thinking` boolean are rejected. The model's chat template determines the effect of each tier; effort does not directly change sampling.

LoRA responses expose the selected `adapter_generation`, including Chat/Completions streaming chunks and completed Responses objects. See [adapter management](/reference/http-api/#lora-adapter-routing-and-management) for runtime requirements and errors.

## Responses API

| Route | Operation |
| --- | --- |
| `POST /v1/responses` | Create a response. |
| `GET /v1/responses/{id}` | Fetch a stored response. |
| `DELETE /v1/responses/{id}` | Delete it. |
| `POST /v1/responses/{id}/cancel` | Cancel an unfinished background response. |

### Implemented

- `input`: messages or a raw prompt string.
- `input_file` content parts with `file_id`, `file_data`, or `file_url`.
- `previous_response_id`: continues a stored conversation.
- `max_output_tokens`: with `max_tokens` and `max_completion_tokens` as aliases.
- `instructions`, `temperature`, `top_p`, `stop`, `stream`, `tools`, `tool_choice`, `text.format`, `response_format`, `logit_bias`, `logprobs`, `top_logprobs`, `presence_penalty`, `frequency_penalty`, `n`, `metadata`, `background`, `store`.

`store` defaults to `true`; `false` makes the response unavailable to retrieval or `previous_response_id`. Use `text.format` for structured output; the legacy `response_format` alias is also accepted. As with Chat Completions, `json_object` is rejected and function tools support `strict: true`.

Responses `tools` accepts:

- Function tools in the Responses flat form: `{"type":"function","name":"...","parameters":{...}}`.
- Function tools in the Chat Completions nested form: `{"type":"function","function":{...}}`.
- `{"type":"web_search", ...}` and `{"type":"web_search_preview"}` for server-side web search.
- `{"type":"code_interpreter","container":{"type":"auto"}}` for server-side Python code execution.
- `{"type":"shell","environment":{"type":"container_auto","skills":[{"type":"skill_reference","skill_id":"skill_...","version":"latest"}]}}` for server-side shell execution and OpenAI-compatible Skills. Skills require the shell executor, so start the server with at least `--enable-shell`; `--agent` is the recommended preset when you also want the full agentic runtime.

`tool_choice: "required"` is accepted and rejects requests that provide no tools. Specific function choices must reference a declared function tool. `tool_choice: {"type":"allowed_tools", ...}` is supported for function tool subsets only. Hosted tool forcing or filtering, such as `{"type":"web_search_preview"}`, `{"type":"code_interpreter"}`, `{"type":"shell"}`, or hosted tools inside `allowed_tools`, is not supported.

### Skills API

Upload, list, and version Skills through `/v1/skills`, then reference them in a shell tool's `environment.skills`. Execution requires `--enable-shell` or `--agent`. See the [Skills guide](/guides/agents/skills/) for upload shapes, version selection, and storage.

### Rejected non-default values

- `parallel_tool_calls` must be `true` (default) or omitted; `false` returns an error.
- `max_tool_calls` returns an error for any value. Use `max_tool_rounds` to cap agentic rounds per request, or `--max-tool-rounds` for the server default.
- `tools[*].type="web_search"` rejects image search (`search_content_types: ["image"]` or `image_settings`) and `external_web_access: false`. Domain filters are supported for up to 100 allowed or blocked domains and include subdomains.
- `tools[*].type="web_search_preview"` rejects `filters` and `return_token_budget`; `external_web_access` is ignored.
- `tools[*].type="code_interpreter"` rejects container ids, `container.file_ids`, and `container.memory_limit`.
- `tools[*].type="shell"` rejects local environments, container references, local skill paths, inline/container-created skills, and OpenAI container lifecycle APIs. Uploaded `skill_reference` skills are supported.

### mistralrs extensions on Responses

`top_k`, `min_p`, `repetition_penalty`, the `dry_*` sampling fields, `grammar`, `seed`, `ignore_eos`, and `adapter` follow the Chat Completions behavior. `files` requests tool-produced output files; `max_tool_rounds` caps agentic rounds. Chat-only `session_id`, `agent_permission`, and `web_search_options` are not accepted as controls here. Use the Responses `tools` array for hosted tools and Skills.

Thinking, reasoning effort, and truncation are not top-level extension fields here; they are controlled through the standard Responses objects. Use `reasoning.effort` with `off`, `low`, `medium`, `high`, or `xhigh`; `none` is accepted as an alias for `off`. Omission enables thinking without choosing an effort. `reasoning.summary` is accepted for compatibility but currently does not change the response. Use `truncation` for sequence truncation. Top-level `enable_thinking`, `reasoning_effort`, and `truncate_sequence` keys are silently ignored on this endpoint.

### Background runs

Send `background: true`, then poll `GET /v1/responses/{id}` or cancel with `POST /v1/responses/{id}/cancel`. [Streaming event names](/reference/http-api/#streaming-responses) and the [Responses example](/examples/server/responses/) cover consuming results.

## Completions (legacy)

`/v1/completions` (non-chat) is supported with a subset of Chat Completions extensions: `top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`, `truncate_sequence`, `adapter`. LoRA accepts the same alias-as-`model`, adapter alias, and exact-generation forms as Chat Completions. The agentic, session, file, web-search, thinking, and reasoning-effort fields are not part of this endpoint's schema and have no effect.

## Embeddings

`input` accepts a string or list of strings. `encoding_format` is `float` (default) or `base64`. `dimensions` is rejected; `user` is ignored. The `truncate_sequence` extension truncates oversized inputs instead of rejecting them.

## Image Generation

Supports `prompt`, `n`, and `response_format`. Format values differ from OpenAI: `"Url"` (default) returns a server-side filename in `url`; `"B64Json"` returns a `data:image/png;base64,...` string in `b64_json`.

Use `height` (default 720) and `width` (default 1280) instead of OpenAI's `size` string. `quality`, `style`, `steps`, and `guidance_scale` are ignored.

## Audio

### `/v1/audio/speech` (TTS)

| Field | Support |
| --- | --- |
| `model`, `input` | Dedicated speech models or chat speech models such as LFM3-Audio/Omni. |
| `response_format` | `mp3` (default), `opus`, `aac`, `flac`, `wav`, `pcm`. Compressed formats require FFmpeg. PCM is signed 16-bit little-endian. |
| `voice` | Omitted or `"default"`; named/custom voices are rejected. |
| `instructions` | Synthesis instructions for chat speech models. Dia rejects nonempty instructions. |
| `speed` | 0.25 to 4.0, default 1.0. Non-default speed requires FFmpeg. |
| `stream_format` | `audio` (default) returns a completed binary file. Chat speech models also support `sse`; Dia rejects it. |

SSE sends incremental audio only for PCM at speed 1.0; other formats/speeds send a completed encoded file. See [speech streaming](/reference/http-api/#streaming-speech) for event schemas.

### `/v1/audio/transcriptions` and `/v1/audio/translations`

Both endpoints accept multipart `file` and `model` fields and route to a loaded model with audio input support. Transcriptions return the spoken text; translations request English output. Model language coverage still applies.

`response_format="json"` (default) returns `{"text":"..."}`; `"text"` returns plain text. `prompt` provides spelling and vocabulary context, and `temperature` accepts values from `0` to `1`. Transcriptions also accept `language` and `stream=true`; streaming requires JSON format. Translations reject `stream`. See [transcription streaming](/reference/http-api/#streaming-transcriptions) for event schemas. Timestamp, diarization, subtitle, verbose JSON, and logprob options return explicit validation errors.

See the [dedicated audio endpoint example](/examples/server/lfm3-audio-endpoints/) and [speech models guide](/guides/models/use-speech-models/). The OpenAI Realtime API and live microphone sessions are not implemented.

## Moderation

Not supported. mistral.rs has no built-in moderation model; run one as a separate service if needed.

## Files and Assistants APIs

`POST /v1/files` accepts multipart uploads with `purpose="user_data"`. Uploaded, inline, URL-fetched, and generated files can be listed, fetched, downloaded, and deleted through `/v1/files` routes. See [file wire semantics](/reference/http-api/#file-wire-schemas-and-semantics).

UTF-8 files provide bounded text previews; active shell/code tools can inspect their full contents. Binary files are stored and accessible to those tools, but there is no automatic PDF/image/spreadsheet extraction pipeline. The Assistants API is unsupported; use [agent sessions](/guides/agents/persist-sessions/) for persistent tool state.

## Fine-tuning and Batch

Not supported. mistral.rs is an inference engine, not a training platform.

## Tokenization

mistral.rs does not expose `/v1/tokenize` or `/v1/detokenize` HTTP endpoints. Tokenizer access is available through the SDKs (`tokenize_text` / `detokenize_text` in Python; `tokenize_with_model` / `detokenize_with_model` in Rust).

## Authentication

The server does not validate API keys. Clients can supply any nonempty key; use a reverse proxy for authentication and TLS. See [HTTP authentication](/reference/http-api/#authentication).

## Response headers

See [HTTP response headers](/reference/http-api/#response-headers) for JSON, SSE, audio, and file content types. Session IDs are in response bodies.
