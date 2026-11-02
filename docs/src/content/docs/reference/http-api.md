---
title: HTTP API semantics
description: Streaming event types, authentication, approval flow, and protocol behavior that the generated OpenAPI reference does not express.
---

Use the [generated HTTP API reference](/reference/http-api-generated/) for endpoint schemas and [OpenAI compatibility](/reference/openai-compatibility/) for supported fields. This page defines streaming formats, routing, and state shared between requests.

## Discovering the API from a running server

- `GET /docs` serves a Swagger UI for the exact build you are running.
- `GET /api-doc/openapi.json` serves the raw OpenAPI document.
- `GET /` is an alias for `GET /health`: returns 200 when the server is up. Neither verifies model load status.

## Authentication

There is none. The server accepts and ignores `Authorization: Bearer ...` (OpenAI clients) and `x-api-key` plus `anthropic-version` headers (Anthropic clients), so SDKs that require a key at initialization work with any non-empty string. For real authentication and TLS, put a reverse proxy in front.

## Model routing

The request `model` field selects among loaded models. `"default"` (or omitting the field) targets the configured default model; with a single `-m` model that is the only model. `GET /v1/models` lists real ids plus per-model `status` (`loaded`, `unloaded`, `reloading`), `tools_available`, `mcp_tools_count`, and `mcp_servers_connected`. See [multiple models](/guides/serve/multiple-models/).

## Decision models

`POST /v1/systemone` evaluates named typed questions with d1 and d1-Omni models. It accepts arbitrary JSON state and optional image, video, and audio inputs, returning probabilities, extracted values, or boxes. See [decision requests](/reference/decisions/) for the wire format and native SDK equivalents.

## LoRA adapter routing and management

Chat Completions, Completions, and Responses can run a request on a loaded LoRA adapter. Each adapter is loaded under an alias, and each set of adapter files gets a generation: an ID computed from the adapter's config and weights. Loading different files under the same alias gives it a new generation.

Select an adapter in any of these ways:

- `"adapter": "<alias>"` uses the alias's current generation.
- `"adapter": {"generation": "<id>"}` uses that exact generation, as long as it's still loaded.
- `model` set to the adapter's ID from `GET /v1/models`, or to its bare alias when only one model has an adapter by that name.

Responses include the `adapter_generation` that ran. Replacing or unloading an alias doesn't affect requests already running.

| Route | Behavior |
| --- | --- |
| `GET /v1/lora_adapters` | Lists aliases, their generations, and capacity limits. Returns 409 `lora_runtime_unavailable` if the model has no dynamic LoRA support. |
| `POST /v1/load_lora_adapter` | Loads an adapter from a directory on the server under an alias. To replace an alias that's already loaded, pass `load_inplace: true`. |
| `POST /v1/unload_lora_adapter` | Removes an alias. |

To avoid overwriting someone else's change, pass the generation you last saw as `expected_generation` to load or unload. The call then fails with 409 `lora_generation_mismatch` if the alias has changed since. On load, `expected_generation` also needs `load_inplace: true`.

Loading and unloading are off by default:

- For `mistralrs serve`, set `MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING=1` (`true`, `yes`, and `on` also work).
- For an embedded server, enable them with `LoraAdapterApiConfig`.

While they're off, the load and unload routes don't exist, and `GET /v1/lora_adapters` hides each adapter's source path.

The server loads one adapter at a time, and a second load sent meanwhile gets 429. A load that has started finishes even if the client disconnects, so after a timeout, check `GET /v1/lora_adapters` before retrying.

See the [LoRA guide](/guides/customize/lora-adapters/) for setup, immutable adapter paths, access controls, and error recovery.

## Streaming

Chat Completions, Responses, Messages, speech, and transcriptions use distinct SSE formats described below. Chat Completions and Responses send SSE keep-alive comments every 10 seconds by default; `KEEP_ALIVE_INTERVAL` sets that interval in milliseconds.

### Streaming: Chat Completions

`stream: true` returns unnamed SSE `data:` payloads in OpenAI chat chunk format, followed by `data: [DONE]`.

Models with chat speech output emit `delta.audio.data` during synthesis when `modalities` includes `"audio"` and `audio.format` is `"pcm16"`. For LFM3-Audio and LFM3-Omni, regular chunks contain about 0.96 seconds of mono, signed 16-bit little-endian PCM at 24 kHz; the final chunk contains the remaining tail. Decode each base64 chunk and append its bytes in arrival order. The chunks share an audio ID, and the final metadata carries `transcript` and `expires_at`. For audio-only replies, mistral.rs transcribes the finished speech to produce `transcript`, so that metadata can arrive a little after the last audio chunk. See [LFM3 speech behavior](/guides/models/use-speech-models/#reply-length-and-speech-sampling) for model defaults and limits.

Agentic requests can interleave named events:

| Event | Body |
|---|---|
| `agentic_tool_call_progress` | Tool-loop progress: `round`, opaque `tool_name`, `phase` (`calling` or `complete`), structured `data`. |
| `agentic_tool_approval_required` | A pending agent approval (see below). |
| `file_produced` | A `File` object, emitted once per file as it is produced. |

Tool-progress `data.tool_type` is `code_execution`, `web_search`, `shell`, or `custom`. Code execution events can include `images_base64` and `video_frames_base64`; shell events include commands, stdout/stderr, exit status, and the shell working directory.

### Streaming: Responses

`stream: true` uses named events: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.content_part.done`, `response.output_item.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`.

Terminal events (exactly one ends the stream):

- `response.completed`: the run finished successfully.
- `response.failed`: the run errored.
- `response.incomplete`: the run stopped early (e.g. token cap).

Errors also stream as a named `error` event. The mistral.rs `agentic_tool_call_progress` and `file_produced` events are also emitted on this endpoint. Shell tool calls are represented as Responses `shell_call` and `shell_call_output` output items.

### Streaming: Speech

Set `stream_format="sse"` on `/v1/audio/speech` with a chat speech model such as LFM3-Audio or LFM3-Omni. SSE `data:` payloads contain `{"type":"speech.audio.delta","audio":"<base64>"}`, then `{"type":"speech.audio.done","usage":{"input_tokens":14,"output_tokens":101,"total_tokens":115}}`. Usage counts are the loaded model's actual prompt and completion tokens.

With `response_format="pcm"` and speed `1.0`, audio arrives during synthesis. Decode and append each chunk to obtain mono, 24 kHz, signed 16-bit little-endian PCM. WAV, compressed formats, and non-default speed wait for synthesis and encoding to finish before sending a complete encoded file in one audio delta. The default `stream_format="audio"` returns a completed binary file. Dedicated speech models such as Dia reject SSE.

### Streaming: Transcriptions

Set multipart `stream=true` and use `response_format=json` (the default). SSE `data:` payloads contain `{"type":"transcript.text.delta","delta":"..."}` while text is generated, then `{"type":"transcript.text.done","text":"..."}` with the complete transcript. This streams recognition results for an uploaded file; it is not a live audio-input connection. The translations endpoint does not accept `stream`.

### Streaming: Anthropic Messages

`stream: true` on `POST /v1/messages` uses Anthropic's named events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, with idle `ping` events. Deltas are `text_delta`, `thinking_delta` (when the model exposes separate reasoning), and `input_json_delta` (tool-call arguments). mistral.rs named events (`agentic_tool_call_progress`, `agentic_tool_approval_required`, `file_produced`) may be interleaved. See the [Anthropic Messages guide](/guides/serve/anthropic-messages-api/).

## Chat response extensions

Non-streaming chat responses carry four mistral.rs fields beyond the OpenAI shape (omitted when empty):

- `session_id` (string): reuse in later requests to keep agentic state across messages.
- `agentic_tool_calls` (array): ordered record of tool calls made during the agentic loop. Each entry has `round`, opaque `name`, `arguments`, `result_content`, plus `result_images_base64` and `file_ids` when present.
- `files` (array of `File` objects): see [file wire schemas](#file-wire-schemas-and-semantics).
- `adapter_generation` (string): exact immutable LoRA generation used for the response; omitted for base-model requests.

The `usage` object is a superset of OpenAI's, adding timing fields such as `avg_tok_per_sec`, `avg_prompt_tok_per_sec`, `avg_compl_tok_per_sec`, and total prompt/completion times.

## Audio requests and history

Chat input uses `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`. WAV and MP3 are accepted; `data` is encoded file bytes without a data-URL prefix. Each decoded file is limited to 64 MiB. The server's default 50 MiB request-body limit also applies, including base64 expansion; embedded servers can change it with `with_max_body_limit`. The `audio_url` extension accepts URLs, data URLs, and other supported audio formats.

Non-streaming output is base64 in `message.audio.data`; streamed PCM arrives in `delta.audio.data`. See [chat streaming](#streaming-chat-completions) for chunk assembly and [OpenAI compatibility](/reference/openai-compatibility/#audio) for dedicated speech/transcription endpoints, formats, and controls. LFM3 uses the checkpoint's voice: omit `voice` or use `"default"`.

Completed HTTP chat responses include `audio.id`, `audio.transcript`, and Unix timestamp `audio.expires_at`. Send `{"role":"assistant","audio":{"id":"..."}}` in a later conversation to restore that transcript as text context. This restores the words only, not the speech itself. For text-and-audio replies, the transcript is the generated text. For audio-only replies, mistral.rs transcribes the finished speech, which can delay the completion metadata.

IDs expire after one hour and may be evicted earlier or lost on server restart. Unknown or expired IDs return an error; keep the transcript as assistant `content` for durable history. If both content and an ID are supplied, content takes precedence but the ID must still be valid. When transcription fails or exceeds resource limits, generated audio is preserved with an empty transcript and an expired ID. This history service is HTTP-only; native SDK callers manage conversation text themselves.

## Agent approval flow

HTTP chat requests with `agent_permission: "ask"` require `stream: true`. An action needing approval emits:

```text
event: agentic_tool_approval_required
data: {"type":"agentic_tool_approval_required","approval_id":"appr_abc123","session_id":"...","round":1,"tool":{"source":"built_in","kind":"code_execution","label":"Python code"},"arguments":{"code":"...","outputs":[]}}
```

Reply to `POST /v1/agent/approvals/{approval_id}` with `decision: "approve"` or `"deny"`. A denial may include a `message` for the model; approval may include `remember_for_session: true`. Unanswered approvals are denied after five minutes.

The response status is `resolved` if a waiting call was released, `queued` if the answer arrived before the call began waiting, or `not_found` with HTTP 404 for an unknown/expired ID. See [permissions and approvals](/guides/agents/permissions-and-approvals/) for policy and [OpenAPI](/reference/http-api-generated/) for schemas.

## File wire schemas and semantics

Chat Completions and Messages return generated files in `files[]` or streamed `file_produced` events. Responses uses `container_file_citation` annotations on assistant `output_text`. The tables below cover file fields not fully described by OpenAPI; see [file inputs](/guides/agents/file-inputs/) for upload and attachment examples.

Requesting files (`files` on chat, Responses, and Anthropic Messages requests):

| Field | Type | Notes |
|---|---|---|
| `files[].name` | string | Filename. Required. |
| `files[].format` | string | Format hint (`png`, `csv`, `json`, ...). Inferred from the extension if omitted. |
| `files[].description` | string | Optional hint surfaced to the model. |

`File` object (response `files[]` entries and `file_produced` event bodies):

| Field | Type | Notes |
|---|---|---|
| `id` | string | Opaque file ID used for retrieval. |
| `name` | string | Filename as written. |
| `format` | string | Open-ended format string. |
| `mime_type` | string | Content-Type. |
| `bytes` | integer | Body size. |
| `created_at` | integer | Unix epoch seconds. |
| `purpose` | string | `agent_output` for generated files, `user_data` for uploaded/request files. |
| `source` | object | `{"tool", "round", "turn"}` attribution. |
| `text` | string | Full text body for text files. Absent if elided. |
| `preview` | string | Short UTF-8 preview for text files. |
| `data_base64` | string | Base64 body for binary files. Absent if elided. |
| `code`, `message` | strings | Present instead of a body if the file failed to materialize. |

- Upload with multipart `POST /v1/files`, using `purpose="user_data"`. Responses `input_file` accepts `file_id`, inline `file_data`, or `file_url`; Chat Completions `file` parts accept IDs and inline data only. Inline data is base64 or a data URL.
- UTF-8 inputs get decoded previews of up to 4,096 characters per file and 32,768 per request. Binary inputs provide metadata only. Active shell/code tools can access the files in their workdirs.
- Output bodies up to 8 MiB appear inline. Fetch larger bodies from `GET /v1/files/{id}/content`. Files expire after 30 minutes and the store retains at most 4,096 entries.
- Content fetches return 200 for a body, 404 for an unknown/expired ID, 410 for an elided body, or 422 for a file error placeholder. `/v1/containers/{container_id}/files/{file_id}/content` is an alias for the same store.
- Responses citations contain `container_id`, `file_id`, and `filename`. Chat tool records attribute files through `agentic_tool_calls[].file_ids`.
- Shell/code tools surface newly created files automatically. Requested outputs that are missing produce error placeholders. See [file inputs](/guides/agents/file-inputs/) for file access and extraction limits.

## Skills

`POST /v1/skills` uploads a zip or files from one top-level directory containing `SKILL.md` with `name` and `description` frontmatter. `GET /v1/skills` lists uploads; `POST /v1/skills/{skill_id}/versions` adds a version. Executing uploaded skills requires shell support (`--enable-shell` or `--agent`). See [Skills](/guides/agents/skills/) for request examples and storage configuration.

## Session semantics

`GET /v1/sessions/{session_id}` exports a `SerializedSession` (404 if missing); `PUT` imports one, replacing any session with the same id; `DELETE` always returns 200 whether the session existed or not. Session lifecycle and splicing behavior are on the [sessions page](/guides/agents/persist-sessions/).

## Metrics

`GET /metrics` exposes Prometheus metrics, enabled by default. It returns 503 before the recorder initializes or when `--disable-metrics` is set. See the [metric catalog](/reference/metrics/) for names and labels, and [observability](/guides/deploy/observability/) for setup and queries.

## Response headers

`Content-Type` matches the endpoint: JSON responses use `application/json`, SSE uses `text/event-stream`, and audio/file downloads use their media types. Text transcription uses `text/plain`. Session IDs are returned in the body's `session_id` field, not a header.

## Compatibility

Field-level OpenAI deviations, silently-ignored fields, and Responses API restrictions: [OpenAI compatibility](/reference/openai-compatibility/). Anthropic field and content-block support: [Anthropic Messages API](/guides/serve/anthropic-messages-api/).
