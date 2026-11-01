---
title: HTTP API semantics
description: Streaming event types, authentication, approval flow, and protocol behavior that the generated OpenAPI reference does not express.
---

Every endpoint, request schema, and response schema is in the [generated HTTP API reference](/reference/http-api-generated/), produced from the server's OpenAPI document. This page covers what a schema cannot say: streaming wire formats, authentication, and cross-request semantics.

## Discovering the API from a running server

- `GET /docs` serves a Swagger UI for the exact build you are running.
- `GET /api-doc/openapi.json` serves the raw OpenAPI document.
- `GET /` is an alias for `GET /health`: returns 200 when the server is up. Neither verifies model load status.

## Authentication

There is none. The server accepts and ignores `Authorization: Bearer ...` (OpenAI clients) and `x-api-key` plus `anthropic-version` headers (Anthropic clients), so SDKs that require a key at initialization work with any non-empty string. For real authentication and TLS, put a reverse proxy in front.

## Model routing

The request `model` field selects among loaded models. `"default"` (or omitting the field) targets the configured default model; with a single `-m` model that is the only model. `GET /v1/models` lists real ids plus per-model `status` (`loaded`, `unloaded`, `reloading`), `tools_available`, `mcp_tools_count`, and `mcp_servers_connected`. See [multiple models](/guides/serve/multiple-models/).

## LoRA adapter routing and management

Chat Completions, Completions, and Responses requests can select dynamic LoRA with the top-level `adapter` field. A string selects a loaded alias; `{"generation":"<generation-id>"}` selects one exact generation. Replacing or unloading an alias does not change an in-flight request.

The read-only status route is always registered, even when mutation is disabled. It returns status only when the target model has a dynamic LoRA runtime; for `mistralrs serve`, enable one with `--enable-lora` or a preload. Otherwise it returns 409 with `lora_runtime_unavailable`. `mistralrs serve` enables mutation endpoints only when `MISTRALRS_ALLOW_RUNTIME_LORA_UPDATING` is `1`, `true`, `yes`, or `on`. Embedded servers can instead configure `LoraAdapterApiConfig` on `MistralRsServerRouterBuilder`:

- `POST /v1/load_lora_adapter` loads an alias from a local adapter directory; replacing one requires `load_inplace: true` and may use `expected_generation` for compare-and-set safety.
- `GET /v1/lora_adapters` lists aliases, generation IDs, capacity, and configured limits for a dynamic LoRA model. Sources are redacted unless mutation is enabled.
- `POST /v1/unload_lora_adapter` removes an alias without interrupting in-flight requests; `expected_generation` prevents stale removal.

Set `MISTRALRS_LORA_ADAPTER_ROOT` to restrict load paths. Keep that root, its ancestors, and every selected adapter directory writable only by the service operator. Publish adapters at new immutable directory paths instead of replacing an existing selected path. Loads are serialized; concurrent attempts return 429. An admitted load continues if its client disconnects, so verify the generation through the list endpoint after a timeout. These routes have no built-in authentication and should not be exposed without an authenticated reverse proxy. Request schemas and response objects are in the [generated HTTP API reference](/reference/http-api-generated/); stable error-code recovery, setup, and support boundaries are in the [LoRA guide](/guides/customize/lora-adapters/).

## Streaming

These endpoints stream, each in its own event dialect:

- `POST /v1/chat/completions`: OpenAI chat-completion chunks plus mistral.rs agentic events.
- `POST /v1/responses`: OpenAI named Responses events.
- `POST /v1/messages`: Anthropic named message events.
- `POST /v1/audio/speech`: audio events from chat models with speech output.
- `POST /v1/audio/transcriptions`: transcript text events for an uploaded audio file.

### Streaming: Chat Completions

`stream: true` on `POST /v1/chat/completions` returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE). Unnamed `data:` lines carry chat completion chunks in OpenAI format; the stream terminates with `data: [DONE]`. SSE keep-alive comments (lines starting with `:`) are sent every 10 seconds by default (`KEEP_ALIVE_INTERVAL` env var, milliseconds).

Models with chat speech output emit `delta.audio.data` during synthesis when `modalities` includes `"audio"` and `audio.format` is `"pcm16"`. For LFM3-Audio and LFM3-Omni, regular chunks contain about 0.96 seconds of mono, signed 16-bit little-endian PCM at 24 kHz; the final chunk contains the remaining tail. Decode each base64 chunk and append its bytes in arrival order. The chunks share an audio ID; completed metadata supplies `transcript` and `expires_at`. The audio-only transcript requires a final ASR pass, which can delay the terminal metadata after the last speech chunk. See [LFM3 speech behavior](/guides/models/model-family-notes/#lfm3-speech-input-and-output) for model defaults and limits.

Named events carry the agentic timeline:

| Event | Body |
|---|---|
| (default `data:`) | Chat completion chunk in OpenAI format. Terminator: `data: [DONE]`. |
| `agentic_tool_call_progress` | Tool-loop progress: `round`, opaque `tool_name`, `phase` (`calling` or `complete`), structured `data`. |
| `agentic_tool_approval_required` | A pending agent approval (see below). |
| `file_produced` | A `File` object, emitted once per file as it is produced. |

Tool-progress `data.tool_type` is `code_execution`, `web_search`, `shell`, or `custom`. Code execution events can include `images_base64` and `video_frames_base64`; shell events include commands, stdout/stderr, exit status, and the shell working directory.

### Streaming: Responses

`stream: true` on `POST /v1/responses` uses OpenAI's named Responses events.

Lifecycle and delta events: `response.created`, `response.in_progress`, `response.output_item.added`, `response.content_part.added`, `response.output_text.delta`, `response.content_part.done`, `response.output_item.done`, `response.function_call_arguments.delta`, `response.function_call_arguments.done`.

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

Chat audio inputs use standard content parts: `{"type":"input_audio","input_audio":{"data":"<base64>","format":"wav"}}`. WAV and MP3 are supported; `data` contains encoded file bytes without a data-URL prefix. Each file is limited to 64 MiB after base64 decoding. The HTTP server also applies a 50 MiB request-body limit by default, including base64 expansion and other fields; multipart uploads share that request-body limit. Embedded servers can configure it with `with_max_body_limit`. The `audio_url` extension accepts other supported audio formats through the existing media loader.

For non-streaming chat output, `audio.format` accepts `wav`, `pcm16`, `mp3`, `flac`, `opus`, or `aac`. The compressed formats require FFmpeg on PATH. LFM3 speech uses one checkpoint voice: specify `audio.voice: "default"` or omit the field. Generated audio is base64 in `message.audio.data`.

Compressed output and speed conversion run in bounded background workers, with up to 32 outstanding conversion jobs, including at most two active workers. A full conversion queue returns an error; retry after earlier requests finish. Output FFmpeg conversions have a 120-second timeout. WAV and PCM at the default speed use native encoding.

Completed HTTP chat responses include `audio.id`, `audio.transcript`, and `audio.expires_at`. A later request can send `{"role":"assistant","audio":{"id":"..."}}`; the server inserts the cached transcript into the conversation. It reuses text context, not the original audio tokens. Mixed output uses the generated text as the transcript; audio-only output is transcribed through the loaded model. The per-router cache holds up to 1,024 entries and 16 MiB of transcripts, with a one-hour lifetime. Eviction, expiry, or restart invalidates IDs; resend the assistant transcript as `content` when an ID is unavailable. This server history cache is not part of the native SDK.

If the transcript pass fails, the audio response is still returned with an empty transcript and an immediately expired ID. When an assistant history message supplies both `content` and an audio ID, its explicit text content takes precedence, but the ID must still be valid.

Audio-only transcript enrichment spools audio to temporary files instead of retaining the waveform in memory. Capture and decoded PCM each have a 64 MiB limit, and each server router runs at most two transcript enrichment jobs at a time. Compressed-audio decoding uses FFmpeg with a 60-second timeout. Exceeding these limits or a decoding/transcription failure leaves the generated audio intact but makes its history ID unavailable. Temporary files are removed on completion or cancellation. Mixed text/audio uses its generated text and discards any unnecessary audio capture.

`POST /v1/audio/speech` accepts text and defaults to MP3. It supports `response_format` values `mp3`, `opus`, `aac`, `flac`, `wav`, and `pcm`; `voice="default"`; synthesis `instructions` for chat speech models; and `speed` from `0.25` to `4.0`. FFmpeg is required for compressed formats and non-default speed. `stream_format="audio"` returns a completed binary file; chat speech models also support the SSE mode described above.

`POST /v1/audio/transcriptions` and `/v1/audio/translations` accept multipart file uploads. `response_format="json"` returns a `text` field; `"text"` returns plain text. Transcriptions also support the SSE mode described above. Translations request English output and do not stream. Timestamp, diarization, subtitle, and logprob controls are rejected. See [OpenAI compatibility](/reference/openai-compatibility/#audio) for supported fields and the [SDK example](/examples/server/lfm3-audio-endpoints/). These routes do not implement the OpenAI Realtime API.

## Agent approval flow

`agent_permission: "ask"` requires `stream: true` on HTTP chat requests; non-streaming requests with `"ask"` return a validation error. When an agent action needs approval, the stream emits:

```text
event: agentic_tool_approval_required
data: {"type":"agentic_tool_approval_required","approval_id":"appr_abc123","session_id":"...","round":1,"tool":{"source":"built_in","kind":"code_execution","label":"Python code"},"arguments":{"code":"...","outputs":[]}}
```

The run pauses until the app answers with `POST /v1/agent/approvals/{approval_id}` (body schema in the [generated reference](/reference/http-api-generated/)). Semantics:

- `decision` is `"approve"` or `"deny"`. A deny may carry `message`, returned to the model as the tool result.
- `remember_for_session: true` on an approve auto-approves later agent actions in the same `session_id`.
- Unanswered approvals are denied after five minutes.
- The response `status` is `"resolved"` (a waiting tool call was released), `"queued"` (the app answered before the runtime started waiting), or `"not_found"` with HTTP 404 (unknown or expired approval id).

Permission levels and how they combine across CLI, HTTP, Python, and Rust are on the [permissions and approvals page](/guides/agents/permissions-and-approvals/).

## File wire schemas and semantics

Agentic runs return typed file outputs as first-class objects. Chat Completions and Anthropic Messages use a `files[]` array in non-streaming responses and `file_produced` events in streams. Responses uses OpenAI-style `container_file_citation` annotations on assistant `output_text` content. User-provided input files can also be uploaded or attached to OpenAI-compatible requests. These schemas are normative because they do not all appear in the OpenAPI document. (The `/v1/files` metadata endpoints *are* in the [generated reference](/reference/http-api-generated/).)

Requesting files (`files` on chat, Responses, and Anthropic Messages requests):

| Field | Type | Notes |
|---|---|---|
| `files[].name` | string | Filename. Required. |
| `files[].format` | string | Format hint (`png`, `csv`, `json`, ...). Inferred from the extension if omitted. |
| `files[].description` | string | Optional hint surfaced to the model. |

`File` object (response `files[]` entries and `file_produced` event bodies):

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable id. Agent outputs use `file_<run>_r<round>_<idx>`; uploaded/request files use `file-...`. |
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

Semantics:

- `POST /v1/files` accepts multipart `file` and `purpose` fields. Use `purpose="user_data"` for OpenAI-compatible request attachments.
- Responses output files are cited with `container_file_citation` annotations containing `container_id`, `file_id`, and `filename`.
- Responses `input_file` supports `file_id`, `filename` + `file_data`, and `file_url`. Chat Completions `file` content parts support `file_id` and `filename` + `file_data`; file URLs are Responses-only.
- `file_data` is decoded from base64 or a Data URL before use. Base64 is never placed in model context.
- Text-like UTF-8 input files get a decoded preview of up to 4096 chars per file and 32768 chars per request. Agentic runs can inspect more text when file access is available. Non-UTF-8/binary input files are metadata-only in prompt context.
- Input files are mounted into shell/code session workdirs when those tools are active.
- Bodies up to 8 MiB ship inline (`text` or `data_base64`); above that the body field is omitted and clients fetch raw bytes from `GET /v1/files/{id}/content`.
- For agent-produced output files, text is surfaced back to the model as metadata plus the existing 1024-byte preview; agentic runs can inspect more text when file access is available.
- Shell and code execution automatically surface files created during a tool call. Explicit `outputs` and request `files` still provide names/metadata and produce error placeholders for expected files that are missing. Shell can also surface files created in earlier calls via `mistralrs_surface_outputs`. Other files remain in the session working directory.
- Files expire 30 minutes after creation (at most 4096 retained).
- `GET /v1/files/{id}/content` status codes: 200 body returned, 404 unknown or expired id, 410 body was elided, 422 the file is an error placeholder.
- `GET /v1/containers/{container_id}/files/{file_id}/content` is an OpenAI-compatible alias backed by the same file store.
- Each `agentic_tool_calls` entry in a chat response carries a `file_ids` array attributing files to that tool round.

For examples and supported file-type behavior, see [OpenAI-compatible file inputs](/guides/agents/file-inputs/).

## Skills

`POST /v1/skills` accepts multipart uploaded OpenAI-compatible Skills. Upload either a zip file or files from one top-level skill directory; the directory must contain `SKILL.md` with `name` and `description` frontmatter. Use `GET /v1/skills` to list uploaded skills and `POST /v1/skills/{skill_id}/versions` to add a new version.

Uploading skills does not require shell execution, but running a Responses request with a `skill_reference` does. Start the server with at least `--enable-shell`; prefer `--agent` when you want the full local agent runtime.

## Session semantics

`GET /v1/sessions/{session_id}` exports a `SerializedSession` (404 if missing); `PUT` imports one, replacing any session with the same id; `DELETE` always returns 200 whether the session existed or not. Session lifecycle and splicing behavior are on the [sessions page](/guides/agents/persist-sessions/).

## Metrics

`GET /metrics` exposes Prometheus text format. HTTP metrics are enabled by default and can be disabled with `--disable-metrics`.

- `http_requests_total` (counter): completed request count, labeled by `method`, `path`, `model`, and `status`.
- `http_request_duration_seconds` (histogram): request latency, labeled by `method`, `path`, `model`, and `status`.
- `http_requests_in_flight` (gauge): requests currently running, labeled by `method`, `path`, and `model`.
- `http_request_body_bytes` (histogram): request body size when the body size is known, labeled by `method`, `path`, and `model`.

Streaming responses additionally record `mistralrs_time_to_first_token_seconds` and `mistralrs_inter_token_latency_seconds` (histograms, same labels as the request-duration histogram). The endpoint also serves the unlabeled `mistralrs_*` engine metrics (throughput, queues, KV cache, prefix cache, speculative decoding). See the [observability guide](/guides/deploy/observability/#prometheus-metrics) for the full table and PromQL examples.

The `path` label is the matched route pattern (e.g. `/v1/responses/{response_id}`), not the concrete URI, so per-request ids do not inflate label cardinality. The `model` label is the resolved model id for inference requests, reads the `model` query parameter for `GET /v1/lora_adapters`, defaults to the server default model when the request omits `model`, uses explicit `model_id` values for model-management requests, uses `unknown` when a required request body cannot be read or parsed as JSON, and uses `none` for routes that do not target a model. Unmatched requests are labeled `<unmatched>`. Health, metrics, docs, UI, and CORS preflight requests are excluded from these HTTP metrics. Returns 503 until the metrics recorder initializes at startup, or when metrics are disabled.

## Response headers

`Content-Type: application/json` for non-streaming responses, `text/event-stream` for streams. The session id (when assigned or matched) is in the response body's `session_id` field, not a header.

## Compatibility

Field-level OpenAI deviations, silently-ignored fields, and Responses API restrictions: [OpenAI compatibility](/reference/openai-compatibility/). Anthropic field and content-block support: [Anthropic Messages API](/guides/serve/anthropic-messages-api/).
