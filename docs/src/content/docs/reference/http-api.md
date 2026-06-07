---
title: HTTP API reference
description: Every endpoint, request schema, response schema, and mistralrs-specific extension.
sidebar:
  order: 4
---

mistral.rs implements the OpenAI Chat Completions API, the Responses API, and a few mistral.rs-specific endpoints. This page lists every path with its request and response shape.

Fields not documented here are either standard OpenAI fields (pass through unchanged) or ignored. mistral.rs-specific extensions are called out explicitly.

## Core endpoints

### `POST /v1/chat/completions`

Chat completion request.

```json
{
  "model": "default",
  "messages": [ ... ],
  "max_tokens": 256,
  "temperature": 0.7,
  "stream": false,
  "tools": [ ... ],
  "tool_choice": "auto",
  "session_id": "optional-string",
  "web_search_options": { ... },
  "enable_code_execution": false,
  "agent_permission": "auto",
  "max_tool_rounds": 4
}
```

`tools` accepts OpenAI-compatible function tool definitions. mistral.rs also honors `tools[*].function.strict: true`, which constrains generated tool arguments to the tool's `parameters` JSON Schema. See [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/).

Response (non-streaming):

```json
{
  "id": "chatcmpl-...",
  "object": "chat.completion",
  "created": 1234567890,
  "model": "default",
  "choices": [ ... ],
  "usage": { ... },
  "session_id": "...",
  "agentic_tool_calls": [ ... ]
}
```

mistral.rs-specific request fields include `session_id`, `web_search_options`, `enable_code_execution`, `agent_permission`, `max_tool_rounds`, and `files`. The server must be started with the corresponding capabilities, such as `--enable-search` or `--enable-code-execution`.

`agent_permission` accepts `"auto"`, `"ask"`, or `"deny"` and applies to server-executed agent actions: code execution, web search, file tools, registered callbacks, and external tool dispatch. `code_execution_permission` is accepted as a compatibility alias. See [agent permissions](/mistral.rs/guides/agents/agentic-runtime/#agent-permissions) for the shared behavior across CLI, HTTP, Python, and Rust.

Over HTTP, `"ask"` requires `stream: true`. The stream emits a named `agentic_tool_approval_required` event when an action needs approval, then waits for the app to approve or deny it with `POST /v1/agent/approvals/{approval_id}`. Non-streaming chat requests with `"ask"` return a validation error.

mistral.rs-specific response fields: `session_id` (string), `agentic_tool_calls` (array of tool-call records from the agentic loop, each with a `file_ids` array), `files` (array of `File` objects produced during the request).

When `stream: true`, the response is Server-Sent Events: unnamed `data:` lines carry chat completion chunks, named `agentic_tool_call_progress` events carry tool-loop milestones, named `agentic_tool_approval_required` events carry pending agent approvals, and named `file_produced` events carry each typed file emitted during the run. Stream terminates with `data: [DONE]`.

Approval event:

```text
event: agentic_tool_approval_required
data: {"approval_id":"appr_abc123","session_id":"...","round":1,"tool":{"source":"built_in","kind":"code_execution","label":"Python code"},"arguments":{"code":"...","outputs":[]}}
```

Resolve the approval:

```http
POST /v1/agent/approvals/{approval_id}
Content-Type: application/json

{"decision":"deny","remember_for_session":false,"message":"Do not run code for this request."}
```

`decision` is `"approve"` or `"deny"`. Set `remember_for_session: true` on an approve response to allow later agent actions in the same `session_id` without another approval event. A deny response may include `message`; that text is returned to the model as the tool result.

Unanswered approvals are denied after five minutes.

The endpoint returns `{"status":"resolved"}` when the waiting tool call was released, `{"status":"queued"}` if the app answered before the runtime started waiting, and `{"status":"not_found"}` for an unknown or expired approval ID.

For app-facing tool timelines, generated media fields, and sessions, see [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).

### `POST /v1/messages`

Anthropic-compatible Messages API. It uses the same local model path as Chat
Completions, but request and response bodies follow Anthropic's Messages shape.
See the [Anthropic Messages API guide](/mistral.rs/guides/serve/anthropic-messages-api/).

```json
{
  "model": "default",
  "max_tokens": 256,
  "system": "You are concise.",
  "messages": [
    {"role": "user", "content": "Hello"}
  ],
  "stream": false,
  "tools": [ ... ],
  "tool_choice": {"type": "auto"},
  "thinking": {"type": "enabled"},
  "response_format": { ... },
  "grammar": null,
  "min_p": 0.05
}
```

Non-streaming response:

```json
{
  "id": "chatcmpl-...",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "..."}
  ],
  "model": "default",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 12,
    "output_tokens": 24
  }
}
```

`stream: true` returns Anthropic-style SSE events: `message_start`,
`content_block_start`, `content_block_delta`, `content_block_stop`,
`message_delta`, `message_stop`, and idle `ping` events. Text deltas use
`text_delta`; separate reasoning streams as `thinking_delta` when available.

The endpoint also accepts Anthropic server-tool declarations:

```json
{
  "tools": [
    {"type": "web_search_20250305", "name": "web_search"},
    {"type": "code_execution_20250825", "name": "code_execution"}
  ],
  "agent_permission": "auto"
}
```

`web_search_*` maps to `web_search_options`; `code_execution_*` maps to
`enable_code_execution`. Server capabilities still apply: start with `--agent`,
`--enable-search`, or `--enable-code-execution` as needed.

The endpoint also routes mistral.rs chat extensions for sampling, constraints,
and reasoning: `logit_bias`, `logprobs`, `top_logprobs`, `presence_penalty`,
`frequency_penalty`, `repetition_penalty`, `min_p`, `top_k`, `response_format`,
`grammar`, `dry_multiplier`, `dry_base`, `dry_allowed_length`,
`dry_sequence_breakers`, `enable_thinking`, and `reasoning_effort`.

### `POST /v1/messages/count_tokens`

Anthropic-compatible token counting for the Messages request shape. Response:

```json
{"input_tokens": 42}
```

The count is produced by the loaded model's tokenizer after chat-template
formatting.

### `POST /v1/completions`

Text completion (non-chat). Schema is OpenAI-compatible. Supported mistralrs extensions: `top_k`, `min_p`, `repetition_penalty`, `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers`, `grammar`, `truncate_sequence`. The chat-only fields (`session_id`, `enable_code_execution`, `agent_permission`, `files`, `web_search_options`, `enable_thinking`, `reasoning_effort`, `max_tool_rounds`) have no effect on this endpoint.

### `POST /v1/embeddings`

Embedding request. `input`, `encoding_format` (`"float"` or `"base64"`) supported. `dimensions` returns an error. Extension: `truncate_sequence`.

### `POST /v1/images/generations`

Image generation. Uses `height` and `width` in place of OpenAI's `size`. `response_format` defaults to `"Url"`. See the [image generation guide](/mistral.rs/guides/models/use-image-generation/).

### `POST /v1/audio/speech`

Text to speech. `model` and `input` supported. `response_format` accepts only `wav` and `pcm`; other OpenAI values return a validation error. `voice`, `speed`, `instructions` are ignored.

### `GET /v1/models`

Lists loaded models.

Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "default",
      "object": "model",
      "created": 1234567890,
      "owned_by": "local",
      "status": "loaded",
      "tools_available": true,
      "mcp_tools_count": 5,
      "mcp_servers_connected": 1
    }
  ]
}
```

Status values: `loaded`, `unloaded`, `reloading`.

## Responses API

### `POST /v1/responses`

OpenAI Responses API. Schema matches OpenAI's spec. See the [Responses guide](/mistral.rs/guides/serve/openai-responses-api/) for supported and unsupported fields.

Function tools in Responses requests also accept `strict: true` and use the same strict tool-calling path as Chat Completions.

### `GET /v1/responses/{id}`

Retrieve a response by id.

### `DELETE /v1/responses/{id}`

Delete a response.

### `POST /v1/responses/{id}/cancel`

Cancel a background response.

## Model management

### `POST /v1/models/unload`

Unload a model, freeing its memory.

```json
{ "model_id": "qwen" }
```

Response: `{ "model_id": "qwen", "status": "unloaded" }`.

### `POST /v1/models/reload`

Reload a previously unloaded model.

```json
{ "model_id": "qwen" }
```

Response: `{ "model_id": "qwen", "status": "loaded" }`.

### `POST /v1/models/status`

Query a model's current status.

```json
{ "model_id": "qwen" }
```

### `POST /v1/models/tune`

Launch a tune run.

## Session management

### `GET /v1/sessions/{session_id}`

Export an agentic session. Response is a `SerializedSession` object with messages, tool-call history, images, and videos. Returns 404 if the session does not exist.

### `PUT /v1/sessions/{session_id}`

Import a session. Body is a `SerializedSession` produced by a previous `GET`. Replaces any existing session with the same id.

### `DELETE /v1/sessions/{session_id}`

Delete a session. Always returns 200 whether the session existed or not.

## System

### `GET /health`

Returns 200 when the server is up. Does not verify model load status.

### `GET /metrics`

Exposes server metrics in the Prometheus text exposition format for scraping. Two metrics are recorded per request: `http_requests_total` (counter) and `http_request_duration_seconds` (histogram), each labeled by `method`, `path`, and `status`. The `path` label uses the matched route pattern (e.g. `/v1/responses/{id}`) rather than the concrete URI, so per-request identifiers do not inflate label cardinality; requests that match no route are labeled `<unmatched>`. Returns 503 until the recorder is initialized at startup.

```bash
curl http://localhost:1234/metrics
```

### `GET /v1/system/info`

Returns system information (OS, memory, GPUs, mistralrs version).

### `POST /v1/system/doctor`

Returns a diagnostic report equivalent to `mistralrs doctor` output.

### `POST /re_isq`

Re-apply ISQ to the loaded model.

## Streaming event types

Streaming responses are Server-Sent Events. Default (unnamed) `data:` lines carry chat completion chunks in OpenAI format; the stream ends with `data: [DONE]`. Named events are used for the agentic timeline:

| Event | Body |
|---|---|
| (default `data:`) | Chat completion chunk in OpenAI format. Stream terminator is `data: [DONE]`. |
| `agentic_tool_call_progress` | Tool-loop progress. Includes `round`, `tool_name`, `phase` (`calling` or `complete`), and structured `data`. |
| `file_produced` | A `File` object emitted during the run. Each file is sent once. |

Tool-progress `data.tool_type` is `code_execution`, `web_search`, or `custom`. Code execution events can include `images_base64` and `video_frames_base64`.

## Files

mistral.rs returns typed file outputs from agentic runs as first-class objects, separate from the model transcript.

### Request schema

| Field | Type | Notes |
|---|---|---|
| `files[].name` | string | Filename. Required. |
| `files[].format` | string | Format hint (`png`, `csv`, `json`, ...). Inferred from the extension if omitted. |
| `files[].description` | string | Optional hint surfaced to the model. |

Example:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Plot sin(x) and save as plot.png."}
  ],
  "enable_code_execution": true,
  "files": [{"name": "plot.png", "format": "png"}]
}
```

### Response schema

The non-streaming response gains a top-level `files` array of `File` objects:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Stable id, format `file_<run>_r<round>_<idx>`. |
| `name` | string | Filename as written. |
| `format` | string | Open-ended format string. |
| `mime_type` | string | Content-Type. |
| `bytes` | integer | Body size. |
| `created_at` | integer | Unix epoch seconds. |
| `source` | object | `{"tool", "round", "turn"}` attribution. |
| `text` | string | Full text body for text files. Absent if elided. |
| `preview` | string | Short UTF-8 preview for text files. |
| `data_base64` | string | Base64 body for binary files. Absent if elided. |
| `code`, `message` | strings | Present if the file failed to materialize. |

Each entry in `agentic_tool_calls` carries a `file_ids` array listing the files attributable to that round.

Example response:

```json
{
  "files": [
    {
      "id": "file_abc_r0_0",
      "name": "plot.png",
      "format": "png",
      "mime_type": "image/png",
      "bytes": 14823,
      "source": {"tool": "mistralrs_execute_python", "round": 0, "turn": 0},
      "data_base64": "iVBORw0KGgo..."
    }
  ],
  "agentic_tool_calls": [
    {
      "round": 0,
      "name": "mistralrs_execute_python",
      "file_ids": ["file_abc_r0_0"]
    }
  ]
}
```

### `file_produced` event

Streaming requests emit each file as soon as it is produced. The body is the same `File` JSON as the non-streaming `files[]` entry.

```text
event: file_produced
data: {"id":"file_abc_r0_0","name":"plot.png","format":"png","mime_type":"image/png","bytes":14823,"source":{"tool":"mistralrs_execute_python","round":0,"turn":0},"data_base64":"iVBORw0KGgo..."}
```

### Size policy

Bodies up to 8 MB ship inline (`text` or `data_base64`). Above the cap, the body field is omitted and the client fetches the raw bytes via `GET /v1/files/{id}/content`. Inside the model's context, text files only ever see the first 1024 bytes as a preview; the model uses `read_file` to inspect more.

### Files API

OpenAI-compatible Files endpoints. Upload (`POST /v1/files`) is not implemented; files arrive via agentic tool calls.

| Method | Path | Returns |
|---|---|---|
| `GET` | `/v1/files` | `{object: "list", data: [<File metadata>]}` |
| `GET` | `/v1/files/{id}` | File metadata JSON |
| `GET` | `/v1/files/{id}/content` | Raw bytes (Content-Type, Content-Length, Content-Disposition) |
| `DELETE` | `/v1/files/{id}` | `{id, object: "file", deleted: bool}` |

File metadata shape:

```json
{
  "id": "file_abc_r0_0",
  "object": "file",
  "bytes": 14823,
  "created_at": 1735632000,
  "filename": "plot.png",
  "purpose": "agent_output",
  "format": "png",
  "mime_type": "image/png",
  "source": {"tool": "mistralrs_execute_python", "round": 0, "turn": 0}
}
```

`/v1/files/{id}/content` response codes:

| Code | Meaning |
|---|---|
| 200 | Body returned. |
| 404 | Unknown or expired file id. |
| 410 | File body was elided. |
| 422 | The file is an error placeholder. |

## OpenAI compatibility

See the [OpenAI compatibility reference](/mistral.rs/reference/openai-compatibility/) for the supported and unsupported fields.
