---
title: HTTP API semantics
description: Streaming event types, authentication, approval flow, and protocol behavior that the generated OpenAPI reference does not express.
---

Every endpoint, request schema, and response schema is in the [generated HTTP API reference](/mistral.rs/reference/http-api-generated/), produced from the server's OpenAPI document. This page covers what a schema cannot say: streaming wire formats, authentication, and cross-request semantics.

## Discovering the API from a running server

- `GET /docs` serves a Swagger UI for the exact build you are running.
- `GET /api-doc/openapi.json` serves the raw OpenAPI document.
- `GET /` is an alias for `GET /health`: returns 200 when the server is up. Neither verifies model load status.

## Authentication

There is none. The server accepts and ignores `Authorization: Bearer ...` (OpenAI clients) and `x-api-key` plus `anthropic-version` headers (Anthropic clients), so SDKs that require a key at initialization work with any non-empty string. For real authentication and TLS, put a reverse proxy in front.

## Model routing

The request `model` field selects among loaded models. `"default"` (or omitting the field) targets the configured default model; with a single `-m` model that is the only model. `GET /v1/models` lists real ids plus per-model `status` (`loaded`, `unloaded`, `reloading`), `tools_available`, `mcp_tools_count`, and `mcp_servers_connected`. See [multiple models](/mistral.rs/guides/serve/multiple-models/).

## Streaming

Three endpoints stream, each in its own event dialect:

- `POST /v1/chat/completions`: OpenAI chat-completion chunks plus mistral.rs agentic events.
- `POST /v1/responses`: OpenAI named Responses events.
- `POST /v1/messages`: Anthropic named message events.

### Streaming: Chat Completions

`stream: true` on `POST /v1/chat/completions` returns [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events) (SSE). Unnamed `data:` lines carry chat completion chunks in OpenAI format; the stream terminates with `data: [DONE]`. SSE keep-alive comments (lines starting with `:`) are sent every 10 seconds by default (`KEEP_ALIVE_INTERVAL` env var, milliseconds).

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

### Streaming: Anthropic Messages

`stream: true` on `POST /v1/messages` uses Anthropic's named events: `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, with idle `ping` events. Deltas are `text_delta`, `thinking_delta` (when the model exposes separate reasoning), and `input_json_delta` (tool-call arguments). mistral.rs named events (`agentic_tool_call_progress`, `agentic_tool_approval_required`, `file_produced`) may be interleaved. See the [Anthropic Messages guide](/mistral.rs/guides/serve/anthropic-messages-api/).

## Chat response extensions

Non-streaming chat responses carry three mistral.rs fields beyond the OpenAI shape (omitted when empty):

- `session_id` (string): reuse in later requests to keep agentic state across messages.
- `agentic_tool_calls` (array): ordered record of tool calls made during the agentic loop. Each entry has `round`, opaque `name`, `arguments`, `result_content`, plus `result_images_base64` and `file_ids` when present.
- `files` (array of `File` objects): see [file wire schemas](#file-wire-schemas-and-semantics).

The `usage` object is a superset of OpenAI's, adding timing fields such as `avg_tok_per_sec`, `avg_prompt_tok_per_sec`, `avg_compl_tok_per_sec`, and total prompt/completion times.

## Agent approval flow

`agent_permission: "ask"` requires `stream: true` on HTTP chat requests; non-streaming requests with `"ask"` return a validation error. When an agent action needs approval, the stream emits:

```text
event: agentic_tool_approval_required
data: {"type":"agentic_tool_approval_required","approval_id":"appr_abc123","session_id":"...","round":1,"tool":{"source":"built_in","kind":"code_execution","label":"Python code"},"arguments":{"code":"...","outputs":[]}}
```

The run pauses until the app answers with `POST /v1/agent/approvals/{approval_id}` (body schema in the [generated reference](/mistral.rs/reference/http-api-generated/)). Semantics:

- `decision` is `"approve"` or `"deny"`. A deny may carry `message`, returned to the model as the tool result.
- `remember_for_session: true` on an approve auto-approves later agent actions in the same `session_id`.
- Unanswered approvals are denied after five minutes.
- The response `status` is `"resolved"` (a waiting tool call was released), `"queued"` (the app answered before the runtime started waiting), or `"not_found"` with HTTP 404 (unknown or expired approval id).

Permission levels and how they combine across CLI, HTTP, Python, and Rust are on the [permissions and approvals page](/mistral.rs/guides/agents/permissions-and-approvals/).

## File wire schemas and semantics

Agentic runs return typed file outputs as first-class objects. Chat Completions and Anthropic Messages use a `files[]` array in non-streaming responses and `file_produced` events in streams. Responses uses OpenAI-style `container_file_citation` annotations on assistant `output_text` content. User-provided input files can also be uploaded or attached to OpenAI-compatible requests. These shapes are serialized from an internal type and do not all appear in the OpenAPI document, so they are normative here. (The `/v1/files` metadata endpoints *are* in the [generated reference](/mistral.rs/reference/http-api-generated/).)

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
- Shell and code execution surface only files named in tool `outputs` or request `files`. Shell can also surface files created in earlier calls via `mistralrs_surface_outputs`. Other files remain in the session working directory.
- Files expire 30 minutes after creation (at most 4096 retained).
- `GET /v1/files/{id}/content` status codes: 200 body returned, 404 unknown or expired id, 410 body was elided, 422 the file is an error placeholder.
- `GET /v1/containers/{container_id}/files/{file_id}/content` is an OpenAI-compatible alias backed by the same file store.
- Each `agentic_tool_calls` entry in a chat response carries a `file_ids` array attributing files to that tool round.

For examples and supported file-type behavior, see [OpenAI-compatible file inputs](/mistral.rs/guides/agents/file-inputs/).

## Skills

`POST /v1/skills` accepts multipart uploaded OpenAI-compatible Skills. Upload either a zip file or files from one top-level skill directory; the directory must contain `SKILL.md` with `name` and `description` frontmatter. Use `GET /v1/skills` to list uploaded skills and `POST /v1/skills/{skill_id}/versions` to add a new version.

Uploading skills does not require shell execution, but running a Responses request with a `skill_reference` does. Start the server with at least `--enable-shell`; prefer `--agent` when you want the full local agent runtime.

## Session semantics

`GET /v1/sessions/{session_id}` exports a `SerializedSession` (404 if missing); `PUT` imports one, replacing any session with the same id; `DELETE` always returns 200 whether the session existed or not. Session lifecycle and splicing behavior are on the [sessions page](/mistral.rs/guides/agents/persist-sessions/).

## Metrics

`GET /metrics` exposes Prometheus text format. Two metrics are recorded per request, both labeled by `method`, `path`, and `status`:

- `http_requests_total` (counter): request count.
- `http_request_duration_seconds` (histogram): request latency.

- The `path` label is the matched route pattern (e.g. `/v1/responses/{response_id}`), not the concrete URI, so per-request ids do not inflate label cardinality. Unmatched requests are labeled `<unmatched>`.
Returns 503 until the metrics recorder initializes at startup.

## Response headers

`Content-Type: application/json` for non-streaming responses, `text/event-stream` for streams. The session id (when assigned or matched) is in the response body's `session_id` field, not a header.

## Compatibility

Field-level OpenAI deviations, silently-ignored fields, and Responses API restrictions: [OpenAI compatibility](/mistral.rs/reference/openai-compatibility/). Anthropic field and content-block support: [Anthropic Messages API](/mistral.rs/guides/serve/anthropic-messages-api/).
