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
  "web_search_options": { ... }
}
```

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

mistral.rs-specific fields: `session_id` (string), `agentic_tool_calls` (array of tool-call records from the agentic loop).

When `stream: true`, the response is Server-Sent Events with event types `data` for chunks and `agentic_tool_call_progress` for tool-loop milestones. Stream terminates with `data: [DONE]`.

### `POST /v1/completions`

Text completion (non-chat). Schema is OpenAI-compatible; no mistralrs extensions.

### `POST /v1/embeddings`

Embedding request. `input`, `encoding_format` (`"float"` or `"base64"`) supported. `dimensions` returns an error. Extension: `truncate_sequence`.

### `POST /v1/images/generations`

Image generation. Uses `height` and `width` in place of OpenAI's `size`. `response_format` defaults to `"url"`. See the [image generation guide](/mistral.rs/guides/models/use-image-generation/).

### `POST /v1/audio/speech`

Text to speech. `model`, `input`, `response_format` supported. `voice`, `speed`, `instructions` are ignored.

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

### `GET /v1/system/info`

Returns system information (OS, memory, GPUs, mistralrs version).

### `POST /v1/system/doctor`

Returns a diagnostic report equivalent to `mistralrs doctor` output.

### `POST /re_isq`

Re-apply ISQ to the loaded model.

## Streaming event types

Streaming responses are Server-Sent Events with these `event` types:

| Event | Body |
|---|---|
| `data` | Chat completion chunk in OpenAI format. Terminal event is `data: [DONE]`. |
| `agentic_tool_call_progress` | Tool-loop progress. Includes `round`, `tool_name`, `phase` (`calling` or `complete`), and structured `data`. |

## OpenAI compatibility

See the [OpenAI compatibility reference](/mistral.rs/reference/openai-compatibility/) for the supported and unsupported fields.
