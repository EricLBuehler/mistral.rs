---
title: Anthropic Messages API
description: Use Anthropic-compatible clients with the mistralrs HTTP server.
sidebar:
  order: 6
---

mistral.rs exposes Anthropic-compatible Messages endpoints at `POST /v1/messages`
and `POST /v1/messages/count_tokens`. They run through the same local model,
scheduler, chat templates, multimodal handling, tool calling, and agentic runtime
as `/v1/chat/completions`.

For Claude Code configuration, see [Use Codex and Claude Code](/mistral.rs/guides/serve/coding-agents/).

## Start the server

```bash
mistralrs serve -m Qwen/Qwen3-4B
```

Use `model: "default"` for a single-model server. In multi-model serving, use the
configured model id exactly as it appears in `GET /v1/models`.

## Basic request

```bash
curl http://localhost:1234/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: not-used' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "default",
    "max_tokens": 128,
    "system": "You are concise.",
    "messages": [
      {"role": "user", "content": "Write a haiku about local inference."}
    ]
  }'
```

Response shape:

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
    "input_tokens": 18,
    "output_tokens": 31
  }
}
```

The server accepts Anthropic headers for client compatibility, but does not validate
`x-api-key`. Put authentication in a reverse proxy when exposing the server to users.

## Streaming

Set `stream: true` to receive Anthropic-style Server-Sent Events:

```bash
curl http://localhost:1234/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: not-used' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "default",
    "max_tokens": 128,
    "stream": true,
    "messages": [{"role": "user", "content": "Count to three."}]
  }'
```

The stream uses `message_start`, `content_block_start`, `content_block_delta`,
`content_block_stop`, `message_delta`, and `message_stop` events. It also emits
Anthropic `ping` events while idle. Text deltas use
`{"type":"text_delta","text":"..."}`. Thinking deltas use
`{"type":"thinking_delta","thinking":"..."}` when the model produces separate
reasoning content. Tool-call argument deltas use
`{"type":"input_json_delta","partial_json":"..."}`.

## Count tokens

Use `POST /v1/messages/count_tokens` with the same request shape to count the
input tokens after chat-template formatting:

```bash
curl http://localhost:1234/v1/messages/count_tokens \
  -H 'content-type: application/json' \
  -H 'x-api-key: not-used' \
  -H 'anthropic-version: 2023-06-01' \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "Count these tokens."}]
  }'
```

Response:

```json
{"input_tokens": 14}
```

## Supported fields

Request fields:

| Anthropic field | Support |
|---|---|
| `model` | Supported. Use `default` or a loaded model id. |
| `messages` | Supports `user` and `assistant` messages with string content or content blocks. |
| `system` | Supported as a top-level string or text-block array. |
| `max_tokens` | Supported. |
| `temperature`, `top_p`, `top_k`, `min_p` | Supported. |
| `stop_sequences` | Supported. |
| `stream` | Supported. |
| `tools` | Client tools are converted to OpenAI-compatible function tools. Anthropic server tools for `web_search_*` and `code_execution_*` map to mistral.rs agentic features. |
| `tool_choice` | `auto`, `none`, and specific client `tool` choices are supported. `any` is accepted as `auto`. Anthropic server-tool choices are accepted as `auto`. |
| `thinking` | `{"type":"enabled"}` maps to mistral.rs thinking mode when the loaded chat template supports it. |
| `enable_thinking`, `reasoning_effort` | Supported as mistral.rs extensions. |
| `logit_bias`, `logprobs`, `top_logprobs` | Supported as mistral.rs extensions. |
| `presence_penalty`, `frequency_penalty`, `repetition_penalty` | Supported as mistral.rs extensions. |
| `response_format`, `grammar` | Supported as mistral.rs extensions. Do not set both in one request. |
| `dry_multiplier`, `dry_base`, `dry_allowed_length`, `dry_sequence_breakers` | Supported as mistral.rs extensions. |
| `metadata` | Accepted for client compatibility. |

Content blocks:

| Block | Support |
|---|---|
| `text` | Supported. |
| `image` | Supports base64 and URL sources. Requires a multimodal model. |
| `tool_use` | Supported on assistant messages. |
| `tool_result` | Supported on user messages. Text results are forwarded as tool messages. |
| `thinking`, `redacted_thinking` | Accepted in request history. Returned when the model exposes separate reasoning content. |

mistral.rs agentic extensions accepted on this endpoint: `session_id`,
`web_search_options`, `enable_code_execution`, `agent_permission`,
`code_execution_permission`, `files`, `max_tool_rounds`, and `truncate_sequence`.

## Tool use

Anthropic tool definitions:

```json
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get weather for a city.",
      "input_schema": {
        "type": "object",
        "properties": {"city": {"type": "string"}},
        "required": ["city"]
      }
    }
  ]
}
```

When the model asks for a tool, the response contains a `tool_use` content block:

```json
{
  "type": "tool_use",
  "id": "call-...",
  "name": "get_weather",
  "input": {"city": "Paris"}
}
```

Return the result in a later user message with a `tool_result` block:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "call-...",
      "content": "Light rain, 12 C."
    }
  ]
}
```

For server-executed tools, use the same mistral.rs agent fields as Chat Completions.
Streaming may include mistral.rs named events such as `agentic_tool_call_progress`,
`agentic_tool_approval_required`, and `file_produced`.

## Agentic server tools

Anthropic web search server-tool declarations enable mistral.rs web search for
the request. The server must be started with search enabled, for example with
`mistralrs serve --agent ...` or `mistralrs serve --enable-search ...`.
`web_search_20260209` is also accepted and enables code execution for the
request because Anthropic's dynamic web-search variant uses code-backed
filtering.

```json
{
  "tools": [
    {
      "type": "web_search_20250305",
      "name": "web_search",
      "user_location": {
        "type": "approximate",
        "city": "New York",
        "country": "US",
        "region": "NY",
        "timezone": "America/New_York"
      }
    }
  ]
}
```

Anthropic code-execution server-tool declarations enable the built-in Python
tool for the request. The server must be started with code execution enabled,
for example with `mistralrs serve --agent ...` or
`mistralrs serve --enable-code-execution ...`.

```json
{
  "tools": [
    {"type": "code_execution_20250825", "name": "code_execution"}
  ],
  "agent_permission": "auto"
}
```

You can combine both server tools. The Anthropic request field
`tool_choice: {"type":"none"}` disables both client and server tools for that
request.

## Examples

Server examples live in `examples/server/`:

| File | What it shows |
|---|---|
| `anthropic_chat.py` | Plain non-streaming Messages request. |
| `anthropic_streaming.py` | Anthropic SSE parsing. |
| `anthropic_tool_calling.py` | Client-side tool use with `tool_use` and `tool_result`. |
| `anthropic_agentic.py` | Anthropic server-tool declarations mapped to mistral.rs web search and code execution. |
