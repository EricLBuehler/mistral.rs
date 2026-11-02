---
title: Anthropic Messages API
description: Use Anthropic-compatible clients with the mistralrs HTTP server.
---

Use Anthropic clients with `http://localhost:1234` as the base URL, without `/v1`. The server exposes `/v1/messages` and `/v1/messages/count_tokens` alongside the OpenAI endpoints. Start it with the [serving guide](/guides/serve/openai-compatible-apis/); see [coding agents](/guides/serve/coding-agents/) for Claude Code setup.

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
  "id": "0",
  "type": "message",
  "role": "assistant",
  "content": [
    {"type": "text", "text": "..."}
  ],
  "model": "Qwen/Qwen3-4B",
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "usage": {
    "input_tokens": 18,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "output_tokens": 31
  }
}
```

`id` is the server's per-request sequence number and `model` echoes the loaded model's name, not the `"default"` alias sent in the request.

The server accepts Anthropic headers for client compatibility, but does not validate
`x-api-key`. Put authentication in a reverse proxy when exposing the server to users.

## Streaming

Add `stream: true` to the request. The server emits Anthropic message/content-block events, including text, thinking, and tool-argument deltas, plus idle pings. See [HTTP streaming semantics](/reference/http-api/#streaming-anthropic-messages) for event names and the [streaming example](/examples/server/anthropic-streaming/) for a parser.

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
| `messages` | Supports `user`, `assistant`, and mid-conversation `system` messages with string content or content blocks. |
| `system` | Supported as a top-level string or text-block array. |
| `max_tokens` | Supported. |
| `temperature`, `top_p`, `top_k`, `min_p` | Supported. |
| `stop_sequences` | Supported. |
| `stream` | Supported. |
| `tools` | Client tools are converted to OpenAI-compatible function tools. Anthropic server tools for `web_search_*` and `code_execution_*` map to mistral.rs agentic features. |
| `tool_choice` | `auto`, `any`, `none`, and specific client `tool` choices are supported. `any` requires a client tool call when client tools are present. Anthropic server-tool choices use local agentic auto-selection. |
| `container.skills` | Supports uploaded custom Skills with `{"type":"custom","skill_id":"...","version":"latest"}`. Anthropic-managed built-in Skills such as `pptx`, `xlsx`, `docx`, and `pdf` are not bundled. |
| `thinking` | `enabled` and `adaptive` map to thinking on; `disabled` maps to thinking off. `display: "omitted"` suppresses reasoning text while preserving the thinking block lifecycle. The loaded chat template determines the effect. |
| `output_config` | Native `effort` and JSON Schema `format` are supported. Effort accepts `low`, `medium`, `high`, `xhigh`, and `max`; `max` maps to local `xhigh`. Do not combine these fields with the equivalent mistral.rs extensions. |
| `enable_thinking`, `reasoning_effort` | Supported as mistral.rs extensions. Effort also accepts `off`; `none` aliases `off`. If all controls are omitted, thinking defaults on with no selected effort. Contradictory native and extension controls return a validation error. |
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
| `tool_result` | Supported on user messages. String and text-block results are forwarded as tool messages. |
| `thinking`, `redacted_thinking` | Accepted in request history. Returned when the model exposes separate reasoning content. |

mistral.rs agentic extensions accepted on this endpoint: `session_id`,
`web_search_options`, `enable_code_execution`, `agent_permission`,
`code_execution_permission`, `files`, `max_tool_rounds`, and `truncate_sequence`.

## Compatibility status

Messages, client tool loops, errors, and token counts use Anthropic wire shapes. Hosted features differ:

| Feature | Local behavior |
|---|---|
| Prompt caching | `cache_control` is accepted. mistral.rs uses its own automatic prefix cache, reports detected hits as `cache_read_input_tokens`, and reports `cache_creation_input_tokens` as `0`; explicit breakpoints and TTLs are not implemented. |
| Extended thinking | Thinking maps to the loaded model's chat-template reasoning controls. `budget_tokens` is accepted but not enforced. Returned thinking blocks include an empty structural signature because mistral.rs cannot produce Anthropic cryptographic signatures. |
| Rich tool results | Text results are supported. Nested image, document, search-result, and citation blocks are not translated as multimodal tool content, and `is_error` is not separately modeled. |
| Server tools | Web search and code execution use mistral.rs agentic implementations. Their extra streaming events are mistral.rs extensions, not Anthropic server-tool result blocks. |
| Beta request controls | Unknown forward-compatible fields are ignored. Anthropic context management and context editing are not implemented. |

`max_tokens` must be greater than zero.

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

## Agentic server tools

Enable the required executors when starting the server, for example with `mistralrs serve --agent ...`. Then include server tools in the request:

```json
{
  "tools": [
    {"type": "web_search_20250305", "name": "web_search"},
    {"type": "code_execution_20250825", "name": "code_execution"}
  ],
  "agent_permission": "auto"
}
```

`web_search_20250305` requires search enabled; `web_search_20260209` also requires code execution for dynamic filtering. Code-execution declarations require the Python executor. Search accepts approximate `user_location` fields such as city, country, region, and timezone.

`tool_choice: {"type":"none"}` disables both client and server tools. Server-tool streams include mistral.rs progress, approval, and file events; see [HTTP semantics](/reference/http-api/#streaming-anthropic-messages).

## Skills

Upload a custom Skill through `/v1/skills`, then reference it in a Messages request:

```json
{
  "container": {
    "skills": [{"type": "custom", "skill_id": "skill_...", "version": "latest"}]
  }
}
```

Execution requires shell support (`--enable-shell` or `--agent`). Adding a code-execution tool also enables Python execution for that request. See the [Skills guide](/guides/agents/skills/) for uploads and the [Messages Skills example](/examples/server/anthropic-skills/) for a complete request.

With Anthropic headers, skill-list responses use `display_title`, `latest_version`, `source`, `has_more`, and `next_page`. `source=custom` lists uploads; `source=anthropic` is empty because built-in Anthropic Skills are not bundled.

Generated files appear in response `files[]` or streamed `file_produced` events. Download them from `/v1/files/{file_id}/content`.

## Examples

| Example | What it shows |
|---|---|
| [anthropic_chat](/examples/server/anthropic-chat/) | Plain non-streaming Messages request. |
| [anthropic_streaming](/examples/server/anthropic-streaming/) | Anthropic SSE parsing. |
| [anthropic_tool_calling](/examples/server/anthropic-tool-calling/) | Client-side tool use with `tool_use` and `tool_result`. |
| [anthropic_agentic](/examples/server/anthropic-agentic/) | Anthropic server-tool declarations mapped to mistral.rs web search and code execution. |
| [anthropic_skills](/examples/server/anthropic-skills/) | Upload a custom Skill and use it from `container.skills`. |
