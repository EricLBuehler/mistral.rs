---
title: Anthropic Messages API
description: Use Anthropic-compatible clients with the mistralrs HTTP server.
---

mistral.rs exposes Anthropic-compatible Messages endpoints at `POST /v1/messages`
and `POST /v1/messages/count_tokens`. They run through the same local model,
scheduler, chat templates, multimodal handling, tool calling, and [agentic runtime](/guides/agents/agentic-runtime/)
as `/v1/chat/completions`. Anthropic clients use `http://localhost:1234` as the
base URL (no `/v1` suffix; the client appends `/v1/messages` itself).

Start the server as for the [OpenAI-compatible API](/guides/serve/openai-compatible-apis/);
both APIs are always served. For Claude Code configuration, see
[Use Codex and Claude Code](/guides/serve/coding-agents/).

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

Core Messages requests and responses, streaming event order, client tool loops,
stop reasons and sequences, token usage, request IDs, structured errors, and
`/v1/messages/count_tokens` use Anthropic-compatible wire shapes.

Some hosted Anthropic features have local equivalents rather than identical
semantics:

| Feature | Local behavior |
|---|---|
| Prompt caching | `cache_control` is accepted. mistral.rs uses its own automatic prefix cache, reports detected hits as `cache_read_input_tokens`, and reports `cache_creation_input_tokens` as `0`; explicit breakpoints and TTLs are not implemented. |
| Extended thinking | Thinking maps to the loaded model's chat-template reasoning controls. `budget_tokens` is accepted but not enforced. Returned thinking blocks include an empty structural signature because mistral.rs cannot produce Anthropic cryptographic signatures. |
| Rich tool results | Text results are supported. Nested image, document, search-result, and citation blocks are not translated as multimodal tool content, and `is_error` is not separately modeled. |
| Server tools | Web search and code execution use mistral.rs agentic implementations. Their extra streaming events are mistral.rs extensions, not Anthropic server-tool result blocks. |
| Beta request controls | Unknown forward-compatible fields are ignored. Anthropic context management and context editing are not implemented. |

`max_tokens` must be greater than zero. Authentication headers are accepted for
client compatibility but are not validated by the local server; use a reverse
proxy when authentication is required.

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
`mistralrs serve --agent ...` or `mistralrs serve --enable-search ...`. Two
declaration types are accepted:

- `web_search_20250305` enables web search only.
- `web_search_20260209` (Anthropic's dynamic web-search variant) also enables
  code execution for the request, because it uses code-backed filtering. The
  server must additionally be started with code execution enabled.

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

## Skills

Anthropic-compatible Skills use the same uploaded skill store as
[OpenAI-compatible Skills](/guides/agents/skills/). Upload a zip or
multipart skill directory to `POST /v1/skills`, then reference the returned
custom skill id from `container.skills` on `/v1/messages`.

Start the server with shell execution enabled. `--agent` is recommended because
it also enables the rest of the local agent runtime:

```bash
mistralrs serve --agent -p 1234 -m Qwen/Qwen3-4B
```

List and upload endpoints accept Anthropic headers. When an Anthropic header is
present, `GET /v1/skills` returns Anthropic-style fields such as
`display_title`, `latest_version`, `source`, `has_more`, and `next_page`.
`source=custom` returns uploaded local Skills; `source=anthropic` returns an
empty list because mistral.rs does not bundle Anthropic-managed built-in Skills.

Use the uploaded Skill in a Messages request:

```bash
curl http://localhost:1234/v1/messages \
  -H 'content-type: application/json' \
  -H 'x-api-key: not-used' \
  -H 'anthropic-version: 2023-06-01' \
  -H 'anthropic-beta: code-execution-2025-08-25,skills-2025-10-02' \
  -d '{
    "model": "default",
    "max_tokens": 1024,
    "agent_permission": "auto",
    "max_tool_rounds": 6,
    "container": {
      "skills": [
        {"type": "custom", "skill_id": "skill_...", "version": "latest"}
      ]
    },
    "tools": [
      {"type": "code_execution_20250825", "name": "code_execution"}
    ],
    "messages": [
      {"role": "user", "content": "Use the uploaded skill to complete the task."}
    ]
  }'
```

Skills are mounted into the shell runtime. The Anthropic code execution tool
declaration is accepted for client compatibility and enables Python code
execution too, but the skill bundle itself is read from the shell workdir under
`skills/<skill-name>/`.

Generated artifacts are available the same way as other mistral.rs agentic
files. Non-streaming responses include a top-level `files` array, and streaming
responses emit `file_produced` events. Download bytes from
`GET /v1/files/{file_id}/content`.

## Examples

| Example | What it shows |
|---|---|
| [anthropic_chat](/examples/server/anthropic-chat/) | Plain non-streaming Messages request. |
| [anthropic_streaming](/examples/server/anthropic-streaming/) | Anthropic SSE parsing. |
| [anthropic_tool_calling](/examples/server/anthropic-tool-calling/) | Client-side tool use with `tool_use` and `tool_result`. |
| [anthropic_agentic](/examples/server/anthropic-agentic/) | Anthropic server-tool declarations mapped to mistral.rs web search and code execution. |
| [anthropic_skills](/examples/server/anthropic-skills/) | Upload a custom Skill and use it from `container.skills`. |
