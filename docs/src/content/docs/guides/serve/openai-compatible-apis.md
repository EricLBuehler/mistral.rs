---
title: OpenAI-compatible APIs
description: Use OpenAI-compatible clients with the mistralrs HTTP server.
sidebar:
  order: 4
---

mistral.rs exposes OpenAI-compatible endpoints under `/v1`. Use
`http://localhost:1234/v1` as the base URL for OpenAI SDKs and compatible
clients.

The same server also exposes the [Anthropic Messages API](/mistral.rs/guides/serve/anthropic-messages-api/)
at `http://localhost:1234`.

## Start the server

```bash
mistralrs serve -m Qwen/Qwen3-4B
```

Use `model: "default"` for a single-model server. In multi-model serving, use
the configured model id exactly as it appears in `GET /v1/models`.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List loaded models. |
| `POST /v1/chat/completions` | OpenAI-compatible chat, streaming, tool calling, multimodal inputs, and mistral.rs agentic extensions. |
| `POST /v1/responses` | OpenAI Responses API for response objects, polling, background runs, and cancellation. |
| `POST /v1/completions` | Legacy text completions. |
| `POST /v1/embeddings` | Embedding generation. |
| `POST /v1/images/generations` | Image generation. |
| `POST /v1/audio/speech` | Text to speech. |
| `GET /v1/files` | List files produced by agentic runs. |

For every path, request schema, and response schema, see the
[HTTP API reference](/mistral.rs/reference/http-api/). For field-level
compatibility notes, see [OpenAI compatibility](/mistral.rs/reference/openai-compatibility/).

## Chat Completions

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-used" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a haiku about local inference."}
    ],
    "max_tokens": 128
  }'
```

The `Authorization` header is accepted for client compatibility but is not
validated. Put authentication in a reverse proxy when exposing the server to
users.

## OpenAI Python client

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-used")

response = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Say hello from mistral.rs."}],
)

print(response.choices[0].message.content)
```

## Responses

Use `/v1/responses` when the client expects OpenAI's Responses shape or needs
response ids, polling, background processing, or cancellation.

```bash
curl http://localhost:1234/v1/responses \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer not-used" \
  -d '{
    "model": "default",
    "input": "Summarize the benefits of local inference.",
    "max_output_tokens": 128
  }'
```

See [OpenAI Responses API](/mistral.rs/guides/serve/openai-responses-api/) for
supported fields and endpoint-specific behavior.

## Tools and agentic features

OpenAI-compatible function tools are supported on Chat Completions and
Responses. mistral.rs also supports `strict: true` inside function definitions
for JSON-Schema-constrained tool arguments.

When the server is started with agentic capabilities, OpenAI-compatible requests
can also use mistral.rs extensions such as `session_id`, `web_search_options`,
`enable_code_execution`, `agent_permission`, `files`, and `max_tool_rounds`.

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

For app-facing tool timelines, generated files, search, code execution, and
session state, see [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).

## Examples

Server examples live in `examples/server/`:

| File | What it shows |
|---|---|
| `chat.py` | Basic Chat Completions request. |
| `streaming.py` | Chat Completions streaming. |
| `tool_calling.py` | OpenAI-compatible function tools. |
| `responses.py` | Responses API request. |
| `responses_vision.py` | Responses API with image input. |
| `web_search.py` | Search through OpenAI-compatible request fields. |
| `codex_config.toml` | Codex provider config for `/v1/responses`. |

For Codex setup, see [Use Codex and Claude Code](/mistral.rs/guides/serve/coding-agents/).
