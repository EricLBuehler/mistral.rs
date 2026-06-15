---
title: Serve an OpenAI-compatible API
description: Run mistralrs serve and use OpenAI SDKs and compatible clients against the local server.
---

`mistralrs serve` puts a local model behind OpenAI-compatible endpoints under `/v1`. OpenAI SDKs and compatible clients work unchanged with `http://localhost:1234/v1` as the base URL.

```bash
mistralrs serve -m Qwen/Qwen3-4B
```

Then send a request:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "Write a haiku about local inference."}
    ],
    "max_tokens": 128
  }'
```

With a single `-m` model, the request `model` is `"default"` (or omitted). In [multi-model serving](/mistral.rs/guides/serve/multiple-models/), use a model id exactly as it appears in `GET /v1/models`.

First time serving a model? The [Quickstart](/mistral.rs/quickstart/) walks through installation, Hugging Face authentication for gated models, and the first run.

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

The `api_key` is required by the client but not validated by the server; see [authentication](/mistral.rs/reference/http-api/#authentication). Set `stream=True` for token-by-token output ([full example](/mistral.rs/examples/server/streaming/)).

## Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List loaded models. |
| `POST /v1/chat/completions` | Chat, streaming, tool calling, multimodal inputs, and mistral.rs agentic extensions. |
| `POST /v1/responses` | OpenAI Responses API: response objects, polling, background runs, cancellation. |
| `POST /v1/messages` | [Anthropic Messages API](/mistral.rs/guides/serve/anthropic-messages-api/) (base URL without `/v1`). |
| `POST /v1/completions` | Legacy text completions. |
| `POST /v1/embeddings` | Embedding generation. |
| `POST /v1/images/generations` | Image generation. |
| `POST /v1/audio/speech` | Text to speech. |
| `GET /v1/files` | Files produced by agentic runs. |

Every path with full request and response schemas is in the [generated HTTP API reference](/mistral.rs/reference/http-api-generated/). Streaming events, authentication, and protocol semantics are in the [HTTP API reference](/mistral.rs/reference/http-api/); field-level compatibility notes (including Responses API restrictions) are in [OpenAI compatibility](/mistral.rs/reference/openai-compatibility/).

:::caution[Compatibility gaps]
Most OpenAI-compatible fields work, but a few common ones have limitations:

- `seed`, `user`, `stream_options`, `metadata`, `parallel_tool_calls` - accepted but ignored.
- `code_interpreter` supports only `{"container":{"type":"auto"}}`; container ids and uploaded file ids are not supported.
- Responses `web_search` does not support image search or `external_web_access: false`.
- `dimensions` (embeddings) - errors rather than truncating.

Full list in [OpenAI compatibility](/mistral.rs/reference/openai-compatibility/).
:::

A live Swagger UI for the running server is at `http://localhost:1234/docs`.

## Tools, structured output, and agentic features

OpenAI-compatible function tools work on Chat Completions and Responses, including `strict: true` for JSON-Schema-constrained tool arguments. See [tool calling](/mistral.rs/guides/agents/tool-calling-basics/).

`response_format` with `json_schema` and the `grammar` extension constrain output server-side. See [structured output](/mistral.rs/guides/serve/structured-output/).

Start the server with agentic capabilities to use server-side tools and agentic fields. Chat Completions uses `web_search_options` for web search and `tools: [{"type":"code_interpreter","container":{"type":"auto"}}]` for code execution. Responses uses hosted tools in the `tools` array for both web search and code execution.

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

For tool timelines, generated files, search, code execution, and session state, see [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).

## Configuration

`-p/--port` (default 1234) and `--host` (default `0.0.0.0`) control the bind address. `--no-ui` disables the [web UI](/mistral.rs/guides/serve/with-web-ui/) at `/ui`. All flags are in the [CLI reference](/mistral.rs/reference/cli/serve/); the equivalent config file for multi-model, repeatable deployments is the [TOML config reference](/mistral.rs/reference/cli-toml-config/), which also covers CORS, body limits, authentication, and logging.

:::caution
The default `--host 0.0.0.0` accepts connections from any host on the network. Use `--host 127.0.0.1` to restrict to the local machine, and put authentication in a reverse proxy before exposing the server.
:::

## Examples

Runnable client scripts live in `examples/server/` and render under [server examples](/mistral.rs/examples/):

| Example | What it shows |
|---|---|
| [chat](/mistral.rs/examples/server/chat/) | Basic Chat Completions request. |
| [streaming](/mistral.rs/examples/server/streaming/) | Chat Completions streaming. |
| [tool_calling](/mistral.rs/examples/server/tool-calling/) | OpenAI-compatible function tools. |
| [openai_response_format](/mistral.rs/examples/server/openai-response-format/) | Structured output via `response_format`. |
| [responses](/mistral.rs/examples/server/responses/) | Responses API request. |
| [responses_tools](/mistral.rs/examples/server/responses-tools/) | Responses hosted tools: web search and code interpreter. |
| [responses_vision](/mistral.rs/examples/server/responses-vision/) | Responses API with image input. |
| [web_search](/mistral.rs/examples/server/web-search/) | Search through OpenAI-compatible request fields. |
| [anthropic_chat](/mistral.rs/examples/server/anthropic-chat/) | Anthropic Messages request. |
| [multi_model_chat](/mistral.rs/examples/server/multi-model-chat/) | Routing requests across loaded models. |

For Codex and Claude Code setup, see [coding agents](/mistral.rs/guides/serve/coding-agents/).
