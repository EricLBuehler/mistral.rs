---
title: Serve an OpenAI-compatible API
description: Run mistralrs serve and use OpenAI SDKs and compatible clients against the local server.
---

`mistralrs serve` puts a local model behind OpenAI-compatible endpoints under `/v1`. Point any OpenAI SDK or compatible client at `http://localhost:1234/v1` as its base URL.

```bash
mistralrs serve -m Qwen/Qwen3-4B --quant 4
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

With a single `-m` model, set `model` to `"default"` or leave it out. When [serving several models](/guides/serve/multiple-models/), use a model ID exactly as `GET /v1/models` lists it.

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

If you've loaded [LoRA adapters](/guides/customize/lora-adapters/), pick one by setting `model` to its alias, such as `model="code"`, or keep the base model and pass `extra_body={"adapter": "code"}`.

The client requires an `api_key`, but the server ignores it, so any non-empty string works; see [authentication](/reference/http-api/#authentication). Set `stream=True` to get tokens as they're generated ([full example](/examples/server/streaming/)).

## Endpoints

| Task | Endpoint | Guide |
| --- | --- | --- |
| Chat, tools, multimodal input | `/v1/chat/completions` | [Tool calling](/guides/agents/tool-calling-basics/) |
| Responses, background runs, cancellation | `/v1/responses` | [Compatibility](/reference/openai-compatibility/#responses-api) |
| Anthropic clients | `/v1/messages` | [Messages API](/guides/serve/anthropic-messages-api/) |
| Embeddings | `/v1/embeddings` | [Embeddings](/guides/models/use-embeddings/) |
| Image generation | `/v1/images/generations` | [Diffusion](/guides/models/use-image-generation/) |
| Speech and recognition | `/v1/audio/speech`, `/v1/audio/transcriptions`, `/v1/audio/translations` | [Speech models](/guides/models/use-speech-models/) |
| Typed decisions | `/v1/systemone` | [Decision requests](/reference/decisions/) |

Each endpoint needs a model of the matching kind loaded, such as an embedding model for `/v1/embeddings`. If you're pointing an existing OpenAI client at mistral.rs, check [OpenAI compatibility](/reference/openai-compatibility/) for fields that behave differently. For example, embeddings reject `dimensions` instead of truncating.

A running server shows an interactive API explorer (Swagger UI) at `http://localhost:1234/docs`. The [generated HTTP reference](/reference/http-api-generated/) lists every route, including model management, legacy Completions, files, LoRA adapters, and Skills. [HTTP semantics](/reference/http-api/) covers routing, streaming, and history.

## Tools, structured output, and agentic features

OpenAI-compatible function tools work on Chat Completions and Responses, including `strict: true` for JSON-Schema-constrained tool arguments. See [tool calling](/guides/agents/tool-calling-basics/).

`response_format` with `json_schema` and the `grammar` extension constrain output server-side. See [structured output](/guides/serve/structured-output/).

To have the server run tools itself, such as web search, code execution, and shell, start it with `--agent`:

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

See [agentic runtime for apps](/guides/agents/agentic-runtime/) for tool progress events, generated files, Skills, and session state.

## Configuration

| Flag | Default | What it does |
|---|---|---|
| `-p`, `--port` | `1234` | Port to listen on. |
| `--host` | `0.0.0.0` | Address to bind. `127.0.0.1` accepts local connections only. |
| `--no-ui` | UI on | Turns off the [web UI](/guides/serve/with-web-ui/) at `/ui`. |

Every flag is in the [CLI reference](/reference/cli/serve/). To keep a deployment in a file, including several models, use a [TOML config](/reference/cli-toml-config/).

:::caution
The default `--host 0.0.0.0` accepts connections from any machine on your network, and the server has no authentication of its own. Use `--host 127.0.0.1` to keep it local, or put an authenticated reverse proxy in front before exposing it.
:::

## Examples

Runnable [server examples](/examples/) cover [streaming](/examples/server/streaming/), [Responses](/examples/server/responses/), [hosted tools](/examples/server/responses-tools/), [Skills](/examples/server/skills/), and [speech/transcription](/examples/server/lfm3-audio-endpoints/).

For Codex and Claude Code setup, see [coding agents](/guides/serve/coding-agents/).
