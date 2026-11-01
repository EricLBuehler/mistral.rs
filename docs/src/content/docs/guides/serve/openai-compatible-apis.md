---
title: Serve an OpenAI-compatible API
description: Run mistralrs serve and use OpenAI SDKs and compatible clients against the local server.
---

`mistralrs serve` puts a local model behind OpenAI-compatible endpoints under `/v1`. Configure your OpenAI client with `http://localhost:1234/v1` as its base URL.

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

With a single `-m` model, the request `model` is `"default"` (or omitted). In [multi-model serving](/guides/serve/multiple-models/), use a model id exactly as it appears in `GET /v1/models`.

For dynamic LoRA aliases and exact generations, see [adapter routing](/reference/http-api/#lora-adapter-routing-and-management).

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

For a loaded LoRA alias, either set `model="code"` like vLLM or keep the base model explicit and pass `extra_body={"adapter": "code"}`. The latter also accepts an exact generation object.

The `api_key` is required by the client but not validated by the server; see [authentication](/reference/http-api/#authentication). Set `stream=True` for token-by-token output ([full example](/examples/server/streaming/)).

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

Each task needs a model that supports it. Field support also differs by endpoint; check [OpenAI compatibility](/reference/openai-compatibility/) when adapting an existing client.

The live Swagger UI is at `http://localhost:1234/docs`. The [generated HTTP reference](/reference/http-api-generated/) lists all routes, including model management, legacy Completions, files, and Skills. [HTTP semantics](/reference/http-api/) defines routing, streaming, and history.

## Tools, structured output, and agentic features

OpenAI-compatible function tools work on Chat Completions and Responses, including `strict: true` for JSON-Schema-constrained tool arguments. See [tool calling](/guides/agents/tool-calling-basics/).

`response_format` with `json_schema` and the `grammar` extension constrain output server-side. See [structured output](/guides/serve/structured-output/).

To execute tools on the server, enable the [agentic runtime](/guides/agents/agentic-runtime/):

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

For tool timelines, generated files, search, code execution, shell, Skills, and session state, see [agentic runtime for apps](/guides/agents/agentic-runtime/).

## Configuration

`-p/--port` (default 1234) and `--host` (default `0.0.0.0`) control the bind address. `--no-ui` disables the [web UI](/guides/serve/with-web-ui/) at `/ui`. All flags are in the [CLI reference](/reference/cli/serve/); the equivalent config file for multi-model, repeatable deployments is the [TOML config reference](/reference/cli-toml-config/), which also covers CORS, body limits, authentication, and logging.

:::caution
The default `--host 0.0.0.0` accepts connections from any host on the network. Use `--host 127.0.0.1` to restrict to the local machine, and put authentication in a reverse proxy before exposing the server.
:::

## Examples

Runnable [server examples](/examples/) cover [streaming](/examples/server/streaming/), [Responses](/examples/server/responses/), [hosted tools](/examples/server/responses-tools/), [Skills](/examples/server/skills/), and [speech/transcription](/examples/server/lfm3-audio-endpoints/).

For Codex and Claude Code setup, see [coding agents](/guides/serve/coding-agents/).
