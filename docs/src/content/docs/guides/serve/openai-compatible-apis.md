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

With a single `-m` model, the request `model` is `"default"` (or omitted). In [multi-model serving](/guides/serve/multiple-models/), use a model id exactly as it appears in `GET /v1/models`.

Loaded dynamic [LoRA adapters](/guides/customize/lora-adapters/) also appear in `GET /v1/models`, so vLLM-compatible clients can put the adapter alias in `model`. Chat Completions, Completions, and Responses additionally accept `adapter` as either an alias string or `{"generation":"<generation-id>"}`. Omit `adapter` and select the base model to run without LoRA. LoRA inference responses expose the exact resolved generation as `adapter_generation` for audit and retry routing.

First time serving a model? The [Quickstart](/quickstart/) walks through installation, Hugging Face authentication for gated models, and the first run.

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

| Endpoint | Purpose |
|---|---|
| `GET /v1/models` | List loaded base models and LoRA alias model cards. |
| `POST /v1/chat/completions` | Chat, streaming, tool calling, multimodal inputs, and mistral.rs agentic extensions. |
| `POST /v1/responses` | OpenAI Responses API: response objects, polling, background runs, cancellation. |
| `POST /v1/skills` | Upload Skills for OpenAI-compatible Responses or Anthropic-compatible Messages. |
| `GET /v1/skills` | List uploaded skills. Anthropic headers return Anthropic-shaped list objects. |
| `GET, POST /v1/skills/{skill_id}/versions` | List or upload versions of an existing skill. |
| `POST /v1/messages` | [Anthropic Messages API](/guides/serve/anthropic-messages-api/) (base URL without `/v1`). |
| `POST /v1/completions` | Legacy text completions. |
| `POST /v1/embeddings` | Embedding generation. |
| `POST /v1/images/generations` | Image generation. |
| `POST /v1/audio/speech` | Text to speech. |
| `POST /v1/files` | Upload OpenAI-compatible user files. |
| `GET /v1/files` | List uploaded and generated files. |
| `POST /v1/load_lora_adapter` | Load a local LoRA adapter, or atomically replace one with `load_inplace`. Available only when runtime updating is enabled. |
| `GET /v1/lora_adapters` | List loaded LoRA aliases, generations, and capacity. The route is always registered, but the target model must have dynamic LoRA enabled. Use `?model=<model-id>` in a multi-model server. |
| `POST /v1/unload_lora_adapter` | Unload a LoRA alias. Available only when runtime updating is enabled. |

Every path with full request and response schemas is in the [generated HTTP API reference](/reference/http-api-generated/). Streaming events, authentication, and protocol semantics are in the [HTTP API reference](/reference/http-api/); field-level compatibility notes (including Responses API restrictions) are in [OpenAI compatibility](/reference/openai-compatibility/).

:::caution[Compatibility gaps]
Most OpenAI-compatible fields work, but a few common ones have limitations:

- `user`, `stream_options`, `metadata`, `parallel_tool_calls` - accepted but ignored. `seed` controls deterministic request-scoped sampling.
- `code_interpreter` supports only `{"container":{"type":"auto"}}`; OpenAI code-interpreter container ids and `container.file_ids` are not supported.
- Responses `web_search` does not support image search or `external_web_access: false`.
- Responses `shell` supports `environment.type = "container_auto"` and uploaded `skill_reference` entries; local environments, container references, and inline container-created skills are not implemented. Anthropic Messages uses the same store through `container.skills` with `type = "custom"`.
- File inputs support uploaded ids, inline base64/Data URLs, and Responses `file_url`, but binary formats are not converted with OpenAI's private PDF/image/spreadsheet extraction pipeline.
- `dimensions` (embeddings) - errors rather than truncating.

Full list in [OpenAI compatibility](/reference/openai-compatibility/).
:::

A live Swagger UI for the running server is at `http://localhost:1234/docs`.

## Tools, structured output, and agentic features

OpenAI-compatible function tools work on Chat Completions and Responses, including `strict: true` for JSON-Schema-constrained tool arguments. See [tool calling](/guides/agents/tool-calling-basics/).

`response_format` with `json_schema` and the `grammar` extension constrain output server-side. See [structured output](/guides/serve/structured-output/).

Start the server with agentic capabilities to use server-side tools and agentic fields. Chat Completions uses `web_search_options` for web search and `tools: [{"type":"code_interpreter","container":{"type":"auto"}}]` for code execution. Responses uses hosted tools in the `tools` array for web search, code execution, shell, and [OpenAI-compatible Skills](/guides/agents/skills/).

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

Runnable client scripts live in `examples/server/` and render under [server examples](/examples/):

| Example | What it shows |
|---|---|
| [chat](/examples/server/chat/) | Basic Chat Completions request. |
| [streaming](/examples/server/streaming/) | Chat Completions streaming. |
| [tool_calling](/examples/server/tool-calling/) | OpenAI-compatible function tools. |
| [allowed_tools](/examples/server/allowed-tools/) | OpenAI-compatible `allowed_tools` function subset selection. |
| [openai_response_format](/examples/server/openai-response-format/) | Structured output via `response_format`. |
| [responses](/examples/server/responses/) | Responses API request. |
| [responses_tools](/examples/server/responses-tools/) | Responses hosted tools: web search and code interpreter. |
| [skills](/examples/server/skills/) | OpenAI-compatible Skills upload and execution. |
| [responses_vision](/examples/server/responses-vision/) | Responses API with image input. |
| [web_search](/examples/server/web-search/) | Search through OpenAI-compatible request fields. |
| [anthropic_chat](/examples/server/anthropic-chat/) | Anthropic Messages request. |
| [multi_model_chat](/examples/server/multi-model-chat/) | Routing requests across loaded models. |

For Codex and Claude Code setup, see [coding agents](/guides/serve/coding-agents/).
