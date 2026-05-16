---
title: Serve a model as an API
description: Take a local model and put it behind an OpenAI-compatible HTTP endpoint. Hit it with curl and with the OpenAI Python client.
sidebar:
  order: 2
---

`mistralrs serve` exposes a model over an OpenAI-compatible HTTP API. The model used here is Google's [Gemma 4](https://huggingface.co/google/gemma-4-E4B-it). If you prefer to stay on the Qwen model from [Tutorial 1](/mistral.rs/tutorials/01-install-and-run/), substitute `Qwen/Qwen3-4B` for `google/gemma-4-E4B-it` throughout and skip the license step below.

## Accepting the Gemma license

Gemma weights are gated on Hugging Face. One-time setup per account:

1. Open [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it), sign in, and accept the license at the top of the page.
2. Create a read-only access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
3. Pass the token to mistral.rs:

```bash
mistralrs login
```

The token is saved to `~/.cache/huggingface/token` and reused for subsequent downloads. If you have already logged in via `huggingface-cli`, skip this step -- both tools read the same token file.

## Starting the server

```bash
mistralrs serve -m google/gemma-4-E4B-it
```

The first run downloads the weights. When loading completes:

```
Server listening on http://0.0.0.0:1234
```

The server binds `0.0.0.0` by default, making it reachable from any host on the network. To restrict it, pass `--host 127.0.0.1`. The port is configurable with `--port`.

Leave the server running and open a second terminal.

## Sending a request with curl

The server implements the OpenAI Chat Completions protocol. Request bodies match OpenAI's, with one difference: the model name is `default`.

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [
      {"role": "user", "content": "In one sentence, what is Gemma?"}
    ]
  }'
```

The response is JSON with a `choices` array, a `usage` block, and the generated text in `choices[0].message.content`.

When the server is started with a single `-m` flag, the model is registered under the reserved name `default`. With multiple models, each is registered under its configured name and `default` is not used.

## Calling it from Python

The official `openai` package works without modification:

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:1234/v1", api_key="not-used")

response = client.chat.completions.create(
    model="default",
    messages=[
        {"role": "user", "content": "In one sentence, what is Gemma?"}
    ],
)

print(response.choices[0].message.content)
```

The `api_key` field is required by the client but not validated by the server.

Streaming uses the OpenAI streaming protocol:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write me a haiku about Rust."}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Enabling the web UI

The `--ui` flag exposes a browser chat interface at `/ui`:

```bash
mistralrs serve --ui -m google/gemma-4-E4B-it
```

Open `http://localhost:1234/ui`. The UI provides a chat window with markdown rendering and controls for sampling parameters and the system prompt.

## Notes

The server implements the Chat Completions, legacy Completions, and Responses APIs. The [OpenAI compatibility reference](/mistral.rs/reference/openai-compatibility/) lists the supported and ignored fields.

The `default` model name is special-cased server-side: when the request's `model` field is `"default"` or absent, the server uses the configured default model. `GET /v1/models` lists the real model id.

## Next steps

- [Tutorial 3](/mistral.rs/tutorials/03-python-sdk/): load a model directly inside a Python program, without an HTTP server.
- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/): enable tool calling, web search, and code execution on the running server.
- The [serving guides](/mistral.rs/guides/serve/http-server/) cover multi-model serving and configuration.
