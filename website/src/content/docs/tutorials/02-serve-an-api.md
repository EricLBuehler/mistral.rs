---
title: Serve a model as an API
description: Take a local model and put it behind an OpenAI-compatible HTTP endpoint. Hit it with curl and with the OpenAI Python client. About ten minutes, including an optional Hugging Face login step.
sidebar:
  order: 2
---

In [Tutorial 1](/mistral.rs/tutorials/01-install-and-run/) you ran a model directly in your terminal. That is useful for experimentation, but most real applications want to talk to a model over HTTP. This tutorial turns a locally loaded model into an OpenAI-compatible API that anything speaking that protocol can use unchanged, including the official OpenAI Python client.

We will use Google's [Gemma 4](https://huggingface.co/google/gemma-4-E4B-it) for this one. It is in the same size class as the Qwen3 model from Tutorial 1, but it comes from a different model family, which is useful because it means any quirks specific to Qwen3 are not going to show up here.

## Accepting the Gemma license

Google releases Gemma under a license that requires you to click an "acknowledge" button on the Hugging Face model page before you can download the weights. This is a one-time step per Hugging Face account.

1. Visit [huggingface.co/google/gemma-4-E4B-it](https://huggingface.co/google/gemma-4-E4B-it) and sign in.
2. At the top of the page there will be a prompt asking you to accept the license. Click through it.
3. Create an access token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens). A read-only token is enough.
4. Give mistral.rs your token:

```bash
mistralrs login
```

It will prompt for the token and save it to `~/.cache/huggingface/token`. Every subsequent download, including gated models like Gemma, picks up that token automatically.

If you already have `huggingface-cli` installed and have logged in through it, you can skip the `mistralrs login` step. The two tools look for the token in the same place.

## Starting the server

```bash
mistralrs serve -m google/gemma-4-E4B-it
```

The first run will download the weights, same as `run` did in the last tutorial. When loading finishes, you will see a line like this:

```
Serving on http://0.0.0.0:1234
```

The server binds on `0.0.0.0` by default, which means anything on your network can reach it. That is fine for local development but worth knowing about before you leave it running on a shared machine. You can override the bind address with `--host 127.0.0.1` if you want to keep it local-only, and the port with `--port`.

Leave the server running and open a second terminal for the next step.

## Sending a request with curl

The server speaks the OpenAI Chat Completions protocol. That means the same request body you would send to OpenAI works here, with one small change: the model name is `default` rather than something like `gpt-4`.

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

You should get back a JSON blob with a `choices` array, a `usage` block showing token counts, and the response text inside `choices[0].message.content`.

Why `"default"` instead of a real model name? When you start the server with a single `-m` flag, mistral.rs registers that model under the reserved name `default`, so clients can point at it without having to know the Hugging Face repo id. If you serve multiple models, each one gets its configured name instead, and `default` is no longer special. That is covered in the multi-model guide, which you do not need to read yet.

## Calling it from Python

Because the API shape matches OpenAI's, the official `openai` package works unchanged:

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

The `api_key` field is required by the OpenAI client but mistral.rs does not check it. Put anything you like there.

Streaming works the same way it does against OpenAI. Pass `stream=True` and iterate over the returned object:

```python
stream = client.chat.completions.create(
    model="default",
    messages=[{"role": "user", "content": "Write me a haiku about Rust."}],
    stream=True,
)

for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
```

## Turning on the web UI

If you add `--ui` to the serve command, the server also exposes a browser chat interface at `/ui`:

```bash
mistralrs serve --ui -m google/gemma-4-E4B-it
```

Open `http://localhost:1234/ui` in a browser and you get a chat window with markdown rendering and a settings panel for things like temperature and system prompt. The UI is the quickest way to sanity-check a running server without writing any code, and later tutorials will use it to show off tool calling and code execution when we get there.

## Before you leave

A few things to know before you move on.

The server speaks the Chat Completions shape, the older Completions shape, and the Responses shape. Whichever one your client library prefers should work. If you are using an OpenAI-compatible wrapper that does not seem to talk to the server, the [OpenAI compatibility reference](/mistral.rs/reference/openai-compatibility/) lists exactly which fields we honor and which we ignore.

You can keep a model loaded across restarts of your client code. The server process holds the weights in GPU memory; stopping and restarting your Python script does not touch that. In other words, development iteration is cheap once the first load completes.

The `default` model name is specific to the single-model case. If you look at the full URL space with `curl http://localhost:1234/v1/models`, you will see `default` listed alongside the real model id. Either string works as the `model` field in a request.

## What to try next

- [Tutorial 3](/mistral.rs/tutorials/03-python-app/) skips the HTTP server entirely and loads a model directly inside a Python program.
- [Tutorial 5](/mistral.rs/tutorials/05-build-an-agent/) takes the server you just started and turns on tool calling, web search, and code execution.
- The [serving guides](/mistral.rs/guides/serve/http-server/) cover things that are out of scope for a tutorial, like CORS configuration, larger request bodies, and running multiple models at once.
