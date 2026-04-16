---
title: Use the built-in web UI
description: What you get when you pass --ui to mistralrs serve, and how to customize it.
sidebar:
  order: 3
---

Passing `--ui` to `mistralrs serve` attaches a browser chat interface to the server at `/ui`. It is the fastest way to interact with a running mistral.rs instance, and it is useful in a few more situations than that:

- Sanity-checking a newly loaded model before you start writing code against it.
- Demoing tool calling and code execution without having to explain the response envelope.
- Giving non-technical teammates a quick way to try a model without exposing anything beyond an HTTP endpoint.

The UI is a single-page app bundled into the binary. Nothing is served from the network; once the server is running, everything works offline.

## Basic usage

```bash
mistralrs serve --ui -m Qwen/Qwen3-4B
```

Open `http://localhost:1234/ui` in a browser. You get:

- A chat panel with markdown rendering, code block syntax highlighting, and streamed output.
- A settings drawer with sampling controls (temperature, top-p, top-k, max tokens) and a system prompt field.
- Inline rendering for thinking tokens, when the model emits them.
- Tool-call visualization when you also pass `--enable-search` or `--enable-code-execution`.
- Multimodal attachments when the loaded model supports images, audio, or video.

## With agents enabled

```bash
mistralrs serve --ui --enable-search --enable-code-execution -m Qwen/Qwen3-4B
```

When the model decides to call a tool, the UI draws a collapsed block in the conversation showing which tool was invoked. Clicking the block expands it to reveal the tool arguments and the result. Search blocks show the queries and the retrieved URLs; code-execution blocks show the exact Python the model ran and any output or images it produced.

A toggle in the settings drawer lets you disable either tool for a given conversation without restarting the server.

## With multimodal models

If the loaded model accepts images, a paperclip icon appears in the input bar. Attaching an image adds it as a `{"type": "image"}` content part in the request. The same works for audio and video on models that support those modalities.

The full list of which modalities which model supports is in the [supported models reference](/mistral.rs/reference/supported-models/).

## System prompt and sampling

The settings drawer is persistent per browser. A system prompt you set in one session sticks around through reloads, so if you are iterating on a prompt you are not re-entering it every time. Sampling parameters persist the same way.

If you want to start fresh, clearing browser local storage for the site resets everything.

## In production

The UI is meant for development and for small internal deployments. For anything bigger you probably want either a dedicated chat frontend (Open WebUI, LibreChat, and similar projects all work against an OpenAI-compatible endpoint like ours) or your own custom client.

Disabling the UI is a matter of not passing `--ui`. The endpoint is only mounted when the flag is present.
