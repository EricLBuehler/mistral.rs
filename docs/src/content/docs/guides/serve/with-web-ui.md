---
title: Use the built-in web UI
description: What you get when you pass --ui to mistralrs serve, and how to customize it.
sidebar:
  order: 3
---

`--ui` mounts a browser chat interface at `/ui`. Use cases:

- Sanity-checking a newly loaded model.
- Demonstrating tool calling and code execution without explaining the response envelope.
- Giving non-technical teammates a way to try a model behind an HTTP endpoint.

The UI is a single-page app bundled into the binary. Nothing is fetched from the network at runtime.

## Basic usage

```bash
mistralrs serve --ui -m Qwen/Qwen3-4B
```

Open `http://localhost:1234/ui`. The UI provides:

- A chat panel with markdown rendering, code block syntax highlighting, and streamed output.
- A settings drawer with sampling controls (temperature, top-p, top-k, max tokens) and a system prompt field.
- Inline rendering of thinking tokens when the model emits them.
- Tool-call visualization when `--enable-search` or `--enable-code-execution` is also set.
- Multimodal attachments when the loaded model supports images, audio, or video.

## With agents enabled

```bash
mistralrs serve --ui --enable-search --enable-code-execution -m Qwen/Qwen3-4B
```

When the model calls a tool, the UI renders a collapsed block in the conversation. Expanding shows the tool arguments and result. Search blocks display queries and retrieved URLs; code-execution blocks display the executed Python and any output or images.

A toggle in the settings drawer disables either tool per conversation without restarting the server.

On Linux and macOS, code execution uses the default [OS-level sandbox](/mistral.rs/reference/sandbox/) unless the server is started with `--sandbox off`. For the server, HTTP, Python, Rust, and sandbox settings, see [enable code execution](/mistral.rs/guides/agents/enable-code-execution/).

## With multimodal models

When the loaded model accepts images, a paperclip icon appears in the input bar. Attaching an image adds a `{"type": "image"}` content part. Audio and video work the same on supporting models.

Modality support per model is in the [supported models reference](/mistral.rs/reference/supported-models/).

## System prompt and sampling

The settings drawer state is persistent per browser. System prompts and sampling parameters survive reloads.

Clearing browser local storage for the site resets all UI state.

## Disabling the UI

Omit `--ui`. The endpoint is mounted only when the flag is present.
