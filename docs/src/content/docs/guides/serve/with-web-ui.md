---
title: Use the built-in web UI
description: What you get from the built-in web UI mounted at /ui, and how to customize it.
---

```bash
mistralrs serve -m Qwen/Qwen3-4B --quant 4
```

When the server is ready, open [http://localhost:1234/ui](http://localhost:1234/ui). Keep the terminal open while chatting; press Ctrl+C to stop. For installation, follow the [quickstart](/quickstart/).

The UI provides:

- A chat panel with markdown rendering, code block syntax highlighting, and streamed output.
- A settings drawer with sampling controls (temperature, top-p, top-k, max tokens) and a system prompt field.
- Inline rendering of thinking tokens when the model emits them, with Default, Off, Low, Medium, High, and XHigh reasoning controls.
- Tool-call visualization when `--enable-search`, `--enable-code-execution`, or `--enable-shell` is also set.
- Tool approval cards when approval mode is `ask`.
- Multimodal attachments when the loaded model supports images, audio, or video.

## With agents enabled

```bash
mistralrs serve --agent -m Qwen/Qwen3-4B
```

See [tool calling](/guides/agents/tool-calling-basics/) for what `--agent` enables and the individual flags behind it.

When the model calls a tool, the UI renders a collapsed block in the conversation. Expanding shows the tool arguments and result. Search blocks display the query and a result count; code-execution blocks display the executed Python and any output or images; shell blocks display the commands, stdout/stderr, and exit status.

The settings drawer controls search, code execution, shell, and tool approval per conversation without restarting the server. Set **Tool approval** to `ask` to approve or deny each agent action inline.

Code and shell execution use the configured [sandbox](/reference/sandbox/). See [code execution](/guides/agents/enable-code-execution/) and [shell](/guides/agents/enable-shell/) for setup.

## With multimodal models

Use the paperclip to attach images, audio, or video supported by the loaded model.

Modality support per model is in the [supported models reference](/reference/supported-models/).

## System prompt and sampling

The settings drawer state is persistent per browser. System prompts and sampling parameters survive reloads.

Clearing browser local storage for the site resets all UI state.

## Disabling the UI

Pass `--no-ui` to `mistralrs serve` to skip mounting the UI router. The HTTP API continues to serve normally.
