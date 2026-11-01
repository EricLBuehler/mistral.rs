---
title: Agentic runtime for apps
description: Consume tool progress, files, and sessions in an agent application.
---

When you turn on tools, mistral.rs runs the whole agent loop inside one request: the model asks for a tool, mistral.rs runs it, adds the result to the conversation, and the model continues until it has an answer. Your app sends one request and gets back the model's text, a record of every tool call, any files the tools wrote, and a session ID for follow-up requests.

[Build an agent](/guides/agents/build-an-agent/) gets a server running. This page covers what your app does with the output.

## When mistral.rs runs tools for you

mistral.rs runs the tool loop for a request when any of these is true:

| Trigger | Example |
|---|---|
| The request turns on a built-in tool | `web_search_options` or `"enable_shell": true` (Chat Completions), or a `web_search`, `code_interpreter`, or `shell` entry in `tools` |
| The request sets a round cap | `"max_tool_rounds": 8` |
| The request attaches [input files](/guides/agents/file-inputs/) | a `file` or `input_file` content part |
| The server is set up for it | [MCP servers](/guides/agents/connect-mcp-server/), SDK tool callbacks, `--tool-dispatch-url`, or `--max-tool-rounds`. Every chat request then uses the loop. |

Each round, the model asks for one tool, mistral.rs runs it, and the model generates again with the result in view. The loop ends when the model replies without asking for a tool, or when it hits `max_tool_rounds` (256 if you don't set one). Set a cap per request or with `--max-tool-rounds`; see [tool calling](/guides/agents/tool-calling-basics/#configuring-the-tool-loop).

Two things to know:

- **One tool per round.** If the model asks for several tools in the same reply, only the first one runs. The model sees that result and can ask for the next tool in the following round.
- **Tools mistral.rs can't run go back to you.** If the model calls a function you defined in `tools` and nothing on the server handles it (no MCP server, callback, or dispatch URL), the loop stops and returns that call in `tool_calls` so your app can run it, as in the standard [client-side flow](/guides/agents/tool-calling-basics/). When streaming, the call arrives in the final chunk's `delta.tool_calls`, with `finish_reason` set to `"tool_calls"`. The same happens if the loop hits `max_tool_rounds` while a call is pending.

## Stream a run over HTTP

Start a server with the tools you want, then send a streaming request:

```bash
mistralrs serve --agent -m google/gemma-4-E4B-it
```

```json
{
  "model": "default",
  "stream": true,
  "messages": [{"role": "user", "content": "Plot sin(x) as plot.png and explain it."}],
  "tools": [{"type": "code_interpreter", "container": {"type": "auto"}}],
  "files": [{"name": "plot.png", "format": "png"}],
  "max_tool_rounds": 4,
  "session_id": "analysis-demo"
}
```

The model's text arrives as ordinary chat-completion chunks. Tool work and files arrive as named Server-Sent Events (SSE) in between:

| Event | When it arrives | What to show |
|---|---|---|
| `agentic_tool_call_progress` with `phase: "calling"` | A tool is about to run. `data` holds its input: the search query, the Python code, or the shell commands. | A running step with its input. |
| `agentic_tool_call_progress` with `phase: "complete"` | The tool finished. `data` holds its result. | The tool's output. |
| `file_produced` | A tool wrote a file. | An attachment with a download link. |
| `agentic_tool_approval_required` | The run is paused for approval. Only sent with `agent_permission: "ask"`. | An approve or deny prompt. See [permissions and approvals](/guides/agents/permissions-and-approvals/). |

```text
event: agentic_tool_call_progress
data: {"type":"agentic_tool_call_progress","round":0,"tool_name":"<tool identifier>","phase":"calling","data":{"tool_type":"code_execution","code":"import numpy as np ..."}}
```

`round` counts up from 0 and pairs each `calling` event with its `complete` event. Treat `tool_name` as an opaque ID and branch on `data.tool_type` instead:

| `tool_type` | Fields in a `complete` event |
|---|---|
| `code_execution` | `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `working_directory`, `execution_time_ms` |
| `shell` | `commands`, `stdout`, `stderr`, `exit_code`, `timed_out`, `status`, `working_directory` |
| `web_search` | `query`, `results_count`, `sources` |
| `custom` (MCP, callbacks, dispatch) | `arguments`, `content` |

A non-streaming request returns the same history in an `agentic_tool_calls` array, one entry per round with `round`, `name`, `arguments`, `result_content`, and `file_ids`. Payload schemas are in the [HTTP API reference](/reference/http-api/), and the [HTTP tool rounds example](/examples/server/agentic-tool-rounds/) has a working client.

## Files

Python code execution and shell return any file they create or change in the session's working directory. Hidden files, dependency and cache folders such as `node_modules` and `__pycache__`, and compiled artifacts are skipped. You don't have to ask for files, but listing them in the request's `files` array does two things: the model is told which files you expect, and any file it fails to write comes back as an error entry instead of silently missing. A request that lists `files` without turning on any tool is rejected. See declaring outputs for [code execution](/guides/agents/enable-code-execution/#declaring-output-files) and [shell](/guides/agents/enable-shell/#file-outputs).

Where files appear in the response depends on the API:

| API | Where files appear |
|---|---|
| Chat Completions | A top-level `files` array, plus a `file_produced` event for each file when streaming. Each `agentic_tool_calls` entry lists the `file_ids` that round wrote. |
| Anthropic Messages | A top-level `files` array, plus `file_produced` events when streaming. |
| Responses | `container_file_citation` annotations on the assistant's text, in the OpenAI style. |

Each file carries an ID, name, format, MIME type, and size. Files up to 8 MiB include their contents inline (`text` or `data_base64`). For bigger files, the body is left out and you download it with `GET /v1/files/{id}/content`; in the SDKs, `File.is_truncated()` tells you the body was left out. The server keeps files for 30 minutes after their session's last request.

## Sessions

Pass a `session_id` so the next request continues where this one stopped. The session keeps the tool calls, their results, and the live Python interpreter with its variables. Keep sending the conversation (user and assistant messages) as usual, and mistral.rs puts the tool steps back in where they happened. Sessions live in server memory, and you can export and re-import them. See [sessions](/guides/agents/persist-sessions/).

## Python and Rust SDKs

Both SDKs run the same loop in-process, with the same features: web search, code execution, shell, callbacks, local Skills, input files, and sessions. Non-streaming responses include `agentic_tool_calls` and `files` as over HTTP. Streaming differs:

| SDK | What streaming gives you |
|---|---|
| Rust | `Model::stream_chat_request` yields every event, including `Response::AgenticToolCallProgress` and files. See the [streaming example](/examples/rust/advanced/agent-streaming/). |
| Python | The stream iterator yields only the model's text chunks. If you need live tool progress, use the HTTP server. See the [agentic tools example](/examples/python/agentic-tools/). |

The built-in web UI renders all of it: tool steps, generated media, files, and approval prompts.

## Security

When you start code or shell execution from the CLI on Linux or macOS, Python and shell commands run in an OS-level [sandbox](/reference/sandbox/). The default `developer` profile limits writes to the session's working directory, lets common toolchains such as Conda, rustup, and Homebrew work, and leaves the network open. To tighten it for users you don't trust:

```bash
# Limit filesystem reads and allow only loopback network
mistralrs serve --agent --sandbox-profile restricted -m <model>

# Also block all network access
mistralrs serve --agent --sandbox-profile restricted --sandbox-network none -m <model>
```

The Python and Rust SDKs don't sandbox anything unless you attach a `SandboxPolicy`.

To decide whether an action may run at all, set `agent_permission` to `auto`, `ask`, or `deny`. A request can tighten the server's setting but never loosen it. See [permissions and approvals](/guides/agents/permissions-and-approvals/). For untrusted users, also run mistral.rs itself in a container or VM as a low-privilege user.
