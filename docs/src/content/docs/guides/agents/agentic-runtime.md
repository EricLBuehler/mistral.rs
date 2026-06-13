---
title: Agentic runtime for apps
description: Build a local agent app around model output, tool execution, generated media, and sessions.
---

mistral.rs can act as a local-first runtime for agent applications. A single runtime request can include:

- Model generation (chat-completion responses and chunks).
- Server-side tool execution.
- Python code execution, [sandboxed by default](/mistral.rs/reference/sandbox/) on Linux and macOS.
- Web search.
- Generated images or video frames from tools.
- Persistent session state.

The most complete app-facing event stream today is `/v1/chat/completions` with `stream: true`. It emits normal OpenAI-compatible chunks plus mistral.rs `agentic_tool_call_progress` Server-Sent Events (SSE).

| Runtime part | What mistral.rs provides |
|---|---|
| Model output | Chat-completion responses and streaming chunks. |
| Tool execution | Built-in search, code execution, [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) tools, callbacks, or HTTP tool dispatch. |
| Generated media | Captured images and video frames from tools as base64 fields. |
| Session state | Reusable `session_id` values for multi-turn tool and code state. |

Use this when an app wants inference and tool execution in one process rather than running its own tool loop around a model server. Built-in runtime tools are [strict by default](/mistral.rs/guides/agents/tool-calling-basics/#strict-tool-calling); whether an action may run at all is governed by [permissions and approvals](/mistral.rs/guides/agents/permissions-and-approvals/).

## How the loop runs

The server-side loop engages for a chat request when any of these hold:

- The request sets `web_search_options` (advertises the web search tools).
- The request sets `enable_code_execution: true` on a server or runner with code execution enabled.
- The request carries `tools` and server-side executors exist for them (SDK `tool_callbacks` or connected MCP tools).
- The request sets `max_tool_rounds`, or the server has a `--tool-dispatch-url`.

Otherwise the request is dispatched normally: the model's `tool_calls` field is returned to the client and the client runs the next round (the standard OpenAI-compatible flow).

Each round:

1. The engine runs inference. The result either contains tool calls or does not.
2. No tool calls: the loop exits and the response is forwarded to the client.
3. The loop emits a progress event with phase `calling` and the tool arguments.
4. The tool is executed through one of the paths above (built-in search, code execution, file helpers, a registered callback, or a POST to the dispatch URL). If the model returns more than one tool call, only the first is executed and a warning is logged.
5. The loop emits a progress event with phase `complete` and the structured result.
6. The message history is extended with the assistant's tool-call message and a `tool`-role response, so the next inference pass sees the outcome.
7. If the round counter reaches the cap, the loop exits without another tool opportunity.

The cap and dispatch URL are configured on the [tool calling page](/mistral.rs/guides/agents/tool-calling-basics/#configuring-the-tool-loop). At termination, the expanded message list is written back to the [session](/mistral.rs/guides/agents/persist-sessions/), so the next request with the same session id sees the synthesized tool messages as history.

## HTTP run stream

Start a server with the tools your app is allowed to use:

```bash
mistralrs serve --agent -m google/gemma-4-E4B-it
```

(`--agent` enables both built-in tools; see [build an agent](/mistral.rs/guides/agents/build-an-agent/).)

Send a streaming chat-completions request:

```json
{
  "model": "default",
  "stream": true,
  "messages": [
    {"role": "user", "content": "Use Python to plot sin(x), then explain the chart."}
  ],
  "enable_code_execution": true,
  "web_search_options": {},
  "max_tool_rounds": 4,
  "session_id": "analysis-demo"
}
```

Model output arrives as standard chat-completion chunks. Tool progress arrives as named SSE events with `round`, `tool_name`, `phase` (`calling` or `complete`), and tool-type-specific `data`:

```text
event: agentic_tool_call_progress
data: {"type":"agentic_tool_call_progress","round":0,"tool_name":"mistralrs_execute_python","phase":"calling","data":{"tool_type":"code_execution","code":"print('hello')"}}
```

Complete events carry tool-type-specific payloads:

- Code execution: `stdout`, `stderr`, `images_base64`, `video_frames_base64`, `working_directory`, `execution_time_ms`.
- Web search: `query`, `results_count`.
- Custom tools: `arguments`, `content`.

The full event tables are in the [HTTP API reference](/mistral.rs/reference/http-api/). Non-streaming responses include the same information as an `agentic_tool_calls` array.

## Files

A `File` is a typed output produced by a tool (typically code execution). Each file has a stable id, a name, a format, a mime type, a size in bytes, and either an inline body or a reference for fetching it. Files are first-class on the wire: they ride alongside the model transcript, not buried inside tool output strings.

Declare required outputs on the request to give the model a contract:

```json
{
  "model": "default",
  "messages": [
    {"role": "user", "content": "Generate a sin(x) plot and a CSV of the samples."}
  ],
  "enable_code_execution": true,
  "files": [
    {"name": "plot.png", "format": "png"},
    {"name": "samples.csv", "format": "csv", "description": "x, sin(x) columns"}
  ]
}
```

The non-streaming response carries produced files in a top-level `files` array; when streaming, each file is emitted as soon as it is produced via a `file_produced` SSE event. Each `agentic_tool_calls[*]` record gains a `file_ids` field listing the files attributable to that round, so apps can correlate files with the tool that wrote them.

Behavior worth designing around:

- Inline vs fetched: bodies up to **8 MB** are inlined (`text` or `data_base64`); larger bodies are elided from the wire and fetched via `GET /v1/files/{id}/content`. `is_truncated()` on the SDK `File` reports an elided body.
- Context preview: inside the model's context, text files expose only the first **1024 bytes**; the model uses the auto-registered `read_file(file_id, start?, end?)` and `list_files()` helper tools to inspect more.
- Undeclared outputs: the Python executor tool accepts an `outputs: [string]` parameter for files the model wrote but the request did not declare. Files declared via `request.files` are surfaced regardless; missing declared files come back as error placeholders.

The exact file schema, metadata endpoint, and content-endpoint status codes are in the [HTTP API reference](/mistral.rs/reference/http-api/).

## Sessions

Use `session_id` when your app needs continuity across requests: message history, tool records, media, and code-execution state. Session behavior, the export/import/delete endpoints, and lifetime rules live in [persist sessions](/mistral.rs/guides/agents/persist-sessions/).

## SDK boundaries

| Surface | Current behavior |
|---|---|
| HTTP | Best surface for live model chunks, tool-progress timelines, files, and agent approval events. |
| Rust SDK | `Model::stream_chat_request` yields raw `Response::AgenticToolCallProgress` events. |
| Python SDK | Supports agentic requests, callbacks, code execution, and sessions. The streaming iterator currently yields model chunks; use HTTP SSE for the full timeline. |
| Web UI | Renders code execution, search, reasoning blocks, generated media, and approval cards inline. |

Full examples: [Rust agent](/mistral.rs/examples/rust/advanced/agent/), [Rust agent streaming](/mistral.rs/examples/rust/advanced/agent-streaming/), [Python agentic tools](/mistral.rs/examples/python/agentic-tools/), [HTTP tool rounds](/mistral.rs/examples/server/agentic-tool-rounds/).

## Security

Code execution runs with the permissions of the configured Python interpreter, inside the [sandbox](/mistral.rs/reference/sandbox/) where enabled. Use `agent_permission: "ask"` or `"deny"` when an app needs tighter control over server-executed actions; a server-wide `ask` or `deny` cannot be loosened by the request (see [permissions and approvals](/mistral.rs/guides/agents/permissions-and-approvals/)). For untrusted users, run mistral.rs in a container or VM, use a low-privilege user, and constrain network access.
