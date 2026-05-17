---
title: Agentic runtime for apps
description: Build a local agent app around model output, tool execution, generated media, and sessions.
sidebar:
  order: 5
---

mistral.rs can act as a local-first runtime for agent applications. A runtime request can include model generation, server-side tool execution, Python code execution that is [sandboxed by default](/mistral.rs/reference/sandbox/) on Linux and macOS, web search, generated images or video frames, and persistent session state.

The most complete app-facing event stream today is `/v1/chat/completions` with `stream: true`. It emits normal OpenAI-compatible chunks plus mistral.rs `agentic_tool_call_progress` events.

Built-in runtime tools use [strict tool calling](/mistral.rs/guides/agents/strict-tool-calling/) by default. Web search, code execution, and file helper calls are constrained to their declared JSON Schemas before dispatch.

## What the runtime does

| Runtime part | What mistral.rs provides |
|---|---|
| Model output | Chat-completion responses and streaming chunks. |
| Tool execution | Built-in search, code execution, MCP tools, callbacks, or HTTP tool dispatch. |
| Generated media | Captured images and video frames from tools as base64 fields. |
| Session state | Reusable `session_id` values for multi-turn tool and code state. |

This is the lane for applications that want local inference and local action in one process, without building a separate tool loop around a raw model server.

## HTTP run stream

Start a server with the tools your app is allowed to use:

```bash
mistralrs serve --agent -m google/gemma-4-E4B-it
```

(`--agent` is a shorthand for `--enable-search --enable-code-execution`; the UI is on by default.)

Send a streaming chat-completions request:

```json
{
  "model": "default",
  "stream": true,
  "messages": [
    {
      "role": "user",
      "content": "Use Python to plot sin(x), then explain the chart."
    }
  ],
  "enable_code_execution": true,
  "web_search_options": {},
  "max_tool_rounds": 4,
  "session_id": "analysis-demo"
}
```

Model output arrives as standard chat-completion chunks. Tool progress arrives as named SSE events:

```text
event: agentic_tool_call_progress
data: {"type":"agentic_tool_call_progress","round":0,"tool_name":"mistralrs_execute_python","phase":"calling","data":{"tool_type":"code_execution","code":"print('hello')"}}
```

A complete code-execution event can include captured output and media:

```json
{
  "type": "agentic_tool_call_progress",
  "round": 0,
  "tool_name": "mistralrs_execute_python",
  "phase": "complete",
  "data": {
    "tool_type": "code_execution",
    "stdout": "saved plot\n",
    "images_base64": ["..."],
    "video_frames_base64": ["..."],
    "video_frame_count": 12,
    "working_directory": "/tmp/mistralrs-code-demo",
    "execution_time_ms": 118
  }
}
```

## Event fields

| Field | Meaning |
|---|---|
| `type` | `agentic_tool_call_progress`. |
| `round` | Agentic loop round. |
| `tool_name` | Tool that started or completed. |
| `phase` | `calling` or `complete`. |
| `data.tool_type` | `code_execution`, `web_search`, or `custom`. |

Tool-specific data:

| Tool type | Fields |
|---|---|
| `code_execution` | `code`, `stdout`, `stderr`, `exception`, `images_base64`, `video_frames_base64`, `video_frame_count`, `working_directory`, `execution_time_ms`. |
| `web_search` | `query`, `results_count`. |
| `custom` | `arguments`, `content`. |

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

The non-streaming response carries the produced files in a top-level `files` array:

```json
{
  "files": [
    {
      "id": "file_abc_r0_0",
      "name": "plot.png",
      "format": "png",
      "mime_type": "image/png",
      "bytes": 14823,
      "source": {"tool": "mistralrs_execute_python", "round": 0, "turn": 0},
      "data_base64": "iVBORw0KGgo..."
    },
    {
      "id": "file_abc_r0_1",
      "name": "samples.csv",
      "format": "csv",
      "mime_type": "text/csv",
      "bytes": 412,
      "source": {"tool": "mistralrs_execute_python", "round": 0, "turn": 0},
      "text": "x,y\n0,0\n..."
    }
  ]
}
```

When streaming, each file is emitted as soon as it is produced via a named SSE event:

```text
event: file_produced
data: {"id":"file_abc_r0_0","name":"plot.png","format":"png","mime_type":"image/png","bytes":14823,"source":{"tool":"mistralrs_execute_python","round":0,"turn":0},"data_base64":"iVBORw0KGgo..."}
```

Each `agentic_tool_calls[*]` record gains a `file_ids` field that lists the ids of files attributable to that round, so apps can correlate files with the tool that wrote them.

### Size policy

| Body | Inline | Above the cap |
|---|---|---|
| Up to 8 MB | Full body in `text` / `data_base64`. | The body field is omitted; fetch via `GET /v1/files/{id}/content`. |

Inside the model's context, text files only ever expose the first 1024 bytes as a preview; the model uses `read_file` to inspect more. `is_truncated()` on the SDK `File` returns true when the wire body was elided.

### Model tools

When the request declares files or `enable_code_execution` is on, the model gets two helper tools registered automatically:

- `read_file(file_id, start?, end?)` — fetch part or all of a file the model has emitted.
- `list_files()` — enumerate files produced so far in this turn.

The Python executor tool also accepts an `outputs: [string]` parameter the model can use to declare what it wrote. Files declared via `request.files` are surfaced regardless.

### Fetching by id

`GET /v1/files/{id}/content` returns the raw bytes with `Content-Type`, `Content-Length`, and `Content-Disposition` set. `GET /v1/files/{id}` returns the OpenAI-style metadata JSON. Status codes for the content endpoint:

| Code | Meaning |
|---|---|
| 200 | Body returned. |
| 404 | Unknown or expired file id. |
| 410 | File body was elided (request via the producing run instead). |
| 422 | The file is an error placeholder. |

## Sessions

Use `session_id` when your app needs continuity across requests. Sessions can preserve tool history, generated media references, and code-execution state.

| Endpoint | Purpose |
|---|---|
| `GET /v1/sessions/{session_id}` | Export a serialized session. |
| `PUT /v1/sessions/{session_id}` | Import or replace a session. |
| `DELETE /v1/sessions/{session_id}` | Delete a session. |

## SDK boundaries

| Surface | Current behavior |
|---|---|
| HTTP | Best surface for live model chunks plus tool-progress timeline. |
| Rust SDK | `Model::stream_chat_request` yields raw `Response::AgenticToolCallProgress` events. |
| Python SDK | Supports agentic requests, callbacks, code execution, and sessions. The streaming iterator currently yields model chunks; use HTTP SSE for the full timeline. |
| Web UI | Renders code execution, search, reasoning blocks, and generated media inline. |

## Security

Code execution runs with the permissions of the configured Python interpreter. For untrusted users, run mistral.rs in a container or VM, use a low-privilege user, and constrain network access.
