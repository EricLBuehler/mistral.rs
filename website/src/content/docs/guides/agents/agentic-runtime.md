---
title: Agentic runtime for apps
description: Build a local agent app around model output, tool execution, generated media, and sessions.
sidebar:
  order: 1
---

mistral.rs can act as a local-first runtime for agent applications. A runtime request can include model generation, server-side tool execution, Python code execution, web search, generated images or video frames, and persistent session state.

The most complete app-facing event stream today is `/v1/chat/completions` with `stream: true`. It emits normal OpenAI-compatible chunks plus mistral.rs `agentic_tool_call_progress` events.

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
mistralrs serve \
  --enable-code-execution \
  --enable-search \
  --ui \
  -m google/gemma-4-E4B-it
```

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
