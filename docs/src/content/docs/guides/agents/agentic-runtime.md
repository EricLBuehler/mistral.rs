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

## Agent permissions

`agent_permission` controls whether mistral.rs may run an agent action after the model asks for one. It applies to all server-executed actions, not just Python: code execution, web search, file tools, registered callbacks, and external tool dispatch.

| Mode | Behavior |
|---|---|
| `auto` | Run the action as soon as the tool call is valid. |
| `ask` | Pause before the action and ask the app, callback, or CLI user to approve it. |
| `deny` | Keep the tool visible to the model, but return a denied tool result instead of running it. |

The server or runner policy is a floor. A request can tighten it, for example from `auto` to `ask` or `deny`, but cannot loosen a server started with `--agent-permission ask` or `--agent-permission deny`. `code_execution_permission` and `--code-exec-permission` are compatibility aliases for code-execution-focused apps; prefer `agent_permission` for new code.

Permissioning is separate from sandboxing. Permission mode decides whether an action may start. The [sandbox](/mistral.rs/reference/sandbox/) controls what generated Python can access after it starts.

CLI, the built-in UI, HTTP, Rust, and Python expose the same approval semantics:

| Concept | Meaning |
|---|---|
| Approve or deny | Allow the action, or return a denied tool result to the model. |
| `message` | Optional deny message returned to the model as the tool result. |
| `remember_for_session` | On approve, skip later approval prompts for the same `session_id`. |

### CLI

In interactive mode, `ask` prompts inline before each agent action. Choosing `always` approves later actions in the same CLI session.

```bash
mistralrs run --agent -m google/gemma-4-E4B-it --agent-permission ask
```

`deny` is useful when you want to inspect proposed actions without letting them run:

```bash
mistralrs run --agent -m google/gemma-4-E4B-it --agent-permission deny
```

### Built-in UI

The built-in UI has a **Tool approval** control in the settings drawer. Set it to `ask` to show approval cards inline before agent actions run, or to `deny` to keep tool calls visible while denying execution.

Approval cards show the tool metadata and decision controls, with collapsible arguments when useful. Choose **Approve**, **Always**, or **Deny**; **Always** sets `remember_for_session` for the current chat session.

### HTTP

For HTTP, `ask` is only supported with `stream: true`. The stream emits `agentic_tool_approval_required`, then waits while the app resolves that approval.

```json
{
  "model": "default",
  "stream": true,
  "messages": [
    {"role": "user", "content": "Use Python to inspect data.csv."}
  ],
  "enable_code_execution": true,
  "agent_permission": "ask",
  "session_id": "analysis-demo"
}
```

The approval event contains stable public metadata for app display and routing:

```text
event: agentic_tool_approval_required
data: {"approval_id":"appr_abc123","session_id":"analysis-demo","round":1,"tool":{"source":"built_in","kind":"code_execution","label":"Python code"},"arguments":{"code":"...","outputs":[]}}
```

Resolve it with `POST /v1/agent/approvals/{approval_id}`:

```json
{"decision":"approve","remember_for_session":true}
```

`decision` is `approve` or `deny`. A deny response may include `message`, which is returned to the model as the tool result. `remember_for_session: true` on an approve response is the HTTP version of "always for this chat": later actions in the same `session_id` do not ask again.

If an approval is not resolved, the action is denied after five minutes.

The approval endpoint returns `{"status":"resolved"}`, `{"status":"queued"}`, or `{"status":"not_found"}`. See the [HTTP API reference](/mistral.rs/reference/http-api/) for the exact wire schema and [the HTTP approval example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/server/code_execution_approval.py) for a complete client.

### Python SDK

For Python, set `agent_permission` and pass an `agent_approval_callback` on the request. The callback receives an `AgentToolApproval` with `approval_id`, `session_id`, `round`, stable `tool` metadata, `arguments_json`, and a convenience `code` field when the action is Python code. Return `True` or `False` for simple callbacks, or return `AgentToolApprovalDecision` for deny messages and `remember_for_session`.

```python
from mistralrs import (
    AgentPermission,
    AgentToolApprovalDecision,
    AgentToolKind,
    ChatCompletionRequest,
)

def approve(call):
    print(call.tool.label)
    if call.tool.kind == AgentToolKind.CodeExecution:
        print(call.code or "")
    else:
        print(call.arguments_json)
    answer = input("Approve? [y/N/a] ").strip().lower()
    if answer == "a":
        return AgentToolApprovalDecision.approve(remember_for_session=True)
    if answer in {"y", "yes"}:
        return AgentToolApprovalDecision.approve()
    return AgentToolApprovalDecision.deny("The user denied this action.")

request = ChatCompletionRequest(
    model="default",
    messages=[{"role": "user", "content": "Plot sin(x)."}],
    enable_code_execution=True,
    agent_permission=AgentPermission.Ask,
    agent_approval_callback=approve,
)
```

See the [Python approval example](https://github.com/EricLBuehler/mistral.rs/blob/master/examples/python/code_execution_approval.py) and the [Python agent approval reference](/mistral.rs/reference/python/agent-approvals/).

### Rust SDK

For Rust, set `AgentPermission::Ask` and pass an `AgentToolApprovalCallback`. The callback receives `approval_id`, `session_id`, `round`, stable `tool` metadata, and JSON `arguments`. Return `AgentToolApprovalDecision::approve()`, `approve_for_session()`, `deny(None)`, or `deny_with_message(...)`.

```rust
use std::sync::Arc;

use mistralrs::{
    AgentPermission, AgentToolApprovalCallback, AgentToolApprovalDecision, RequestBuilder,
};

let approval: AgentToolApprovalCallback = Arc::new(|approval| {
    println!("{}", approval.tool.label);
    println!("{}", approval.arguments);
    AgentToolApprovalDecision::approve_for_session()
});

let request = RequestBuilder::from(messages)
    .with_code_execution()
    .with_agent_permission(AgentPermission::Ask)
    .with_agent_approval_callback(approval);
```

Rust also supports `with_agent_approval_async_callback` for approval flows backed by async state, such as a UI event, database row, or message queue.

Both Rust callback forms use the same approval semantics. If an approval handler fails, the action is denied.

```rust
let request = RequestBuilder::from(messages)
    .with_code_execution()
    .with_agent_permission(AgentPermission::Ask)
    .with_agent_approval_async_callback(|approval| async move {
        println!("{}", approval.tool.label);
        AgentToolApprovalDecision::approve()
    });
```

See the [Rust approval example](https://github.com/EricLBuehler/mistral.rs/blob/master/mistralrs/examples/advanced/code_execution_approval/main.rs).

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
  "agent_permission": "auto",
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
| HTTP | Best surface for live model chunks, tool-progress timelines, files, and agent approval events. |
| Rust SDK | `Model::stream_chat_request` yields raw `Response::AgenticToolCallProgress` events. |
| Python SDK | Supports agentic requests, callbacks, code execution, and sessions. The streaming iterator currently yields model chunks; use HTTP SSE for the full timeline. |
| Web UI | Renders code execution, search, reasoning blocks, generated media, and approval cards inline. |

## Security

Code execution runs with the permissions of the configured Python interpreter. Use `agent_permission: "ask"` or `"deny"` per request when an app needs tighter control over any server-executed agent action; a server-wide `--agent-permission ask` or `deny` cannot be loosened by the request. HTTP `"ask"` approval is app-driven over SSE, not a server terminal prompt. For untrusted users, run mistral.rs in a container or VM, use a low-privilege user, and constrain network access.
