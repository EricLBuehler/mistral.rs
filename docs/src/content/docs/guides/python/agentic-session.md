---
title: Agentic sessions from Python
description: Multi-turn agent state with the Python SDK.
sidebar:
  order: 3
---

Sessions on the HTTP server are keyed by session id and persist message history, tool-call records, images, and (when applicable) the Python code-execution subprocess. See the [persist-sessions guide](/mistral.rs/guides/agents/persist-sessions/) for the underlying behavior.

## In-process with Runner

`Runner` exposes the same session operations as the HTTP endpoints:

```python
from mistralrs import Runner, Which

runner = Runner(which=Which.Plain(model_id="Qwen/Qwen3-4B"))

ids = runner.list_session_ids()

exported = runner.export_session("user-42-chat-abc")  # JSON string or None
if exported is not None:
    runner.import_session("user-42-chat-new-id", exported)

runner.delete_session("user-42-chat-abc")
```

Each method takes an optional `model_id` keyword argument for multi-model setups.

`ChatCompletionRequest` can carry a `session_id`, so in-process Python requests can reuse agentic state:

```python
from mistralrs import ChatCompletionRequest

response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="default",
        messages=[{"role": "user", "content": "Continue the analysis."}],
        session_id="user-42-chat-abc",
    )
)
```

Use HTTP when a Python application also needs the live `agentic_tool_call_progress` timeline.

## Example: HTTP from Python

```python
import requests

# Create a session implicitly
r = requests.post("http://localhost:1234/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Research recent Rust releases."}],
    "session_id": "user-42-chat-abc",
})
print(r.json()["choices"][0]["message"]["content"])

# Continue the same session
r = requests.post("http://localhost:1234/v1/chat/completions", json={
    "model": "default",
    "messages": [{"role": "user", "content": "Summarize what you found."}],
    "session_id": "user-42-chat-abc",
})
```

## Export, import, delete

```python
# Export
exported = requests.get(
    "http://localhost:1234/v1/sessions/user-42-chat-abc"
).json()

# Import elsewhere
requests.put(
    "http://localhost:1234/v1/sessions/user-42-chat-abc",
    json=exported,
)

# Delete
requests.delete("http://localhost:1234/v1/sessions/user-42-chat-abc")
```

## Lifetime

Sessions are in-memory with a 30-minute idle TTL and 128-entry capacity (LRU). They do not survive a server restart unless exported and re-imported.

## Code execution subprocess

If the session has an active Python subprocess (code execution), the subprocess is not part of the exportable state.
