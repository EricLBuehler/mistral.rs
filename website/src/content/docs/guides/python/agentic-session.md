---
title: Agentic sessions from Python
description: Multi-turn agent state with the Python SDK.
sidebar:
  order: 3
---

Sessions on the HTTP server are keyed by session id and persist message history, tool-call records, images, and (when applicable) the Python code-execution subprocess. See the [persist-sessions guide](/mistral.rs/guides/agents/persist-sessions/) for the underlying behavior.

## Availability

The Python SDK does not currently expose `export_session`, `import_session`, `delete_session`, or `list_session_ids` methods on `Runner`. The `ChatCompletionRequest` dataclass in the .pyi stub does not include a `session_id` field.

For multi-turn agentic state today:

- Run `mistralrs serve` and call the HTTP API from Python (e.g., with the OpenAI client or `requests`).
- Use the `/v1/chat/completions` endpoint with a `session_id` field on the request body.
- Use `GET / PUT / DELETE /v1/sessions/{id}` to export, import, and delete sessions.

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
