---
title: Agentic sessions from Python
description: Keep multi-turn agent state alive across calls. Export, import, and delete sessions using the Python SDK.
sidebar:
  order: 3
---

Sessions keep agent state coherent across calls: message history, tool-call records, and the Python code-execution subprocess if enabled. The [persist-sessions guide](/mistral.rs/guides/agents/persist-sessions/) covers the concept; this page covers the Python API.

## Passing a session id

Add `session_id` to the request:

```python
from mistralrs import Runner, Which, ChatCompletionRequest

runner = Runner(
    Which.Plain(model_id="Qwen/Qwen3-4B"),
    in_situ_quant="4",
    enable_search=True,
)

# First call creates the session.
response = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Research recent Rust releases."}],
    )
)
session_id = response.session_id
print(f"Session created: {session_id}")

# Second call continues the same session. Tool history, message history,
# and any loaded search context carry over.
response2 = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Summarize what you found."}],
        session_id=session_id,
    )
)
```

Starting with a generated id is not required. Any string creates a new session under that name if it does not exist.

## Export and import

The Python Runner exposes the same export and import operations as the HTTP API:

```python
# Capture current state.
serialized = runner.export_session(session_id)

import json
with open("session.json", "w") as f:
    json.dump(serialized, f)

# Later, in a different process:
with open("session.json") as f:
    serialized = json.load(f)

runner.import_session("new-session-id", serialized)
# Requests against "new-session-id" continue from the original.
```

The serialized object is a plain dict, safe to pickle, store, or transfer. Images and videos are base64-encoded.

## Deleting a session

```python
runner.delete_session(session_id)
```

Frees session memory. Any associated Python code-execution subprocess is terminated. Subsequent requests against the deleted id create a new, empty session.

## Listing sessions

```python
ids = runner.list_session_ids()
for sid in ids:
    print(sid)
```

For admin UIs and bookkeeping in long-running applications.

## When to persist and when not to

Keep sessions in memory when:

- Conversations are ephemeral (chat UI with refresh-to-restart).
- Loss on server restart is acceptable.

Export to disk or database when:

- Conversations span a long time and users return to them.
- Multiple servers must continue the same conversation.
- Audit trails of agent behavior are required.

Common pattern for longer-lived applications: maintain a mapping from user-visible conversation ids to mistralrs session ids, export-and-store on a timer or after significant events, restore on demand.

## Interaction with code execution

When code execution is enabled and a session has an active Python subprocess, the subprocess lives as long as the session is in memory. Exporting a session does not export the subprocess state; the importing server starts a fresh subprocess on the next code-execution call in that session.

For agents relying on long-lived Python state (large datasets, opened files, initialized models), either prompt the agent to rebuild state at the start of each restored session, or keep the session in memory across the workload's lifetime.
