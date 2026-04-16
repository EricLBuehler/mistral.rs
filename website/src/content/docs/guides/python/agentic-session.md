---
title: Agentic sessions from Python
description: Keep multi-turn agent state alive across calls. Export, import, and delete sessions using the Python SDK.
sidebar:
  order: 3
---

Sessions are how mistralrs keeps agent state coherent across multiple calls: message history, tool-call records, and the Python code-execution subprocess if you have that turned on. The [persist-sessions guide](/mistral.rs/guides/agents/persist-sessions/) covers the concept; this page covers the Python-specific mechanics.

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

# Second call continues the same session. Tool history, message history, and
# any loaded search context carry over.
response2 = runner.send_chat_completion_request(
    ChatCompletionRequest(
        model="Qwen/Qwen3-4B",
        messages=[{"role": "user", "content": "Summarize what you found."}],
        session_id=session_id,
    )
)
```

You do not have to start with a generated id. Pass any string and mistralrs will create a new session under that name if one does not already exist.

## Export and import

The Python Runner exposes the same export and import endpoints the HTTP API does:

```python
# Capture current state.
serialized = runner.export_session(session_id)
# It is a JSON-serializable dict; save however you like.

import json
with open("session.json", "w") as f:
    json.dump(serialized, f)

# Later, in a different process:
with open("session.json") as f:
    serialized = json.load(f)

runner.import_session("new-session-id", serialized)
# Now requests against "new-session-id" pick up where the original left off.
```

The serialized object is a plain Python dict that is safe to pickle, store in a database, or send over the wire. Images and videos inside the session are base64-encoded in the serialization.

## Deleting a session

```python
runner.delete_session(session_id)
```

This frees the memory held by that session. If there is a Python code-execution subprocess associated with the session, it is terminated. Subsequent requests against the deleted id will create a new, empty session.

## Listing sessions

```python
ids = runner.list_session_ids()
for sid in ids:
    print(sid)
```

Useful for building admin UIs or for bookkeeping in long-running applications.

## When to persist and when not to

Keep sessions in memory when:

- The conversation is ephemeral (a chat UI where users can refresh and start over).
- You can tolerate losing the state on a server restart.

Export and persist sessions to disk or a database when:

- The conversation spans a long time and users expect to come back to it.
- Multiple servers need to be able to continue the conversation.
- You need audit trails of agent behavior.

A common pattern for longer-lived applications: maintain a mapping from user-visible conversation ids to mistralrs session ids, export-and-store sessions on a timer or after significant events, and restore them on demand. The mistralrs side of that is just the three methods above.

## Interaction with code execution

When code execution is enabled and a session has an active Python subprocess, that subprocess stays alive as long as the session is in memory. Exporting a session does not export the subprocess state; the new server that imports the session will start a fresh subprocess the first time code execution runs in that session.

If your agent relies on long-lived Python state (large loaded datasets, opened files, initialized models), this matters. Either prompt the agent to rebuild state at the start of each restored session, or keep the session in memory across the entire life of the workload.
