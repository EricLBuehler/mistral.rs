---
title: Persist agent sessions
description: Keep agent state (messages, tool history, code execution subprocess) alive across independent HTTP requests.
sidebar:
  order: 6
---

Agent conversations accumulate state: message history, prior tool calls, Python variables in code-execution sessions. mistralrs holds this state in memory keyed by session id and makes it available to any request that supplies the id.

This guide covers session matching, export and import (for restart and migration), and session lifetime.

## Session identification

Every agentic request has a session id. Two cases:

**No id passed.** The server generates a fresh UUID, runs the request against an empty state, and returns the new id (as `session_id` in the JSON body, and in the final streaming chunk).

**Explicit id passed.** The server looks it up. If found, the request continues from prior state. If not found, a new session is created under that id.

Example:

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2 to the 10th?"}],
    "session_id": "user-42-chat-abc"
  }'
```

## Implicit matching for clients without ids

Clients that cannot pass `session_id` (older OpenAI SDKs, etc.) fall back to message-prefix matching. If the messages match the start of an existing session's user-visible history, that session is picked up automatically.

This is transparent in most cases. The failure mode: two users sending identical opening messages may match the same session. Pass the session id explicitly when correctness matters.

## Exporting a session

The GET endpoint serializes a session to JSON, including message history, tool calls, and any images or videos:

```bash
curl http://localhost:1234/v1/sessions/user-42-chat-abc
```

The response is a `SerializedSession` with full state. Save it or transfer it elsewhere.

## Importing a session

The PUT endpoint accepts a serialized session and installs it under the given id:

```bash
curl -X PUT http://localhost:1234/v1/sessions/user-42-chat-abc \
  -H "Content-Type: application/json" \
  -d @saved-session.json
```

Subsequent requests with that id pick up from the imported state. This is the primitive for moving sessions between servers, post-restart restoration, and conversation handoff.

## Deleting sessions

```bash
curl -X DELETE http://localhost:1234/v1/sessions/user-42-chat-abc
```

Always returns 200 regardless of session existence.

## Lifetime

Sessions are in-memory. Three terminators:

- **Idle expiry** — sessions untouched for 30 minutes are evicted.
- **Capacity** — at most 128 sessions; the 129th evicts the oldest.
- **Server restart** — memory wipes on process exit. Export before shutdown for persistence.

Long-lived persistence across restarts uses the export-and-restore pattern: a background job exports active sessions periodically and restores them on startup. There is no first-party database integration.

## What the stored state contains

A session holds:

- Message history, including server-generated tool calls and tool responses the client never explicitly sent.
- Images and videos attached to past messages.
- A pointer to the Python code-execution subprocess, if one exists.

The Python subprocess is not exportable. After import on another server, the new server starts a fresh subprocess on first code-execution call in the restored session. Variables and imports from the original session are gone.

Models can usually rebuild Python state on request; prompt for re-import if needed.

## Sessions and the Python SDK

The Python SDK works the same way. `session_id` on `ChatCompletionRequest` accepts the same string. The runner exposes `export_session`, `import_session`, and `delete_session` mirroring the HTTP endpoints. See the [Python agentic session guide](/mistral.rs/guides/python/agentic-session/).
