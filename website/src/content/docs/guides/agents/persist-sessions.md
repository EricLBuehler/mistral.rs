---
title: Persist agent sessions
description: Keep agent state across HTTP requests via the sessions API.
sidebar:
  order: 6
---

Agentic requests on the HTTP server are stateful. State is keyed by session id with LRU eviction at 128 entries and a 30-minute idle TTL.

## Session id

Two cases:

- **Explicit `session_id` on the request** — the server looks it up. Existing session continues; missing id creates a new one.
- **No `session_id`** — a new id is created and returned in the response.

```bash
curl http://localhost:1234/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "default",
    "messages": [{"role": "user", "content": "What is 2 to the 10th?"}],
    "session_id": "user-42-chat-abc"
  }'
```

The response body includes a top-level `session_id` field.

## Exporting

```bash
curl http://localhost:1234/v1/sessions/user-42-chat-abc
```

Returns 404 if the session does not exist.

## Importing

```bash
curl -X PUT http://localhost:1234/v1/sessions/user-42-chat-abc \
  -H "Content-Type: application/json" \
  -d @saved-session.json
```

Body is a `SerializedSession` produced by a previous `GET`. Replaces any existing session with the same id.

## Deleting

```bash
curl -X DELETE http://localhost:1234/v1/sessions/user-42-chat-abc
```

Always returns 200 regardless of session existence.

## Lifetime

- **Idle expiry** — 30 minutes of inactivity.
- **Capacity** — 128-session cap with LRU eviction.
- **Server restart** — full loss.

## Code execution subprocess

Sessions with code execution hold a Python subprocess. The subprocess is not part of the exportable state. After import on another server, the new server starts a fresh subprocess on the next code execution call.

## SDK availability

Session export, import, delete, and list are HTTP-only. The Python and Rust SDKs do not expose equivalent methods on `Runner` / `Model`. To use sessions from those SDKs, run the HTTP server alongside.

`session_id` can be sent on a request via the SDKs (Python: not on `ChatCompletionRequest` directly; Rust: `RequestBuilder::with_session_id`).
