---
title: Session memory
description: How agentic session state is stored, matched, and reconciled with incoming requests.
sidebar:
  order: 3
---

Agentic sessions hold tool-call records, tool responses, and multimodal payloads from earlier turns. mistralrs stores this state in memory and reconciles it with each new request.

## Store

The session store is bounded:

- 128-session capacity, with least-recently-used eviction once exceeded.
- 30-minute idle TTL per session.
- Process memory only: sessions do not survive a server restart unless explicitly exported.

Each session holds:

- The full message history, including tool-role entries and synthesized assistant messages with tool calls.
- Multimodal payloads (images, videos) from earlier turns.
- A handle to the Python code-execution subprocess, if any.

## Matching

A request matches an existing session in one of two ways:

1. **Explicit `session_id`**, direct lookup.
2. **Content matching**, when no `session_id` is provided, the store searches for a session whose user-visible message prefix matches the incoming messages. The longest match wins. Tool-role entries in the stored session are skipped during comparison.

Content matching is the fallback for clients that cannot pass `session_id`. When two clients send identical opening messages, content matching can route them to the same session. Pass an explicit `session_id` in correctness-sensitive deployments.

## Splicing

On match, the engine merges the stored session's history with the incoming request so that:

- Tool-role entries and assistant-with-tool-calls entries from the stored history are preserved.
- User and assistant messages from the incoming request take precedence wherever they differ from the stored version.
- When the incoming messages diverge from the stored ones, the engine stops consuming stored history at the divergence point and appends the remaining incoming messages unchanged.

The effect: editing a previous turn works (the new content takes effect), while tool-call history from before the edit is retained.

Images and videos from the session are re-attached to the request after merging, and the request is upgraded to multimodal shape if it was plain-text.

## Post-turn save

At the end of a successful agentic turn, the expanded message list is written back to the session. Subsequent requests with the same id see the synthesized tool messages as part of history.

## Excluded from session state

- Sampling parameters. Each request specifies its own.
- Tool schemas. Taken from the current request's `tools` field or the server's configured built-in tools.
- The Python code-execution subprocess state. The handle travels with the in-memory session but is not serialized for export.

## Export and import

`GET /v1/sessions/{id}` returns a serialized session: messages, images, and videos. `PUT /v1/sessions/{id}` installs a serialized session under the given id. Use this to persist across restarts or move a session between servers.

## See also

- Guide: [persist sessions](/mistral.rs/guides/agents/persist-sessions/).
- Reference: [HTTP API `/v1/sessions/{id}`](/mistral.rs/reference/http-api/).
