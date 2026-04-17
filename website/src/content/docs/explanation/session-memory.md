---
title: Session memory
description: How mistralrs keeps multi-turn agent state coherent across independent HTTP requests. The splicing algorithm, content matching, and edge cases.
sidebar:
  order: 3
---

A multi-turn agent conversation has more state than the client can see. The client sees the messages it sent plus the final reply. The server saw all of that plus every tool call, every tool response, every subprocess state change, and every image or video flowing through.

When the next request arrives, the client cannot send all of that back. The server has to remember it and merge it back in. This page covers how that merging works.

## The naive approach does not work

The obvious design: on each request, look up the previous session by id, take its full message history as the conversation, and add only new messages from the client.

This fails for several reasons.

**Clients cannot always pass a session id.** Older OpenAI SDKs predate the concept. Requiring a session id breaks compatibility with existing client code.

**Clients may legitimately edit older messages.** A client that wants to revise a previous turn should be able to. Blindly overwriting client messages with server-stored versions defeats this.

**Clients and servers may disagree about the conversation.** On client retry or reorder, the server's stored history may not match what the client thinks happened.

The right approach merges the two views, keeping the server's extra information (tool calls, images) while respecting the client's ground truth on user and assistant messages.

## The splicing algorithm

On every request, the server has two sequences:

- **Incoming** — messages from the request body. User and assistant only; clients never include tool messages.
- **Stored** — messages from the matched session. User, assistant, and tool messages interleaved.

Goal: produce a merged sequence including the stored tool messages at the right positions, ending with the client's new messages. The algorithm:

1. Walk both sequences in parallel.
2. For each stored message, check if it is tool-related (role=tool, or assistant with tool_calls).
3. If so, splice it into output and advance the stored pointer only.
4. Otherwise (user or assistant), check if it matches the next incoming message (same role, same content).
5. On match, include and advance both pointers.
6. On divergence, the conversation has changed from the stored version. Stop splicing and append the remaining incoming messages as-is.

The output interleaves stored tool messages with fresh incoming messages. Client edits to older messages take precedence.

## Content matching for clients without session ids

For clients that cannot pass a session id, content matching is the fallback. The engine scans stored sessions for one whose user-visible messages match the incoming request as a prefix. A match: every stored user or assistant message equals its corresponding incoming message, and the stored session has no extra user-visible messages beyond the incoming.

On match, that session is used. Multiple matches resolve by longest match (most specific). No match creates a fresh session.

This is reliable for the common case of a single user running a coherent conversation. It can misbehave when two users send identical opening messages to the same server, possibly matching the same stored session. For correctness-critical deployments, pass an explicit session id.

## Images and videos

Multimodal messages carry binary payloads not practical to embed in message history as strings. mistral.rs stores them separately in the session, indexed positionally with the messages.

On splice, the engine attaches any session images or videos to the request before handing it to the pipeline. Transparent to the client: an image sent in turn 1 is still available in turn 3.

## The Python subprocess

Code-execution sessions hold a reference to a Python subprocess. On splice, the subprocess is reused as-is — not cloned or serialized. Subsequent code executions in the session see the same globals, imports, and file handles.

On export-and-import across server restarts, the subprocess does not transfer. The new server starts a fresh subprocess on the next code execution. Python state from the old subprocess is gone.

A deliberate tradeoff. Serializing a running Python subprocess is hard and error-prone; most workloads can rebuild necessary state on demand; persistent-state workloads can keep the session in memory on the original server.

## Lifetime

Sessions are in-memory. Three terminators:

- **Idle expiry** — 30 minutes of inactivity.
- **Capacity** — 128-session cap with LRU eviction.
- **Server restart** — full loss.

For cross-restart persistence, the [export and import endpoints](/mistral.rs/guides/agents/persist-sessions/) serialize sessions to JSON. The serialized form includes message history and multimodal payloads, not the subprocess.

## What the session does not include

- Sampling parameters from previous requests. Each request specifies independently.
- Tool schemas. Taken from the current request's `tools` field or the server's configured built-in tools.
- User identity. Session ids are opaque strings; mistral.rs has no user concept.
- Context outside the chat message sequence. System prompts live in messages — send them in every request to persist.

## Design consequences

**Sessions are a performance and coherence tool, not a security boundary.** Two requests with the same session id share state. Guessable or leaking session ids cause cross-conversation contamination. Treat session ids as secrets.

**Editing old messages is not surprising.** To rewrite a previous turn (e.g., a UI "edit" button), send the new version. The splice algorithm respects it.

**Forking a conversation needs a new session id.** To continue from an old state while preserving the original, either export and import under a new id, or use a new session id and send the full history on the first request.

## When splicing can be wrong

The algorithm is conservative: when incoming diverges from stored, it stops splicing at the divergence point and appends incoming unchanged. Tool-call history from before the divergence remains, which is usually what is wanted.

If a client sends an abbreviated history (only the latest turn), the algorithm sees immediate divergence and splices nothing. The result is a fresh conversation reusing a session id.

This is rarely what the client wants. In practice, clients passing session ids also send full message history, so the case is uncommon. Clients sending abbreviated histories should not pass session ids.
