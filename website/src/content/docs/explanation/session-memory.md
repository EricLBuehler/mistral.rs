---
title: Session memory
description: How mistralrs keeps multi-turn agent state coherent across independent HTTP requests. The splicing algorithm, content matching, and edge cases.
sidebar:
  order: 3
---

A multi-turn agent conversation has more state than the client can see. The client sees the messages it sent plus the final reply. The server saw all of that plus every tool call, every tool response, every subprocess state change, and every image or video that flowed through in either direction.

When the next request arrives, the client cannot send all of that back because it does not have it. The server has to remember it and merge it back in. This page is about how that remembering and merging work.

## The naïve approach does not work

The obvious design is: on each request, look up the previous session by session id, take its full message history, and use that as the conversation. Send the client's incoming messages only for the new turn.

This fails for a few reasons.

**Clients cannot always pass a session id.** Older OpenAI SDKs predate the concept. If you want mistralrs to work with existing client code, you cannot require that they send one.

**The client may legitimately change older messages.** A client that wants to edit a previous turn should be able to. If we blindly overwrite their messages with ours, that edit does not take effect.

**Clients and servers may disagree about the conversation.** If the client retries or reorders, the server's stored history may no longer match what the client thinks happened.

The right approach is to merge the two views of the conversation, keeping the server's extra information (tool calls, images) while respecting the client's ground truth about user and assistant messages.

## The splicing algorithm

On every request, the server has two sequences:

- **Incoming**: the messages the client sent in the request body. These are user and assistant messages only; clients never include tool-role messages.
- **Stored**: the messages from the matched session. These include user, assistant, and tool messages interleaved.

The job is to produce a merged sequence that includes the stored tool messages at the right positions and ends with the client's new messages. The algorithm:

1. Walk the stored sequence and the incoming sequence in parallel.
2. For each stored message, check if it is a tool-related message (role=tool, or an assistant message with tool_calls).
3. If it is, splice it into the merged output and advance the stored pointer only.
4. If it is a user or assistant message, check if it matches the next incoming message (same role, same content).
5. If they match, include the message and advance both pointers.
6. If they diverge, the conversation has changed from the stored version. Stop splicing and append the remaining incoming messages as-is.

The output is a single sequence where tool messages from the stored session are interleaved with the fresh incoming messages, but client-provided changes to older messages take precedence.

## Content matching for clients without session ids

For clients that cannot pass a session id, we fall back to content matching. The engine scans the stored sessions for one whose user-visible messages match the incoming request as a prefix. A match means: every stored user or assistant message equals its corresponding incoming message, and the stored session has no user-visible messages beyond what the incoming has.

If a match is found, that session is used. If several sessions could match, the longest match wins (most specific). If nothing matches, a fresh session is created.

This works reliably for the common case where a single user runs a coherent conversation. It can misbehave if two users send the same opening messages to the same server, because both might match the same stored session. For correctness-critical deployments, passing an explicit session id is the safe choice.

## Images and videos

Multimodal messages carry binary payloads that are not practical to embed in the message history as strings. mistralrs stores them separately in the session, indexed positionally with the messages.

On splice, if the merged session has any images or videos, the engine attaches them to the request before handing it to the pipeline. This is transparent to the client: as far as they are concerned, they sent an image once in turn 1 and now turn 3 behaves as if the image is still there.

## The Python subprocess

Code-execution sessions hold a reference to a Python subprocess. On splice, this subprocess is not cloned or serialized; it is reused as-is. That means subsequent code executions in the same session see the same globals, imports, and file handles the earlier ones left behind.

On export-and-import across server restarts, the subprocess does not come with. The new server starts a fresh subprocess when code execution first runs in the restored session. Python state from the old subprocess is gone.

This is a deliberate tradeoff. Serializing a running Python subprocess is hard and error-prone; most workloads can rebuild necessary state on demand; and the ones that cannot can keep the session in memory on the original server.

## Lifetime

Sessions live in memory. Three things end a session:

- **Idle expiry**: 30 minutes of inactivity.
- **Capacity**: we keep at most 128 sessions in memory; beyond that, least-recently-used gets evicted.
- **Server restart**: everything is gone.

For persistence across server lifetimes, the [export and import endpoints](/mistral.rs/guides/agents/persist-sessions/) let you serialize a session to JSON and restore it later. That serialized form includes the message history and multimodal payloads but not the subprocess.

## What the session does not include

For clarity, a session does not store:

- Sampling parameters from previous requests. Every request specifies these independently.
- Tool schemas. These are taken from the current request's `tools` field or from the server's configured built-in tools.
- User identity. mistralrs does not have a concept of users; session ids are opaque strings.
- Any context outside the chat message sequence. System prompts live in the messages; if you need them to persist across requests, send them in every request.

## Design consequences

The splicing approach has a few implications for how you should think about sessions:

**Sessions are a performance and coherence tool, not a security boundary.** Two requests with the same session id share state. If session ids are guessable or leak between users, conversations can cross. Treat session ids as secrets.

**Editing old messages is not surprising.** If you want to rewrite a previous turn (common pattern: a UI "edit" button), just send the new version. The splice algorithm respects it.

**Forking a conversation needs a new session id.** If you want to continue from an old state while also preserving the original thread, either export the session and import it under a new id, or just use a new session id and send the full history on the first request.

## When splicing can be wrong

The algorithm is conservative: when the incoming messages diverge from the stored ones, it stops splicing at the divergence point and appends incoming messages unchanged. This means some tool-call history from the stored session can end up strictly before the divergence, which is usually what you want.

But if a client sends an abbreviated history (only the latest turn, no earlier context), the algorithm sees this as "the conversation diverged immediately" and does not splice anything in. The effect is a fresh conversation that happens to reuse a session id.

That is almost never what the client wants. In practice, clients that pass session ids also send the full message history, so this case does not come up. Clients that send abbreviated histories should not pass session ids either.
