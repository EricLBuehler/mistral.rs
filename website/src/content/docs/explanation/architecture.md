---
title: Architecture
description: How mistralrs is organized. Request flow, threading, and how pieces interact.
sidebar:
  order: 1
---

## The three layers

From the outside in:

**Server layer.** HTTP endpoints, MCP endpoints, CORS, body limits, routing. Knows about HTTP and OpenAI wire formats; does not know about model internals.

**Engine layer.** Request queue, scheduler, tool loop, session store. Drives pipelines without knowing about specific model architectures.

**Pipeline layer.** Model implementations, tokenization, quantization, attention kernels. One pipeline per model type, conforming to a shared trait.

Requests enter at the server layer and flow down. New model architectures touch only the pipeline layer; new API surfaces touch only the server layer.

## Engine threads

Startup spawns one engine thread per loaded model. Requests enter through a channel; the thread drives the loop that turns requests into token generation.

A single engine thread is the unit of concurrency. Concurrent requests to the same model share a queue and are batched by the scheduler. Requests to different models go to different threads and run independently.

## Scheduling

Each engine thread has a scheduler that decides which sequences to generate tokens for on each pass. Default scheduling is continuous batching: every active sequence with an available slot is included on every decoding step. With paged attention, a slot is a KV cache block rather than a full sequence's cache, so many more sequences can coexist.

Speculative decoding alternates drafting and verification passes. MCP tool calls pause a sequence during execution and resume it when the result arrives.

## Tool loop

Server-side tool calling runs in the outer engine loop:

1. Request arrives with tool schemas.
2. The engine runs inference until the model emits a tool call.
3. The engine runs the tool.
4. The tool output is appended to message history.
5. The engine resumes inference with the updated history.
6. The loop repeats until the model produces a non-tool-call response or the round cap is hit.

All of this happens inside one HTTP request. See [the agentic loop](/mistral.rs/explanation/agentic-loop/).

## Session store

Agentic requests are stateful. State lives in an in-memory store keyed by session id, with a 128-session cap and 30-minute idle TTL.

On each request, an explicit `session_id` triggers a session lookup. Without one, a content-based match on user-visible messages is the fallback; if nothing matches, a new session is created.

See [session memory](/mistral.rs/explanation/session-memory/) for the merging behavior.

## Multi-model

A multi-model config loads several pipelines at once. Each gets its own engine thread. A top-level router dispatches HTTP requests to the right thread by `model` field.

Unloaded models remain in the router with a stopped engine thread. A request for an unloaded model triggers rehydration before dispatch.

## See also

- [The agentic loop](/mistral.rs/explanation/agentic-loop/).
- [Session memory](/mistral.rs/explanation/session-memory/).
- [PagedAttention](/mistral.rs/explanation/paged-attention/).
- [Device mapping](/mistral.rs/explanation/device-mapping/).
