---
title: Architecture
description: How the mistralrs engine is put together. Pipelines, the request loop, the threading model.
sidebar:
  order: 1
---

A walkthrough of what happens inside mistral.rs when a request arrives. Useful for reading the codebase, debugging unusual failures, and informed performance decisions.

## The three layers

From the outside in:

**The server layer.** HTTP endpoints, MCP endpoints, CORS, body limits, routing. Lives in `mistralrs-server-core`. Knows about HTTP, OpenAI wire formats, streaming. Does not know about model internals.

**The engine layer.** Request queue, scheduler, tool loop, session store. Lives in `mistralrs-core::engine`. Knows about pipelines and how to drive them; not about specific model architectures.

**The pipeline layer.** Model implementations, tokenization, quantization, attention kernels. One pipeline per model type, all conforming to a shared `Pipeline` trait. Lives in `mistralrs-core::pipeline`.

A request enters at the top, flows through the middle, and the middle calls into the bottom. The split allows piece-wise replacement: new model architectures touch only the pipeline layer; new API surfaces touch only the server layer.

## The Pipeline trait

Every supported model implements `Pipeline`. The trait is deliberately narrow — methods to encode prompts, generate tokens, apply ISQ, and report configuration. Transformer internals are not exposed. Implementations sharing backbone code delegate to helpers; the trait surface only requires the public shape.

This makes the engine layer architecture-agnostic. The scheduler asks a pipeline to work without knowing whether the underlying model is GGUF, a full checkpoint with ISQ, a LoRA adapter, or a speech model. Implementing the trait is sufficient.

## The engine thread

Startup spawns one engine thread per loaded pipeline. Requests enter through an `mpsc` channel; the thread drives the inner loop that turns requests into token generation.

A single engine thread is the unit of concurrency inside mistral.rs. Concurrent requests to the same model share a queue and are batched by the scheduler. Requests to different models go to different threads and run independently.

The engine thread has an outer `Request` loop consuming the channel and an inner token-generation loop cycling through active sequences. The inner loop handles batching, paged attention, and speculative decoding. The outer loop handles tool calling and session management.

## The scheduler

Each engine thread has a scheduler that decides which sequences to generate tokens for on each pass. Two goals: keep the GPU busy (batch as many sequences as possible), and respect fairness (no starvation).

Default scheduling is continuous batching: each decoding step includes every active sequence with a slot. With paged attention, a slot is a KV cache block rather than a full sequence's cache, so many more sequences can coexist than with contiguous cache.

For speculative decoding, the scheduler alternates drafting and verification passes. For MCP tool calls, the scheduler pauses a sequence during the tool run and resumes when results arrive.

## The tool loop

Server-side tool calling (`--enable-search`, `--enable-code-execution`, MCP) runs in the outer engine loop:

1. Request arrives with tool schemas.
2. Engine runs inference until the model emits a tool call.
3. Engine parses the call, looks up the tool, and runs it.
4. Tool output is appended to message history as a `tool` message.
5. Engine resumes inference with the updated history.
6. Loop until the model produces a non-tool-call response or the round cap is hit.

All of this happens inside one HTTP request. The client sees the final response plus an `agentic_tool_calls` array.

See the [agentic loop explanation](/mistral.rs/explanation/agentic-loop/) for more.

## The session store

Agentic requests are stateful. State lives in an in-memory store keyed by session id, with LRU eviction capped at 128 entries.

A session's state:

- Full message history, including server-generated tool calls and tool responses the client never sent.
- Images and videos passed in earlier turns.
- A handle to the Python code-execution subprocess, if any.

On each request, an explicit `session_id` triggers a session lookup and merges its state into the request. Without one, content-based matching is the fallback; if no match, a new session is created.

See the [session memory explanation](/mistral.rs/explanation/session-memory/) for the algorithm.

## Multi-model

`MultiModelBuilder` or a multi-model config loads several pipelines at once. Each gets its own engine thread. A top-level router dispatches HTTP requests to the right thread by `model` field.

Unloaded models remain in the router with a stopped engine thread. A request for an unloaded model triggers rehydration before dispatch.

## Where things live

For code readers:

- `mistralrs-core/src/engine/` — scheduler, tool loop, session store, request handlers.
- `mistralrs-core/src/pipeline/` — `Pipeline` trait and per-type implementations.
- `mistralrs-core/src/models/` — model-architecture-specific code (transformer blocks, attention variants).
- `mistralrs-core/src/vision_models/` — same for multimodal.
- `mistralrs-server-core/` — HTTP layer.
- `mistralrs-quant/` — ISQ implementations.
- `mistralrs-mcp/` — MCP client and server code.

[CLAUDE.md](https://github.com/EricLBuehler/mistral.rs/blob/master/CLAUDE.md) in the repo root is the up-to-date map.
