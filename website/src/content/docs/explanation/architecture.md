---
title: Architecture
description: How the mistralrs engine is put together. Pipelines, the request loop, the threading model.
sidebar:
  order: 1
---

This page is a tour of what happens inside mistralrs when you send a request. It is not a tutorial or a reference; it exists so that reading the codebase, debugging an unusual failure, or making an informed performance decision is easier.

## The three layers

From the outside in:

**The server layer.** HTTP endpoints, MCP endpoints, CORS and body limits, routing. Lives in `mistralrs-server-core`. Knows about HTTP, OpenAI wire formats, and streaming. Does not know about model internals.

**The engine layer.** Request queue, scheduler, tool loop, session store. Lives in `mistralrs-core::engine`. Knows about pipelines and how to drive them but does not know about specific model architectures.

**The pipeline layer.** Model implementations, tokenization, quantization, attention kernels. One pipeline per model type, all conforming to a shared `Pipeline` trait. Lives in `mistralrs-core::pipeline`.

A request enters the top layer, flows through the middle, and the middle orchestrates calls into the bottom. The three-layer split lets us swap pieces without cascading changes: adding a new model architecture touches only the pipeline layer; adding a new API surface touches only the server layer.

## The Pipeline trait

Every supported model implements `Pipeline`. The trait is deliberately narrow: it exposes methods to encode prompts, generate tokens, apply ISQ, and report configuration. It does not expose the transformer internals directly. Implementations that share backbone code (most of them) delegate to helper modules; what the trait surface demands is just the public shape.

This is what makes the engine layer architecture-agnostic. The scheduler asks a pipeline to do work without caring whether the model underneath is a GGUF file, a full checkpoint with ISQ, a LoRA-adapted model, or a speech model. As long as the pipeline implements the trait, it plugs in.

## The engine thread

When mistralrs starts, it spawns a dedicated engine thread per loaded pipeline. Requests are sent into this thread through an `mpsc` channel; the thread drives the inner loop that turns requests into token generation.

A single engine thread is the unit of concurrency inside mistralrs. Concurrent requests to the same model go into the same thread's queue and are batched by the scheduler. Requests to different models go to different threads and run independently.

The engine thread itself has an outer `Request` loop that consumes the channel, and an inner token-generation loop that cycles through active sequences. The inner loop is where batching, paged attention, and speculative decoding all happen; the outer loop is where tool calling and session management happen.

## The scheduler

Inside each engine thread, a scheduler decides which sequences to generate tokens for on each pass. The scheduler has two jobs: keep the GPU busy (batch as many sequences as possible), and respect fairness (do not starve any one request).

Default scheduling is "continuous batching": on every decoding step, include every active sequence that has a slot. With paged attention, a slot is a block of KV cache rather than a whole sequence's worth, so many more sequences can coexist than with contiguous cache.

For speculative decoding, the scheduler alternates drafting and verification passes. For MCP tool calls, the scheduler pauses a sequence while the tool runs, then resumes it when the result arrives.

## The tool loop

Server-side tool calling (`--enable-search`, `--enable-code-execution`, MCP) runs inside the outer engine loop. The flow:

1. Request arrives with tool schemas attached.
2. Engine runs inference until the model emits a tool call.
3. Engine parses the tool call, looks up which tool was named, and runs it.
4. Tool output is appended to the message history as a `tool` message.
5. Engine resumes inference with the updated history.
6. Loop until the model produces a non-tool-call response or the round cap is hit.

All of this happens inside a single HTTP request from the client's point of view. The client sees the final response plus an `agentic_tool_calls` array documenting what happened.

The explanation page on [the agentic loop](/mistral.rs/explanation/agentic-loop/) goes into this in more detail.

## The session store

Agentic requests are stateful. mistral.rs keeps that state in an in-memory store keyed by session id, with an LRU eviction policy capped at 128 entries.

A session's state includes:

- The full message history, including tool call and response messages that never appeared in the client's request.
- Images and videos that have been passed through in earlier turns.
- A handle to the Python code-execution subprocess, if any.

On each request, if the client provides a `session_id`, the engine looks up the matching session and merges its state into the request. If the client does not provide one, the engine uses content-based matching as a fallback and, failing that, creates a new session.

The [session memory explanation](/mistral.rs/explanation/session-memory/) has the algorithm details.

## Multi-model

With `MultiModelBuilder` or a multi-model config file, mistralrs loads several pipelines at once. Each gets its own engine thread. A top-level router receives HTTP requests and dispatches them to the right thread based on the request's `model` field.

Unloaded models are still represented in the router but their engine thread is not running. When a request arrives for an unloaded model, the router rehydrates it before dispatching.

## Where things live

For people reading the code:

- `mistralrs-core/src/engine/` has the scheduler, tool loop, session store, and request handlers.
- `mistralrs-core/src/pipeline/` has the `Pipeline` trait and the implementations per model type.
- `mistralrs-core/src/models/` has the model-architecture-specific code (transformer blocks, attention variants).
- `mistralrs-core/src/vision_models/` is the same but for multimodal models.
- `mistralrs-server-core/` has the HTTP layer.
- `mistralrs-quant/` has ISQ implementations.
- `mistralrs-mcp/` has the MCP client and server code.

The [CLAUDE.md](https://github.com/EricLBuehler/mistral.rs/blob/master/CLAUDE.md) file in the repo root is an up-to-date summary of where to look for specific changes.
