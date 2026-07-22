---
title: Architecture
description: How mistralrs is organized. Request flow, threading, and how pieces interact.
---

## The three layers

From the outside in:

**Server layer.** HTTP endpoints, [MCP (Model Context Protocol)](/mistral.rs/guides/agents/connect-mcp-server/) endpoints, CORS, body limits, routing. Knows about HTTP and OpenAI wire formats; does not know about model internals.

**Engine layer.** Request queue, scheduler, tool loop, session store. Drives pipelines without knowing about specific model architectures.

**Pipeline layer.** Model implementations, tokenization, quantization, attention kernels. One pipeline per model type, conforming to a shared trait.

Requests enter at the server layer and flow down. New model architectures touch only the pipeline layer; new API surfaces touch only the server layer.

## Engine threads

Startup spawns one engine thread per loaded model. Requests enter through a channel; the thread drives the loop that turns requests into token generation.

A single engine thread is the unit of concurrency. Concurrent requests to the same model share a queue and are batched by the scheduler. Requests to different models go to different threads and run independently. Multi-model routing and rehydration (reloading) of unloaded models are covered in [running multiple models](/mistral.rs/guides/serve/multiple-models/).

## Scheduling

Each engine thread has a scheduler that decides which sequences to generate tokens for on each pass. Default scheduling is continuous batching: every active sequence with an available slot is included on every decoding step. With [paged attention](/mistral.rs/guides/perf/paged-attention/), a slot is a KV cache block rather than a full sequence's cache, so many more sequences can coexist.

[Speculative decoding](/mistral.rs/guides/perf/speculative-decoding/) alternates drafting and verification passes. MCP tool calls pause a sequence during execution and resume it when the result arrives.

On CUDA, supported paged-attention decode steps can be replayed through [CUDA graphs](/mistral.rs/guides/perf/paged-attention/#cuda-graphs) by default. Graph capture lives in the pipeline layer; the scheduler still selects the active sequences normally.

## Tool loop and sessions

Server-side tool calling runs in the outer engine loop, inside one HTTP request:

1. Run inference until a tool call.
2. Execute the tool.
3. Append the result to history.
4. Resume.

Repeat until the model returns a non-tool-call response or the round cap is hit. Loop semantics, entry conditions, and configuration live in [agentic runtime for apps](/mistral.rs/guides/agents/agentic-runtime/).

Agentic requests are stateful; state lives in an in-memory session store. Matching, splicing, and eviction are documented in [session memory](/mistral.rs/developer/session-memory/), the user-facing workflow in [persist sessions](/mistral.rs/guides/agents/persist-sessions/).

## See also

- [Session memory](/mistral.rs/developer/session-memory/).
- [cuTile setup](/mistral.rs/developer/moe-backends/).
- [The multimodal pipeline](/mistral.rs/developer/multimodal-pipeline/).
