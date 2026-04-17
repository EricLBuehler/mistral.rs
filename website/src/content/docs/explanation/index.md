---
title: Explanation
description: Concepts, design decisions, and the reasoning behind mistral.rs.
---

These pages explain how things work, not how to do them. Each one covers a piece of mistral.rs internals, the design tradeoffs, and the situations where a feature applies or does not.

For action-oriented documentation, see the other sections: [Tutorials](/mistral.rs/tutorials/) for end-to-end workflows, [Guides](/mistral.rs/guides/) for "how do I do X" answers, [Reference](/mistral.rs/reference/) for flag and API lookups.

## What is here

**Architecture.** Pipeline composition, request flow, and how the main loop interacts with model threads.

**The agentic loop.** Why tool-calling loops run on the server, what that enables, and when to bypass it.

**Session memory.** The splicing algorithm used to maintain multi-turn agent state across independent HTTP requests.

**Quantization tradeoffs.** Where each quantization method sits on the speed/quality/memory triangle.

**PagedAttention.** Block-based KV caching, the workloads it benefits, and the workloads it does not.

**Multi-head Latent Attention (MLA).** What DeepSeek's attention variant changes relative to standard attention, and what it costs.

**Device mapping.** How automatic device placement chooses a layout, and when to override it.

**The multimodal pipeline.** How pixels, video frames, and audio clips reach the model from an HTTP request.

**Code execution design.** Why the Python executor runs in a subprocess, how sessions are isolated, and the stdin/stdout protocol.
