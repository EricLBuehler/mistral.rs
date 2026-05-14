---
title: Explanation
description: Concepts, design decisions, and the reasoning behind mistral.rs.
---

Explanation pages cover mistral.rs internals and design tradeoffs.

For action-oriented documentation: [Tutorials](/mistral.rs/tutorials/), [Guides](/mistral.rs/guides/), [Reference](/mistral.rs/reference/).

## Contents

**Architecture.** Pipeline composition, request flow, and how the main loop interacts with model threads.

**The agentic loop.** Why tool-calling loops run on the server, what that enables, and when to bypass it.

**Session memory.** The splicing algorithm used to maintain multi-turn agent state across independent HTTP requests.

**Quantization tradeoffs.** Where each quantization method sits on the speed/quality/memory triangle.

**PagedAttention.** Block-based KV caching, the workloads it benefits, and the workloads it does not.

**Multi-head Latent Attention (MLA).** What DeepSeek's attention variant changes relative to standard attention, and what it costs.

**Device mapping.** How automatic device placement chooses a layout, and when to override it.

**The multimodal pipeline.** How pixels, video frames, and audio clips reach the model from an HTTP request.

**Code execution design.** Why the Python executor runs in a subprocess, how sessions are isolated, and the stdin/stdout protocol.
