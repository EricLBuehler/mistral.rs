---
title: Explanation
description: Concepts, design decisions, and the reasoning behind mistral.rs.
---

These pages are for when you want to understand how something works, not do something with it. They are essays rather than recipes. Most of them explain a piece of mistral.rs internals, the tradeoffs we made while building it, and the situations where a particular feature does or does not apply.

If you arrived here looking for a concrete action to take, one of the other sections is probably a better fit. [Tutorials](/mistral.rs/tutorials/) walk through full workflows end to end. [Guides](/mistral.rs/guides/) answer "how do I do X" questions. [Reference](/mistral.rs/reference/) is for looking up specific flags and APIs.

## What is here

**Architecture.** How pipelines are put together, how requests flow through the engine, and how the main loop talks to model threads.

**The agentic loop.** Why we run tool-calling loops on the server rather than the client, what that buys you, and when you would want to bypass it.

**Session memory.** The splicing algorithm mistral.rs uses to keep multi-turn agent state coherent across independent HTTP requests.

**Quantization tradeoffs.** Where each quantization method sits on the speed/quality/memory triangle, and why some of them exist when others are already "good enough."

**PagedAttention.** What block-based KV caching actually does, which workloads benefit from it, and which do not.

**Multi-head Latent Attention (MLA).** What DeepSeek's attention variant changes relative to standard attention, and what it costs.

**Device mapping.** How automatic device placement picks a layout across your GPUs, and when to override it manually.

**The multimodal pipeline.** How pixels, video frames, and audio clips get from an HTTP request all the way to the model.

**Code execution design.** Why the Python executor runs in a subprocess, how sessions are isolated, and what the stdin/stdout protocol looks like.
