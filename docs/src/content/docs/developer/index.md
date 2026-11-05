---
title: Developer Guide
description: Internals and contributor documentation for mistral.rs.
---

How mistral.rs works inside:

- [Inference runtime](/developer/inference-runtime/) - scheduling, cache budgets, and accelerator execution.
- [Architecture](/developer/architecture/) - how the server, engine, and pipeline layers fit together.
- [Quantized execution](/developer/quantized-execution/) - FP8/FP4 backend behavior and kernel benchmarks.
- [cuTile setup](/developer/moe-backends/) - install optional cuTile acceleration on supported NVIDIA GPUs.
- [Multimodal pipeline](/developer/multimodal-pipeline/) - how image, video, and audio inputs reach the model.
- [Session memory](/developer/session-memory/) - how agentic conversation state is stored and matched.
- [Build from source](/developer/from-source/) - local build setup and feature flags.
