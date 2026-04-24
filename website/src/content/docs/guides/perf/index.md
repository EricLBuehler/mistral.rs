---
title: Performance
description: Get the most out of the hardware you have. Quantization, attention kernels, multi-GPU, and auto-tuning.
---

Guides for tuning throughput, memory, and latency.

- [Pick a quantization method](/mistral.rs/guides/perf/pick-a-quantization/): fit a model on a smaller card or run faster on the same one.
- [Let the tune command decide for you](/mistral.rs/guides/perf/auto-tune/): automated benchmarking.
- [Use flash attention](/mistral.rs/guides/perf/use-flash-attention/): faster attention kernels on NVIDIA GPUs.
- [Use paged attention](/mistral.rs/guides/perf/use-paged-attention/): higher concurrency and stable memory use.
- [Multi-GPU tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/): split a model across cards on one host.
- [Multi-machine inference with the ring backend](/mistral.rs/guides/perf/multi-machine-ring/): for models too large for a single node.
- [Topology](/mistral.rs/guides/perf/topology/): fine-grained per-layer placement.
- [Speculative decoding](/mistral.rs/guides/perf/speculative-decoding/): pair a draft model with a target model.
- [UQFF for pre-quantized models](/mistral.rs/guides/perf/use-uqff/): serialized ISQ output for fast reload.

Underlying concepts (paged attention design, what quantization changes, MLA) live in the [Explanation](/mistral.rs/explanation/) section.
