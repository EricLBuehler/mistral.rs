---
title: Performance
description: Get the most out of the hardware you have. Quantization, attention kernels, multi-GPU, and auto-tuning.
---

mistral.rs goes reasonably fast out of the box, but a few decisions can make a large difference for a specific workload. These guides cover the choices that usually matter most.

- [Pick a quantization method](/mistral.rs/guides/perf/pick-a-quantization/) if you want one model to run on a smaller card, or faster on the same card.
- [Let the tune command decide for you](/mistral.rs/guides/perf/auto-tune/) if you would rather not think about it.
- [Use flash attention](/mistral.rs/guides/perf/use-flash-attention/) on NVIDIA GPUs for faster attention kernels.
- [Use paged attention](/mistral.rs/guides/perf/use-paged-attention/) for higher concurrency and stable memory use.
- [Multi-GPU tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) to split a model across cards on the same box.
- [Multi-machine inference with the ring backend](/mistral.rs/guides/perf/multi-machine-ring/) for models too large for a single node.
- [Topology](/mistral.rs/guides/perf/topology/) for fine-grained per-layer placement.
- [Speculative decoding](/mistral.rs/guides/perf/speculative-decoding/) to pair a draft model with a target model for a throughput win.
- [UQFF for pre-quantized models](/mistral.rs/guides/perf/use-uqff/) when you want the loading speed of GGUF with the quality of our native quantization.

Concepts behind all of these (why paged attention helps, what quantization actually changes, what MLA does) live in the [Explanation](/mistral.rs/explanation/) section.
