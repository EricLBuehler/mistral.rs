---
title: Performance
description: Get the most out of the hardware you have. Quantization, attention kernels, multi-GPU, and auto-tuning.
---

Guides for tuning throughput, memory, and latency.

## Choose by constraint

| If you need to... | Start here |
|---|---|
| Fit a model into less memory | [Pick a quantization method](/mistral.rs/guides/perf/pick-a-quantization/) |
| Let mistral.rs benchmark the host | [Let the tune command decide for you](/mistral.rs/guides/perf/auto-tune/) |
| Improve attention throughput on NVIDIA GPUs | [Use flash attention](/mistral.rs/guides/perf/use-flash-attention/) |
| Improve high-concurrency serving memory use | [Use paged attention](/mistral.rs/guides/perf/use-paged-attention/) |
| Split one model across local GPUs | [Multi-GPU tensor parallelism](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) |
| Split one model across machines | [Multi-machine inference with the ring backend](/mistral.rs/guides/perf/multi-machine-ring/) |
| Place layers manually | [Topology](/mistral.rs/guides/perf/topology/) |
| Reduce decode latency with MTP | [Speculative decoding](/mistral.rs/guides/perf/speculative-decoding/) |
| Use Gemma 4 assistant checkpoints for MTP | [Gemma 4 MTP](/mistral.rs/guides/perf/gemma4-mtp/) |
| Save an ISQ result for faster reloads | [UQFF for pre-quantized models](/mistral.rs/guides/perf/use-uqff/) |

Underlying concepts (paged attention design, what quantization changes, MLA) live in the [Explanation](/mistral.rs/explanation/) section.
