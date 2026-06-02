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
| Reduce CUDA decode launch overhead | [Use CUDA graphs](/mistral.rs/guides/perf/use-cuda-graphs/) |
| Compare multi-GPU and distributed modes | [Multi-GPU and distributed inference](/mistral.rs/guides/perf/multi-gpu-distributed/) |
| Split one model across local GPUs | [Single-machine multi-GPU](/mistral.rs/guides/perf/multi-gpu-tensor-parallel/) |
| Run NCCL across machines | [Multi-node NCCL inference](/mistral.rs/guides/perf/multi-node-nccl/) |
| Use the ring backend | [Ring backend inference](/mistral.rs/guides/perf/multi-machine-ring/) |
| Place layers manually | [Topology](/mistral.rs/guides/perf/topology/) |
| Reduce decode latency with MTP | [Speculative decoding](/mistral.rs/guides/perf/speculative-decoding/) |
| Use Gemma 4 assistant checkpoints for MTP | [Gemma 4 MTP](/mistral.rs/guides/perf/gemma4-mtp/) |
| Save an ISQ result for faster reloads | [UQFF for pre-quantized models](/mistral.rs/guides/perf/use-uqff/) |

Underlying concepts (paged attention design, what quantization changes, MLA) live in the [Explanation](/mistral.rs/explanation/) section.
